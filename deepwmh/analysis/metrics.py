##
## something about evaluation metrics and utilities used during evaluation process.
##

from typing import Union
from deepwmh.utilities.plot import PlotCanvas
import random
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from copy import deepcopy
from deepwmh.utilities.data_io import SimpleExcelWriter, SimpleExcelReader, load_nifti, load_nifti_simple, save_nifti, targz_compress
from deepwmh.utilities.file_ops import cp, dir_exist, file_exist, join_path, ls, mkdir, gd
from deepwmh.analysis.image_ops import connected_components
from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.misc import contain_duplicates, ignore_SIGINT, minibar, printi
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy.interpolate
import numpy as np
import statsmodels.api as sm


def hard_dice_binary(y_true: np.ndarray, y_pred: np.ndarray):
    # calculating hard Dice for binary segmentation labels x and y.
    assert y_true.shape == y_pred.shape, 'shapes %s, %s are not match.' % (str(y_true.shape),str(y_pred.shape))
    # hard label
    y_true = (y_true>0.5).astype('float32')
    y_pred = (y_pred>0.5).astype('float32')
    return 2 * np.sum(y_true*y_pred) / ( np.sum(y_true) + np.sum(y_pred) + 0.000001 )

def voxel_precision_recall(y_true: np.ndarray, y_pred: np.ndarray):
    yt_binary = (y_true > 0.5)
    yp_binary = (y_pred > 0.5)
    tp = np.sum((yt_binary == True) * (yp_binary == True))
    tn = np.sum((yt_binary == False) * (yp_binary == False))
    fp = np.sum((yt_binary == False) * (yp_binary == True))
    fn = np.sum((yt_binary == True) * (yp_binary == False))
    # calculate metrics
    ppv = tp / (tp+fp) # precision
    tpr = tp / (tp+fn) # recall
    return ppv, tpr

def inst_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    _, yt_part = connected_components(y_true)
    _, yp_part = connected_components(y_pred)
    tp, fp, fn = 0, 0, 0
    for i in range( np.max(yt_part) ):
        if np.sum((yt_part == i) * yp_part) > 0:
            tp += 1 # gt has and pred has
        else:
            fn += 1 # gt has but pred hasn't
    for j in range( np.max(yp_part) ):
        if np.sum((yp_part == j) * y_true) < 1:
            fp += 1 # pred has but gt hasn't
    # tp + fn should equal with np.max(y_true)
    smooth = 1e-6
    tpr = tp / (tp+fn+smooth) # recall
    # fnr = fn / (tp+fn+smooth)
    ppv = tp / (tp+fp+smooth) # positive predictive value, aka precision
    precision = ppv
    recall = tpr
    #
    f1 = 2*recall*precision/(recall+precision+smooth)
    # For a classifier that is really really bad, the F1 score can 
    # be very close to zero (but never be zero in theory). In this 
    # case, we simply set it to zero to simplify calculation. Since
    # the F1 score is usually the final evaluation result for most
    # of the evaluation tasks, such approximation will not affect
    # much. 
    return precision,recall,f1

def instance_F1(y_true: np.ndarray, y_pred: np.ndarray):
    return inst_confusion_matrix(y_true, y_pred)[2]

def _parallel_paired_evaluation(params):
    subject, eval_function, file_A, file_B = params
    data_A = load_nifti_simple(file_A) if file_A is not None else None
    data_B = load_nifti_simple(file_B) if file_B is not None else None
    if data_A is None:
        data_A = np.zeros_like(data_B)
    elif data_B is None:
        data_B = np.zeros_like(data_A)
    if data_A.shape != data_B.shape:
        raise RuntimeError('subject "%s" - shapes not equal: "%s", "%s".' % (subject, file_A, file_B))
    metric = eval_function(data_A, data_B)
    return (subject, metric)

###
### Base Evaluator Class
###

class PairedEvaluation(object):
    '''
    Description
    --------------
    Base template class for all paired evaluators.

    Usage
    --------------
    * collect all patient names as a single list 

    * define eval_function

    >>> def f(y1: np.ndarray, y2: np.ndarray): -> Any
    >>>     ...
    
    For example:

    >>> def dice_binary(y_true: np.ndarray, y_pred: np.ndarray):
    >>>     y_true = (y_true>0.5).astype('float32')
    >>>     y_pred = (y_pred>0.5).astype('float32')
    >>>     num = 2 * np.sum(y_true*y_pred)
    >>>     den = np.sum(y_true) + np.sum(y_pred)
    >>>     return num / (den + 0.000001 )

    * define map_function for each method:

    >>> def m(patient_name): -> str # map patient name to patient file path
    >>>     patient_seg_path = ... # find patient segmentation result based on patient_name 
    >>>     return patient_seg_path
    
    * add method to evaluator:

    >>> evaluator = PairedEvaluation(all_patient_names, eval_function)
    >>> evaluator.add_method( 'method_name1', m1 )
    >>> evaluator.add_method( 'method_name2', m2 )
    >>> evaluator.add_method( ... )

    * evaluate

    >>> eval_list = evaluator.run_eval('method_name1', 'method_name2')
    
    or using parallel version to speed up evaluation
    
    >>> eval_list = evaluator.run_eval_parallel('method_name1', 'method_name2', num_workers=4)

    * after evaluation, collect results for further use

    >>> for subject_name, subject_eval in zip( all_patient_names, eval_list )
    >>>     # do further operations
    >>>     ...

    '''
    def __init__(self, subjects: list, eval_function: callable):
        '''
        eval_function: callable
            evaluation function with prototype f(y1, y2) -> float | list[float]
        subjects: list of string
            all subject names
        '''
        if contain_duplicates(subjects):
            raise RuntimeError('subject list contains duplicates, please remove them.')
        self.eval_function = eval_function
        self.subjects = subjects
        self.method_mappings = {}
    
    def add_method(self, name: str, map_function: callable):
        '''
        name: str
            method name
        map_function: define mapping from patient name to patient segmentation file,
            must have the following function prototype:
        >>> def map_function(patient_name):
        >>>     patient_seg = ... # do something here
        >>>     return patient_seg
        '''
        self.method_mappings[name] = map_function
    
    def get_subject_list(self):
        return deepcopy(self.subjects)

    def run_eval(self, method_A:str, method_B:str, allow_null = False):
        '''
        Start paired voxel-wise evaluation for two methods A and B.  
        '''

        print('** running evaluation for "%s" vs "%s"...' % (method_A, method_B))
        print('   evaluator class name:', type(self).__name__)
        all_methods = list(self.method_mappings.keys())
        assert method_A in all_methods and method_B in all_methods

        all_subjects = self.subjects
        
        # make eval dict to store all evaluated results
        eval_list = []
        
        sid = 0
        for subject in all_subjects:
            print('[%d/%d] subject "%s"' % (sid+1,len(all_subjects), subject))
            map_function_A = self.method_mappings[method_A]
            map_function_B = self.method_mappings[method_B]
            file_A = map_function_A(subject)
            file_B = map_function_B(subject)
            if file_A is not None and file_exist(file_A) == False:
                raise RuntimeError('subject "%s", method "%s" - file not exist: "%s".' % (subject, method_A, file_A))
            if file_B is not None and file_exist(file_B) == False:
                raise RuntimeError('subject "%s", method "%s" - file not exist: "%s".' % (subject, method_B, file_B))
            if not allow_null:
                if file_A is None or file_B is None:
                    raise RuntimeError('subject "%s", method "%s" - NULL is not allowed.' % (subject, method_B))
            else:
                if file_A is None and file_B is None:
                    raise RuntimeError('subject "%s", method "%s" - no valid file found for evaluation.' % (subject, method_A))

            data_A = load_nifti_simple(file_A) if file_A is not None else None
            data_B = load_nifti_simple(file_B) if file_B is not None else None
            if data_B is None:
                data_B = np.zeros_like(data_A)
            elif data_A is None:
                data_A = np.zeros_like(data_B)

            if data_A.shape != data_B.shape:
                raise RuntimeError('subject "%s" - shapes not equal: "%s", "%s".' % (subject, file_A, file_B))
            metric = self.eval_function(data_A, data_B)
            eval_list.append(metric)
            sid+=1

        return eval_list
    
    def run_eval_parallel(self, method_A: str, method_B: str, num_workers: int=8, allow_null = False):
        '''
        Run paired voxel-wise evaluation in parallel to save time if 
        evaluation process is time consuming.
        '''
        print('** running evaluation for "%s" vs "%s"...' % (method_A, method_B))
        print('   evaluator class name:', type(self).__name__)
        all_methods = list(self.method_mappings.keys())
        assert method_A in all_methods and method_B in all_methods, 'unknown method name.'

        all_subjects = self.subjects
        
        task_list = []
        for subject in all_subjects:  
            map_function_A = self.method_mappings[method_A]
            map_function_B = self.method_mappings[method_B]
            file_A = map_function_A(subject)
            file_B = map_function_B(subject)
            if file_A is not None and file_exist(file_A) == False:
                raise RuntimeError('subject "%s", method "%s" - file not exist: "%s".' % (subject, method_A, file_A))
            if file_B is not None and file_exist(file_B) == False:
                raise RuntimeError('subject "%s", method "%s" - file not exist: "%s".' % (subject, method_B, file_B))
            if not allow_null:
                if file_A is None or file_B is None:
                    raise RuntimeError('subject "%s", method "%s" - NULL is not allowed.' % (subject, method_B))
            else:
                if file_A is None and file_B is None:
                    raise RuntimeError('subject "%s", method "%s" - no valid file found for evaluation.' % (subject, method_A))
            task_list.append( (subject, self.eval_function, file_A, file_B) )
        
        finished_jobs = run_parallel(_parallel_paired_evaluation, task_list, num_workers, 'running')
        # sort results based on subject order, because parallelization shuffles the result
        subj_eval_dict = {}
        for tup in finished_jobs:
            subj_eval_dict[tup[0]] = tup[1]
        eval_list = []
        for subject in self.subjects:
            eval_list.append( subj_eval_dict[subject] )

        return eval_list
    
class BinaryDiceEvaluation(PairedEvaluation):
    def __init__(self, subjects: list):
        super().__init__(subjects, hard_dice_binary)

class VoxelPrecisionRecallEvaluation(PairedEvaluation):
    '''
    Evaluate precision(y_true, y_pred) and recall(y_true, y_pred).

    Note (important!)
    ----------------
    Be aware of the input order!
    '''
    def __init__(self, subjects: list):
        super().__init__(subjects, voxel_precision_recall)

class InstancePrecisionRecallEvaluation(PairedEvaluation):
    def __init__(self, subjects: list):
        '''
        note: keep in mind that the two operands are not 
        equivalent and they cannot change order !!!
        
        first is y_true then is y_pred.
        '''
        super().__init__(subjects, inst_confusion_matrix)

class InstanceF1Evaluation(PairedEvaluation):
    def __init__(self, subjects: list):
        '''
        note: keep in mind that the two operands are not 
        equivalent and they cannot change order !!!
        
        first is y_true then is y_pred.
        '''
        print('** Note: ground truth should be placed first. **')
        super().__init__(subjects, instance_F1)

###
### Binary Component Dice Evaluation
###

def binary_component_dice(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape, 'shape not equal.'
    yt = (y_true > 0.5).astype('int') # ensure binary segmentation
    yp = (y_pred > 0.5).astype('int')
    nT, lT = connected_components(yt)
    _, lP = connected_components(yp)
    
    e = []

    for iT in range(1,nT+1):
        cT = (lT == iT).astype('int')
        labels = list(np.unique(lP * cT))
        if 0 in labels: labels.remove(0)
        mP = np.zeros(yp.shape).astype('int')
        for z in labels:
            mP += (lP==z).astype('int')
        cP = ((mP - (yt - cT)) > 0.5).astype('int')
        dTP = hard_dice_binary(cT, cP)
        e.append( ( np.sum(cT), dTP ) )
    
    return sorted( e, key=lambda x: x[0] )

class BinaryComponentDiceEvaluation(PairedEvaluation):
    def __init__(self, subjects: list):
        '''
        Note
        ------
        Keep in mind that the two operands of component dice are not 
        equivalent and they cannot change order !!! The first operand
        must be ground truth / manual label result and the second 
        operand must be prediction from networks.
        '''
        super().__init__(subjects, binary_component_dice)

    @staticmethod
    def plot_scatter(eval_result, voxel_size_in_mm3, save_file, title='', figsize=(4,3), dpi=300,
        tight=False, null_plot=False):
        '''
        Description
        -------------
        Visualizing evaluated binary component dice result using 
        scatter plot.

        Parameters
        -------------
        eval_result: <internal object>
            Evaluated result returned directly from self.run_eval(...)
            format: [(x1,y1), (x2,y2), ...]
        voxel_size_in_mm3: float
            Voxel size in mm^3, will be used in volume calculation
        save_file: str
            Image save path, can be "*.png" or "*.pdf", please ensure
            the directory of the target save file actually exist.
        title: str
            Figure title (default='', no title).
        figsize: tuple
            Figure size in inches (default=(6,4)).
        dpi: int
            Dot per inch (DPI), default=300. Ignore this if you want to
            save a vectorized drawing (for example "*.pdf").
        tight: bool
            if you want to remove white spaces around your plot you can 
            set this to True (default=False)
        null_plot: sometimes if you dont want to draw the scatter plot
            and CI95 interval, instead you just want to draw a background 
            grid of the plot, you can set this to True.
        '''
        # convert points
        points = []
        for i in eval_result: 
            for j in i: 
                points.append(j)
        X, Y = np.array([i[0] for i in points]), np.array([i[1] for i in points])

        # user settings
        vox_range = [2.5, np.max(X)] # voxels
        #ml_ticks_desired = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        ml_ticks_desired = [0.05,0.1,0.2,0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
        dice_range = [0,1] # dice

        # functions
        def ml_to_vox(ml):
            return (ml * 1000.0 / voxel_size_in_mm3)
        def vox_to_ml(vox):
            return (vox * voxel_size_in_mm3 / 1000.0)

        # calculate actual mL ticks
        ml_range = [ vox_to_ml(vox_range[0]), vox_to_ml(vox_range[1]) ]
        ml_ticks_actual = []
        for ml_tick in ml_ticks_desired:
            if ml_tick > ml_range[0] and ml_tick < ml_range[1]:
                ml_ticks_actual.append(ml_tick) # remove any out of bound tick value 
        ml_ticks_actual = sorted(ml_ticks_actual + ml_range) # add lower/upper bound to ticks
        if ml_ticks_actual[1] / ml_ticks_actual[0] < 2.0: # this tick is too close to lower bound, delete
            ml_ticks_actual.remove(ml_ticks_actual[1])
        if ml_ticks_actual[-2] / ml_ticks_actual[-1] > 0.5: # this tick is too close to upper bound, delete 
            ml_ticks_actual.remove(ml_ticks_actual[-2])

        fig = plt.figure('figure', figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0.12,0.13,0.83,0.64])

        ax.grid(True,ls='--',alpha=0.4)
        ax.set_xlim(vox_range[0], vox_range[1])
        ax.set_xscale('log')
        ax.set_xlabel('Lesion component volume (voxel)', labelpad=0)
        ax.set_ylim(dice_range[0],dice_range[1])
        ax.set_ylabel('Dice coefficient', labelpad=0)
        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(base=10,subs=(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),numticks=100))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.minorticks_on()

        def ticklist_to_labels(ticks):
            s = []
            for tick in ticks:
                formatter = ''
                if tick<0.01: formatter = '%.3f'
                elif tick<0.1: formatter = '%.2f'
                elif tick<1: formatter = '%.1f'
                else: formatter = '%d'
                s0 = formatter % tick
                s.append(s0)
            s[0] = '<' + s[0]
            s[-1] = '>' + s[-1]
            return s

        ax2 = ax.twiny()
        ax2.set_xlim(vox_range[0], vox_range[1])
        ax2.set_ylim(dice_range[0],dice_range[1])
        ax2.set_xscale('log')
        ax2.set_xlabel('Lesion component volume (mL)', labelpad=4)
        ax2.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10))
        ax2.set_xticks([ ml_to_vox(ml) for ml in ml_ticks_actual  ] )
        ax2.set_xticklabels(ticklist_to_labels(ml_ticks_actual), rotation=0)
        ax2.minorticks_off()
        ax2.tick_params(axis='both', labelsize=8)
        ax2.tick_params(axis='x', pad=1)

        if not null_plot:
            ax.scatter(X,Y,marker='o',alpha=0.2,s=10,lw=0.8,facecolors='royalblue', edgecolors='none', zorder = 30)

        ##
        ## Start LOWESS regression
        ##

        lowess_frac = 0.20
        n_bootstrap = 200
        bootstrap_ratio = 0.25
        #bootstrap_plot_alpha = 0.05
        grid_density = 200
        #line_color = [217/255,68/255,69/255]
        line_color = 'darkgreen'

        
        def jitter(vals, offset):
            return np.array( [val + (np.random.rand()*2-1)*offset for val in vals] )

        def remove_dup_x(X, Y, step=0.001):
            XY = [(float(x),float(y)) for x, y in zip(X,Y)]
            XY = sorted(XY,key=lambda x: x[0])
            x_cur, x_end = np.min(X), np.max(X)
            XY_new = []
            I=0
            while I<len(XY):
                XY_new.append(XY[I])
                x_cur = XY[I][0] + step
                while I<len(XY) and XY[I][0] < x_cur:
                    I+=1
            if XY_new[-1][0] < x_end:
                XY_new.append(XY[-1])
            X_new, Y_new = zip(*XY_new)
            return X_new, Y_new

        def bootstrap(x, y, sample_ratio):
            samples = np.random.choice(len(x), int(len(x)*sample_ratio), replace=False)
            x_s, y_s = x[samples], y[samples]
            x_sm, y_sm = lowess(jitter(y_s,0.01),jitter(x_s, 1), frac=lowess_frac, it=5, return_sorted = True).T
            x_sm, y_sm = remove_dup_x(x_sm, y_sm)
            return x_sm, y_sm

        # plot confidence interval using bootstrapping, see
        # https://james-brennan.github.io/posts/lowess_conf/

        with ignore_SIGINT():
            x_grid = np.logspace(np.log10( np.min(X) ), np.log10( np.max(X) ), num=grid_density, base=10.0, endpoint=True)
            y_grid = []
            for k in range(n_bootstrap):
                x_sm, y_sm = bootstrap(X, Y, bootstrap_ratio)
                y_grid.append( scipy.interpolate.interp1d(x_sm, y_sm, fill_value='extrapolate')(x_grid) )
                minibar(msg='bootstrap', a=k+1, b=n_bootstrap)
            print('')
            y_grid = np.stack(y_grid).T
            # compute mean \mu and standard error \sigma of the LOWESS model
            mu = np.nanmean(y_grid, axis=1)
            sigma = np.nanstd(y_grid, axis=1, ddof=0)

            if not null_plot:
                # plot 95% confidence interval
                plt.fill_between(x_grid, mu-1.96*sigma, mu+1.96*sigma, color=line_color, alpha=0.35, zorder=19) 
                plt.plot(x_grid, mu, color=line_color,alpha=0.8, zorder=20)

            legend_items = [Patch(facecolor=line_color,fill=True, alpha=0.35, label='CI95')]
            ax.legend(handles=legend_items,loc='lower right',fontsize=8)

        # # plot main LOWESS regression line
        # lowess_x, lowess_y = lowess(jitter(Y,0.01), jitter(X, 1), frac=lowess_frac, it=5, return_sorted = True).T
        # ax.plot(lowess_x,lowess_y,lw=1.0,color='darkgreen',alpha=0.8,zorder=20)

        if title!='':
            ax.set_title(title, pad=10)
                
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_facecolor([0.95,0.95,0.95])

        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.set_facecolor([0.95,0.95,0.95])

        # save and close
        mkdir(gd(save_file))
        if tight:
            plt.savefig(save_file, bbox_inches='tight',pad_inches=0)
        else:
            plt.savefig(save_file)
        plt.close(fig)

        print('saved figure as "%s".' % save_file)

class VisualScoreEvaluation(object):
    def __init__(self, subjects: list, output_folder: str, dataset_name: str):
        if contain_duplicates(subjects):
            raise RuntimeError('subject list contains duplicates, please remove them.')
        self.subjects = subjects
        self.method_mappings = {}
        self.data_mapping = None
        self.output_folder = output_folder
        self.dataset_name = dataset_name
    
    def add_seg_mapping(self, method_name: str, map_function: callable):
        '''
        name: str
            method name
        map_function: define mapping from patient name to patient segmentation file
        '''
        self.method_mappings[method_name] = map_function
    
    def add_data_mapping(self, map_function: callable):
        self.data_mapping = map_function

    def get_subject_list(self):
        return deepcopy(self.subjects)
    
    def get_method_list(self):
        return list(self.method_mappings.keys())

    def gen_eval_data(self, pack=True, to_grayscale=False):
        '''
        Generate data used for evaluation.

        pack: automatically pack data to *.tar.gz
        to_grayscale: convert to uint8 grayscale [0~255] format to significantly reduce file size.
        '''
        if dir_exist(self.output_folder) and len(ls(self.output_folder)) > 0:
            raise RuntimeError('Folder "%s" is not empty! Please change to a new empty folder!' % self.output_folder)
        mkdir(self.output_folder)
        
        # generate method mappings
        anonymous_names = []
        method_names = list(self.method_mappings.keys())
        for i in range(len(method_names)):
            anonymous_names.append('seg_%d' % (i+1))
        shuffled_anonymous_names = deepcopy(anonymous_names)

        # create excel file
        output_excel = join_path(self.output_folder, '%s.xlsx' % self.dataset_name)
        worksheet_names = ['Score', 'Mapping']
        xlsx = SimpleExcelWriter(output_excel, worksheet_names=worksheet_names)
        dark_gray_cell = xlsx.new_format( font_color='#FFFFFF', bg_color='#808080' )
        light_gray_cell = xlsx.new_format( font_color='#000000', bg_color='#D9D9D9' )
        # write table head
        for worksheet_name in worksheet_names:
            xlsx.write( (0,0), 'case', worksheet_name=worksheet_name, format=dark_gray_cell )
            for i in range(len(anonymous_names)):
                xlsx.write( (0,i+1), anonymous_names[i], worksheet_name=worksheet_name, format=dark_gray_cell )

        # copy files to destination (segmentations & original images) 
        all_cases = self.subjects   
        for i in range(len(all_cases)):
            case = all_cases[i]

            random.shuffle(shuffled_anonymous_names)
            mapping = {}
            for ii in range(len( method_names )):
                method_name = method_names[ii]
                anonymous_name = shuffled_anonymous_names[ii]
                print( method_name, '->', anonymous_name )
                mapping[anonymous_name] = method_name

            xlsx.write((i+1,0), case, worksheet_name='Score',format=light_gray_cell)
            xlsx.write((i+1,0), case, worksheet_name='Mapping',format=light_gray_cell)
            for ii in range(len( anonymous_names )):
                xlsx.write((i+1,ii+1), mapping[anonymous_names[ii]], worksheet_name='Mapping')

            destination_folder = mkdir(join_path(self.output_folder, self.dataset_name, case))
            original_image = self.data_mapping(case)
            
            dest_image = join_path(destination_folder, 'original_image.nii.gz')
            if to_grayscale == False:
                cp( original_image, dest_image )
            else:
                # since original image is stored as float32 or float64,
                # which can consume a lot of disk space, for evaluation
                # we dont need such precision, just convert them to 0~255
                # integer range would be OK
                img, hdr = load_nifti(original_image)
                img = (img - img.min()) / (img.max() - img.min()) * 255.0
                img = img.astype('int') 
                # save nifti to target path
                save_nifti(img, hdr, dest_image )
            print(dest_image)

            for method_name in list(self.method_mappings.keys()):
                src = self.method_mappings[method_name](case)
                anon_name = [aname for aname in mapping if mapping[aname]==method_name][0]
                dst = join_path(destination_folder, '%s.nii.gz' % anon_name)
                orig_data, orig_header = load_nifti(original_image)
                # src can be None, in this case we need to fill a completely blank segmentation
                if src is not None:
                    label_dat = load_nifti_simple(src)
                    assert label_dat.shape == orig_data.shape, 'image and segmentation shape not equal.'
                    save_nifti(label_dat > 0.5, orig_header, dst)
                    print(src,'>>>', dst )
                else:
                    label_dat = np.zeros_like(orig_data)
                    save_nifti(label_dat, orig_header, dst)
                    print('WARNING: save blank segmentation to "%s".' % dst)

        xlsx.save_and_close()

        if pack:
            print('** packing all data into a "*.tar.gz" file, this may take a while, please wait...')
            targz_file = join_path(self.output_folder, '%s.tar.gz' % self.dataset_name)
            targz_compress(self.output_folder, targz_file)

    @staticmethod
    def check_worksheet_exists(xlsx_file, worksheet_name):
        try:
            xlsx = SimpleExcelReader(xlsx_file)
            xlsx.max_row(worksheet_name)
        except KeyError:
            return False
        else:
            return True

    @staticmethod
    def parse_sheet(xlsx_file, worksheet_name = "Score", return_methods_and_subjects = False, verbose = False):

        assert file_exist(xlsx_file), 'file "%s" not exist.' % xlsx_file
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, worksheet_name), \
            'file "%s" does not contain worksheet named "%s".' % (xlsx_file, worksheet_name)
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, "Mapping"), \
            'Cannot find worksheet named "Mapping" in file "%s".' % xlsx_file
        
        xlsx = SimpleExcelReader(xlsx_file)
        rows, columns = xlsx.max_row(worksheet_name), xlsx.max_column(worksheet_name)

        method_scores = {}
        all_methods, all_cases = [], []

        for i in range(1,columns):
            method_name = xlsx.read((1,i), worksheet_name="Mapping")
            if isinstance(method_name,str):
                all_methods.append(method_name)
                method_scores[method_name] = {}

        if verbose:
            print('all methods:', all_methods)

        for i in range(1, rows):
            case_name = xlsx.read((i,0), worksheet_name="Mapping")
            if isinstance(case_name,str):
                all_cases.append(case_name)
                if verbose:
                    printi('%s ' % case_name)

        
        for i in range(1,rows):
            case_name = str( xlsx.read((i,0), worksheet_name=worksheet_name) )
            assert case_name in all_cases, 'case "%s" is not in mapping.' % case_name
            mapping_row = all_cases.index(case_name) + 1
            contain_n_a = False # if this row contains any N/A
            for j in range(1, columns):
                method_name = str( xlsx.read((mapping_row,j), worksheet_name='Mapping') )
                if method_name not in all_methods:
                    continue
                method_score_for_this_case = xlsx.read((i,j), worksheet_name=worksheet_name)
                try:
                    method_score_for_this_case = str(method_score_for_this_case)
                    _ = int(method_score_for_this_case) # checking if this is n/a, N/A etc.
                except (TypeError, ValueError):
                    method_score_for_this_case = 'n/a'
                    contain_n_a = True
                method_scores[method_name][case_name] = method_score_for_this_case
                if verbose:
                    printi(method_score_for_this_case+' ')
            if contain_n_a:
                # if one of the method score is n/a, change all score to n/a
                for method in all_methods:
                    method_scores[method][case_name] = 'n/a'
        
        if verbose:
            print('\n')
        
        if return_methods_and_subjects:
            return all_methods, all_cases
        else:
            return method_scores

    @staticmethod
    def plot_hist(normalized_scores: Union[np.ndarray, list], n_max: int, save_file:str, font_file:str="",
        color_palette = 'red', null_plot = False):
        '''
        plot histogram for visualization, data range should be [0,1] (normalized score)
        '''

        assert np.max(normalized_scores) < 1.001 and np.min(normalized_scores) > -0.001, \
            'Scores aren\'t normalized. Please normalize them to [0,1]. Got value range [%f, %f].' % \
            (np.min(normalized_scores), np.max(normalized_scores))
        if color_palette not in ['red', 'blue']:
            warnings.warn('Unknown color palette "%s", switching to default "red" color palette.' % str(color_palette))
            color_palette = 'red'
        if not null_plot:
            assert file_exist(font_file), 'Font file "%s" not exist, please assign a valid file.' % font_file

        #avg_marker_color = [142/255,110/255,172/255] # purple
        avg_marker_color = [0,0,0] # black

        if color_palette == 'red':
            bar_color, line_color = [228/255,140/255,141/255], [217/255, 68/255, 69/255]
        elif color_palette == 'blue':
            bar_color, line_color = [136/255,180/255,213/255], [ 57/255,128/255,171/255]

        bins = [0.0,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1.0]
        hist, _ = np.histogram(normalized_scores, bins = bins)

        # reverse bins and hist, since we want to draw bars from top to bottom
        bins.reverse()
        hist = hist[::-1]

        if np.max(hist) > n_max:
            warnings.warn("Maximum bar height (%d) > n_max (%d), bar will be truncated." % (np.max(hist), n_max))

        # global settings
        pagesize = (2.8,4) # cm
        cv = PlotCanvas(save_file, "%fcm*%fcm" % (pagesize[0], pagesize[1]))
        if not null_plot:
            cv.register_font(font_file,"font")

        # global shape settings
        bottom_hline_y = pagesize[1] * 0.02
        top_hline_y = pagesize[1] * 0.98
        hline_x = (pagesize[0]*0.0,pagesize[0]*1.0)
        l1_color = [0.80,0.80,0.80]
        l1_lw = 1.2
        center_x = (hline_x[0] + hline_x[1]) / 2.0
        n_bars = len(bins)-1

        # calculate size of each bar
        bar_heights = [0] * n_bars
        bar_widths = [0] * n_bars
        # height
        for i in range(n_bars):
            bins_range = bins[0] - bins[-1]
            bar_heights[i] = ( top_hline_y - bottom_hline_y ) * ((bins[i]-bins[i+1])/bins_range)
        # width
        for i in range(n_bars):
            norm_val = (hist[i] / n_max) * (np.min(bar_heights) / bar_heights[i] )
            bar_widths[i] = norm_val * (hline_x[1] - hline_x[0])

        # draw background box
        for w in [0.0,0.4,0.8]:
            w_y = bottom_hline_y + (top_hline_y - bottom_hline_y) * w
            w_h = (top_hline_y - bottom_hline_y) * 0.2
            cv.rect((hline_x[0], w_y),(hline_x[1], w_y+w_h),0,None, [0.95,0.95,0.95])
        cv.line((hline_x[0], bottom_hline_y),(hline_x[1], bottom_hline_y), l1_lw, l1_color)
        cv.line((hline_x[0], top_hline_y),(hline_x[1], top_hline_y), l1_lw, l1_color)
        cv.line((center_x, top_hline_y),(center_x, bottom_hline_y), l1_lw, l1_color,alpha=0.6)

        if not null_plot:
            # draw bars
            y_cur = top_hline_y
            for i in range(n_bars):
                bar_w, bar_h = bar_widths[i], bar_heights[i]
                position_start = ( center_x - bar_w/2, y_cur - bar_h)
                position_end = (position_start[0] + bar_w, position_start[1] + bar_h)
                if hist[i] > 0:
                    cv.rect(position_start, position_end, 0, line_color=None, fill_color=bar_color)
                    # dont draw lines for out of bound values
                    cv.line(
                        (position_start[0], position_start[1]),
                        (position_start[0], position_start[1] + bar_h),
                        1, line_color=line_color)
                    cv.line(
                        (position_end[0], position_end[1]),
                        (position_end[0], position_end[1] - bar_h),
                        1, line_color=line_color)
                    cv.text("%d" % hist[i], (position_end[0]+0.04,position_start[1]+bar_h/2-0.115),"font", 9,
                        font_color=[0,0,0])
                y_cur -= bar_h

            # visualize average scores
            avg_score = np.mean( normalized_scores )
            avg_score_fig_y =  bottom_hline_y + (top_hline_y - bottom_hline_y) * avg_score
            cv.line((hline_x[0],avg_score_fig_y), (hline_x[1],avg_score_fig_y), 2, 
                line_color=avg_marker_color,alpha=0.6, dashed=True,dash_pattern=(5,4))
            if avg_score < 0.5:
                cv.text("%.2f" % avg_score,(hline_x[0] + 0.04, avg_score_fig_y + 0.06),"font", 10, 
                    font_color=avg_marker_color, alpha=1.0)
            else:
                cv.text("%.2f" % avg_score,(hline_x[0] + 0.04, avg_score_fig_y - 0.32),"font", 10, 
                    font_color=avg_marker_color, alpha=1.0)

        cv.save()
    
    @staticmethod
    def parse_xlsx_TianTan_format(xlsx_file):
        '''
        all_methods, valid_subjects, final_scores = parse_xlsx_TianTan_format('example.xlsx')
        '''
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, "Cerebral_small") == True, \
            'cannot find sheet "Cerebral_small" in file "%s".' % xlsx_file
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, "Cerebral_large") == True, \
            'cannot find sheet "Cerebral_large" in file "%s".' % xlsx_file
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, "Cerebellum_and_brainstem") == True, \
            'cannot find sheet "Cerebellum_and_brainstem" in file "%s".' % xlsx_file
        assert VisualScoreEvaluation.check_worksheet_exists(xlsx_file, "Mapping") == True, \
            'cannot find sheet "Mapping".'

        cerebral_small_lesions = VisualScoreEvaluation.parse_sheet(xlsx_file, worksheet_name="Cerebral_small")
        cerebral_large_lesions = VisualScoreEvaluation.parse_sheet(xlsx_file, worksheet_name="Cerebral_large")
        cerebellum_and_brainstem_lesions = VisualScoreEvaluation.parse_sheet(xlsx_file, worksheet_name="Cerebellum_and_brainstem")

        all_methods, all_subjects = VisualScoreEvaluation.parse_sheet(xlsx_file, worksheet_name="Mapping", return_methods_and_subjects=True)
        valid_subjects = []
        final_scores = {}

        for method in all_methods:
            final_scores[method] = {}
            for subject in all_subjects:

                score_cbrl_small = cerebral_small_lesions[method][subject]
                score_cbrl_large = cerebral_large_lesions[method][subject]
                score_cbrm_bstm = cerebellum_and_brainstem_lesions[method][subject]

                maximum_score = 0

                if score_cbrl_small == 'n/a':
                    score_cbrl_small = 0  
                else:
                    score_cbrl_small = float(score_cbrl_small)
                    maximum_score += 2

                if score_cbrl_large == 'n/a':
                    score_cbrl_large = 0  
                else:
                    score_cbrl_large = float(score_cbrl_large)
                    maximum_score += 2
                    
                if score_cbrm_bstm == 'n/a':
                    score_cbrm_bstm = 0  
                else:
                    score_cbrm_bstm = float(score_cbrm_bstm)
                    maximum_score += 2
                
                if maximum_score == 0:
                    print('Subject "%s" of method "%s" do not have any valid score. This subject will be ignored.' % (subject, method))
                else:
                    score = (score_cbrl_small + score_cbrl_large + score_cbrm_bstm) / maximum_score
                    
                    final_scores[method][subject] = score
                    if subject not in valid_subjects:
                        valid_subjects.append(subject)
        
        return all_methods, valid_subjects, final_scores


def linreg(data_dict:dict, x_key:str, y_key:str, debug = False):
    '''
    Description
    -----------
    * Simple linear regression with covariances (nuisance regression)

    Parameters
    -----------
    * data_dict: a dictionary representing the data. Format:
        { 'var1': [...], 'var2': [...], ... , 'varN': [...] }
    * x_key: key name for independent variable.
    * y_key: key name for dependent variable.
    * other keys will be treated as covariances.

    Returns
    -----------
    r: Pearson r value
    p: p value
    X_star: multiple observations of independent variable X
    Y_star: corrected dependent variable Y (covariances removed)
    k, b: a regression line representing Y = kX + b.

    Example
    -----------
    >>> data_dict = { 
    >>>     'x1': [0,1,2,3],
    >>>     'x2': [2,3,1,4], 
    >>>     'x3': [4,5,6,7],
    >>>      'y': [4,3,2,1]
    >>> }
    >>> linreg(data_dict, 'x2', 'y') # x1, x3 as covariances
    >>> linreg(data_dict, 'x1', 'y') # x2, x3 as covariances
    '''

    assert x_key in data_dict, 'key "%s" is not in data_dict.'
    assert y_key in data_dict, 'key "%s" is not in data_dict.'

    all_keys = list(data_dict.keys())
    cov_keys = [ key for key in all_keys if (key != y_key) and (key != x_key) ]
    ind_cov_keys = [x_key] + cov_keys
    num_samples = len(data_dict[y_key])

    if num_samples < 3:
        raise RuntimeError("number of samples too small (<3), abort.")
    for key in all_keys:
        assert len(data_dict[key]) == num_samples, 'number of samples = %d, expected independent '\
            'variable "%s" to have length %d, but got %d.' % (num_samples, key, num_samples, len(data_dict[key]))

    X = np.zeros([num_samples, len(all_keys)-1])
    Y = np.zeros([num_samples])

    # fill data
    for i in range(num_samples):
        Y[i] = data_dict[y_key][i]
        for key in ind_cov_keys:
            j = ind_cov_keys.index(key)
            X[i,j] = data_dict[key][i]

    if debug:
        print('X =', X)
        print('Y =', Y)

    # linear regression
    coeffs = sm.OLS(Y, sm.add_constant(X, has_constant='add')).fit().params
    C, A = coeffs[0], coeffs[1:] # y = C + A[0]x[0] + A[1]x[1] + ... + A[N]x[N]

    if debug:
        print('C =',C)
        print('A =',A)

    # get rid of covariances
    Y_star = np.zeros([num_samples])
    for i in range(num_samples):
        sum_of_covs = 0.0 # weighted sum of covariance(s)
        for cov_key in cov_keys:
            j = ind_cov_keys.index(cov_key)
            sum_of_covs += A[j]*X[i,j]
        Y_star[i] = Y[i] - sum_of_covs
    
    # collect independent variable
    X_star = np.zeros([num_samples])
    for i in range(num_samples):
        j = ind_cov_keys.index(x_key)
        X_star[i] = X[i,j]

    r, p = stats.pearsonr(X_star, Y_star)

    coeffs = sm.OLS(Y_star, sm.add_constant(X_star, has_constant='add')).fit().params
    k, b = coeffs[1], coeffs[0]

    if debug:
        print('X* =', X_star)
        print('Y* =', Y_star)
        print('r =', r)
        print('p =', p)
        print('k =', k)
        print('b =', b)

    return r, p, X_star, Y_star, k, b

##
## some plot functions
##

def boxplot_2x(evalA, evalB, names, save_file, legends=None ,title='',
    statistic_test = 'wilcoxon', vline=None, vlinetext=None, figsize=(3.0,2.1), dpi=300):
    
    '''
    Parameters
    ----------
    evalA / evalB: evaluation results from two independent experiments (or two raters).
            format: [
                        [0.4,0.5,...,0.8], 
                        [0.3,0.9,...,0.1], 
                        ..., 
                        [0.4,0.5,...,0.1]
                    ]
    names: label names. Will be displayed in x axis labels.
    save_file: output file path. "*.png" for a rasterized image and "*.pdf" for a vector file.
    statistic_test: perform statistical significance test. Can be "wilcoxon" or "paired t",
        calculate p-values for (evalA[0], evalA[i]), (evalB[0], evalB[i]).
    figsize: figure size measured in inches (recommend using default value).
    dpi: dot per inches. For saving a vector file this can be ignored and set to default. 
            "figsize" and "dpi" determine the image size (in pixels).
    legends: provide legends for both boxes. Example:
            legends = ['text for 1', 'text for 2']
    '''
    
    assert len(evalA) == len(evalB), 'invalid parameter length.'
    assert len(evalA) == len(names), 'invalid parameter length.'

    assert statistic_test in ['wilcoxon', 'paired t']

    num_methods = len(evalA)

    linewidth = 0.7
    boxspace = 0.35
    groupspace = 0.1
    boxsidepad = 0.4
    boxwidth=0.28

    colorA1, colorA2 = [217/255,68/255,69/255], [228/255,140/255,141/255] # R
    #colorA1, colorA2 = [45/255,101/255,50/255], [166/255,200/255,166/255] # G
    colorB1, colorB2 = [60/255,127/255,170/255], [136/255,180/255,211/255] # B

    fig = plt.figure('figure', figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0.20,0.09,0.75,0.8])
    
    # change outlier appearance
    median_style = { 'linestyle':'-', 'linewidth':linewidth, 'color': [1.0,1.0,1.0] }
    whisker_style = {'linestyle': '-', 'linewidth': linewidth, 'color': [0.6,0.6,0.6]}
    
    # prepare positions and data
    positions = []
    cursor = 0.0
    for i in range(num_methods):
        positions.append(cursor)
        cursor += boxspace
    cursor += groupspace
    for i in range(num_methods):
        positions.append(cursor)
        cursor += boxspace

    evalBA = []
    for i in range(num_methods): # from bottom to top
        evalBA.append(evalB[i])
    for i in range(num_methods):
        evalBA.append(evalA[i])

    # axvline
    axlinecolor = [142/255,110/255,172/255]
    if vline is not None:
        plt.axvline(vline, linestyle='--', color=axlinecolor, 
            linewidth=2.0, zorder=2,alpha=0.6)
    if vlinetext is not None:
        plt.text( vline + 0.07 , cursor-boxspace+boxsidepad/2 ,
            vlinetext,rotation='horizontal', horizontalalignment='center',
            verticalalignment='center',fontsize=7,
            color=axlinecolor,zorder=2,weight="bold",alpha=0.6)

    bp = ax.boxplot(evalBA, patch_artist=True, vert=0,notch=False, widths=boxwidth, 
        positions=positions, whiskerprops=whisker_style,
        medianprops=median_style,showcaps=False,showfliers=False)
    ax.set_zorder(2)

    ax.tick_params(axis='x',labelsize=7)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_yticks(positions)
    ax.set_yticklabels(names+names)
    plt.grid(True,ls='--',alpha=0.4,axis='x',zorder=1)
    ax.set_xlim(0,1)
    ax.set_ylim(positions[0]-boxsidepad, positions[-1]+boxsidepad)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_facecolor([0.95,0.95,0.95])
    
    # change appearance
    cursor = 0.0
    bi=0
    toggle = False
    for box in bp['boxes']:
        if bi<num_methods: 
            box.set(color = colorB1, facecolor=colorB2, linewidth=linewidth,alpha=1.0)
        else:
            box.set(color = colorA1, facecolor=colorA2, linewidth=linewidth,alpha=1.0)
            if toggle == False:
                cursor += groupspace
            toggle = True

        if statistic_test == 'paired t':
            if bi < num_methods:
                q3q1 = (np.quantile(evalB[bi],0.75) + np.quantile(evalB[bi],0.25))/2
                pval = stats.ttest_rel(evalB[bi], evalB[0]).pvalue
            else:
                q3q1 = (np.quantile(evalA[bi-num_methods],0.75) + np.quantile(evalA[bi-num_methods],0.25))/2
                pval = stats.ttest_rel(evalA[bi-num_methods], evalA[0]).pvalue

            if bi==0 or bi==num_methods:
                pass # dont calculate significance for proposed method
            else:
                if pval > 0.05:
                    text = ''
                elif pval <= 0.05 and pval > 0.01:
                    text = '*'
                elif pval <= 0.01 and pval > 0.001:
                    text = '**'
                else:
                    text = '***'
                plt.text( q3q1, cursor-0.05, text, 
                    horizontalalignment='center',verticalalignment='center',fontsize=6,color=[0,0,0],zorder=5,
                    weight='bold')

        elif statistic_test == 'wilcoxon':
            if bi==0 or bi==num_methods:
                pass # dont calculate significance for proposed method
            else:
                if bi < num_methods:
                    q3q1 = (np.quantile(evalB[bi],0.75) + np.quantile(evalB[bi],0.25))/2
                    pval = stats.wilcoxon(evalB[bi], evalB[0]).pvalue
                else:
                    q3q1 = (np.quantile(evalA[bi-num_methods],0.75) + np.quantile(evalA[bi-num_methods],0.25))/2
                    pval = stats.wilcoxon(evalA[bi-num_methods], evalA[0]).pvalue

                if pval > 0.05:
                    text = ''
                elif pval <= 0.05 and pval > 0.01:
                    text = '*'
                elif pval <= 0.01 and pval > 0.001:
                    text = '**'
                else:
                    text = '***'
                plt.text( q3q1, cursor-0.05, text, 
                    horizontalalignment='center',verticalalignment='center',fontsize=6,color=[0,0,0],zorder=5,
                    weight='bold')

        bi+=1
        cursor += boxspace
    bi=0
    for whisker in bp['whiskers']:
        if bi<2*num_methods: whisker.set(color = colorB1, linewidth=linewidth,alpha=1.0)
        else:       whisker.set(color = colorA1, linewidth=linewidth,alpha=1.0)
        bi+=1
    bi=0
    for flier in bp['fliers']:
        if bi<num_methods: flier.set(color = colorB1, alpha=1.0)
        else:       flier.set(color = colorA1, alpha=1.0)
        bi+=1
    bi=0
    for median in bp['medians']:
        if bi<num_methods: median.set(color = colorB1, linewidth=linewidth,alpha=1.0)
        else:       median.set(color = colorA1, linewidth=linewidth,alpha=1.0)
        bi+=1

    if len(title)>0: plt.title(title,fontsize=8)

    if legends is not None:
        legend_elements = [ Patch(facecolor=colorA2,edgecolor=colorA1,label=legends[0],linewidth=linewidth),
            Patch(facecolor=colorB2,edgecolor=colorB1,label=legends[1],linewidth=linewidth) ]
        ax.legend(handles=legend_elements, loc='lower left',prop={'size': 6},frameon=False, handlelength=0.55 * 1.2, handleheight=0.5 * 1.2)

    plt.tick_params(left = False) # hide y tick marks

    def random1(): # generate random numbers in [-1,+1]
        return float(np.random.rand(1)) * 2 - 1

    cursor = -0.02
    for i in range(num_methods):
        for j in range(len(evalB[i])):
            ax.plot(evalB[i][j], cursor+0.8*boxwidth/2*random1(),marker='.',
                markersize=2,markeredgewidth=0.0,markerfacecolor=colorB1,alpha=1.0, zorder=3)
            plt.text(0.95, cursor, '%.2f'  % np.mean(evalB[i]),
                horizontalalignment='center',verticalalignment='center',fontsize=6,color=colorB1,zorder=4)
        cursor += boxspace
    cursor += groupspace
    for i in range(num_methods):
        for j in range(len(evalA[i])):
            ax.plot(evalA[i][j], cursor+0.8*boxwidth/2*random1(),marker='.',
                markersize=2,markeredgewidth=0.0,markerfacecolor=colorA1,alpha=1.0, zorder=3)
            plt.text(0.95, cursor, '%.2f'  % np.mean(evalA[i]),
                horizontalalignment='center',verticalalignment='center',fontsize=6,color=colorA1,zorder=4)
        cursor += boxspace

    mkdir(gd(save_file))
    plt.savefig(save_file)
    plt.close(fig)
