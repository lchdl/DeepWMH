import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from freeseg.analysis.image_ops import average_contiguous_labels, component_filtering, group_mean, group_std, mean_std_grid, remove_3mm_sparks, z_score, median_3mm
from freeseg.utilities.file_ops import cp, file_exist, gn, join_path, mkdir
from freeseg.utilities.data_io import get_nifti_header, get_nifti_pixdim, load_nifti, load_nifti_simple, load_pkl, save_nifti, save_pkl, try_load_nifti
from freeseg.utilities.parallelization import run_parallel
from freeseg.utilities.misc import SimpleTxtLog, TimeStamps

# utility function to plot histogram curves
def hist_plot(x, y, r, rs, save_file, fig_size=(8,6), dpi=144, thresholds=None, simple_plot = False):
    if simple_plot == False:
        plt.figure('figure',figsize=fig_size,dpi=dpi,frameon=True)
        if thresholds is not None: # threshold marks
            for value in thresholds:
                plt.axvline(x=value,ls='--',lw=1,color=[0/255,0/255,0/255])
        for r0 in rs:
            plt.plot(x, r0, color=[100/255,100/255,100/255],ls='-',lw=0.5)
        plt.plot(x, y, color=[235/255,64/255,52/255],label='input',ls='-',lw=1.5)    
        plt.plot(x, r, color=[52/255,64/255,235/255],label='refs',ls='-',lw=1.5)
        plt.title('Histogram curve plot (log scale)')
        plt.xlabel('anomaly score')
        plt.ylabel('exponent value')
        plt.grid(which='both',ls='--',lw=1,color=[200/255,200/255,200/255])
        plt.legend()
    else: # simple plot just for demo
        plt.figure('figure',figsize=(3,3),dpi=300,frameon=True)
        plt.xlim((-2,45))
        plt.ylim((-0.2,6))
        plt.plot(x, y, color=[183/255,84/255,82/255],label='input',ls='-',lw=2.5)
        plt.plot(x, r, color=[130/255,179/255,102/255],label='refs',ls='-',lw=2.5)
        plt.fill_between(x,y,color=[247/255,206/255,205/255])
        plt.fill_between(x,r,color=[213/255,232/255,213/255])

    plt.savefig(save_file)
    plt.close('figure')

def hist_curve(data,bins,log_y=False, mask=None):
    if mask is None:
        hist, bin_edges = np.histogram(data,bins=bins)
    else:
        hist, bin_edges = np.histogram(data[mask>0.5],bins=bins)
    bin_centers = np.array([ (bin_edges[i]+bin_edges[i+1])/2 for i in range(len(hist))])
    if log_y:
        hist = np.where(hist==0,0.001,hist)
        hist = np.log10(hist)
        hist = np.where(hist<0,0,hist)
    return bin_centers, hist

def histogram_analysis(a_prime, a_refs, bins=None, mask=None):
    if isinstance(a_refs,list) == False:
        a_refs = [a_refs] # wrap it in a list

    if bins is None: # using default bins
        assert mask is not None, 'must provide mask when "bins" is None.'
        assert mask.shape == a_prime.shape
        print('automatically generating histogram bins..')
        ref_means = []
        for i in range(len(a_refs)):
            a_filtered = a_refs[i][mask>0.5]
            a_filtered = a_filtered[a_filtered>0]
            ref_means.append(a_filtered.mean())
        bin_width = np.array(ref_means).mean() / 4
        num_bins = 400
        print('* bin_width=%.2f, num_bins=%d.' % (bin_width, num_bins))
        hist_start = 0.0
        hist_end = hist_start + num_bins * bin_width
        print('* hist_start=%.2f, end=%.2f.' % (hist_start, hist_end))
        bins = np.linspace(hist_start, hist_end, num=num_bins+1)

    # calculate histogram curve for image
    x,y = hist_curve(a_prime,bins,log_y=True)
    r = np.zeros_like(x)
    rs = []
    for i in range(len(a_refs)):
        _, r0 = hist_curve(a_refs[i],bins,log_y=True)
        r += r0
        rs.append(r0)
    r = r / len(a_refs)
    return x, y, r, rs

def nll(x_prime, x_refs, min_std = None, side=None, return_all=False, use_mask=False):
    assert side in [None, '+', '-']

    if use_mask:
        x_ref_masks = []
        for x_i in x_refs:
            x_ref_masks.append(np.where(x_i>threshold_otsu(x_i), 1, 0))
        mu = group_mean(x_refs, masks=x_ref_masks)
        sigma = group_std(x_refs, masks=x_ref_masks)
    else:
        mu = group_mean(x_refs)
        sigma = group_std(x_refs)

    if min_std == None: 
        sigma += 1e-6 # add a small number to avoid division by zero
    else: 
        sigma = np.where(sigma<min_std, min_std, sigma)
    
    # calculate anomaly scores by taking the negative log likelihood of the fitted Gaussian distribution
    anomaly = np.power(x_prime-mu,2) / (2*np.power(sigma,2)) + np.log(sigma*2.506)  # sqrt(2*pi)=2.506

    anomaly = np.nan_to_num(anomaly, nan=0.0)
    if side == '+':
        anomaly = anomaly * (x_prime > mu).astype('float32')
    elif side == '-':
        anomaly = anomaly * (x_prime < mu).astype('float32')
    if return_all == False:
        return anomaly
    else:
        return anomaly, mu, sigma

def nll_analysis(case_info, apply_otsu=True, intensity_prior=None, case_output_folder=None, mean_correction=True, debug=False):

    assert intensity_prior in [None, '+', '-'], 'Unknown intensity prior "%s".' % str(intensity_prior)
    assert case_output_folder is not None, 'must assign case output folder.'

    source_image_path = case_info['x'] 
    ref_image_paths = case_info['r']
    ref_label1_paths = case_info['m']
    ref_label2_paths = case_info['y']

    # calculate image patch size based on physical patch size
    physical_patch_size = [50,50,50] # mm
    physical_voxel_size = get_nifti_pixdim(source_image_path) # since images are registered, 
                                                              # it is also the voxel size of reference images
    image_patch_size = list(np.ceil(
                        [physical_patch_size[0] / physical_voxel_size[0], 
                        physical_patch_size[1] / physical_voxel_size[1],
                        physical_patch_size[2] / physical_voxel_size[2]]).astype('int'))

    print('voxel size = %.2fx%.2fx%.2f mm^3' % ( physical_voxel_size[0], physical_voxel_size[1], physical_voxel_size[2] ))
    print('patch size = %dx%dx%d voxels' % (image_patch_size[0], image_patch_size[1], image_patch_size[2]))

    # calculate rough brain mask and valid score mask
    print('loading labels...')
    m_i = [(load_nifti_simple(item) > 0.5).astype('float32') for item in ref_label1_paths]    
    m_prob_brain = group_mean(m_i)
    m_rough_brain = (m_prob_brain > 0.5).astype('int')
    x_prime, hdr = load_nifti(source_image_path)
    x_prime = z_score(x_prime, mask=m_rough_brain)
    if apply_otsu:    # otsu thresholding
        otsu_thr = threshold_otsu(np.where(m_rough_brain<0.5,x_prime.min(),x_prime))
        m_otsu = np.where(x_prime > otsu_thr, 1, 0)
    else:
        m_otsu = np.ones_like(x_prime)
    m_valid_score = m_rough_brain * m_otsu

    print('image size = %dx%dx%d voxels' % (x_prime.shape[0],x_prime.shape[1],x_prime.shape[2]))

    # process target image
    print('processing target image')
    tissue_min = np.ma.masked_array(x_prime,mask=1-m_rough_brain).min()
    x_prime = np.where( m_rough_brain < 0.5, tissue_min, x_prime )
    
    # process reference images
    print('processing reference images')
    sample_size = len(ref_image_paths)
    x_i = []
    for i in range(sample_size):
        t = load_nifti_simple(ref_image_paths[i])
        t = z_score(t, mask=m_rough_brain)
        tissue_min = np.ma.masked_array(t,mask=1-m_rough_brain).min()
        t = np.where( m_rough_brain < 0.5, tissue_min, t ) # replace intensities in invalid regions to tissue_min
        x_i.append(t)

    # align local mean value
    print('aligning local mean values')
    x_prime_local_mu, _ = mean_std_grid(x_prime, image_patch_size, mask=m_valid_score)
    if mean_correction:
        for i in range(sample_size):
            x_i_local_mu, _ = mean_std_grid(x_i[i], image_patch_size, mask=m_valid_score)
            x_i[i] = x_i[i] - x_i_local_mu + x_prime_local_mu # align reference to target
    x_prime_local_mu = x_prime_local_mu * m_valid_score # mask invalid regions to make better visualization

    # calculate anomaly score and apply masks
    print('calculate anomaly score and apply masks')
    min_std = 0.03 # setting up min_std, usually calculated stds would not be that low 
                   # (usually they are around 0.1), but in case something unexpected 
                   # happens, a minimum value is still required.
    anomaly, x_mean, x_std = nll(x_prime,x_i, min_std=min_std,side=intensity_prior, return_all=True)
    anomaly = anomaly * component_filtering(m_valid_score, physical_voxel_size)

    # calculate anomaly score for reference samples
    print('calculate anomaly scores for reference samples')
    anomaly_refs = []
    for i in range(len(x_i)):
        s = x_i[i]
        anomaly_ref = nll(s,x_i, min_std=min_std,side=intensity_prior)
        anomaly_ref = anomaly_ref * m_valid_score
        anomaly_refs.append(anomaly_ref)

    # compute histogram curves
    print('computing histograms...')
    curve_x, curve_y, curve_r, curve_rs = histogram_analysis(anomaly,anomaly_refs, mask=m_valid_score)
    hist_save_file = join_path(case_output_folder, 'histogram_curves.png')
    hist_plot(curve_x, curve_y, curve_r, curve_rs, hist_save_file,thresholds=None)

    # determine segmentation thresholds
    anomaly_threshold = None
    zero_crossings = []
    for i in range(len(curve_rs)):
        for j in range(len(curve_rs[i])-1,0,-1):
            if curve_rs[i][j] > 0.01:
                zero_crossings.append(curve_x[j])
                break
    print('zero crossing points:')
    zero_crossings = np.sort( zero_crossings )
    for i in zero_crossings:
        print('%.2f ' % i,end='')
    print('')
    anomaly_threshold = np.median(zero_crossings)
    print('segmentation threshold (initial guess): %.2f' % anomaly_threshold)

    # average prior masks and mask out background
    y_i = [ load_nifti_simple(item) for item in ref_label2_paths ]
    averaged_label = average_contiguous_labels( y_i )
    anomaly = anomaly * (averaged_label > 0.5).astype('float32')

    ################################################################
    ## Apply priors.                                              ##
    ## * You need to provide an image label describing the tissue ##
    ##   type of each voxel.                                      ##
    ##     0: background                                          ##
    ##     1: cerebrum (without cortex)                           ##
    ##     2: cerebellum and brainstem                            ##
    ##     3: cerebral cortex                                     ##
    ## "examples/Brain_label.nii.gz" shows an example of this.    ##
    ################################################################
    
    print('using 3mm median filtering for cerebellum and brainstem.')
    cb_mask = ((1.5 < averaged_label) * (averaged_label < 2.5)).astype('float32')
    anomaly_cb = median_3mm( anomaly, physical_voxel_size )
    anomaly = np.where( cb_mask > 0.5, anomaly_cb, anomaly )

    print('masking non brain tissue (accurate & majority voting)...')
    tissue_sum = np.zeros(m_valid_score.shape).astype('float32')
    for t in y_i:
        tissue_sum += (t>0.5).astype('float32')
    tissue_sum = (tissue_sum > (sample_size/2)).astype('float32')
    anomaly = anomaly * tissue_sum

    # calculate intensity threshold from anomaly score    
    d = 2*(anomaly_threshold - np.log(x_std*2.506))
    d = np.where(d<0, np.nan, d) # replace negative values to nan 
                                 # (sqrt will produce complex numbers for negatives)
    x_thr = x_mean + x_std * np.sqrt(d)
    x_thr = x_thr * m_valid_score
    x_std = x_std * m_valid_score

    # save results for visualization
    save_nifti(x_prime,          hdr, join_path(case_output_folder, 'normalized_input.nii.gz'))
    save_nifti(anomaly,          hdr, join_path(case_output_folder, 'anomaly_score.nii.gz'))
    save_nifti(m_valid_score,    hdr, join_path(case_output_folder, 'valid_mask.nii.gz'))

    if debug: # note: debug mode will save lots of intermediate results
        print('saving intermediate results...')
        save_nifti(x_thr,            hdr, join_path(case_output_folder, 'intensity_thr.nii.gz'))
        save_nifti(m_rough_brain,    hdr, join_path(case_output_folder, 'rough_brain.nii.gz'))
        save_nifti(x_prime_local_mu, hdr, join_path(case_output_folder, 'local_mean.nii.gz'))
        save_nifti(x_mean,           hdr, join_path(case_output_folder, 'mean_value.nii.gz'))
        save_nifti(x_std,            hdr, join_path(case_output_folder, 'std_value.nii.gz'))
        save_nifti(averaged_label,   hdr, join_path(case_output_folder, 'averaged_label.nii.gz'))
        ref = mkdir(join_path(case_output_folder, 'references'))
        for p, x, a in zip(ref_image_paths, x_i, anomaly_refs):
            n = gn(p,no_extension=True)            
            m, _ = mean_std_grid(x, image_patch_size, mask=m_valid_score)
            tissue_min = np.ma.masked_array(x,mask=1-m_rough_brain).min()
            x = np.where( m_rough_brain < 0.5, tissue_min, x )
            m = m * m_valid_score 
            a = a * m_valid_score
            save_nifti(x, hdr, join_path(ref,n+'.nii.gz'))
            save_nifti(m, hdr, join_path(ref,n+'_local_mean.nii.gz'))
            save_nifti(a, hdr, join_path(ref,n+'_anomaly.nii.gz'))

    return anomaly, m_valid_score, curve_x, curve_y, curve_r, anomaly_threshold

def _parallel_lesion_analysis(params):
    case, data_dict, output_folder, intensity_prior, \
    normalization_method, apply_otsu, class_name, debug = params

    print(case)

    case_info = data_dict[case]
    case_output_folder = mkdir(join_path(output_folder,case))
    summary_save_path = join_path(case_output_folder,'summary.pkl')
    
    if file_exist(summary_save_path):
        print('file "%s" already exists, skip.' % summary_save_path)
        return
    else:
        _, _, curve_x, curve_y, curve_r, segmentation_threshold = \
            nll_analysis(case_info, apply_otsu=apply_otsu,
            intensity_prior=intensity_prior, case_output_folder=case_output_folder,debug=debug)
        
        # save original image to this folder
        cp(case_info['x'], join_path(case_output_folder, 'preprocessed_image.nii.gz'))
        
        # generate summary for this case and save it as pkl
        case_summary={
            'preprocessed_image': case_info['x'],
            'analyzer_name': class_name,
            'normalization_method': normalization_method,
            'apply_otsu': apply_otsu,
            'output_folder': case_output_folder,
            'anomaly_score': join_path(case_output_folder, 'anomaly_score.nii.gz'),
            'otsu_mask': join_path(case_output_folder, 'otsu_mask.nii.gz') if apply_otsu == True else None,
            'histogram_curves' : {
                'x': curve_x,
                'y': curve_y,
                'r': curve_r
            },
            'autoseg_threshold': segmentation_threshold
        }
        save_pkl(case_summary, summary_save_path)
        print(case,'OK.')

def _parallel_post_processing(params):
    case_name, preprocessed_image, input_segmentation, output_segmentation = params
    print('>> processing case "%s"' % case_name)
    if file_exist(output_segmentation):
        print('done. pass')
        return
    phys_voxel_size = get_nifti_pixdim(preprocessed_image)
    seg3d = load_nifti_simple(input_segmentation)
    seg3d_pp = remove_3mm_sparks(seg3d, phys_voxel_size)
    save_nifti(seg3d_pp, get_nifti_header(preprocessed_image), output_segmentation)
    print('post-processed segmentation saved to "%s".' % output_segmentation)

def _parallel_segmentation(params):
    case, output_folder = params
    case_output_folder = join_path(output_folder, case)
    seg_path = join_path(case_output_folder, 'segmentation.nii.gz')
    summary_file = join_path(case_output_folder, 'summary.pkl')

    if try_load_nifti(seg_path) == False:
        summary = load_pkl(summary_file)
        # segment image
        preprocessed_image = summary['preprocessed_image']
        nll_score = summary['anomaly_score']
        segmentation_threshold = summary['autoseg_threshold']
        segmentation = (load_nifti_simple(nll_score) > segmentation_threshold).astype('float32')
        # save segmentation result
        save_nifti(segmentation, get_nifti_header(preprocessed_image), seg_path)
        # update summary and overwrite
        summary['final_threshold'] = segmentation_threshold
        summary['segmentation_file'] = seg_path
        save_pkl(summary, summary_file)
        with open(join_path(case_output_folder, 'segmentation.txt'), 'w') as f:
            f.write('case name: %s\n' % case)
            f.write('segmentation threshold: %.4f\n' % segmentation_threshold)

    else:
        return

class LesionAnalyzer(object):

    def log(self, msg, print_to_console=True):
        if self.logger is not None:
            self.logger.write(msg,timestamp=True)
        if print_to_console:
            print(msg)

    def __init__(self, output_folder, num_workers=8, logger = None):
        self.data_dict = {}
        self.output_folder = mkdir(output_folder)
        self.normalization_method = 'z_score'
        self.apply_otsu = True
        self.num_workers = num_workers
        self.time_stamps = TimeStamps()
        if logger is not None:
            assert isinstance(logger, SimpleTxtLog)
            self.logger = logger
        else:
            self.logger = None
        
        ### set this to True to save intermediate results ###
        self.debug = False
    
    def add_case(self, name, x_input, x_refs, label1, label2):        
        self.data_dict[name] = { 'x': x_input, 'r': x_refs, 'm': label1, 'y': label2}
    
    def analyze_and_do_segmentation(self, intensity_prior = None, do_postprocessing=True):
        assert intensity_prior in [None, '+', '-']
        self.time_stamps.record('segmentation_start')

        #
        # analyze
        #
        self.log('')
        self.log('+-------------------------------------+')
        self.log('|  Step I: computing anomaly scores   |')
        self.log('| and set thresholds for segmentation |')
        self.log('+-------------------------------------+')
        self.log('analyzer class name: "%s".' % type(self).__name__)
        self.log('start to analyze images using negative log likelihood...')
        all_cases = list(self.data_dict.keys())
        self.log('total number of case(s) need to analyze: %d' % len(all_cases))
        self.log('output folder is: "%s".' % self.output_folder)
        self.log('start time: %s' % self.time_stamps.get('segmentation_start'))
        self.log('')
        
        self.log('** parallel computation starts (%d workers)...' % self.num_workers)

        # computing NLL scores serially in a single CPU core are a bit slow 
        # (about 2~5 min per case) so I changed them to execute in parallel.
        task_list = []
        for case in all_cases:
            task_args = (case, self.data_dict, self.output_folder,
                        intensity_prior, self.normalization_method, self.apply_otsu, 
                        type(self).__name__,self.debug)
            task_list.append(task_args)
        
        if self.debug:
            # force single thread when debugging and enable outputs
            print('DEBUG MODE IS ON. Parallelism is disabled.')
            run_parallel(_parallel_lesion_analysis, task_list, 1, 'analyzing', print_output=True)
        else:
            run_parallel(_parallel_lesion_analysis, task_list, self.num_workers, 'analyzing')

        self.log('** parallel computation finished.')

        # do segmentation
        self.log('+-------------------------------+')
        self.log('|  Step II: image segmentation  |')
        self.log('+-------------------------------+')
        tasks = []
        for case in all_cases:
            tasks.append( (case, self.output_folder) )
        run_parallel(_parallel_segmentation, tasks, self.num_workers, 'lesion segmentation')                
        self.log('segmentation finished.')
        
        # do post-processing
        if do_postprocessing:
            self.log('+-----------------------------+')
            self.log('|  Step III: post-processing  |')
            self.log('+-----------------------------+')
            self.do_postprocessing()

        self.time_stamps.record('segmentation_end')
        self.log('all finished.')
        self.log('--------------------------')
        self.log('>> begins at : %s' % self.time_stamps.get('segmentation_start'))
        self.log('>> ends at   : %s' % self.time_stamps.get('segmentation_end'))
        self.log('--------------------------')
        
    def do_postprocessing(self):
        # remove sparks
        all_cases = list(self.data_dict.keys())
        task_list = []
        for case in all_cases:
            preprocessed_image = join_path(self.output_folder, case, 'preprocessed_image.nii.gz')
            input_segmentation = join_path(self.output_folder, case, 'segmentation.nii.gz')
            output_segmentation = join_path(self.output_folder, case, 'segmentation_pp.nii.gz')
            task_args = (case, preprocessed_image, input_segmentation, output_segmentation)
            task_list.append(task_args)

        run_parallel(_parallel_post_processing, task_list, self.num_workers, 'post-processing')

