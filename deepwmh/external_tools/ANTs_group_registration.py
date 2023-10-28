from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.external_call import run_shell, try_shell
from deepwmh.utilities.file_ops import abs_path, cwd, gd, join_path, make_unique_dir, mkdir, mv, rm
from deepwmh.utilities.data_io import load_csv_simple, try_load_mat, try_load_nifti
import argparse
import warnings

###################################################################################
# This script wraps the registration command into a simple Python function call   #
# (they are: "image_registration_with_label(...)" and "image_registration(...)"). #
#                                                                                 #
# For more tutorials about medical image registration using ANTs, please          #
# visit: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call.  # 
# "antsRegistration" can be used to register two images even if their imaging     #
# modalities are different from each other.                                       #
###################################################################################


def get_ANTs_version_string():
    stdout, _ = try_shell('antsRegistration --version', stdio=True)
    stdout = stdout.replace('\n',' ')
    return stdout

def antsRegistration(source, target, warped, 
    interpolation_method='Linear', use_histogram_matching=False,
    deform_type='Elastic',advanced_config=None) -> str:
    '''
    Description
    ------------
    Generates bash call from Python function to register a pair of medical
    images using ANTs. Combining it with run_shell(...) to launch the task.
    Deformation fields will be saved under the directory where the warped
    image is located.

    Usage
    ------------
    >>> source = '/path/to/source.nii.gz'
    >>> target = '/path/to/target.nii.gz'
    >>> warped = '/path/to/warped.nii.gz'
    >>> run_shell( antsRegistration(source, target, warped, interpolation_method='Linear') )

    Parameters
    ------------
    source: str
        source image path
    target: str
        target image path
    warped: str
        warped output image path
    deform_type: str
        deform type can be either "Linear" or "Elastic" (default).
        Linear = Rigid + Affine transforms,
        Elastic = Rigid + Affine + SyN transforms.
    advanced_config: dict
        advanced configurations for deformable registration method 
        "SyN". Usually you don't need to manually adjust this.
    '''

    assert deform_type in ['Elastic', 'Linear'], 'unknown deformation type.'
    assert interpolation_method in ['Linear', 'NearestNeighbor'], 'unknown interpolation method.'
    assert use_histogram_matching in [True, False], 'invalid parameter setting for "use_histogram_matching".'
    
    output_directory = gd(abs_path(warped))
    mkdir(output_directory)
    output_transform_prefix = join_path(output_directory,'warp_')

    # fill in default configurations
    config = {
        'SyN_gradientStep' : 0.1,
        'SyN_updateFieldVarianceInVoxelSpace' : 3.0,
        'SyN_totalFieldVarianceInVoxelSpace' : 0.0,
        'SyN_CC_neighborVoxels': 4,
        'SyN_convergence' : '100x70x50x20',
        'SyN_shrinkFactors' : '8x4x2x1',
        'SyN_smoothingSigmas' : '3x2x1x0'
    }
    if advanced_config is not None:
        for key in advanced_config:
            if key in config:
                config[key] = advanced_config[key] # override setting
            else:
                warnings.warn('Unknown config setting "%s".' % key, UserWarning)

    # generate registration command
    command = 'antsRegistration '
    command += '--dimensionality 3 '                                     # 3D image
    command += '--float 1 '                                              # 0: use float64, 1: use float32 (save mem)
    command += '--collapse-output-transforms 1 '
    command += '--output [%s,%s] ' % (output_transform_prefix,warped)
    command += '--interpolation %s ' % interpolation_method
    command += '--use-histogram-matching %s ' % ( '0' if use_histogram_matching == False else '1')
    command += '--winsorize-image-intensities [0.005,0.995] '
    command += '--initial-moving-transform [%s,%s,1] ' % (target,source) # initial moving transform
    command += '--transform Rigid[0.1] '                                 # rigid transform
    command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
    command += '--convergence [1000x500x250x0,1e-6,10] '
    command += '--shrink-factors 8x4x2x1 '
    command += '--smoothing-sigmas 3x2x1x0vox '
    command += '--transform Affine[0.1] '                                # affine transform
    command += '--metric MI[%s,%s,1,32,Regular,0.25] ' % (target,source)
    command += '--convergence [1000x500x250x0,1e-6,10] '
    command += '--shrink-factors 8x4x2x1 '
    command += '--smoothing-sigmas 3x2x1x0vox '
    if deform_type == 'Elastic':
        # For medical image registration with large image deformations, maybe adding a
        # "TimeVaryingVelocityField" transform is better (see https://github.com/stnava/C).
        # But for robustness here I will only use SyN (which is the most commonly used
        # method) for deformable registration.
        command += '--transform SyN[%f,%f,%f] ' % \
            (config['SyN_gradientStep'], config['SyN_updateFieldVarianceInVoxelSpace'], \
            config['SyN_totalFieldVarianceInVoxelSpace'])
        command += '--metric CC[%s,%s,1,%d] ' % (target,source, config['SyN_CC_neighborVoxels'])
        command += '--convergence [%s,1e-6,10] ' % (config['SyN_convergence'])
        command += '--shrink-factors %s ' % config['SyN_shrinkFactors']
        command += '--smoothing-sigmas %svox ' % config['SyN_smoothingSigmas']
    elif deform_type == 'Linear':
        # No need to use "SyN" algorithm here as Linear = Rigid + Affine transforms.
        pass
    return command

def antsApplyTransforms(source,reference,transform,output,
    interpolation_method='Linear',inverse_transform=False):

    assert interpolation_method in ['Linear', 'NearestNeighbor'], 'unknown interpolation method.'
    assert inverse_transform in [True,False], 'invalid parameter setting for "inverse_transform".'

    command = 'antsApplyTransforms '
    command += '-d 3 --float --default-value 0 '
    command += '-i %s ' % source
    command += '-r %s ' % reference
    command += '-o %s ' % output
    command += '-n %s ' % interpolation_method
    command += '-t [%s,%s] ' % (transform, ( '0' if inverse_transform == False else '1')  )

    return command
    
def image_registration(moving, fixed, moved, save_deformations_to = None, advanced_config = None):
    try:
        output_directory = mkdir(gd(moved))
        run_shell(antsRegistration(moving, fixed, moved, 
            interpolation_method='Linear', use_histogram_matching=False, advanced_config=advanced_config))
    except:
        raise # if anything bad happens, the temporary file will not be deleted,
              # user need to delete them manually.
    finally:
        # remove temporary files
        if save_deformations_to is not None:
            mkdir(save_deformations_to)
            mv( join_path(output_directory, 'warp_0GenericAffine.mat'), join_path(save_deformations_to, 'warp_0GenericAffine.mat') )
            mv( join_path(output_directory, 'warp_1Warp.nii.gz'), join_path(save_deformations_to, 'warp_1Warp.nii.gz') )
            mv( join_path(output_directory, 'warp_1InverseWarp.nii.gz'), join_path(save_deformations_to, 'warp_1InverseWarp.nii.gz') )
        else:
            rm(join_path(output_directory, 'warp_0GenericAffine.mat'))
            rm(join_path(output_directory, 'warp_1Warp.nii.gz'))
            rm(join_path(output_directory, 'warp_1InverseWarp.nii.gz'))


def _parallel_registration(args):
    # unpack args
    moving, fixed, save_dir, work_dir, \
        allow_large_deformations, allow_quick_registration, keep_deformation = args

    moving_case, moving_img = moving
    fixed_case, fixed_img = fixed
    output_case_name = '%s_to_%s' % (moving_case, fixed_case)
    output_file = join_path(save_dir, '%s.nii.gz' % output_case_name)
    output_deformations = [
        join_path(save_dir, output_case_name, 'warp_0GenericAffine.mat'), # rigid + affine
        join_path(save_dir, output_case_name, 'warp_1Warp.nii.gz'),       # forward elastic deformation (S->T)
        join_path(save_dir, output_case_name, 'warp_1InverseWarp.nii.gz') # backward elastic deformation (T->S)
    ]

    # check if this registration can be skipped (finished from previous run)

    files_need_to_exist = [output_file] if keep_deformation == False else [output_file] + output_deformations
    skip = True
    for file in files_need_to_exist:
        if file.endswith('.nii.gz'):
            if try_load_nifti(file) == False:
                skip = False
                break
        if file.endswith('.mat'):
            if try_load_mat(file) == False:
                skip = False
                break
                    
    if skip: 
        return

    try:
        advanced_config = {}
        if allow_large_deformations:
            advanced_config['SyN_gradientStep'] = 0.3
            advanced_config['SyN_updateFieldVarianceInVoxelSpace'] = 3
            advanced_config['SyN_convergence'] = '200x100x50x25'
            advanced_config['SyN_shrinkFactors'] = '8x4x2x1'
            advanced_config['SyN_smoothingSigmas'] = '3x2x1x0'
        if allow_quick_registration:
            advanced_config['SyN_convergence'] = '200x100x50'
            advanced_config['SyN_shrinkFactors'] = '8x4x2'
            advanced_config['SyN_smoothingSigmas'] = '3x2x1'
        temp_dir = make_unique_dir(basedir=work_dir)
        print(temp_dir)
        temp_output = join_path(temp_dir,'%s.nii.gz' % output_case_name)
        if not keep_deformation:
            image_registration( moving_img, fixed_img, temp_output , save_deformations_to=None, advanced_config=advanced_config)
        else:
            deformation_dir = mkdir(join_path(save_dir, output_case_name))
            image_registration( moving_img, fixed_img, temp_output , save_deformations_to=deformation_dir, advanced_config=advanced_config)
        mv(temp_output, output_file)
        
    except:
        raise

    else:
        print('cleaning up...')
        rm(temp_dir)

# launch from class instance
class ANTsGroupRegistration(object):
    def __init__(self, sources, targets, save_dir, work_dir, num_workers, 
        allow_large_deformations, allow_quick_registration, keep_deformation):
        
        '''
        sources: [(case1, img1), (case2, img2), ... , (caseN, imgN)]
        targets: [(case1, img1), (case2, img2), ... , (caseN, imgN)]
        '''

        self.sources = sources
        self.targets = targets
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.allow_large_deformations = allow_large_deformations
        self.allow_quick_registration = allow_quick_registration
        self.keep_deformation = keep_deformation
        self.work_dir = work_dir

    def _organize_tasks(self, distributed = 'none'):

        task_list = []
        for i in range(len(self.sources)):
            for j in range(len(self.targets)):
                task_args = (self.sources[i], self.targets[j], self.save_dir, self.work_dir, \
                    self.allow_large_deformations, self.allow_quick_registration, 
                    self.keep_deformation)
                task_list.append(task_args)
            
        if distributed != 'none':
            sub_task, total_task = distributed.split('/')
            sub_task, total_task = int(sub_task), int(total_task)
            if sub_task < 1 or sub_task > total_task:
                raise RuntimeError('parameter error : "distributed"')
            sub_task_list = []
            for i in range(len(task_list)):
                if i % total_task == sub_task - 1:
                    sub_task_list.append(task_list[i])
            task_list = sub_task_list # override task list
            print('Distributed task %d/%d, %d image registration operations.' % (sub_task, total_task, len(task_list)))
        else:
            print('Running full task (%d image registration operations).' % len(task_list))
        return task_list

    def launch(self, task_list = None):

        mkdir(self.save_dir)
        mkdir(self.work_dir)

        if task_list == None:
            task_list = self._organize_tasks()

        print('start image registration.')
        run_parallel(_parallel_registration, task_list, self.num_workers, "registration")
        print('registration finished.')


# or launch from bash console
def main():

    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, 
                      argparse.RawDescriptionHelpFormatter):  pass

    parser = argparse.ArgumentParser(
        description=
            'antsGroupRegistration: A utility program used to register two sets of images '
            'using antsRegistration. '
            'You need to install ANTs toolkit (https://github.com/ANTsX/ANTs) before using '
            'this script. The two sets of images are indicated by the contents of the two '
            'comma separated files (*.csv).\n\n'
            '* The contents of source csv should be something like this: \n\n'
            'case,data\n'
            '0001,/path/to/nii1.nii.gz\n'
            '0002,/path/to/nii2.nii.gz\n'
            '...\n\n'
            '* The contents of target csv should be something like this: \n\n'
            'case,data\n'
            '0001,/path/to/nii1.nii.gz\n'
            '0002,/path/to/nii2.nii.gz\n'
            '...\n\n'
            '--------------------\n'
            'For more tutorials about medical image registration using ANTs, please '
            'visit: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call, '
            'the link above gives a detailed explanation of ANTs registration command and '
            'some basic principles of medical image registration. '
            '"antsRegistration" can be used to register two images even if their imaging '
            'modalities are different from each other.',formatter_class=MyFormatter)
    parser.add_argument('-s', '--source', help='Source image dataset.',type=str,required=True)
    parser.add_argument('-t', '--target', help='Target image dataset.',type=str,required=True)
    parser.add_argument('-o', '--output-dir',required=True, 
        help='Target output directory used for saving all the registered images.', type=str)
    parser.add_argument('-j', '--num-workers', 
        help=
            'Number of processes used for registration. This applies to servers '
            'that has lots of CPU cores. Although antsRegistration is multithreaded, '
            'it doesn\'t use all the computational powers of CPU cores. Changing '
            'number of workers to an integer larger than 1 can further speed up the '
            'registration process.', type=int, default=2)
    parser.add_argument('--allow-large-deformations', 
        help='Allowing large image deformations when doing registration. '
            'This applies to images that have large variations on ventricle sizes.',
        action='store_true',
        default=False)        
    parser.add_argument('--allow-quick-registration',
        help='Allowing quick image registration by down-sampling images (2x). '
             'About 30%%~40%% performance gain is expected. However, you may need '
             'to manually check the quality of registered images.',action='store_true',
        default=False)
    parser.add_argument('--keep-deformation',
        help='Keep deformation fields once the registration is finished. The saved deformations '
             'can be further used later (such as using antsApplyTransforms). By default it is set '
             'to "False" because the deformation fields can consume lots of disk space (~6x).',
        action='store_true',
        default=False)
    parser.add_argument('-d', '--distributed',type=str,default='none',
        help=
            'Distribute the whole task into several sub-tasks that can be manually '
            'assigned for multiple servers. Format is "a/b". For example, if you want '
            'to divide the whole registration into three parts and this is the second '
            'part, you should write "--distributed 2/3". In this way, you can divide '
            'the whole registration task into multiple parts that can be executed '
            'concurrently on different machines.')
    parser.add_argument('-z', '--working-directory',type=str,
        help=
            'During registration the program will generates some temporary output files. '
            'You can change this directory to anywhere you want. All the intermediate '
            'files will be saved to this directory and will be automatically deleted once '
            'the registration is finished.',
        required = False, default='none')

    args = parser.parse_args()

    # passing arguments
    source_csv = args.source
    target_csv = args.target
    save_dir = args.output_dir
    num_workers = args.num_workers
    allow_large_deformations = args.allow_large_deformations
    allow_quick_registration = args.allow_quick_registration
    distributed = args.distributed
    work_dir = cwd() if args.working_directory == 'none' else args.working_directory
    keep_deformation = args.keep_deformation

    # checking if ANTs is installed, print tips if not
    if try_shell('antsRegistration --version') == 127:
        print('ANTs toolkit is not properly installed in your system! '
              'Please install ANTs before using this tool!\n'
              'Also note that you need to manually add \n\n'
              'export ANTSPATH=/usr/local/ANTs-*.*/bin\n'
              'export PATH="/usr/local/ANTs-*.*/bin:$PATH"\n\n'
              'in your ~/.bashrc after installation. You need to '
              'replace "*.*" with the actual ANTs version number installed '
              'in your local machine (e.g.: 2.3, 3.0, ...).')
        exit()

    print('antsGroupRegistration : an utility to register two sets of images using ANTs toolkit.')
    print('source images dataset: "%s"' % source_csv)
    print('target images dataset: "%s"' % target_csv)
    print('allow large deformations ? %s' % ('True' if allow_large_deformations else 'False'))
    print('allow quick registration ? %s' % ('True' if allow_quick_registration else 'False'))
    print('keep deformation ? %s' % ('True' if keep_deformation else 'False'))
    print('number of workers : %d' % num_workers)

    source_dataset = load_csv_simple(source_csv,key_names=['case','data'])
    target_dataset = load_csv_simple(target_csv,key_names=['case','data'])

    S = [ item for item in zip(source_dataset['case'], source_dataset['data']) ]
    T = [ item for item in zip(target_dataset['case'], target_dataset['data']) ]

    img_regist = ANTsGroupRegistration(S, T, save_dir, work_dir, num_workers, 
        allow_large_deformations, allow_quick_registration, keep_deformation)
    task_list = img_regist._organize_tasks(distributed = distributed)
    img_regist.launch(task_list)
