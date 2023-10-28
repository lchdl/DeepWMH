from deepwmh.utilities.file_ops import abs_path, file_exist, gd, join_path, dir_exist, mkdir, rm
from deepwmh.utilities.data_io import load_nifti, save_nifti, try_load_nifti
from deepwmh.utilities.external_call import run_shell, try_shell
from deepwmh.utilities.parallelization import run_parallel
import numpy as np
import os

# FreeSurfer aseg utility
# if you installed FreeSurfer, you can import and call the functions below to generate aseg labels.

def check_FreeSurfer_install():
    '''
    Check if FreeSurfer is successfully installed.
    '''
    if try_shell('recon-all -version') != 0:
        return False
    if try_shell('mri_vol2vol') != 1:
        return False
    return True

def get_FreeSurfer_version_string():
    stdout, _ = try_shell('recon-all -version', stdio=True)
    stdout = stdout.replace('\n','')
    return stdout

def _parallel_FreeSurfer_aseg(params):
    subject_name, t1_path, aseg_path, additional_param = params
    if try_load_nifti(aseg_path) == True:
        return
    SUBJECT_DIR = os.environ['SUBJECTS_DIR']
    if dir_exist(join_path(SUBJECT_DIR, subject_name)):
        rm(join_path(SUBJECT_DIR, subject_name))
    recon_all = 'recon-all -s %s -i %s %s -autorecon1 -gcareg -canorm -careg -rmneck -skull-lta -calabel' % \
        (subject_name,t1_path, additional_param)
    run_shell(recon_all, print_command=False, print_output=False)
    aseg_auto_noCCseg = join_path(SUBJECT_DIR, subject_name, 'mri', 'aseg.auto_noCCseg.mgz')
    mkdir( gd( aseg_path) )
    mri_vol2vol = 'mri_vol2vol '          \
                '--mov %s '               \
                '--targ %s '              \
                '--o %s '                 \
                '--regheader --nearest' % (aseg_auto_noCCseg, t1_path, aseg_path)
    run_shell(mri_vol2vol, print_command=False, print_output=False)

def run_FreeSurfer_aseg(subject_names, t1_paths, aseg_outputs, 
    SUBJECTS_DIR = None, num_workers = 8, additional_param = ''):
    '''
    Run "recon-all" in parallel to generate rough brain segmentations 
    for a group of T1w images.
    '''
    # if FreeSurfer is not installed i will print an error message and quit
    if check_FreeSurfer_install() == False:
        print('[!] FreeSurfer is not installed!')
        exit()
    else:
        print('[*] FreeSurfer version: "%s".' % get_FreeSurfer_version_string())
    # check input parameters
    assert len(subject_names) == len(t1_paths) and \
        len(t1_paths) == len(aseg_outputs), 'cannot zip lists with different lengths.'
    for t1_path in t1_paths:
        assert file_exist(t1_path), 'file not exist or have no access: "%s".' % t1_path
    # making tasks and setting SUBJECT_DIR
    task_list = []
    for name, t1, aseg in zip(subject_names, t1_paths, aseg_outputs):
        task_list.append( (name, t1, aseg, additional_param) )
    if SUBJECTS_DIR is not None:
        print('Setting FreeSurfer SUBJECTS_DIR to "%s".' % abs_path(SUBJECTS_DIR))
        os.environ['SUBJECTS_DIR'] = abs_path(SUBJECTS_DIR)
        mkdir(abs_path(SUBJECTS_DIR))
    # start FreeSurfer
    run_parallel( _parallel_FreeSurfer_aseg, task_list, num_workers, "FreeSurfer aseg")

def convert_FreeSurfer_aseg(aseg_auto_noCCseg_file, output_label, convert_type='cbstemcor'):
    '''
    Convert recon-all aseg result to label index accepted by the pipeline.
    '''
    assert convert_type == 'cbstemcor', 'invalid convert_type.'
    # load label
    aseg_data, header = load_nifti(aseg_auto_noCCseg_file)
    aseg_data = np.around( aseg_data ).astype('int')
    output = np.zeros(aseg_data.shape).astype('int')
    max_label = np.max(aseg_data)
    # convert
    for i in range(1, max_label+1):
        if i in [7,8,46,47]: # cerebellum
            output[aseg_data == i] = 2
        elif i in [15,16]: # brainstem
            output[aseg_data == i] = 2
        elif i in [3,42]: # cortex
            output[aseg_data == i] = 3
        else: # cerebrum white matter
            output[aseg_data == i] = 1
    # save
    save_nifti(output, header, output_label)

