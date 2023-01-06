from freeseg.external_tools.ANTs_group_registration import get_ANTs_version_string
import os, platform
from freeseg.utilities.external_call import try_shell
from freeseg.utilities.misc import printv
from freeseg.utilities.file_ops import file_exist, join_path
from freeseg.external_tools.FreeSurfer_aseg import check_FreeSurfer_install, get_FreeSurfer_version_string

def check_system_integrity(verbose=True, 
    ignore_ANTs = False, 
    ignore_nnUNet = False,
    ignore_FreeSurfer = False,
    ignore_FSL = False,
    ignore_ROBEX = False,
    ignore_GPU_check = False):
    
    printv('[*] Checking system integrity, please wait...' ,verbose=verbose)
    success = True

    system = platform.system()
    if system != 'Linux':
        printv('[!] Warning: you are trying to run this pipeline on a non Linux-based operating system.', 
        verbose=verbose)

    # ANTs
    if not ignore_ANTs:
        ANTs_failed = False
        if try_shell('antsRegistration -h') != 0:
            success = False
            ANTs_failed = True
            printv('[X] Cannot find "antsRegistration". Please compile & install ANTs from '
                '"https://github.com/ANTsX/ANTs".', verbose=verbose)
        if try_shell('antsGroupRegistration -h') != 0:
            success = False
            ANTs_failed = True
            printv('[X] Cannot find "antsGroupRegistration". Try re-install this tool to fix the problem.', 
                verbose=verbose)
        if try_shell('N4BiasFieldCorrection -h') != 0:
            success = False
            ANTs_failed = True
            printv('[X] Cannot find "N4BiasFieldCorrection". Please compile & install ANTs from '
                '"https://github.com/ANTsX/ANTs".', verbose=verbose)
        
        if ANTs_failed:
            printv('    ** After installation you need to append:        \n\n'
                   '       export ANTSPATH=/usr/local/ANTs-*.*/bin       \n'
                   '       export PATH="/usr/local/ANTs-*.*/bin:$PATH"   \n\n'
                   '       to your ~/.bashrc, and run "source ~/.bashrc" \n'
                   '       to update your settings (replace "*.*" to     \n'
                   '       actual version number of ANTs you installed on\n'
                   '       your machine, such as "2.1", "2.3", etc.).',verbose=verbose)
        else:
            ANTs_version = get_ANTs_version_string()
            printv('[*] ANTs version: "%s".' % ANTs_version, verbose=verbose)

    # nnU-Net
    if not ignore_nnUNet:
        if try_shell('nnUNet_train -h') != 0:
            success = False
            printv('[X] Cannot find "nnUNet_train". Please install nnU-Net from '
                '"https://github.com/lchdl/nnUNet".', verbose=verbose)
        if try_shell('nnUNet_predict -h') != 0:
            success = False
            printv('[X] Cannot find "nnUNet_predict". Please install nnU-Net from '
                '"https://github.com/lchdl/nnUNet".', verbose=verbose)

    # FreeSurfer
    if not ignore_FreeSurfer:
        if check_FreeSurfer_install() == False:
            success = False
            printv('[!] Warning: cannot launch FreeSurfer. You can install FreeSurfer from '
                '"https://surfer.nmr.mgh.harvard.edu/". Note that you may also need to install '
                '"csh" and "tcsh" shell by executing "sudo apt-get install csh tcsh" command.', verbose=verbose)
        else:
            FreeSurfer_version = get_FreeSurfer_version_string()
            printv('[*] FreeSurfer version: "%s".' % FreeSurfer_version, verbose=verbose)
    
    # FSL
    if not ignore_FSL:
        FSL_failed = False
        if try_shell('flirt') != 1:
            success = False
            FSL_failed = True
            printv('[X] Cannot find "flirt". Please install FSL from '
                '"https://fsl.fmrib.ox.ac.uk/fsl/fslwiki".', verbose=verbose)
        if try_shell('robustfov') != 1:
            success = False
            FSL_failed = True
            printv('[X] Cannot find "robustfov". Please install FSL from '
                '"https://fsl.fmrib.ox.ac.uk/fsl/fslwiki".', verbose=verbose)
        if try_shell('bet') != 1:
            success = False
            FSL_failed = True
            printv('[X] Cannot find "bet". Please install FSL from '
                '"https://fsl.fmrib.ox.ac.uk/fsl/fslwiki".', verbose=verbose)
        
        if not FSL_failed:
            FSL_version, _ = try_shell('flirt -version', stdio=True)
            FSL_version = FSL_version.replace('\n', ' ')
            printv('[*] FSL: "%s".' % FSL_version, verbose=verbose)
            
    # ROBEX
    if not ignore_ROBEX:
        if os.environ.get('ROBEX_DIR', None) == None:
            success = False
            printv('[X] Environment variable "ROBEX_DIR" not set. ROBEX can be downloaded from "https://www.nitrc.org/projects/robex". '
                'Also please add:\n\nexport ROBEX_DIR="..."\n\nin your ~/.bashrc. '
                'Recommended version >=1.2. '
                'For Windows, you need to create an environment variable named as "ROBEX_DIR".',
                verbose=verbose)
        else:
            ROBEX_DIR = os.environ['ROBEX_DIR']
            ROBEX_SH = join_path(ROBEX_DIR, 'runROBEX.sh')
            ROBEX_BIN = join_path(ROBEX_DIR, 'ROBEX')
            if file_exist(ROBEX_SH) == False:
                success = False
                printv('[X] Cannot find "runROBEX.sh" in directory "%s".' % ROBEX_DIR, verbose=verbose)
            elif file_exist(ROBEX_BIN) == False:
                success = False
                printv('[X] Cannot find "ROBEX" executable in directory "%s".' % ROBEX_DIR, verbose=verbose)
            else:
                if try_shell(ROBEX_SH) != 0:
                    success = False
                    printv('[X] Cannot launch ROBEX. Make sure the ROBEX_DIR is '
                        'correct and both "ROBEX" and "runROBEX.sh" have '
                        'executable permissions (+x).', verbose=verbose)

    # checking if GPU is available
    if not ignore_GPU_check:
        try:
            import torch
        except ImportError:
            success = False
            printv('[X] Cannot import PyTorch. Maybe you need to install nnU-Net from '
                '"https://github.com/lchdl/nnUNet" before using this tool.', verbose = verbose)
        except BaseException as e:
            if isinstance(e, Exception):
                # general exceptions
                success = False
                printv('[X] Unknown error occurred when importing PyTorch.')
            else:
                # system exceptions
                raise e
        else:
            if torch.cuda.is_available() == False:
                printv('[!] Warning: cannot find a GPU that supports CUDA. Maybe your GPU driver version '
                'is not compatible with the PyTorch CUDA requirements. Try upgrade your GPU driver or '
                're-install PyTorch compiled from older CUDA versions from '
                '"https://pytorch.org/get-started/previous-versions/". Now the pipeline will use CPU for '
                'training & inference instead, which is pretty slow.', verbose=verbose)
    
    if not success:
        printv('[X] System integrity check failed. '
            'Make sure you activated the correct environment and '
            'installed all external dependencies.', verbose=verbose)
    else:
        printv('[*] OK.', verbose=verbose)

    return success

def check_dataset(dataset):
    def _is_valid(s):
        '''
        check if a string has invalid characters
        '''
        for ch in s:
            if ch.isalnum() or ch in ['.', '_', '-']:
                continue
            else: # return status and invalid character, space ' ' is considered invalid.
                return False, ch
        return True, None

    class DatasetError(Exception):
        pass

    success = True
    try:
        case_list = dataset['case']
        data_list = dataset['flair']
        mask_list = dataset['label1'] if 'label1' in dataset else []
        prior_list = dataset['label2'] if 'label2' in dataset else []

        # check case names

        for case in case_list:
            success, ch = _is_valid(case)
            if not success:
                raise DatasetError(
                    "\"%s\": case name can only contains numbers (0~9), "
                    "letters ('A'~'Z', 'a'~'z') and special characters ('.', '_', '-'). "
                    "Got '%s'." % (case, ch))
            if case.find('_to_')!=-1:
                raise DatasetError(
                    '"%s": case name contains invalid keyword "_to_", '
                    'please change it to another name.')

        # check file names

        file_list = data_list + mask_list + prior_list

        for data in file_list:
            success, ch = _is_valid(case)
            if not success:
                raise DatasetError(
                    "\"%s\": file path can only contains numbers (0~9), "
                    "letters ('A'~'Z', 'a'~'z') and special characters ('.', '_', '-'). "
                    "Got '%s'." % (data, ch))
            if file_exist(data) == False:
                raise DatasetError('cannot find file "%s".' % data)
    except DatasetError as e:
        success = False
        msg = str(e)
        print('[!]', msg)
    except:
        raise
    
    return success
