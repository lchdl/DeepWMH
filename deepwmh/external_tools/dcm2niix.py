import argparse
from deepwmh.utilities.external_call import run_shell, try_shell
from deepwmh.utilities.file_ops import gd, laf, mkdir

def main():
    parser = argparse.ArgumentParser(
        description='dcm2niix_py: simple python wrapper for dcm2niix. '
        'For more information about "dcm2niix", please visit: '
        'https://github.com/rordenlab/dcm2niix. This script is just a simple '
        'utility script. You need to manually complie dcm2niix before using '
        'this utility.')

    parser.add_argument('-i', '--input-folder', 
                                            help='DICOM root folder. This script will automatically '
                                            'find all *.dcm files recursively and convert them into '
                                            'different NIFTI files.', 
                                            type=str,required=True)

    parser.add_argument('-o', '--output-folder',
                                            help='This is where all the output files (*.nii.gz and *.json) '
                                            'will be saved. *.nii.gz contains converted data, *.json contains '
                                            'scanning protocol, scanner information and so on.',
                                            type=str, required=True)
    args = parser.parse_args()

    #
    input_folder = args.input_folder
    output_folder = mkdir(args.output_folder)

    # checking if the provided binary executable actually works...
    print('-- checking binary executable "dcm2niix"...')
    if try_shell('dcm2niix -h') != 0:
        print('Failed.')
        raise RuntimeError('Cannot find "dcm2niix", please compile dcm2niix from '
                           '"https://github.com/rordenlab/dcm2niix" and add it to "PATH" '
                           'environment variable.')
    else:
        print('OK.')
    
    #
    print('start conversion...')
    print('listing all dicom files recursively in folder "%s", please wait...' % input_folder)
    all_dicom_files = laf(input_folder)
    all_folders = []
    for dicom in all_dicom_files:
        dicom_folder = gd(dicom)
        if dicom_folder not in all_folders:
            all_folders.append(dicom_folder)
    print('%d dicom files found in %d different folders.' % (len(all_dicom_files), len(all_folders)))
    
    #
    success, failed = 0, 0
    for folder in all_folders:
        conversion_command = 'dcm2niix -9 -o %s -i y -z i %s' % (output_folder, folder)
        # -9     : max compression
        # -o ... : output folder
        # -i y   : ignore derived, localizer and 2D images
        # -z i   : gz compress images [i=internal:zlib]
        state = run_shell(conversion_command, force_continue=True)
        if state != 0: 
            failed += 1
        else:
            success += 1
    
    print('========')
    print('Conversion complete. %d success, %d failed.' % (success, failed))



    







