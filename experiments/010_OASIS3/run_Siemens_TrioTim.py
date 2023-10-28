import sys, time
from deepwmh.utilities.file_ops import files_exist, gd, join_path, file_exist, mkdir
from deepwmh.utilities.data_io import try_load_nifti, write_csv_simple
from deepwmh.utilities.external_call import run_shell
from deepwmh.external_tools.FreeSurfer_aseg import convert_FreeSurfer_aseg, run_FreeSurfer_aseg
from deepwmh.main.integrity_check import check_system_integrity

print(
'''
#############################################################################
#                                                                           #
#                     >>> IMPORTANT! READ ME PLEASE <<<                     #
#                                                                           #
# * Because everyone will organize dataset structure in their own ways, the #
# following script is just an example of how to orgainze OASIS-3 dataset in #
# order to train the pipeline for annotation-free WMH segmentation. However,#
# I recommend you to organize your OASIS-3 dataset structure as follows:    #
#                                                                           #
# .../Siemens_TrioTim_TR_9.00_TE_0.08_TI_2.50/        <= later defined as   #
#      |                                                 "OASIS3_folder"    #
#      +--OAS30027_MR_d2394/                          <= case name          #
#      |  +--t1w_raw.nii.gz                           <= raw T1w image      #
#      |  +--t2flair_raw.nii.gz                       <= raw T2-FLAIR image #
#      |                                                                    #
#      +--OAS30074_MR_d1871/                                                #
#      |  +--t1w_raw.nii.gz                                                 #
#      |  +--t2flair_raw.nii.gz                                             #
#      |                                                                    #
#      ... (other cases)                                                    #
#                                                                           #
# * For each case, a T1w scan named as "t1w_raw.nii.gz" and a FLAIR scan    #
# named as "t2flair_raw.nii.gz" are needed. These two files must in the     #
# same directory. Note that all file paths must not contain any space (' ').#
# The spatial dimensions of T1w and FLAIR images can be different, as this  #
# script will automatically register T1w to FLAIR.                          #
#                                                                           #
# * This script can be used as a template when applying the pipeline to     #
# other datasets.                                                           #
#                                                                           #
# * Image registration is time consuming and it is highly suggested that    #
# you run the whole pipeline on a CPU cluster with at least one CUDA GPU    #
# available.                                                                #
#                                                                           #
#############################################################################
'''
)

# define name of this experiment (without space ' ')
experiment_name = 'Siemens_TrioTim'

# dataset location (change to your actual dataset location)
OASIS3_folder = '/data6/chenghaoliu/Dataset/OASIS3/specific_use/wmh/Siemens_TrioTim_TR_9.00_TE_0.08_TI_2.50/'

# running on first GPU by default. Change this if you want to run on another GPU.
GPU_id = 2

# Here I defined all the case names that will be needed when doing this experiment.
# If you want to quickly go through the whole pipeline you can lower the number of 
# training subjects from 50 to about 10.
OASIS3_subjects_info = {
    'Siemens_TrioTim_reference':
    [   # 10 in total
        'OAS30113_MR_d3502', 'OAS30131_MR_d1901', 'OAS30132_MR_d1392', 'OAS30227_MR_d0000', 'OAS30242_MR_d0137',
        'OAS30484_MR_d1065', 'OAS30531_MR_d0108', 'OAS31103_MR_d1829', 'OAS30499_MR_d1164', 'OAS31047_MR_d1165'
    ],
    'Siemens_TrioTim_training':
    [   # 50 in total
        'OAS30027_MR_d2394', 'OAS30074_MR_d1871', 'OAS30080_MR_d0048', 'OAS30092_MR_d3727', 'OAS30142_MR_d1279', 
        'OAS30146_MR_d3322', 'OAS30175_MR_d3219', 'OAS30198_MR_d0083', 'OAS30204_MR_d0020', 'OAS30232_MR_d2324', 
        'OAS30246_MR_d1591', 'OAS30283_MR_d0797', 'OAS30335_MR_d2770', 'OAS30342_MR_d0001', 'OAS30357_MR_d1195', 
        'OAS30369_MR_d5872', 'OAS30403_MR_d1232', 'OAS30414_MR_d1175', 'OAS30443_MR_d2432', 'OAS30487_MR_d1338', 
        'OAS30492_MR_d0090', 'OAS30535_MR_d0139', 'OAS30559_MR_d2422', 'OAS30589_MR_d1525', 'OAS30596_MR_d2477', 
        'OAS30612_MR_d0039', 'OAS30615_MR_d2022', 'OAS30625_MR_d0033', 'OAS30685_MR_d0032', 'OAS30710_MR_d2323', 
        'OAS30713_MR_d2308', 'OAS30735_MR_d2484', 'OAS30743_MR_d2309', 'OAS30755_MR_d1540', 'OAS30757_MR_d2279', 
        'OAS30765_MR_d2798', 'OAS30818_MR_d1214', 'OAS30857_MR_d2255', 'OAS30869_MR_d2290', 'OAS30876_MR_d1670', 
        'OAS30899_MR_d2324', 'OAS30975_MR_d0008', 'OAS30978_MR_d1207', 'OAS31006_MR_d1106', 'OAS31019_MR_d0076', 
        'OAS31034_MR_d0203', 'OAS31058_MR_d3519', 'OAS31060_MR_d0083', 'OAS31092_MR_d3113', 'OAS31168_MR_d1566'
    ],

    'reference_description':
    [
        'HCwoWMH_male_62',   'HCwoWMH_male_54',   'HCwoWMH_male_71',   'HCwoWMH_female_58', 'HCwoWMH_female_65',
        'HCwoWMH_female_64', 'HCwoWMH_female_52', 'HCwoWMH_female_60', 'HCwoWMH_female_60', 'HCwoWMH_female_70'
    ],

    'training_description':
    [   # write description of each training case here (such as diagnosis, sex, age, etc.)
        'WMH_male_75',       'WMH_female_76',     'WMH_female_62',     'WMH_female_68',     'WMH_male_69',
        'WMH_female_77',     'WMH_female_81',     'WMH_male_88',       'WMH_male_69',       'WMH_female_72',
        'WMH_female_77',     'WMH_female_76',     'WMH_female_73',     'WMH_male_79',       'WMH_male_76', 
        'WMH_female_80',     'WMH_female_61',     'WMH_male_76',       'WMH_male_74',       'WMH_female_75',
        'WMH_male_57',       'WMH_female_52',     'WMH_male_74',       'WMH_female_78',     'WMH_female_79', 
        'WMH_female_64',     'WMH_male_73',       'WMH_male_68',       'WMH_female_68',     'WMH_female_71', 
        'WMH_male_76',       'WMH_female_64',     'WMH_male_78',       'WMH_female_71',     'WMH_male_69', 
        'WMH_female_74',     'WMH_male_73',       'WMH_male_56',       'WMH_female_73',     'WMH_female_67',
        'WMH_male_81',       'WMH_male_82',       'WMH_male_72',       'WMH_male_71',       'WMH_female_67',
        'WMH_male_67',       'WMH_male_69',       'WMH_male_77',       'WMH_male_80',       'WMH_male_69'
    ]
}

T1w_filename = 't1w_raw.nii.gz'
T2FLAIR_filename = 't2flair_raw.nii.gz'

##############################################################################
###########                    >>>   NOTE   <<<                    ###########
###########    NORMALLY YOU DON'T NEED TO CHANGE THE CODE BELOW    ###########
###########    IF YOU ORGANIZED THE DATASET STRUCTURE CORRECTLY    ###########
##############################################################################

print('Dataset location is "%s".' % OASIS3_folder)
print('** Please change it to your actual dataset location before running the following code :)\n\n' )

time.sleep(5)

print('Selected GPU index: %d' % GPU_id)

script_dir = sys.path[0]

# this is where all outputs are saved
output_folder = join_path(script_dir, experiment_name)

# this folder saves the outputs of FSL and FreeSurfer
FSL_FreeSurfer_outputs = join_path(output_folder, 'FSL_FreeSurfer')

if check_system_integrity(verbose=True) == False:
    print('\n\n** Some external softwares are missing, please install them '
    'before running this experiment.\n\n')
    exit(1)

# 1. do skull-stripping for reference healthy samples
##############################################################################

for case in OASIS3_subjects_info['Siemens_TrioTim_reference']:
    subject_folder = join_path(OASIS3_folder, case)
    fsl_folder = mkdir(join_path(FSL_FreeSurfer_outputs, case))
    t1w = join_path( subject_folder, T1w_filename )
    t2flair_raw = join_path(subject_folder, T2FLAIR_filename)
    # remove neck region to improve the quality of skull-stripping
    t1w_nr = join_path( fsl_folder, 't1w_NR.nii.gz' ) # t1w neck removed
    if not file_exist(t1w_nr):
        run_shell('robustfov -i %s -r %s' % (t1w, t1w_nr))
    # then do skull-stripping for t1w image
    t1w_nr_b = join_path( fsl_folder, 't1w_NR_brain.nii.gz' ) # t1w neck removed (brain only)
    t1w_nr_bm = join_path( fsl_folder, 't1w_NR_brain_mask.nii.gz' ) # t1w neck removed (brain mask)
    if not file_exist(t1w_nr_bm):
        run_shell('bet %s %s -m -v -n' % (t1w_nr, t1w_nr_b))
    # affine register t1w image to flair image
    t1w_nr_affine = join_path(fsl_folder, 't1w_affine.nii.gz')
    affine_mat = join_path(fsl_folder, 'affine_mat.mat')
    if not file_exist(affine_mat):
        run_shell('flirt -in %s -ref %s -out %s -omat %s -v' % (t1w_nr,t2flair_raw,t1w_nr_affine,affine_mat))
    # then apply affine transformation for t1w brain mask, to obtain FLAIR brain mask
    t2flair_raw_bm = join_path(fsl_folder, 't2flair_raw_brain_mask.nii.gz')
    if not file_exist(t2flair_raw_bm):
        run_shell('flirt -in %s -ref %s -applyxfm -init %s -interp nearestneighbour -out %s -v ' % \
                    (t1w_nr_bm, t2flair_raw, affine_mat, t2flair_raw_bm))

# 2. compute priors using FreeSurfer
##############################################################################

FreeSurfer_aseg_tasks = {
    'subject_names':[],
    't1_paths':[],
    'aseg_outputs':[],

    'SUBJECTS_DIR': join_path(FSL_FreeSurfer_outputs, 'FreeSurfer_SUBJECTS'),
    'num_workers': 10
}

for case in OASIS3_subjects_info['Siemens_TrioTim_reference']:
    t1w = join_path( FSL_FreeSurfer_outputs, case, 't1w_NR.nii.gz' )
    aseg_output = join_path(FSL_FreeSurfer_outputs, case, 't1w_NR_aseg.nii.gz')
    FreeSurfer_aseg_tasks['subject_names'].append(case)
    FreeSurfer_aseg_tasks['t1_paths'].append(t1w)
    FreeSurfer_aseg_tasks['aseg_outputs'].append(aseg_output)

run_FreeSurfer_aseg(**FreeSurfer_aseg_tasks) # 3.5h per case

for case, aseg_file in zip(FreeSurfer_aseg_tasks['subject_names'], FreeSurfer_aseg_tasks['aseg_outputs']):
    print(aseg_file)
    converted_labels = join_path( gd(aseg_file), 'cbstemcor.nii.gz')
    if try_load_nifti(converted_labels) == False:
        convert_FreeSurfer_aseg( aseg_file , converted_labels ,convert_type='cbstemcor')

    # transform all labels to T2FLAIR space

    brain_prior_labels = join_path(gd(converted_labels), 't2flair_raw_brain_priors.nii.gz')
    t2flair_raw = join_path(OASIS3_folder, case, T2FLAIR_filename)
    affine_mat = join_path(gd(converted_labels), 'affine_mat.mat')
    
    if not file_exist(brain_prior_labels):
        run_shell('flirt -in %s -ref %s -applyxfm -init %s -interp nearestneighbour -out %s -v ' % \
                    (converted_labels, t2flair_raw, affine_mat, brain_prior_labels))

# 3. collect information of reference images
##############################################################################

reference_csv = join_path(output_folder, 'Siemens_TrioTim_reference.csv')

with open(reference_csv, 'w') as f:
    f.write('case,desc,flair,label1,label2\n') 
    # self note: remember that label1 is the brain mask generated by FSL BET and label2 is the prior brain label
    # generated by FreeSurfer. 
with open(reference_csv, 'a') as f:
    for case in OASIS3_subjects_info['Siemens_TrioTim_reference']:
        flair = join_path(OASIS3_folder, case, T2FLAIR_filename)
        desc = OASIS3_subjects_info['reference_description'][ OASIS3_subjects_info['Siemens_TrioTim_reference'].index(case) ]
        label1 = join_path(FSL_FreeSurfer_outputs, case, 't2flair_raw_brain_mask.nii.gz')
        label2 = join_path(FSL_FreeSurfer_outputs, case, 't2flair_raw_brain_priors.nii.gz')
        assert files_exist([flair, label1, label2]), 'some files are missing.'
        f.write('%s,%s,%s,%s,%s\n' % (case,desc,flair,label1,label2))


# 4. collect training set images
##############################################################################

training_csv = join_path(output_folder, 'Siemens_TrioTim_training.csv')

with open(training_csv, 'w') as f:
    f.write('case,desc,flair\n')

for case in OASIS3_subjects_info['Siemens_TrioTim_training']:
    subject_folder = join_path(OASIS3_folder, case)
    t2flair_raw = join_path(subject_folder, T2FLAIR_filename)
    desc = OASIS3_subjects_info['training_description'][OASIS3_subjects_info['Siemens_TrioTim_training'].index(case)]
    assert file_exist(t2flair_raw)
    with open(training_csv, 'a') as f:
        f.write('%s,%s,%s\n' % (case, desc, t2flair_raw))

# 5. now launch the pipeline and wait it to finish 
##############################################################################

run_shell('DeepWMH_train --reference %s --training %s -j %d --output-folder %s --gpu %d --skip-integrity-check' % \
    (reference_csv, training_csv, 4, output_folder, GPU_id))

print('\n\n** Pipeline training is now finished! **\n\n')
Training_fit = join_path(output_folder, '004_WMH_pipeline', 'Stage_3_DCNN_training', '002_training_fit', '3mm_postproc')
print('Final segmentation can be found in "%s".' % Training_fit)

# 6. calculate Dice between network training fits and expert annotations 
##############################################################################

print('**     Now starting evaluation...     **')
Expert_annotation = join_path(script_dir, 'Manual_annotations', 'Siemens_TrioTim')

from deepwmh.analysis.metrics import BinaryDiceEvaluation

def find_rater1(patient_name):
    filepath = join_path(Expert_annotation, patient_name, 'rater_1.nii.gz')
    return filepath

def find_rater2(patient_name):
    filepath = join_path(Expert_annotation, patient_name, 'rater_2.nii.gz')
    return filepath

def find_proposed(patient_name):
    filepath = join_path(Training_fit, '%s.nii.gz' % patient_name)
    return filepath

evaluator = BinaryDiceEvaluation(OASIS3_subjects_info['Siemens_TrioTim_training'])
evaluator.add_method('manual1', find_rater1)
evaluator.add_method('manual2', find_rater2)
evaluator.add_method('proposed', find_proposed)

proposed_vs_rater1 = evaluator.run_eval( 'proposed', 'manual1' ) # compare final fit to rater 1
proposed_vs_rater2 = evaluator.run_eval( 'proposed', 'manual2' ) # compare final fit to rater 2
rater1_vs_rater2 = evaluator.run_eval('manual1', 'manual2') # measure intra-rater variability

csv_dict = {
    'case': evaluator.get_subject_list(),
    'intra-rater_variability': rater1_vs_rater2,
    'proposed_vs_rater1': proposed_vs_rater1,
    'proposed_vs_rater2': proposed_vs_rater2
}

Evaluation_csv = join_path(output_folder, 'Evaluation_training_fit.csv')

write_csv_simple( Evaluation_csv , csv_dict )

# 7. main experiment is finished, print summary
##############################################################################

print('------------')
print('** Trained model is saved to: "%s".' % join_path(output_folder, '004_WMH_pipeline', 'Model_release'))
print('** Final evaluation is saved to: "%s".' % Evaluation_csv)
print('------------')



