import sys, time
from freeseg.utilities.file_ops import files_exist, gd, join_path, file_exist, mkdir
from freeseg.utilities.data_io import try_load_nifti, write_csv_simple
from freeseg.utilities.external_call import run_shell
from freeseg.external_tools.FreeSurfer_aseg import convert_FreeSurfer_aseg, run_FreeSurfer_aseg
from freeseg.main.integrity_check import check_system_integrity

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
# .../Siemens_Biograph_mMR_TR_9.00_TE_0.09_TI_2.50/   <= later defined as   #
#      |                                                 "OASIS3_folder"    #
#      +--OAS30005_MR_d2384/                          <= case name          #
#      |  +--t1w_raw.nii.gz                           <= raw T1w image      #
#      |  +--t2flair_raw.nii.gz                       <= raw T2-FLAIR image #
#      |                                                                    #
#      +--OAS30056_MR_d3491/                                                #
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
# * image registration is time consuming and it is highly suggested that    #
# you run the whole pipeline on a CPU cluster with at least one CUDA GPU    #
# available.                                                                #
#                                                                           #
#############################################################################
'''
)

# define name of this experiment (without space ' ')
experiment_name = 'Siemens_Biograph_mMR'

# dataset location (change to your actual dataset location)
OASIS3_folder = '/data6/chenghaoliu/Dataset/OASIS3/specific_use/wmh/Siemens_Biograph_mMR_TR_9.00_TE_0.09_TI_2.50/'

# running on first GPU by default. Change this if you want to run on another GPU.
GPU_id = 3

# Here I defined all the case names that will be needed when doing this experiment.
# If you want to quickly go through the whole pipeline you can lower the number of 
# training subjects from 100 to about 10.
OASIS3_subjects_info = {
    'Siemens_Biograph_mMR_reference':
    [   # 10 in total
        'OAS30005_MR_d2384', 'OAS30056_MR_d3491', 'OAS30113_MR_d4437', 'OAS30220_MR_d1165', 'OAS30230_MR_d3855',
        'OAS30304_MR_d0027', 'OAS30411_MR_d3025', 'OAS30514_MR_d1526', 'OAS30531_MR_d2584', 'OAS30568_MR_d2326'
    ],
    
    'Siemens_Biograph_mMR_training':
    [   # 100 in total, lower the count if you just want to quickly go through the pipeline :-)
        'OAS30003_MR_d3731', 'OAS30006_MR_d3386', 'OAS30010_MR_d0068', 'OAS30011_MR_d1671', 'OAS30026_MR_d0696', 
        'OAS30039_MR_d0103', 'OAS30050_MR_d1530', 'OAS30066_MR_d2006', 'OAS30071_MR_d0018', 'OAS30080_MR_d1318', 
        'OAS30089_MR_d0001', 'OAS30098_MR_d0036', 'OAS30105_MR_d0056', 'OAS30117_MR_d4155', 'OAS30123_MR_d0122', 
        'OAS30134_MR_d1642', 'OAS30155_MR_d0785', 'OAS30167_MR_d1340', 'OAS30208_MR_d1703', 'OAS30212_MR_d3043', 
        'OAS30257_MR_d3773', 'OAS30263_MR_d2477', 'OAS30272_MR_d3087', 'OAS30279_MR_d0136', 'OAS30281_MR_d0042', 
        'OAS30291_MR_d1979', 'OAS30307_MR_d2362', 'OAS30315_MR_d0124', 'OAS30318_MR_d3298', 'OAS30346_MR_d1685', 
        'OAS30350_MR_d1201', 'OAS30355_MR_d0861', 'OAS30364_MR_d0110', 'OAS30369_MR_d5880', 'OAS30391_MR_d1547', 
        'OAS30403_MR_d2378', 'OAS30407_MR_d2862', 'OAS30414_MR_d0363', 'OAS30438_MR_d2358', 'OAS30464_MR_d2848', 
        'OAS30468_MR_d0069', 'OAS30475_MR_d0062', 'OAS30486_MR_d1300', 'OAS30515_MR_d0044', 'OAS30516_MR_d4192', 
        'OAS30527_MR_d0006', 'OAS30535_MR_d1336', 'OAS30538_MR_d0105', 'OAS30558_MR_d4493', 'OAS30567_MR_d0040', 
        'OAS30574_MR_d1917', 'OAS30577_MR_d0067', 'OAS30580_MR_d1531', 'OAS30585_MR_d0065', 'OAS30587_MR_d4511', 
        'OAS30589_MR_d3191', 'OAS30590_MR_d0085', 'OAS30592_MR_d0087', 'OAS30607_MR_d0117', 'OAS30637_MR_d0079', 
        'OAS30663_MR_d0051', 'OAS30685_MR_d1552', 'OAS30691_MR_d0056', 'OAS30706_MR_d0060', 'OAS30723_MR_d2568', 
        'OAS30728_MR_d0516', 'OAS30735_MR_d3515', 'OAS30746_MR_d0035', 'OAS30749_MR_d1996', 'OAS30762_MR_d1002', 
        'OAS30808_MR_d3453', 'OAS30812_MR_d0055', 'OAS30827_MR_d1875', 'OAS30839_MR_d1394', 'OAS30841_MR_d3499', 
        'OAS30852_MR_d6963', 'OAS30858_MR_d2100', 'OAS30867_MR_d4407', 'OAS30896_MR_d3528', 'OAS30910_MR_d1028', 
        'OAS30950_MR_d0063', 'OAS30978_MR_d0059', 'OAS30982_MR_d1708', 'OAS31006_MR_d0120', 'OAS31012_MR_d4024', 
        'OAS31013_MR_d0628', 'OAS31015_MR_d0222', 'OAS31019_MR_d1370', 'OAS31028_MR_d1285', 'OAS31037_MR_d6061', 
        'OAS31041_MR_d1426', 'OAS31042_MR_d3618', 'OAS31048_MR_d2385', 'OAS31054_MR_d2787', 'OAS31071_MR_d0068', 
        'OAS31090_MR_d3565', 'OAS31096_MR_d1308', 'OAS31115_MR_d0466', 'OAS31127_MR_d2140', 'OAS31150_MR_d1416'
    ],

    'reference_description':
    [
        'HCwoWMH_female_54', 'HCwoWMH_female_59', 'HCwoWMH_male_65',   'HCwoWMH_male_67',   'HCwoWMH_female_56',
        'HCwoWMH_male_68',   'HCwoWMH_male_55',   'HCwoWMH_female_68', 'HCwoWMH_female_59', 'HCwoWMH_female_61'
    ],

    'training_description':
    [   # write description of each training case here (such as diagnosis, sex, age, etc.)
        'WMH_female_68',     'WMH_male_71',       'WMH_female_68',     'WMH_female_83',     'WMH_male_82', 
        'WMH_female_73',     'WMH_female_74',     'WMH_female_79',     'WMH_male_72',       'WMH_female_66', 
        'WMH_male_78',       'WMH_female_65',     'WMH_female_69',     'WMH_male_78',       'WMH_male_74', 
        'WMH_male_75',       'WMH_female_71',     'WMH_male_75',       'WMH_female_80',     'WMH_female_87', 
        'WMH_male_77',       'WMH_female_77',     'WMH_female_74',     'WMH_female_73',     'WMH_male_73', 
        'WMH_female_71',     'WMH_male_83',       'WMH_male_77',       'WMH_male_72',       'WMH_female_76', 
        'WMH_female_83',     'WMH_male_69',       'WMH_male_71',       'WMH_female_80',     'WMH_male_87', 
        'WMH_female_65',     'WMH_female_83',     'WMH_male_74',       'WMH_female_79',     'WMH_female_68', 
        'WMH_female_74',     'WMH_female_72',     'WMH_male_58',       'WMH_male_76',       'WMH_female_78', 
        'WMH_female_74',     'WMH_female_55',     'WMH_female_71',     'WMH_female_76',     'WMH_female_74', 
        'WMH_female_77',     'WMH_male_80',       'WMH_male_71',       'WMH_male_80',       'WMH_female_76', 
        'WMH_female_83',     'WMH_male_58',       'WMH_male_80',       'WMH_female_86',     'WMH_male_73', 
        'WMH_female_69',     'WMH_female_72',     'WMH_female_69',     'WMH_male_80',       'WMH_male_76', 
        'WMH_male_78',       'WMH_female_66',     'WMH_female_75',     'WMH_male_78',       'WMH_male_70', 
        'WMH_female_90',     'WMH_female_62',     'WMH_male_77',       'WMH_female_74',     'WMH_male_87', 
        'WMH_female_81',     'WMH_female_75',     'WMH_female_88',     'WMH_male_91',       'WMH_male_75', 
        'WMH_male_65',       'WMH_male_69',       'WMH_female_78',     'WMH_male_68',       'WMH_female_79', 
        'WMH_female_81',     'WMH_male_86',       'WMH_female_71',     'WMH_female_70',     'WMH_female_81', 
        'WMH_male_81',       'WMH_male_79',       'WMH_male_62',       'WMH_male_69',       'WMH_male_74', 
        'WMH_female_92',     'WMH_male_85',       'WMH_male_88',       'WMH_female_86',     'WMH_male_68'
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

print('Selected GPU: %d' % GPU_id)

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

for case in OASIS3_subjects_info['Siemens_Biograph_mMR_reference']:
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

for case in OASIS3_subjects_info['Siemens_Biograph_mMR_reference']:
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

reference_csv = join_path(output_folder, 'Siemens_Biograph_mMR_reference.csv')

with open(reference_csv, 'w') as f:
    f.write('case,desc,flair,label1,label2\n') 
    # self note: remember that label1 is the brain mask generated by FSL BET and label2 is the prior brain label
    # generated by FreeSurfer. 
with open(reference_csv, 'a') as f:
    for case in OASIS3_subjects_info['Siemens_Biograph_mMR_reference']:
        flair = join_path(OASIS3_folder, case, T2FLAIR_filename)
        desc = OASIS3_subjects_info['reference_description'][ OASIS3_subjects_info['Siemens_Biograph_mMR_reference'].index(case) ]
        label1 = join_path(FSL_FreeSurfer_outputs, case, 't2flair_raw_brain_mask.nii.gz')
        label2 = join_path(FSL_FreeSurfer_outputs, case, 't2flair_raw_brain_priors.nii.gz')
        assert files_exist([flair, label1, label2]), 'some files are missing.'
        f.write('%s,%s,%s,%s,%s\n' % (case,desc,flair,label1,label2))


# 4. collect training set images
##############################################################################

training_csv = join_path(output_folder, 'Siemens_Biograph_mMR_training.csv')

with open(training_csv, 'w') as f:
    f.write('case,desc,flair\n')

for case in OASIS3_subjects_info['Siemens_Biograph_mMR_training']:
    subject_folder = join_path(OASIS3_folder, case)
    t2flair_raw = join_path(subject_folder, T2FLAIR_filename)
    desc = OASIS3_subjects_info['training_description'][OASIS3_subjects_info['Siemens_Biograph_mMR_training'].index(case)]
    assert file_exist(t2flair_raw)
    with open(training_csv, 'a') as f:
        f.write('%s,%s,%s\n' % (case, desc, t2flair_raw))

# 5. now launch the pipeline and wait it to finish 
##############################################################################

run_shell('freeseg_WMH_train --reference %s --training %s -j %d --output-folder %s --gpu %d --skip-integrity-check' % \
    (reference_csv, training_csv, 4, output_folder, GPU_id))

print('\n\n** Pipeline training is now finished! **\n\n')
Training_fit = join_path(output_folder, '004_WMH_pipeline', 'Stage_3_DCNN_training', '002_training_fit', '3mm_postproc')
print('Final segmentation can be found in "%s".' % Training_fit)

# 6. calculate Dice between network training fits and expert annotations 
##############################################################################

print('**     Now starting evaluation...     **')
Expert_annotation = join_path(script_dir, 'Manual_annotations', 'Siemens_Biograph_mMR')

from freeseg.analysis.metrics import BinaryDiceEvaluation

def find_rater1(patient_name):
    filepath = join_path(Expert_annotation, patient_name, 'rater_1.nii.gz')
    return filepath

def find_rater2(patient_name):
    filepath = join_path(Expert_annotation, patient_name, 'rater_2.nii.gz')
    return filepath

def find_proposed(patient_name):
    filepath = join_path(Training_fit, '%s.nii.gz' % patient_name)
    return filepath

evaluator = BinaryDiceEvaluation(OASIS3_subjects_info['Siemens_Biograph_mMR_training'])
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



