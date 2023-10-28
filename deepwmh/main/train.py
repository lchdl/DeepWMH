import argparse
from deepwmh.external_tools.ANTs_group_registration import ANTsGroupRegistration, antsApplyTransforms
from deepwmh.pipeline.DCNN_multistage import Pipeline_DCNN_Multistage_nnUNet
from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.external_call import run_shell
from deepwmh.utilities.file_ops import abs_path, chmod, gd, gn, join_path, ls, mkdir, rm
from deepwmh.utilities.data_io import load_csv_simple, try_load_nifti, write_csv_simple
from deepwmh.main.integrity_check import check_system_integrity, check_dataset


# utility function to collect image registration pairs
def _collect_reg_pairs( folder, source_cases, target_cases ):
    d = {}

    reg_folder = join_path(folder, '002_Registration')
    xfm_folder = join_path(folder, '003_Transformed')

    all_pair_paths = [join_path(xfm_folder, '%s_to_%s' % (s,t)) for s in source_cases for t in target_cases]

    for pair_path in all_pair_paths:

        pair = gn(pair_path)
        _, case_to = pair.split('_to_')

        if case_to not in d:
            d[case_to]={'flair':[], 'label1':[], 'label2':[]}

        flair = join_path(reg_folder, '%s.nii.gz' % pair)
        label1 = join_path(xfm_folder, pair, 'label1.nii.gz')
        label2 = join_path(xfm_folder, pair, 'label2.nii.gz')

        d[case_to]['flair'].append( flair )
        d[case_to]['label1'].append( label1 )
        d[case_to]['label2'].append( label2 )

    return d

def _parallel_do_N4_for_image(params):
    raw_image_path, output_image_path = params
    if try_load_nifti(output_image_path) == False:
        # default N4 correction parameter to correct large bias fields 
        run_shell('N4BiasFieldCorrection -d 3 -i %s -o %s -c [50x50x50,0.0] -s 2' % \
            (raw_image_path, output_image_path), print_command=False, print_output=False)

def _parallel_apply_xfms(params):
    in_files, ref_file, affine, elastic, out_files, interp_method = params
    temp_file = join_path(gd(out_files[0]), 'temp.nii.gz')

    skip = True
    for out_file in out_files:
        if try_load_nifti(out_file) == False:
            skip = False
            break
    
    if not skip:
        for in_file, out_file in zip(in_files, out_files):
            mkdir(gd(out_file))
            run_shell(antsApplyTransforms(in_file, ref_file, affine, temp_file, interp_method))
            run_shell(antsApplyTransforms(temp_file, ref_file, elastic, out_file, interp_method))
            rm(temp_file)

def main():
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, 
                      argparse.RawDescriptionHelpFormatter):  pass

    parser = argparse.ArgumentParser(
        description =
            'Training full segmentation pipeline end-to-end, '
            'including data pre-processing and image registrations.',
        formatter_class = MyFormatter)
    parser.add_argument('-s', '--reference', 
        help='Reference images (need providing both images and labels, '
            'orgainzed in a CSV file, see "examples/Example_reference.csv" '
            'for more details).',type=str,required=True)
    parser.add_argument('-t', '--training', 
        help='Training images (need providing image paths, '
            'orgainzed in a CSV file, see "examples/Example_training.csv" '
            'for more details).',type=str,required=True)
    parser.add_argument('-o', '--output-folder', 
        help='Target output directory used for saving output files.', 
        type=str, required=True)
    parser.add_argument('-j', '--num-CPU-cores', 
        help='Maximum number of CPU cores used by the pipeline.', 
        type=int, default=8)
    parser.add_argument('-g', '--gpu', 
        help='GPU id.', type=int, default=0)
    parser.add_argument('-r', '--release-model',
        help='Collect trained model to a custom location for future use.',
        type=str, required=False)

    # advanced options below

    parser.add_argument('--core-folder', 
        help= '[Advanced] Change core folder to a custom location. Usually we save '
        'initial/post-processed segmentations and all trained models to this directory. '
        'If not specified, core folder will located in folder specified in --output-folder. '
        'You don\'t need to set this in normal use.', 
        type=str, required=False)

    parser.add_argument('--skip-integrity-check', 
        help= '[Advanced] Skip system integrity check.', 
        action='store_true')

    # passing & parse arguments

    args = parser.parse_args()
    reference_csv = args.reference
    training_csv  = args.training
    output_folder = args.output_folder
    num_cores     = args.num_CPU_cores
    gpu           = args.gpu

    if args.core_folder == None:
        core_folder = mkdir( join_path( output_folder, '004_WMH_pipeline' ) )
    else:
        core_folder = mkdir(args.core_folder)
    if args.release_model == None:
        release_folder = mkdir(join_path(core_folder, 'Model_release'))
    else:
        release_folder = mkdir(abs_path(args.release_model))

    if not args.skip_integrity_check:
        if check_system_integrity(verbose=True) == False:
            exit(1)

    # load dataset
    reference_dataset = load_csv_simple(reference_csv,key_names=['case', 'desc', 'flair', 'label1', 'label2'])
    training_dataset = load_csv_simple(training_csv,key_names=['case', 'desc', 'flair'])
    training_cases = training_dataset['case']
    print('number of reference cases: %d' % len(reference_dataset['case']))
    print('number of training cases: %d' % len(training_cases))
    print('checking dataset...')
    if check_dataset(reference_dataset) == False or check_dataset(training_dataset) == False:
        exit(1)
    
    output_folder = mkdir(output_folder)

    # dump command
    with open(join_path(output_folder, 'train_%s.sh' % gn(core_folder)), 'w') as f:
        cached_command = \
            '##################################################\n'        \
            '# Cached command for training the whole pipeline #\n'        \
            '##################################################\n\n'      \
            'DeepWMH_train \\\n'                                          \
            '-s %s \\\n'                                                  \
            '-t %s \\\n'                                                  \
            '-o %s \\\n'                                                  \
            '--core-folder %s \\\n'                                       \
            '--release-model %s \\\n'                                     \
            '-j %d \\\n'                                                  \
            '-g %d ' % (abs_path(reference_csv), abs_path(training_csv), 
                        abs_path(output_folder), core_folder, release_folder, 
                        num_cores, gpu)
        f.write(cached_command)
        chmod(join_path(output_folder, 'train_%s.sh' % gn(core_folder)), '755')

    # pre-processing images
    print('Pre-processing images, please wait...')
    preproc_folder = mkdir(join_path( output_folder, '001_Preprocessed'))
    pre_source_dataset = {'case':[], 'flair':[], 'label1':[], 'label2':[]}
    pre_target_dataset = {'case':[], 'flair':[]}
    tasks = []
    for case, flair, label1, label2 in zip( reference_dataset['case'], reference_dataset['flair'], reference_dataset['label1'], reference_dataset['label2']):
        output_preproc_image = join_path(preproc_folder, '%s.nii.gz' % case)
        tasks.append(  (flair, output_preproc_image)  )
        pre_source_dataset['case'].append(case)
        pre_source_dataset['flair'].append(output_preproc_image)
        pre_source_dataset['label1'].append(label1)
        pre_source_dataset['label2'].append(label2)
    preproc_training_imgs = {}
    for case, flair in zip(training_dataset['case'], training_dataset['flair']):
        output_preproc_image = join_path(preproc_folder, '%s.nii.gz' % case)
        tasks.append(  (flair, output_preproc_image)  )
        pre_target_dataset['case'].append(case)
        pre_target_dataset['flair'].append(output_preproc_image)
        preproc_training_imgs[case] = output_preproc_image
    run_parallel(_parallel_do_N4_for_image, tasks, 4, 'N4 correction') ###################

    # image registration
    registration_folder = mkdir(join_path( output_folder, '002_Registration' ))
    registration_temp_dir = mkdir(join_path( output_folder, '__running_jobs__' ))
    S = [item for item in zip(pre_source_dataset['case'], pre_source_dataset['flair'])]
    T = [item for item in zip(pre_target_dataset['case'], pre_target_dataset['flair'])]
    registration_procnum = 4 # number of concurrent processes spawned when registration
    img_regist = ANTsGroupRegistration(S, T, registration_folder, registration_temp_dir, registration_procnum, True, True, True)
    with open(join_path(output_folder, 'run_registration.sh'), 'w') as f:
        # dump registration command into shell script to help users distribute 
        # registration task into several machines to improve speed.
        f.write('####################################################\n'
                '#   ** This script is automatically generated **   #\n'
                '#                                                  #\n'
                '# If your server have several machines that are    #\n'
                '# independent from each other, you can distribute  #\n'
                '# the following command to different machines to   #\n'
                '# speed up image registration process (by adding   #\n'
                '# "--distributed X/X" at the end of the command    #\n'
                '# and send them into different machines manually). #\n'
                '#                                                  #\n'
                '# For more info about distributing registration    #\n'
                '# tasks into several machines, please type and run #\n'
                '# "antsGroupRegistration -h"                       #\n'
                '####################################################\n'
                '\n')
        regsource = join_path(output_folder, 'regsource.csv')
        regtarget = join_path(output_folder, 'regtarget.csv')
        regsource_dataset = {'case': pre_source_dataset['case'],'data': pre_source_dataset['flair']}
        regtarget_dataset = {'case': pre_target_dataset['case'],'data': pre_target_dataset['flair']}
        write_csv_simple(regsource, regsource_dataset)
        write_csv_simple(regtarget, regtarget_dataset)
        shell_cmd = 'antsGroupRegistration \\\n'
        shell_cmd += '-s %s \\\n' % regsource
        shell_cmd += '-t %s \\\n' % regtarget
        shell_cmd += '-o %s \\\n' % registration_folder
        shell_cmd += '-j %d \\\n' % registration_procnum
        shell_cmd += '--allow-large-deformations --allow-quick-registration --keep-deformation \n'
        f.write(shell_cmd)
        f.write('\n'
                '####################################################\n'
                '# When registration is finished, run this pipeline #\n'				
                '# using the same command as before to continue     #\n'
                '# (training command is cached to "train_XXX.sh").  #\n'
                '####################################################\n')
        chmod(join_path(output_folder, 'run_registration.sh'), '755')
    img_regist.launch()
    rm(registration_temp_dir)

    # transform prior labels
    transformed_folder = mkdir(join_path( output_folder, '003_Transformed' ))
    print('transforming labels...')
    tasks = []
    for case_from in pre_source_dataset['case']:
        for case_to in pre_target_dataset['case']:
            pair = '%s_to_%s' % (case_from, case_to)
            # create output dir
            transform_output = join_path(transformed_folder, gn(pair))
            # find transforms
            transform_dir = join_path(registration_folder, pair)
            affine_transform = join_path( transform_dir, 'warp_0GenericAffine.mat' )
            elastic_transform = join_path( transform_dir, 'warp_1Warp.nii.gz' )
            # find labels that need to be deformed
            s_index = pre_source_dataset['case'].index(case_from)
            source_label1 = pre_source_dataset['label1'][s_index]
            source_label2 = pre_source_dataset['label2'][s_index]
            t_index = pre_target_dataset['case'].index(case_to)
            # find target image
            target_img = pre_target_dataset['flair'][t_index]
            # define output files
            output_label1 = join_path(transform_output, 'label1.nii.gz')
            output_label2 = join_path(transform_output, 'label2.nii.gz')
            # make task
            tasks.append( ([source_label1, source_label2], target_img, 
                affine_transform, elastic_transform, [output_label1, output_label2], 
                'NearestNeighbor') )
    run_parallel( _parallel_apply_xfms, tasks, 4, 'transform' ) ###################

    reg_pairs = _collect_reg_pairs( output_folder, pre_source_dataset['case'], pre_target_dataset['case'] )

    # configure pipeline
    pipeline = Pipeline_DCNN_Multistage_nnUNet(output_folder=core_folder, num_CPU_cores=num_cores)
    for case in training_cases:
        train_flair = preproc_training_imgs[case]
        all_ref_flairs = reg_pairs[case]['flair']
        all_label1 = reg_pairs[case]['label1']
        all_label2 = reg_pairs[case]['label2']
        patient_desc = training_dataset['desc'][ training_dataset['case'].index(case) ]
        pipeline.add_training_case(case,train_flair,all_ref_flairs,all_label1,all_label2,description=patient_desc)

    # finally, train the pipeline!
    pipeline.run_training(gpu=gpu)

    # collect trained model for release
    if len(ls(release_folder))>0:
        print('[!] Cannot release model to a non-empty folder! Maybe your model has '
              'already released to this folder from a previous run.')
    else: 
        print('trained model will be released to "%s".' % release_folder)
        pipeline.release_model(release_folder)
        
