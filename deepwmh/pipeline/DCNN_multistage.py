from copy import deepcopy
import os
import json
import datetime
import numpy as np
from deepwmh.analysis.image_ops import remove_3mm_sparks
from deepwmh.analysis.metrics import hard_dice_binary
from deepwmh.analysis.lesion_analysis import LesionAnalyzer
from deepwmh.utilities.file_ops import abs_path, cp, file_exist, files_exist, gn, join_path, laf, mkdir, ls, rm
from deepwmh.utilities.nii_preview import nii_as_gif, nii_slice_range
from deepwmh.utilities.data_io import get_nifti_header, get_nifti_pixdim, load_nifti, load_nifti_simple, load_pkl, \
    save_nifti, save_pkl, targz_compress, try_load_gif, try_load_nifti
from deepwmh.utilities.misc import SimpleTxtLog, Checkpoints
from deepwmh.utilities.parallelization import run_parallel
from deepwmh.utilities.external_call import run_shell

#####################################
# some utility functions for nnunet #
#####################################

def _convert_nnunet_TaskXXX_to_task_id(task_name):
    return int(task_name[4:7])

def _parse_and_apply_augmentations(image_data, augmentations: str):
    dat = deepcopy(image_data)
    augs = augmentations.replace(' ', '').split(',')
    for opval in augs:
        op,val = opval.split('=')
        if op=='noise':
            # apply image noise
            val = float(val)
            q5 = np.percentile(image_data, 5)
            q95 = np.percentile(image_data, 95)
            dat = dat + np.random.normal(scale = val*(q95-q5),size=dat.shape)
        else:
            raise RuntimeError('unknown augmentation operation: "%s".' % opval)
    return dat

def _prepare_nnunet_training_data(train_list, raw_data_folder, image_modality="Modality_01",
    name="", description="", reference="", license="", release="v1.0", tensorImageSize="3D",
    augmentation=""):
    '''
    prepare training samples with single modality image and binary label

    augmentation: apply image augmentation strategies before network training.
    default: "" (no augmentation)

    '''
    num_samples = len(train_list)
    imagesTr_folder = mkdir( join_path(raw_data_folder, 'imagesTr') )
    labelsTr_folder = mkdir( join_path(raw_data_folder, 'labelsTr') )

    json_train_list = []

    for i in range(len(train_list)):
        name, image, label, mask = train_list[i]
        print('[%d/%d] %s' % (i+1, num_samples, name))
        image_dat, image_hdr = load_nifti(image)
        label_dat = (load_nifti_simple(label) > 0.5).astype('float32') # binarize label in case grayscale {0,255} image given
        if mask is not None:
            mask_dat = (load_nifti_simple(mask) > 0.5).astype('float32')
            label_dat = label_dat * mask_dat        
        assert image_dat.shape == label_dat.shape, \
            'image and label shapes are different. image shape is %s, label shape is %s.' % \
            (str(image_dat.shape), str(label_dat.shape))
        out_image = join_path(imagesTr_folder, '%s_0000.nii.gz' % name)
        out_dummy = join_path(imagesTr_folder, '%s.nii.gz' % name) # dummy image path
        out_label = join_path(labelsTr_folder, '%s.nii.gz' % name)
        # apply augmentation to image if needed
        if augmentation != "":
            image_dat = _parse_and_apply_augmentations(image_dat, augmentation)
            save_nifti(image_dat, image_hdr, out_image)
        else:
            cp(image, out_image) # direct copy image to destination
        save_nifti(label_dat, image_hdr, out_label) # write label to destination
        json_train_list.append({ "image": out_dummy, "label": out_label })

    # write json file
    json_file = {
        "name":name, "description":description, "reference":reference,
        "license":license, "release":release, "tensorImageSize":tensorImageSize,
        "modality":{ "0": image_modality},
        "labels":{
            "0":"background",
            "1":"lesion"
        },
        "numTraining":len(json_train_list),
        "numTest":0,
        "training":json_train_list,
        "test":[]
    }
    with open(join_path(raw_data_folder,'dataset.json'),'w') as f:
        json.dump(json_file, f,indent=4)

def _remove_duplicates_in_list(l:list):
    return list(dict.fromkeys(l))

######################
# parallel functions #
######################

def _parallel_softmax_masking(params):
    in_path, valid_mask, out_path = params
    if file_exist(out_path) and try_load_nifti(out_path)==True:
        return
    x = load_nifti_simple(in_path)
    m = load_nifti_simple(valid_mask)
    y = 1-(m*(1-x)) # note: we are saving the inversed softmax, so -> 1 means background, -> 0 means foreground
    save_nifti(y, get_nifti_header(in_path), out_path)

def _parallel_ensembling(params):
    in_softmaxes, prior_label, seg_pp, out_file, out_seg, shape, phys_res = params
    if try_load_nifti(out_file) and try_load_nifti(out_seg):
        return
    scalar_field = np.zeros(shape).astype('float32')
    for file in in_softmaxes:
        scalar_field += load_nifti_simple(file)
    scalar_field = scalar_field/len(in_softmaxes)
    refined_label = (scalar_field < 0.5).astype('float32') # scalar_field is inversed, so regions < 0.5 means lesion
    # remove sparks with volume less than 3mm^3
    refined_label = remove_3mm_sparks(refined_label, phys_res)
    nifti_header = get_nifti_header(in_softmaxes[0])

    save_nifti(scalar_field, nifti_header, out_file)
    save_nifti(refined_label, nifti_header, out_seg)

def _parallel_3mm_postproc(params):
    in_seg, valid_mask, out_seg = params
    if try_load_nifti(out_seg):
        return
    label, header = load_nifti(in_seg)
    mask = load_nifti_simple(valid_mask)
    label = label * mask
    vox_size = get_nifti_pixdim(in_seg)
    label0 = remove_3mm_sparks(label, vox_size)
    save_nifti(label0, header, out_seg)

def _parallel_generate_final_GIF(params):
    in_image, in_seg, out_gif = params
    if try_load_gif(out_gif) == False:
        axis = 'axial'
        # get rid of empty slices
        data, _ = load_nifti(in_image)
        slice_start, slice_end = nii_slice_range(in_image, axis=axis, value = np.min(data) + 0.001, percentage = 0.999)
        nii_as_gif(in_image, out_gif, axis=axis, lesion_mask=in_seg, side_by_side=True, slice_range=[slice_start, slice_end])

class Pipeline_DCNN_Multistage_nnUNet(object):

    def log(self, msg, print_to_console=True):
        self.logger.write(msg,timestamp=True)
        if print_to_console:
            print(msg)

    def __init__(self, output_folder = None, intensity_prior = '+', num_CPU_cores:int=8, gpu: int=0):
        
        assert output_folder is not None,  'must specify output_folder.'
        assert intensity_prior in ['+', '-', None], 'Unknown intensity prior.'

        ##########################
        # setup basic parameters #
        ##########################
        self.pipeline_folder = mkdir(abs_path(output_folder)) # root folder where all the pipeline outputs will be saved to.
        self.log_path = join_path(self.pipeline_folder, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.txt')
        self.logger = SimpleTxtLog(self.log_path)
        self.log('Configuring pipeline...')
        
        self.gpu = gpu
        self.train_dict = {}                    # image scans used for training the pipeline
        self.intensity_prior = intensity_prior  # intensity prior applied to NLL score, currently i only tested '+' (hyperintense lesions)
        self.num_CPU_cores = num_CPU_cores      # do not set "num_CPU_cores" too high
        self.stage_1_output_folder = mkdir(join_path(self.pipeline_folder, 'Stage_1_initial_segmentation'))
        self.stage_2_output_folder = mkdir(join_path(self.pipeline_folder, 'Stage_2_label_denoising'))
        self.stage_3_output_folder = mkdir(join_path(self.pipeline_folder, 'Stage_3_DCNN_training'))
        self.checkpoints = Checkpoints(join_path(self.pipeline_folder, 'Checkpoints'))

        #########################
        # setup lesion analyzer #
        #########################
        self.lesion_analyzer = LesionAnalyzer(self.stage_1_output_folder, num_workers=self.num_CPU_cores, logger=None)

        #########################
        # setup DCNN parameters #
        #########################

        self.DCNN_output_folder = mkdir(join_path(self.pipeline_folder, 'DCNN_Outputs'))

        # setup environment variables for nnU-Net.
        os.environ['nnUNet_raw_data_base'] = mkdir(join_path( self.DCNN_output_folder, '001_raw_data' ))
        os.environ['nnUNet_preprocessed'] = mkdir(join_path( self.DCNN_output_folder, '002_preprocessed_data' ))
        os.environ['RESULTS_FOLDER'] = mkdir(join_path( self.DCNN_output_folder, '003_trained_models' ))
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % self.gpu

        # DO NOT change the string constants below unless you know what you are doing :-)
        self.DCNN_stage_2_task_name = 'Task001_LabelDenoising'
        self.DCNN_stage_3_task_name = 'Task002_FinalModel'
        self.DCNN_stage_2_epochs = 50
        self.DCNN_stage_3_epochs = 100
        self.DCNN_batches_in_each_epoch = 150
        self.DCNN_network_config = '3d_fullres'
        self.DCNN_trainer_name = 'nnUNetTrainerV2'
        self.DCNN_planner_name = 'nnUNetPlansv2.1'
        self.DCNN_fold = 'all'

        self.DCNN_stage_2_model_folder = join_path(os.environ['RESULTS_FOLDER'], 'nnUNet', 
                self.DCNN_network_config, 
                self.DCNN_stage_2_task_name, 
                '%s__%s' % (self.DCNN_trainer_name, self.DCNN_planner_name), 
                self.DCNN_fold)
        self.DCNN_stage_3_model_folder = join_path(os.environ['RESULTS_FOLDER'], 'nnUNet', 
                self.DCNN_network_config, 
                self.DCNN_stage_3_task_name, 
                '%s__%s' % (self.DCNN_trainer_name, self.DCNN_planner_name), 
                self.DCNN_fold)
        self.DCNN_ensemble_epochs = int(0.1*self.DCNN_stage_2_epochs)
        if self.DCNN_ensemble_epochs < 1:
            self.DCNN_ensemble_epochs = 1

    def _do_initial_segmentation(self):
        
        self.log('+---------------------------------+')
        self.log('|  Stage I: initial segmentation  |')
        self.log('+---------------------------------+')

        if not self.checkpoints.is_finished('STAGE_1_INITIAL_SEGMENTATION'):

            do_postprocessing = True
            self.log('initial segmentation starts.')
            self.log('Used analyzer class is "%s"' % type(self.lesion_analyzer).__name__)
            self.lesion_analyzer.analyze_and_do_segmentation(intensity_prior=self.intensity_prior, 
                do_postprocessing=do_postprocessing)
        
            self.checkpoints.set_finish('STAGE_1_INITIAL_SEGMENTATION')
        
        self.log('stage 1 complete.')

    def _do_label_denoising(self):

        self.log('+-----------------------------+')
        self.log('|  Stage II: label denoising  |')
        self.log('+-----------------------------+')

        ###############################
        # preparing data for training #
        ###############################
        raw_label    = mkdir(join_path(self.stage_2_output_folder, '001_initial_seg'))
        raw_label_pp = mkdir(join_path(self.stage_2_output_folder, '002_initial_seg_pp'))
        if not self.checkpoints.is_finished('STAGE_2-1_PREPARE_DATA'):
            self.log('Preparing training data for label denoising...')
            self.log('Label data (raw) is stored in "%s".' % raw_label)
            self.log('Label data (post-processed) is stored in "%s".' % raw_label_pp)
            #
            nnUNet_raw_data_base = os.environ['nnUNet_raw_data_base']
            tasks_raw_data_base_dir = mkdir(join_path(nnUNet_raw_data_base,'nnUNet_raw_data'))

            # make task specific directory
            task_raw_data_dir = mkdir(join_path(tasks_raw_data_base_dir, self.DCNN_stage_2_task_name))
            train_list = []

            all_cases = ls(self.stage_1_output_folder)
            for case_id in range(len(all_cases)):
                case = all_cases[case_id]
                preprocessed_image = join_path(self.stage_1_output_folder, case, 'preprocessed_image.nii.gz')
                initial_seg = join_path(self.stage_1_output_folder, case, 'segmentation.nii.gz')
                initial_seg_pp = join_path(self.stage_1_output_folder, case, 'segmentation_pp.nii.gz')
                mask_image = join_path( self.stage_1_output_folder, case, 'valid_mask.nii.gz' )
                # copy segmentation files
                label0 = join_path(raw_label, '%s.nii.gz' % case)
                label1 = join_path(raw_label_pp, '%s.nii.gz' % case)
                cp(initial_seg, label0)
                cp(initial_seg_pp, label1)
                train_list.append( ( case, preprocessed_image, initial_seg_pp, mask_image ) )

            _prepare_nnunet_training_data(train_list, task_raw_data_dir)
            self.checkpoints.set_finish('STAGE_2-1_PREPARE_DATA')

        #######################
        # plan and preprocess #
        #######################
        if not self.checkpoints.is_finished('STAGE_2-2_PLAN_AND_PREPROCESS'):
            self.log('start preparing data...')
            preproc_command = 'nnUNet_plan_and_preprocess -t %d --verify_dataset_integrity' % _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_2_task_name)
            self.log('>>> '+ preproc_command, print_to_console=False)
            run_shell(preproc_command)
            self.log('finished preprocessing')
            self.checkpoints.set_finish('STAGE_2-2_PLAN_AND_PREPROCESS')

        ##################
        # start training #
        ##################
        if not self.checkpoints.is_finished('STAGE_2-3_TRAINING_DENOISER'):
            self.log('start training DCNN for denoising labels...')
            training_command = 'nnUNet_train %s %s %d %s -e %d -b %d --noval --save_every_epoch --disable_postprocessing_on_folds' % \
                (self.DCNN_network_config, 
                self.DCNN_trainer_name,
                _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_2_task_name), 
                self.DCNN_fold, 
                self.DCNN_stage_2_epochs, 
                self.DCNN_batches_in_each_epoch)
            self.log('looking for existing model(s) in folder "%s"...' % self.DCNN_stage_2_model_folder)
            model_latest = join_path(self.DCNN_stage_2_model_folder, 'model_latest.model') 
            if file_exist(model_latest):
                self.log('found existing model file "%s".' % model_latest)
                self.log('adding "-c" to training command (to continue from previous training).')
                training_command += ' -c'
            else:
                self.log('No I didn\'t find anything. Training this stage from scratch...')
            self.log('train the network now.')
            self.log('>>> '+ training_command, print_to_console=False)
            run_shell(training_command)
            self.log('training finished.')
            
            self.checkpoints.set_finish('STAGE_2-3_TRAINING_DENOISER')

        #########################################################
        # ensemble predictions and obtain softmax probabilities #
        #########################################################
        raw_softmax = mkdir(join_path(self.stage_2_output_folder, '003_raw_softmax'))
        if not self.checkpoints.is_finished('STAGE_2-4_RAW_SOFTMAX'):
            self.log('calculating softmax probabilities in each epoch.')
            for epoch in range(self.DCNN_stage_2_epochs-self.DCNN_ensemble_epochs+1, self.DCNN_stage_2_epochs+1):
                epoch_model_name = 'model_ep_%04d' % epoch
                epoch_softmax_folder = mkdir(join_path(raw_softmax, 'epoch_%04d' % epoch))
                cases_already_done = _remove_duplicates_in_list([gn(item, no_extension=True) for item in ls(epoch_softmax_folder) if item[-9:]!='_0.nii.gz' ])
                cases_remaining = [item for item in list(self.train_dict.keys()) if item not in cases_already_done]
                if len(cases_remaining) == 0:
                    continue
                cases_remaining0 = ''
                for t in list(self.train_dict.keys()):
                    cases_remaining0 += ' %s ' % t
                imagesTr_folder = join_path(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data', self.DCNN_stage_2_task_name, 'imagesTr')
                predict_command = 'nnUNet_predict -i %s -o %s -tr %s -m %s -p %s -t %s -f %s -chk %s --selected_cases %s --save_softmax --disable_post_processing ' % \
                    (imagesTr_folder, epoch_softmax_folder, self.DCNN_trainer_name, self.DCNN_network_config,
                    self.DCNN_planner_name, self.DCNN_stage_2_task_name, self.DCNN_fold, epoch_model_name, cases_remaining0)
                disable_tta = True # disable test time augmentation? (8x faster for 3D data and 4x faster for 2D)
                if disable_tta:
                    predict_command += ' --disable_tta '
                self.log('Running predictions for epoch %d' % epoch)
                self.log('case(s) still need to be predicted: %d' % len(cases_remaining))
                self.log(predict_command, print_to_console=False)
                # saving raw softmax data will consume lots of disk space (~50MB per foreground channel and ~10MB per
                # background channel). So here i changed the nnunet code to only save softmax values of the background.
                # since we are dealing with a binary segmentation problem (for example, segmenting WMH lesions), softmax
                # values of the foreground channels will be obtained by one minus the background.
                run_shell(predict_command)
            self.checkpoints.set_finish('STAGE_2-4_RAW_SOFTMAX')

        #####################
        # softmax filtering #
        #####################
        masked_softmax = mkdir(join_path(self.stage_2_output_folder, '004_masked_softmax'))
        if not self.checkpoints.is_finished('STAGE_2-5_MASKED_SOFTMAX'):
            tasks = []
            self.log('masking softmax values...')
            all_cases = list(self.train_dict.keys())
            for epoch in range(self.DCNN_stage_2_epochs-self.DCNN_ensemble_epochs+1, self.DCNN_stage_2_epochs+1):
                in_softmax_folder = mkdir(join_path(raw_softmax, 'epoch_%04d' % epoch))
                out_softmax_folder = mkdir(join_path(masked_softmax, 'epoch_%04d' % epoch))
                for case in all_cases:
                    in_softmax = join_path(in_softmax_folder, '%s_0.nii.gz' % case)
                    out_softmax = join_path(out_softmax_folder, '%s_0.nii.gz' % case)
                    valid_mask = join_path(self.stage_1_output_folder, case, 'valid_mask.nii.gz')
                    assert files_exist([in_softmax, valid_mask])
                    tasks.append( (in_softmax, valid_mask, out_softmax)  )
            if len(laf(masked_softmax)) < len(tasks):
                run_parallel(_parallel_softmax_masking, tasks, self.num_CPU_cores, 'Masking')
            self.checkpoints.set_finish('STAGE_2-5_MASKED_SOFTMAX')

        ###############################################
        # ensemble softmaxes to obtain refined labels #
        ###############################################
        refined_label = mkdir(join_path(self.stage_2_output_folder, '005_refined_label'))
        if not self.checkpoints.is_finished('STAGE_2-6_ENSEMBLING'):
            self.log('averaging segmentations from last %d epoch(s).' % self.DCNN_ensemble_epochs)
            all_cases = list(self.train_dict.keys())
            case_softmax = {}
            for epoch in range(self.DCNN_stage_2_epochs-self.DCNN_ensemble_epochs+1, self.DCNN_stage_2_epochs+1):
                for case in all_cases:
                    if case not in case_softmax:
                        case_softmax[case]=[]
                    softmax = join_path(masked_softmax, 'epoch_%04d' % epoch, '%s_0.nii.gz' % case)
                    case_softmax[case].append(softmax)
            tasks = []
            for case in all_cases:
                case_dir = mkdir(join_path(refined_label, case))
                shape = load_nifti_simple( join_path(self.stage_1_output_folder, case, 'preprocessed_image.nii.gz') ).shape
                phys_res = get_nifti_pixdim( join_path(self.stage_1_output_folder, case, 'preprocessed_image.nii.gz') )
                prior_label = join_path(self.stage_1_output_folder, case, 'averaged_label.nii.gz')
                seg_pp = join_path(self.stage_1_output_folder, case, 'segmentation_pp.nii.gz')
                out_file = join_path(case_dir, 'softmax_ensembled.nii.gz')
                out_seg = join_path(case_dir, 'label_ensembled.nii.gz')
                in_softmaxes = case_softmax[case]
                tasks.append(  (in_softmaxes, prior_label, seg_pp, out_file, out_seg, shape, phys_res)  )
            run_parallel(_parallel_ensembling, tasks, self.num_CPU_cores, 'Ensembling softmax')
            self.checkpoints.set_finish('STAGE_2-6_ENSEMBLING')
        
        self.log('stage 2 complete.')

    def _do_DCNN_training(self):

        self.log('+----------------------------+')
        self.log('|  Stage III: DCNN training  |')
        self.log('+----------------------------+')

        partition_folder = mkdir(join_path(self.stage_3_output_folder, '001_data_partitions'))
        train_fit_folder = mkdir(join_path(self.stage_3_output_folder, '002_training_fit'))
        preview_folder = mkdir(join_path(self.stage_3_output_folder, '003_final_preview'))

        nnUNet_raw_data_base = os.environ['nnUNet_raw_data_base']
        tasks_raw_data_base_dir = mkdir(join_path(nnUNet_raw_data_base,'nnUNet_raw_data'))
        task_raw_data_dir = mkdir(join_path(tasks_raw_data_base_dir, self.DCNN_stage_3_task_name))
        
        train_cases, val_cases = [], []

        if not self.checkpoints.is_finished('STAGE_3-1_DATA_SPLIT'):
            
            self.log('picking samples to form training and validation set...')
            self.log('calculating Dice coefficients between refined labels and NLL initial segmentations...')
        
            all_cases = list(self.train_dict.keys())
            case_dice_pairs = []
            for case in all_cases:
                case_NLL_seg = join_path(self.stage_1_output_folder, case, 'segmentation_pp.nii.gz')
                case_fit_seg = join_path(self.stage_2_output_folder, '005_refined_label', case, 'label_ensembled.nii.gz')
                dice = hard_dice_binary( load_nifti_simple(case_NLL_seg), load_nifti_simple(case_fit_seg) )
                case_dice_pairs.append( ( case, dice ) )
            case_dice_pairs = sorted(case_dice_pairs, reverse=True, key=lambda x: x[1]) # sort dice from high to low

            self.log('start picking samples to form training & validation set...')

            validation_set_perc = 0.05 # Don't set it too high :-). 5% is enough for most of the tasks.

            self.log('%d%% of the samples will be put into validation set.' % int(validation_set_perc*100))
            val_target_num = int(len(all_cases) * validation_set_perc)
            if val_target_num < 1: 
                val_target_num = 1
                self.log('Warning: validation set is too small, please increase your dataset samples to stablize '
                         'training process.')
            for i in range(len(case_dice_pairs)):
                case, dice = case_dice_pairs[i]
                if len(val_cases) < val_target_num:
                    if i%2==0:
                        train_cases.append(case)
                    else:
                        val_cases.append(case)
                else: # samples in validation set is enough, just put it into training set
                    train_cases.append(case)
            self.log('training set size: %d' % len(train_cases))
            self.log('validation set size: %d' % len(val_cases))
            self.log('case(s) used in validation: ')
            self.log('  '.join(val_cases))

            save_pkl(train_cases, join_path(partition_folder, 'train_cases.pkl'))
            save_pkl(val_cases, join_path(partition_folder, 'val_cases.pkl'))
        
            self.checkpoints.set_finish('STAGE_3-1_DATA_SPLIT')

        else:

            train_cases = load_pkl(join_path(partition_folder, 'train_cases.pkl'))
            val_cases = load_pkl(join_path(partition_folder, 'val_cases.pkl'))

        all_cases = train_cases + val_cases

        #############################
        # prepare data for training #
        #############################
        if not self.checkpoints.is_finished('STAGE_3-2_PREPARE_DATA'):
            self.log('preparing data...')

            train_list = []
            for case_id in range(len(all_cases)):
                case = all_cases[case_id]
                preprocessed_image = join_path(self.stage_1_output_folder, case, 'preprocessed_image.nii.gz')
                seg = join_path(self.stage_2_output_folder, '005_refined_label', case, 'label_ensembled.nii.gz')
                mask = join_path(self.stage_1_output_folder, case, 'valid_mask.nii.gz')
                train_list.append( (case, preprocessed_image, seg, mask) )
            _prepare_nnunet_training_data(train_list, task_raw_data_dir)

            self.log('finished.')
            self.checkpoints.set_finish('STAGE_3-2_PREPARE_DATA')

        #######################
        # plan and preprocess #
        #######################
        if not self.checkpoints.is_finished('STAGE_3-3_PLAN_AND_PREPROCESS'):
            self.log('start preparing data...')
            preproc_command = 'nnUNet_plan_and_preprocess -t %d --verify_dataset_integrity' % _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_3_task_name)
            self.log('>>> '+ preproc_command, print_to_console=False)
            run_shell(preproc_command)
            self.log('finished preprocessing')
            self.checkpoints.set_finish('STAGE_3-3_PLAN_AND_PREPROCESS')

        ##################
        # start training #
        ##################
        if not self.checkpoints.is_finished('STAGE_3-4_TRAINING'):
            self.log('start training DCNN for final segmentation...')
            val_cases0 = ''
            for val_case in val_cases:
                val_cases0 += ' %s ' % val_case
            training_command = 'nnUNet_train %s %s %d %s -e %d -b %d --disable_postprocessing_on_folds --custom_val_cases %s ' % \
                (self.DCNN_network_config, 
                self.DCNN_trainer_name,
                _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_3_task_name), 
                self.DCNN_fold, 
                self.DCNN_stage_3_epochs, 
                self.DCNN_batches_in_each_epoch,
                val_cases0)
            self.log('looking for existing model(s) in folder "%s"...' % self.DCNN_stage_3_model_folder)
            model_latest = join_path(self.DCNN_stage_3_model_folder, 'model_latest.model') 
            self.log('try finding latest model "%s"...' % model_latest)
            if file_exist(model_latest):
                self.log('found existing model file "%s".' % model_latest)
                self.log('adding "-c" to training command (to continue from previous training).')
                training_command += ' -c'
            else:
                self.log('No I didn\'t find anything. Training this stage from scratch...')
            self.log('train the network now.')
            self.log('>>> '+ training_command, print_to_console=False)
            run_shell(training_command)
            self.log('training finished.')
            
            self.checkpoints.set_finish('STAGE_3-4_TRAINING')

        #################################
        # retrieve training set outputs #
        #################################
        imagesTr_dir = mkdir(join_path(task_raw_data_dir,'imagesTr'))
        if not self.checkpoints.is_finished('STAGE_3-5_FINAL_FIT'):
            self.log('start to retrieve training set outputs...')
            predict_command = 'nnUNet_predict -i %s -o %s -tr %s -m %s -p %s -t %s -f %s -chk model_best --disable_post_processing ' % \
                (imagesTr_dir, train_fit_folder, self.DCNN_trainer_name, self.DCNN_network_config,
                self.DCNN_planner_name, self.DCNN_stage_3_task_name, self.DCNN_fold)
            self.log('>>> ' + predict_command)
            run_shell(predict_command, print_command=False)

            all_segs = [ item for item in ls(train_fit_folder, full_path=True) if item[-7:]=='.nii.gz' ]
            out_3mm_postseg = mkdir(join_path(train_fit_folder, '3mm_postproc'))
            task_list = []
            for seg in all_segs:
                case_name = gn(seg, no_extension=True)
                valid_mask = join_path(self.stage_1_output_folder, case_name, 'valid_mask.nii.gz')
                out_seg = join_path( join_path(out_3mm_postseg, '%s.nii.gz' % ( gn(seg, no_extension=True) )) )
                task_list.append(
                    (seg, valid_mask, out_seg)
                )
            run_parallel(_parallel_3mm_postproc, task_list, self.num_CPU_cores, "post-processing")
            
            # generate GIF animations to quickly preview segmentation quality
            all_cases = list(self.train_dict.keys())
            task_list = []
            for case in all_cases:
                in_image = join_path( self.stage_1_output_folder, case, 'normalized_input.nii.gz' )
                in_seg = join_path(out_3mm_postseg, '%s.nii.gz' % case)
                out_gif = join_path(preview_folder, '%s_image+seg.gif' % case)
                task_list.append(
                    (in_image, in_seg, out_gif)
                )
            run_parallel( _parallel_generate_final_GIF, task_list, self.num_CPU_cores, "generating GIF" )

            self.checkpoints.set_finish('STAGE_3-5_FINAL_FIT')

        self.checkpoints.set_finish('PIPELINE_TRAINING_COMPLETE')
        self.log('stage 3 complete.')

    def add_training_case(self, name: str, x_train: str, x_refs: list, label1: list, label2: list, description=None):
        self.train_dict[name] = {'description': description}
        self.lesion_analyzer.add_case(name, x_train, x_refs, label1, label2)

    def run_training(self, gpu:int=0, run_stages='full'):

        '''
        Train the whole pipeline.
        '''

        assert run_stages in ['initseg', 'denoise', 'full'], 'unknown "run_stages=%s".' % str(run_stages)

        # setup GPU
        self.gpu = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % self.gpu
        self.log('running pipeline using GPU %d.' % self.gpu)

        # for debug purpose, you can only run part of these steps
        if run_stages == 'initseg': # just do initial segmentation (noisy)
            self._do_initial_segmentation() # stage 1
        elif run_stages == 'denoise': # do initial segmentation, then denoise using DCNN
            self._do_initial_segmentation() # stage 1
            self._do_label_denoising()      # stage 2
        elif run_stages == 'full': # run the full pipeline
            self._do_initial_segmentation() # stage 1
            self._do_label_denoising()      # stage 2
            self._do_DCNN_training()        # stage 3
            self.log('training complete.')

    def release_model(self, output_folder):

        '''
        Release the trained model.
        '''

        self.log('collecting models...')
        if not self.checkpoints.is_finished('PIPELINE_TRAINING_COMPLETE') and \
            not self.checkpoints.is_finished('MIXED_COHORT_3_MODEL_TRAINING'):
            
            self.log('Pipeline is not fully trained! You need to train the pipeline before inference. '
                    'Use "self.run_training(gpu=XXX)" to train the pipeline.')
            return
        release_model_path = mkdir(join_path(output_folder, 'nnUNet', self.DCNN_network_config, self.DCNN_stage_3_task_name,
            '%s__%s' % (self.DCNN_trainer_name, self.DCNN_planner_name), self.DCNN_fold))
        for file in ls(self.DCNN_stage_3_model_folder):
            if file in ['model_best.model', 'model_best.model.pkl']:
                cp( join_path(self.DCNN_stage_3_model_folder, file), join_path(release_model_path, file) )
        # copy plans
        cp(
            join_path(os.environ['RESULTS_FOLDER'], 'nnUNet', self.DCNN_network_config, self.DCNN_stage_3_task_name,
            '%s__%s' % (self.DCNN_trainer_name, self.DCNN_planner_name), 'plans.pkl'),
            join_path(output_folder, 'nnUNet', self.DCNN_network_config, self.DCNN_stage_3_task_name,
            '%s__%s' % (self.DCNN_trainer_name, self.DCNN_planner_name), 'plans.pkl')
        )
        cp(
            join_path(os.environ['nnUNet_preprocessed'], self.DCNN_stage_3_task_name, 'dataset_properties.pkl'),
            join_path(output_folder, 'dataset_properties.pkl')
        )
        cp(
            join_path(os.environ['nnUNet_preprocessed'], self.DCNN_stage_3_task_name, '%s_plans_3D.pkl' % self.DCNN_planner_name),
            join_path(output_folder, '%s_plans_3D.pkl' % self.DCNN_planner_name)
        )
        cp(
            join_path(os.environ['nnUNet_preprocessed'], self.DCNN_stage_3_task_name, '%s_plans_2D.pkl' % self.DCNN_planner_name),
            join_path(output_folder, '%s_plans_2D.pkl' % self.DCNN_planner_name)
        )
        # compress file to a single package (*.tar.gz format) which is convenient for later installation
        model_targz = join_path(output_folder, 'model_release.tar.gz')
        if file_exist(model_targz):
            self.log('removing previously compressed model pack...')
            rm(model_targz)
        self.log('compressing model for release...')
        targz_compress( output_folder, model_targz )

        self.log('finished.')
        self.log('Training is complete! Released model is stored to "%s".' % model_targz)

    def mixed_cohort_training(self, data_dict, val_cases, add_noise = True, model_release_folder = None):

        '''
        Train the network using data from multiple cohorts.
        
        data_dict: {
            'case_001': ['/path/to/train/image001.nii.gz', '/path/to/seg001.nii.gz'],
            'case_002': ['/path/to/train/image002.nii.gz', '/path/to/seg002.nii.gz'],
            ...
        }

        val_cases: ['case_001', ...]

        '''
        
        for val_case in val_cases:
            assert val_case in data_dict.keys(), 'validation case "%s" is not in data_dict.' % val_case

        self.log('started mixed cohort training...')


        if not self.checkpoints.is_finished('MIXED_COHORT_1_PREPARE_DATA'):

            data_list = []
            for case_name in data_dict:
                data_list.append(  (case_name, data_dict[case_name][0], data_dict[case_name][1], None)  )
            
            nnUNet_raw_data_base = os.environ['nnUNet_raw_data_base']
            tasks_raw_data_base_dir = mkdir(join_path(nnUNet_raw_data_base,'nnUNet_raw_data'))
            task_raw_data_dir = mkdir(join_path(tasks_raw_data_base_dir, self.DCNN_stage_3_task_name))

            augstring = ""
            if add_noise:
                augstring = "noise=0.1" # adding Gaussian noise N(0, sigma) to each training image, sigma=0.1*(q95-q5)

            _prepare_nnunet_training_data(data_list, task_raw_data_dir,description="Mixed cohort training",augmentation=augstring)

            self.checkpoints.set_finish('MIXED_COHORT_1_PREPARE_DATA')

        # plan and preprocess
        if not self.checkpoints.is_finished('MIXED_COHORT_2_PLAN_AND_PREPROCESS'):

            preproc_command = 'nnUNet_plan_and_preprocess -t %d --verify_dataset_integrity' % _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_3_task_name)
            self.log('>>> '+ preproc_command, print_to_console=False)
            run_shell(preproc_command)

            self.checkpoints.set_finish('MIXED_COHORT_2_PLAN_AND_PREPROCESS')

        # train
        if not self.checkpoints.is_finished('MIXED_COHORT_3_MODEL_TRAINING'):
            training_command = 'nnUNet_train %s %s %d %s -e %d -b %d --disable_postprocessing_on_folds --custom_val_cases %s ' % \
                (self.DCNN_network_config, 
                self.DCNN_trainer_name,
                _convert_nnunet_TaskXXX_to_task_id(self.DCNN_stage_3_task_name), 
                self.DCNN_fold, 
                self.DCNN_stage_3_epochs, 
                self.DCNN_batches_in_each_epoch,
                ' '.join(val_cases))
            
            self.log('looking for existing model(s) in folder "%s"...' % self.DCNN_stage_3_model_folder)
            model_latest = join_path(self.DCNN_stage_3_model_folder, 'model_latest.model') 
            self.log('try finding latest model "%s"...' % model_latest)
            if file_exist(model_latest):
                self.log('found existing model file "%s".' % model_latest)
                self.log('adding "-c" to training command (to continue from previous training).')
                training_command += ' -c'
            else:
                self.log('No I didn\'t find anything. Training this stage from scratch...')
            self.log('train the network now.')
            self.log('>>> '+ training_command, print_to_console=False)
            run_shell(training_command)
            self.log('training finished.')
            
            self.checkpoints.set_finish('MIXED_COHORT_3_MODEL_TRAINING')
        
        # release model if needed
        if model_release_folder is not None:
            self.release_model(model_release_folder)
