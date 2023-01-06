# a simple wrapper for VoxelMorph
import os
import argparse
import numpy as np
from typing import Union
from scipy.ndimage import zoom
from freeseg.utilities.parallelization import run_parallel
from freeseg.utilities.misc import ignore_SIGINT
from freeseg.utilities.external_call import run_shell, try_shell
from freeseg.utilities.data_io import get_nifti_pixdim, load_csv_simple, load_nifti, load_nifti_simple, \
	save_nifti_simple, save_nifti, try_load_nifti
from freeseg.utilities.file_ops import cp, file_exist, gd, gn, join_path, laf, mkdir, rm
from freeseg.pipeline.DCNN_multistage import Checkpoints
from freeseg.external_tools.ANTs_group_registration import antsApplyTransforms, antsRegistration

def determine_internal_shape(expected_shape, template_shape, template_resolution):
	def _parse_triplet(s):
		return [int(word) for word in s.split('x')]
	if expected_shape == "original":
		return template_shape
	else:
		return _parse_triplet(expected_shape)

def zoom_internal(data: np.ndarray, output_shape: Union[list, tuple]):
	data_shape = np.array(data.shape)
	zoom_factors = [ output_shape[0]/data_shape[0], output_shape[1]/data_shape[1], output_shape[2]/data_shape[2] ]
	zoomed = zoom(data,zoom_factors)
	return zoomed

# important: must pre-process images before sending into network
def preprocess_image(image_path, internal_shape, mask_path = None, normalize_intensity=True):
	image_data = load_nifti_simple(image_path)
	# zoom
	image_data = zoom_internal(image_data, internal_shape)
	# normalize intensity
	if normalize_intensity:
		# do z-score normalization
		mu, sigma = np.mean(image_data), np.std(image_data)
		image_data = (image_data - mu) / sigma
	#
	if mask_path is not None:
		mask_data = load_nifti_simple(mask_path)
		mask_data = zoom_internal(mask_data, internal_shape)
		mask_data = (mask_data > 0.5).astype('float32')

	if mask_path is None:
		return image_data
	else:
		return image_data, mask_data

def _parallel_FSL_affine_s(params):
	data_path, template_output, out_data_path, affine_mat, mask_path, out_mask_path,\
		internal_shape, pre_data_path, pre_mask_path, normalize_intensity = params
	if try_load_nifti(pre_data_path) == True and try_load_nifti(pre_mask_path) == True:
		return
	run_shell(antsRegistration(data_path, template_output, out_data_path,deform_type='Linear'), print_command=False, print_output=False)
	affine_mat = join_path(  gd( out_data_path ), 'warp_0GenericAffine.mat'  )
	run_shell(antsApplyTransforms(mask_path, template_output, affine_mat, out_mask_path, interpolation_method='Linear'), print_command=False, print_output=False)
	pre_data, pre_mask = preprocess_image(out_data_path, internal_shape, mask_path=out_mask_path,
		normalize_intensity=normalize_intensity)
	save_nifti_simple(pre_data, pre_data_path)
	save_nifti_simple(pre_mask, pre_mask_path)

def _parallel_FSL_affine_t(params):
	data_path, template_output, out_data_path, affine_mat, out_data_path, internal_shape, pre_data_path,\
		normalize_intensity = params
	if try_load_nifti(pre_data_path) == True:
		return
	run_shell(antsRegistration(data_path, template_output, out_data_path,deform_type='Linear'), print_command=False, print_output=False)
	affine_mat = join_path(  gd( out_data_path ), 'warp_0GenericAffine.mat'  )
	pre_data = preprocess_image(out_data_path, internal_shape, normalize_intensity=normalize_intensity)
	save_nifti_simple(pre_data, pre_data_path)

def _vxm_select_latest_model(model_folder):
	all_models = [( int(gn(item,no_extension=True)) ,item) for item in laf(model_folder) if item[-3:]=='.pt']
	if len(all_models) == 0:
		return None, None
	else:
		latest_model = sorted(all_models, key=lambda x: x[0], reverse=True)[0]
		model_epoch, model_path = latest_model
		return model_epoch, model_path


def vxm_end2end():

	class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, 
					argparse.RawDescriptionHelpFormatter):  pass

	parser = argparse.ArgumentParser(
		description=
            'vxmEnd2End: configure, train, and register two image sets using VoxelMorph. '
			'You need to install\n'
			'1. VoxelMorph (https://github.com/lchdl/voxelmorph),\n'
			'2. ANTs (https://github.com/ANTsX/ANTs)\n'
			'before using this script.', formatter_class = MyFormatter)
	parser.add_argument('-s', '--source', help='Source image dataset (csv path).',type=str,required=True)
	parser.add_argument('-t', '--target', help='Target image dataset (csv path).',type=str,required=True)
	parser.add_argument('-j', '--num-workers', help='Number of CPU cores.', type=int, required=False, default=4)
	parser.add_argument('-e', '--epochs', help='Number of epochs for training network.', type=int, required=False, default=1000)
	parser.add_argument('-n', '--normalize-intensity', action='store_true', 
		help='Using z-score normalization to normalize/scale image intensity before sending into network. '
		'This only affects training. During final registration the original images are used instead of '
		'pre-processed images. Useful when the intensities of input images are too high (>10000). '
		'If training fails, try add this option and run the whole pipeline from scratch again.', 
		required=False, default=False)
	parser.add_argument('-m', '--vxm-folder', 
        help=
            'This option tells vxmEnd2End where to store pre-processed images, models and registered images.',
        type=str,required=True)
	parser.add_argument('--internal-shape', 
		help=
			'Internal image size used in training. Format is: "XxYxZ" (sagittal x coronal x axial). '
			'Image size must satisfy all three requirements listed as follows: '
			'1) the image size in each dimension is divisible by 16 (for example, "192x32x256", "512x512x16"); '
			'2) the image should not be too large, which will cause GPU run out of memory; '
			'3) all the images should have the same image shape. '
			'Default value is "128x128x128".',
		type=str, default="128x128x128", required=True)
	parser.add_argument('-g', '--gpu', help='GPU id',type=int, default=0)

	args = parser.parse_args()	

	source_csv = args.source
	target_csv = args.target
	vxm_folder = args.vxm_folder
	num_workers = args.num_workers
	num_epochs = args.epochs
	normalize_intensity = args.normalize_intensity
	gpu = args.gpu

	#############################################
	######### checking system integrity #########
	#############################################

	print('checking system integrity...')

	# checking if VoxelMorph is installed
	try:
		import voxelmorph
		vxm_train_loc    = join_path(gd(gd( voxelmorph.__file__ )), 'scripts' ,'torch', 'train.py')
		vxm_register_loc = join_path(gd(gd( voxelmorph.__file__ )), 'scripts' ,'torch', 'register.py')
		if file_exist(vxm_train_loc) == False:
			raise FileNotFoundError('cannot find file "train.py" in folder "%s", '
				'please make sure your code is downloaded and unzipped from "https://github.com/lchdl/voxelmorph" '
				'and install voxelmorph again.' % gd(vxm_train_loc))
		if file_exist(vxm_register_loc) == False:
			raise FileNotFoundError('cannot find file "register.py" in folder "%s", '
				'please make sure your code is downloaded and unzipped from "https://github.com/lchdl/voxelmorph" '
				'and install voxelmorph again.' % gd(vxm_train_loc))
	except ImportError:
		raise ImportError('cannot import voxelmorph, please install "VoxelMorph" from '
			'"https://github.com/lchdl/voxelmorph" in order to use this script.')
	except:
		raise
	else:
		print('VoxelMorph is successfully installed.')

	# checking if torch is compatible with existing GPU and CUDA version
	try:
		import torch
		if torch.cuda.is_available() == False:
			raise RuntimeError(
				'CUDA is not available. Please install PyTorch with another version or '
				'update your GPU driver. Older PyTorch versions can be downloaded from '
				'"https://pytorch.org/get-started/previous-versions/".')
	except ImportError:
		raise ImportError('cannot import torch, install torch from "pip install pytorch==x.x" or *.whl. ')
	except:
		raise
	else:
		print('PyTorch is installed with CUDA & GPU support.')
	
	# checking if ANTs is installed
	if os.environ.get('ANTSPATH', None) == None:
		raise RuntimeError(
			'"ANTSPATH" environment variable is not set. Please add \n'
			'export ANTSPATH="/path/to/your/ANTs-x.x/bin/"\n'
			'in your ~/.bashrc and\n'
			'>>> source ~/.bashrc\n'
			'to update your environment variable.\n')	
	if try_shell('${ANTSPATH}/antsMultivariateTemplateConstruction2.sh -h') != 1:
		raise RuntimeError(
			'cannot execute shell script "${ANTSPATH}/antsMultivariateTemplateConstruction2.sh", '
			'make sure the file exists and has execute permission!')
	if try_shell('antsRegistration -h') != 0:
		raise RuntimeError(
			'Cannot run antsRegistration! Make sure you have installed ANTs and set "ANTSPATH" environment '
			'variable in your "~/.bashrc" file.')
	if try_shell('antsApplyTransforms -h') != 0:
		raise RuntimeError(
			'Cannot run antsRegistration! Make sure you have installed ANTs and set "ANTSPATH" environment '
			'variable in your "~/.bashrc" file.')

	print('ANTs is installed.')
	
	######################################
	######### configure pipeline #########
	######################################

	# load source and target datasets
	source_dataset = load_csv_simple(source_csv,key_names=['case','data','mask'])
	target_dataset = load_csv_simple(target_csv,key_names=['case','data'])

	# using source images to build template
	source_images = source_dataset['data']

	checkpoints = Checkpoints(mkdir( join_path(vxm_folder, 'checkpoints') ))

	###############################################
	######### start template construction #########
	###############################################

	print('constructing template (rough & quick), this may take a while, please wait...')
	'''
	One thing need to notice is that during template construction, 
	the program will invoke "antsMultivariateTemplateConstruction2.sh" 
	which is part of the ANTs toolkit. This shell script spawns some 
	child processes to construct template for us. However, if we press 
	"Ctrl+C" when the program is running, the SIGINT signal will sent
	to that shell process, which traps the signal and ignores it, but
	our main program will still exit. This is OK for most of the cases,
	but if you want to construct the template again on the same machine,
	with these undead child processes still running, the program is 
	not safe anymore (because the temporary NIFTI image files can be
	overwritten and destroyed by those processes) and the results can 
	be unexpected. So for safety reason I will disable KeyboardInterrupt 
	for now. 
	'''
	with ignore_SIGINT():
		print('keyboard interrupt is temporarily disabled during template construction for safety reason.')
		template_folder = mkdir(join_path(vxm_folder, 'template'))
		output_prefix = template_folder + '/T_'
		template_output = join_path(template_folder,'T_template0.nii.gz')
		if checkpoints.is_finished('TEMPLATE_CONSTRUCTION') == False:
			if len(source_images) > 1:
				# 1. -d 3: 3D image
				# 2. -r 1: use rigid transformations for constructing the initial template
				# 3. -i 0: no iteration to refine the template (can be more but not worth it, we just need to average them)
				# 4. -t Affine: use affine registration only (not using SyN for saving time)
				# 5. -o xxx: output prefix
				# 6. -c 2 -j xxx: number of CPU cores used
				template_construction_command = '${ANTSPATH}/antsMultivariateTemplateConstruction2.sh '\
					'-d 3 -o %s -t Affine -r 1 -i 0 -c 2 -j %d   %s  ' % \
					(output_prefix, num_workers, ' '.join(source_images))
				run_shell(template_construction_command, print_command=False ,print_output=False)

				for item in laf(template_folder):
					if item != template_output and item[-4:]!='.pkl':
						rm(item)
			else: # only one image in source set, don't need to run template construction.
				cp( source_images[0], template_output ) # directly use this image as template
			# set finish flag
			checkpoints.set_finish('TEMPLATE_CONSTRUCTION')

	#######################################
	######### affine registration #########
	#######################################

	# determine internal shape based on template image
	template_shape = load_nifti_simple(template_output).shape
	template_resolution = get_nifti_pixdim(template_output)
	internal_shape = determine_internal_shape(args.internal_shape, template_shape, template_resolution)
	print('internal shape is set as : %s.' % str(internal_shape))

	# use ANTs to affinely register images to rough template
	print('affine registration & pre-processing')

	affine_folder = mkdir(join_path( vxm_folder, 'affine' ))
	preproc_folder = mkdir(join_path( vxm_folder, 'preproc' ))
	train_imgs_txt = join_path( preproc_folder, 'train_imgs.txt' )

	if not checkpoints.is_finished('FSL_AFFINE_REG'):

		pre_source_list, pre_target_list = [], []

		parallel_tasks = []
		for case_name, data_path, mask_path in zip( source_dataset['case'], source_dataset['data'], source_dataset['mask'] ):
			caes_affine_folder = mkdir(join_path(affine_folder, case_name))
			out_data_path = join_path( caes_affine_folder, '%s_data.nii.gz' % case_name )
			out_mask_path = join_path( caes_affine_folder, '%s_mask.nii.gz' % case_name )
			pre_data_path = join_path( preproc_folder, '%s_data.nii.gz' % case_name )
			pre_mask_path = join_path( preproc_folder, '%s_mask.nii.gz' % case_name )
			if file_exist(pre_data_path)==False or file_exist(pre_mask_path)==False:
				affine_mat = join_path( affine_folder, '%s.mat' % case_name )
				parallel_tasks.append( (data_path, template_output, out_data_path, affine_mat, mask_path, out_mask_path,\
					internal_shape, pre_data_path, pre_mask_path, normalize_intensity) )				
			pre_source_list.append( pre_data_path )
		if len(pre_source_list)>0:
			run_parallel(_parallel_FSL_affine_s, parallel_tasks, num_workers, 'affine reg s.')

		parallel_tasks = []
		for case_name, data_path in zip( target_dataset['case'], target_dataset['data'] ):
			caes_affine_folder = mkdir(join_path(affine_folder, case_name))
			out_data_path = join_path( caes_affine_folder, '%s_data.nii.gz' % case_name )
			pre_data_path = join_path( preproc_folder, '%s_data.nii.gz' % case_name )
			if file_exist(pre_data_path) == False:
				affine_mat = join_path( affine_folder, '%s.mat' % case_name )
				parallel_tasks.append( (data_path, template_output, out_data_path, affine_mat, out_data_path, \
					internal_shape, pre_data_path, normalize_intensity)  )
			pre_target_list.append( pre_data_path )
		if len(pre_target_list)>0:
			run_parallel(_parallel_FSL_affine_t, parallel_tasks, num_workers, 'affine reg t.')

		print('writing data...')
		with open(train_imgs_txt, 'w') as f:
			for item in pre_source_list + pre_target_list:
				f.write(item+'\n')

		checkpoints.set_finish('FSL_AFFINE_REG')

	################################
	######### VxM training #########
	################################

	model_folder = mkdir(join_path( vxm_folder, 'models' ))
	
	print('start training...')	
	if checkpoints.is_finished('VXM_TRAINING') == False:
		model_epoch, model_path = _vxm_select_latest_model(model_folder)
		if model_path == None: # no model saved, train from scratch
			vxm_train_command = 'python %s --img-list %s --model-dir %s --gpu %d --epochs %d ' % \
				(vxm_train_loc, train_imgs_txt, model_folder, gpu, num_epochs)
		else:
			print('restoring latest model...')
			print('continue from epoch %d model file "%s".' % (model_epoch, model_path))
			vxm_train_command = 'python %s --img-list %s --model-dir %s --gpu %d --initial-epoch %d '\
				'--load-model %s --epochs %d' % \
				(vxm_train_loc, train_imgs_txt, model_folder, gpu, model_epoch, model_path, num_epochs)
		run_shell(vxm_train_command)
		checkpoints.set_finish('VXM_TRAINING')
	print('training finished.')

	######################################
	######### image registration #########
	######################################	
	
	print('start registration...')
	regfinal_folder = mkdir(join_path(vxm_folder, 'regfinal'))

	template_path = join_path(vxm_folder, 'template', 'T_template0.nii.gz')
	template_data, template_header = load_nifti(template_path)
	template_shape = template_data.shape
	print('using template: "%s".' % template_path)
	print('internal shape is: %s.' % str(internal_shape))

	if not checkpoints.is_finished('END2END_REGISTRATION'):
		model_epoch, model_path = _vxm_select_latest_model(model_folder)
		if model_path == None:
			raise RuntimeError('cannot find model file in "%s".' % model_folder)
		print('selecting model for registration..')
		print('using epoch %d model file "%s".' % (model_epoch, model_path))
		a, b = 0, len(source_dataset['case']) * len(target_dataset['case'])
		for source_case, _, _ in zip( source_dataset['case'], source_dataset['data'], source_dataset['mask'] ):
			for target_case, target_image in zip( target_dataset['case'], target_dataset['data'] ):
				a+=1
				print('[%d/%d] %s >>> %s' % (a, b, source_case, target_case))
				
				final_moved_d = join_path(regfinal_folder, '%s_to_%s_data.nii.gz' % (source_case, target_case))
				final_moved_m = join_path(regfinal_folder, '%s_to_%s_mask.nii.gz' % (source_case, target_case))

				if try_load_nifti(final_moved_d) and try_load_nifti(final_moved_m):
					continue

				temporary_folder = mkdir(join_path(vxm_folder, '__temp__'))

				# register image in internal template space
				moving_pre_d = join_path(  vxm_folder, 'preproc', '%s_data.nii.gz' % source_case )
				fixed_pre_d = join_path(  vxm_folder, 'preproc', '%s_data.nii.gz' % target_case )
				moving_pre_m = join_path(  vxm_folder, 'preproc', '%s_mask.nii.gz' % source_case )
				moved_pre_m = join_path(  temporary_folder , 'moved_pre_m.nii.gz' )
				moving_orig_d = join_path(  temporary_folder, 'moving_orig_d.nii.gz' )
				moved_orig_d = join_path(  temporary_folder, 'moved_orig_d.nii.gz' )
				save_nifti_simple(preprocess_image(
					join_path(vxm_folder, 'affine', source_case, '%s_data.nii.gz' % source_case), 
					internal_shape,normalize_intensity=False), moving_orig_d)
				run_shell(  'python %s --moving %s --fixed %s --aux-moving %s %s --aux-moved %s %s '\
					'--model %s --gpu %d '\
					% (vxm_register_loc, moving_pre_d, fixed_pre_d, 
					moving_orig_d, moving_pre_m, moved_orig_d, moved_pre_m,
					model_path, gpu )  )
				
				# zoom image to align with template, then save image with template header
				tplt_d = join_path(  temporary_folder, 'template_d.nii.gz' ) # data in template space
				tplt_m = join_path(  temporary_folder, 'template_m.nii.gz' ) # mask in template space
				zoom_factors = [ template_shape[0]/internal_shape[0], template_shape[1]/internal_shape[1], template_shape[2]/internal_shape[2] ]
				zoomed_d = zoom( load_nifti_simple(moved_orig_d) , zoom_factors)
				zoomed_m = (zoom( load_nifti_simple(moved_pre_m) , zoom_factors) > 0.5).astype('float32')
				save_nifti( zoomed_d, template_header, tplt_d )
				save_nifti( zoomed_m, template_header, tplt_m )
				
				# inverse affine transform to target image
				affine_mat = join_path( vxm_folder, 'affine', target_case, 'warp_0GenericAffine.mat' )
				run_shell(antsApplyTransforms( tplt_d, target_image, affine_mat, final_moved_d, interpolation_method='Linear', inverse_transform=True))
				run_shell(antsApplyTransforms( tplt_m, target_image, affine_mat, final_moved_m, interpolation_method='NearestNeighbor', inverse_transform=True))				
				rm(temporary_folder)

		checkpoints.set_finish('END2END_REGISTRATION')

