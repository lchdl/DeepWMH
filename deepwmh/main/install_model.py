#
# install pretrained model (*.tar.gz) into your machine.
# the installed model can be used using "DeepWMH_predict" command
#

import argparse
import os
from deepwmh.utilities.file_ops import cp, dir_exist, file_exist, join_path, ls, mkdir, rm
from deepwmh.utilities.data_io import load_pkl, save_pkl, targz_compress, targz_uncompress

def pack_and_release_nnUNet_model(task_name, output_folder, 
    nnUNet_name = 'nnUNet', network_config = '3d_fullres', trainer_name = 'nnUNetTrainerV2',
    planner_name = 'nnUNetPlansv2.1', fold = 'all'):

    if os.environ.get('RESULTS_FOLDER', None) == None:
        raise RuntimeError('please set "RESULTS_FOLDER" environment variable.')

    release_model_path = mkdir(join_path(output_folder, nnUNet_name, network_config, task_name,
        '%s__%s' % (trainer_name, planner_name), fold))
    source_model_path = join_path(os.environ['RESULTS_FOLDER'], nnUNet_name, network_config, task_name,
        '%s__%s' % (trainer_name, planner_name), fold)
    for file in ls(source_model_path):
        if file in ['model_best.model', 'model_best.model.pkl']:
            cp( join_path(source_model_path, file), join_path(release_model_path, file) )
    # copy plans
    cp(
        join_path(os.environ['RESULTS_FOLDER'], nnUNet_name, network_config, task_name,
        '%s__%s' % (trainer_name, planner_name), 'plans.pkl'),
        join_path(output_folder, nnUNet_name, network_config, task_name,
        '%s__%s' % (trainer_name, planner_name), 'plans.pkl')
    )
    cp(
        join_path(os.environ['nnUNet_preprocessed'], task_name, 'dataset_properties.pkl'),
        join_path(output_folder, 'dataset_properties.pkl')
    )
    cp(
        join_path(os.environ['nnUNet_preprocessed'], task_name, '%s_plans_3D.pkl' % planner_name),
        join_path(output_folder, '%s_plans_3D.pkl' % planner_name)
    )
    cp(
        join_path(os.environ['nnUNet_preprocessed'], task_name, '%s_plans_2D.pkl' % planner_name),
        join_path(output_folder, '%s_plans_2D.pkl' % planner_name)
    )
    # compress file to a single package which is convenient for later installation
    model_targz = join_path(output_folder, 'model_release.tar.gz')
    if file_exist(model_targz):
        print('removing previously compressed model pack...')
        rm(model_targz)
    print('compressing model for release...')
    targz_compress( output_folder, model_targz )


def main():
	class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, 
					  argparse.RawDescriptionHelpFormatter):  pass

	parser = argparse.ArgumentParser(
		description = 'Install pre-trained segmentation model.',
		formatter_class = MyFormatter)
	parser.add_argument('-m', '--model-targz', 
		help='Pre-trained model file (in *.tar.gz format)',
        type=str, required=True)
	parser.add_argument('-o', '--install-location',
		help='Model install location.',
		type=str, required=True)
	parser.add_argument('-f', '--force',
		help = 'Overwrite if model already exists.',
		action='store_true', required=False)
	args = parser.parse_args()

	print('model file is: "%s".' % args.model_targz)
	print('install location: "%s".' % args.install_location)
	mkdir(args.install_location)
	if len(ls(args.install_location)) > 0:
		if args.force == False:
			raise Exception('Model can be only installed in an empty directory! '
				'Add "-f" to overwrite existing model (not recommended).')
		else:
			print('WARNING: overwriting previously installed model...')

	print('installing model...')
	targz_uncompress(args.model_targz, args.install_location)
	
	print('setting up paths...')
	DCNN_backend = 'nnUNet'
	DCNN_network_config = '3d_fullres'
	DCNN_trainer_name = 'nnUNetTrainerV2'
	DCNN_planner_name = 'nnUNetPlansv2.1'
	DCNN_stage_3_task_name = 'Task002_FinalModel'
	DCNN_fold = 'all'
	model_pkl = join_path(args.install_location, DCNN_backend, DCNN_network_config, DCNN_stage_3_task_name,
		'%s__%s' % (DCNN_trainer_name, DCNN_planner_name), DCNN_fold, 'model_best.model.pkl')
	model_dict = load_pkl(model_pkl)
	model_init_params = model_dict['init']
	new_model_init_params = (
		join_path(args.install_location, '%s_plans_3D.pkl' % DCNN_planner_name),
		model_init_params[1],
		join_path(args.install_location, DCNN_backend, DCNN_network_config, DCNN_stage_3_task_name,
			'%s__%s' % (DCNN_trainer_name, DCNN_planner_name)),
		'',
		*model_init_params[4:]
	)
	model_dict['init'] = new_model_init_params
	assert file_exist(new_model_init_params[0])
	assert dir_exist(new_model_init_params[2])
	save_pkl(model_dict, model_pkl) # overwrite and save

	print('installation complete. ')

	