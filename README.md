# annotation_free_wmh_seg
Annotation free white matter hyperintensities (WMH) segmentation using deep learning.

# Table of contents

- [About](#about)
- [Requirements](#requirements)
- [Quick start: how to use our pretrained model (only for Linux-based systems)](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems)
- [Advanced: how to train a model using data of my own? (only for Linux-based systems)](#advanced-how-to-train-a-model-using-data-of-my-own-only-for-linux-based-systems)

## About

This package is used for accurately segmenting WMH lesions using T2FLAIR images without labeled data.
Besides, you can treat this package as a set of APIs which ensembles lots of existing tools.
If you use our code in your work, please cite the following paper:
<...>

## Requirements

Before installation, you may need to update your Python version if it is lower than "3.7.1".
Also, this tool is based on Python 3, Python 2 is deprecated and should no longer be used anymore.

## Quick start: how to use our pretrained model (only for Linux-based systems)

The fastest way of applying our tool to your research is by directly using our pre-trained model.
To use our pre-trained model for inference, please carefully follow the steps listed below:

1.  create a new Python 3 virtual environment using "python3 -m venv /path/to/your/virtual/env/"
    for more detailed information about how to create a Python 3 virtual environment, please
    refer to the following link: "https://docs.python.org/3/library/venv.html". For simplicity,
    here I assume you want to create your environment under "/home/<username>/freeseg/venv_freeseg/".
    To achieve this you need to execute the following commands (replace "<username>" with your actual 
    user name):

    ```bash
    python3 -m pip install --user virtualenv
    python3 -m venv /home/<username>/freeseg/venv_freeseg/
    ```

2.  activate your newly created virtual environment:

    ```bash
    source /home/<username>/freeseg/venv_freeseg/bin/activate
    ```

    then update pip, setuptools, and wheel:

    ```bash
    pip install -U pip
    pip install -U setuptools
    pip install wheel
    ```

    NOTE: the virtual environment should ALWAYS be activated during the following steps. 

3.  download nnU-Net source code from "https://github.com/lchdl/nnUNet" (PLEASE download the forked 
    version above, DO NOT download from https://github.com/MIC-DKFZ/nnUNet as the forked version has 
    made some necessary changes). Unzip the code to "/home/<username>/freeseg/external/nnunet_custom/". 
    Make sure "setup.py" is in this directory, such as "/home/<username>/freeseg/external/nnunet_custom/setup.py"

4.  "cd" into the directory where setup.py is located, then execute the following command:

    ```bash
    pip install -e .
    ```

    This will install customized nnU-Net and all its dependencies into your environment.
    As nnU-Net uses PyTorch, this command will also download and install PyTorch. Make sure your 
    CUDA/cuDNN version is compatible with your GPU driver version.

5.  download main toolkit from "https://github.com/lchdl/annotation_free_wmh_seg", unzip the code under 
    "/home/<username>/freeseg/main/", make sure "setup.py" is located in 
    "/home/<username>/freeseg/main/setup.py", then

    ```bash
    cd /home/<username>/freeseg/main/
    pip install -e .
    ```

6.  download and unzip ROBEX from "https://www.nitrc.org/projects/robex", then add:

    ```bash
    export ROBEX_DIR="/path/to/your/ROBEX/dir/
    ```
    
    in your ~/.bashrc, make sure "runROBEX.sh" is in this directory, then 

    ```bash
    source ~/.bashrc
    ```

    to update the change.

7.  (optional) compile & install ANTs toolkit from "https://github.com/ANTsX/ANTs". This is mainly for intensity
    correction. You can also skip this step if you don't want to install it. However, the segmentation performance
    can be seriously affected if the image is corrupted by strong intensity bias due to magnetic field inhomogeneity.
    For optimal performance I strongly recommend you to install ANTs.

8.  (optional) verify your install:
    1) activate your virtual environment
    2) enter Python by typing and running:
    
    ```bash
    python
    ```

    3) then, enter & run the following script line by line:

    ```bash
    from freeseg.main.integrity_check import check_system_integrity
    check_system_integrity(verbose=True, ignore_ANTs=True, ignore_FreeSurfer=True, ignore_FSL=True)
    ```

    4) if something is missing, you will see some error messages popping up. Some tips about how to fix the
       errors are also given. You can follow the tips to fix those problems and repeat Step 8 to verify your
       install until no error occurs.

9.  after installation, run

    ```bash
    freeseg_WMH_predict -h
    ```

    if no error occurs, then the installation is complete!

10. download our pre-trained model from "https://github.com/lchdl/annotation_free_wmh_seg". Then use

    ```bash
    freeseg_WMH_install -m <tar_gz_file> -o <model_install_dir>
    ```
    
    to install model (as indicated by <tar_gz_file>) to a specific location (as indicated by <model_install_dir>).

11. using pre-trained model to segment WMH lesions from FLAIR images with the following command:

    ```bash
    freeseg_WMH_predict -i <input_images> -n <subject_names> -m <model_install_dir> -o <output_folder> -g <gpu_id>
    ```
    
    > if you don't have ANTs toolkit installed on your machine (see Step 7 for how to install ANTs), you 
    > need to add "--skip-preprocessing" to the end of the command, such as:

    > <pre>freeseg_WMH_predict -i <...> -n <...> -m <...> -o <...> -g <...> </b>--skip-preprocessing</b></pre>
    
    > the segmentation performance can be seriously affected if the image is corrupted by strong intensity 
    > bias due to magnetic field inhomogeneity

    NOTE: you can specify multiple input images. The following command gives a complete example of this:

    ```bash
    freeseg_WMH_predict \
    -i /path/to/FLAIR_1.nii.gz /path/to/FLAIR_2.nii.gz /path/to/FLAIR_3.nii.gz \
    -n subject_1               subject_2               subject_3 \
    -m /path/to/your/model/dir/ \
    -o /path/to/your/output/dir/ \
    -g 0
    ```

## Advanced: how to train a model using data of my own? (only for Linux-based systems)

1.  follow the Steps 1--9 in Quick start section. Note that if you want to train a custom model, you must 
    download and compile ANTs toolkit, Step 7 in Quick start section is no longer optional.

2.  download and install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). Note that you may also need to 
    install "csh" and "tcsh" shell by executing "sudo apt-get install csh tcsh" command.

3.  download and install [FSL]("https://fsl.fmrib.ox.ac.uk/fsl/fslwiki").

4.  (optional) verify your install:
    1) activate your virtual environment
    2) enter Python by typing and running:
    
    ```bash
    python
    ```

    3) then, enter & run the following script line by line:

    ```python
    from freeseg.main.integrity_check import check_system_integrity
    check_system_integrity()
    ```

    4) if something is missing, you will see some error messages popping up. Some tips about how to fix the
       errors are also given. You can follow the tips to fix those problems and repeat Step 4 to verify your
       install until no error occurs.

5.  here we provided two examples of using a public dataset ([OASIS-3](https://www.oasis-brains.org/)) 
    to train a model from scratch, see 
    
    ```
    experiments/010_OASIS3/run_Siemens_Biograph_mMR.py
    experiments/010_OASIS3/run_Siemens_TrioTim.py
    ```
