# === NOTE ===
<i><b>The relevant paper is currently under review. The complete code and pretrained model are now uploaded for the review process.</b></i>

# Table of contents

- [About](#about)
- [Requirements](#requirements)
- [Quick start: how to use our pretrained model (only for Linux-based systems)](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems)
- [Advanced: how to train a model using data of my own? (only for Linux-based systems)](#advanced-how-to-train-a-model-using-data-of-my-own-only-for-linux-based-systems)

## About

Annotation free white matter hyperintensities (WMH) segmentation using deep learning.
This package is used for accurately segmenting WMH lesions using T2FLAIR images without labeled data.

## Requirements

Before installation, you may need to update your Python version if it is lower than "3.7.1".
Also, this tool is based on Python 3, Python 2 is deprecated and should no longer be used anymore.

## Quick start: how to use our pretrained model (only for Linux-based systems)

The fastest way of applying our tool to your research is by directly using our pre-trained model.
To use our pre-trained model for inference, please carefully follow the steps listed below:

1.  create a new Python 3 virtual environment using 
    ```bash
    python3 -m venv /path/to/your/virtual/env/
    ```
    for more detailed information about how to create a Python 3 virtual environment, please
    refer to this [link](https://docs.python.org/3/library/venv.html). For simplicity,
    here I assume you want to create your environment under "/home/\<username\>/deepwmh/venv_deepwmh/".
    To achieve this you need to execute the following commands (replace "\<username\>" with your actual 
    user name):

    ```bash
    python3 -m pip install --user virtualenv
    python3 -m venv /home/<username>/deepwmh/venv_deepwmh/
    ```

2.  activate your newly created virtual environment:

    ```bash
    source /home/<username>/deepwmh/venv_deepwmh/bin/activate
    ```

    then update pip, setuptools, and wheel:

    ```bash
    pip install -U pip
    pip install -U setuptools
    pip install wheel
    ```

    NOTE: the virtual environment should <b><i>always</i></b> be activated during the following steps. 

3.  download nnU-Net source code from "https://github.com/lchdl/nnUNet" (PLEASE download the forked 
    version above, DO NOT download from https://github.com/MIC-DKFZ/nnUNet as the forked version has 
    made some necessary changes). Unzip the code to "/home/\<username\>/deepwmh/external/nnunet_custom/". 
    Make sure "setup.py" is in this directory, such as "/home/\<username\>/deepwmh/external/nnunet_custom/setup.py"

4.  "cd" into the directory where setup.py is located, then execute the following command:

    ```bash
    pip install -e .
    ```

    This will install customized nnU-Net and all its dependencies into your environment.
    As nnU-Net uses PyTorch, this command will also download and install PyTorch. Make sure your 
    CUDA/cuDNN version is compatible with your GPU driver version.

5.  download main toolkit from "https://github.com/lchdl/DeepWMH", unzip the code under 
    "/home/\<username\>/deepwmh/main/", make sure "setup.py" is located in 
    "/home/\<username\>/deepwmh/main/setup.py", then

    ```bash
    cd /home/<username>/deepwmh/main/
    pip install -e .
    ```

6.  download and unzip ROBEX from "https://www.nitrc.org/projects/robex", then add:

    ```bash
    export ROBEX_DIR="/path/to/your/unzipped/ROBEX/dir/"
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

    > **Verify your install**: to see whether ANTs is installed correctly on your system, after the installation you need to type in
    > ```
    > antsRegistration --version
    > ```
    > and
    > ```
    > N4BiasFieldCorrection --version
    > ```
    > in your console. It should produce output such as:
    > ```
    > ANTs Version: 3.0.0.0.dev13-ga16cc
    > Compiled: Jan 22 2019 00:23:29
    > ```
    > Then test if `antsApplyTransforms` can work:
    > ```
    > antsApplyTransforms
    > ```
    > if no error shows, then ANTs is successfully installed.

8.  (optional) verify your install:
    1) activate your virtual environment
    2) enter Python by typing and running:
    
    ```bash
    python
    ```

    3) then, enter & run the following script line by line:

    ```bash
    from deepwmh.main.integrity_check import check_system_integrity
    check_system_integrity(verbose=True, ignore_ANTs=True, ignore_FreeSurfer=True, ignore_FSL=True)
    ```

    4) if something is missing, you will see some error messages popping up. Some tips about how to fix the
       errors are also given. You can follow the tips to fix those problems and repeat Step 8 to verify your
       install until no error occurs.

9.  after installation, run

    ```bash
    DeepWMH_predict -h
    ```

    if no error occurs, then the installation is complete! Now you are ready to use our pretrained model for
    segmentation.

10. download our pre-trained model (~200 MB) from 

    1) "https://drive.google.com/drive/folders/1CDJkY5F95sW638UGjohWDqXvPtBTI1w3?usp=share_link" or
    2) "https://pan.baidu.com/s/1j7aESa4NEcu95gsHLR9BqQ?pwd=yr3o"
    
    Then use

    ```bash
    DeepWMH_install -m <tar_gz_file> -o <model_install_dir>
    ```
    
    to install model (as indicated by <tar_gz_file>) to a specific location (as indicated by <model_install_dir>).

11. using pre-trained model to segment WMH lesions from FLAIR images with the following command:

    ```bash
    DeepWMH_predict -i <input_images> -n <subject_names> -m <model_install_dir> -o <output_folder> -g <gpu_id>
    ```
    
    > if you don't have ANTs toolkit installed on your machine (see Step 7 for how to install ANTs), you 
    > need to add "--skip-preprocessing" to the end of the command, such as:
    > <pre>DeepWMH_predict -i <...> -n <...> -m <...> -o <...> -g <...> <b>--skip-preprocessing</b></pre>
    > the segmentation performance can be seriously affected if the image is corrupted by strong intensity 
    > bias due to magnetic field inhomogeneity

    NOTE: you can specify multiple input images. The following command gives a complete example of this:

    ```bash
    DeepWMH_predict \
    -i /path/to/FLAIR_1.nii.gz /path/to/FLAIR_2.nii.gz /path/to/FLAIR_3.nii.gz \
    -n subject_1               subject_2               subject_3 \
    -m /path/to/your/model/dir/ \
    -o /path/to/your/output/dir/ \
    -g 0
    ```

## Advanced: how to train a model using data of my own? (only for Linux-based systems)

1.  follow the Steps 1--9 in [Quick start](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems) section. Note that if you want to train a custom model, you <b><i>must</i></b> 
    download and compile ANTs toolkit, Step 7 in [Quick start](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems) section is no longer optional.

2.  download and install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). Note that you may also need to 
    install "csh" and "tcsh" shell by running 
    
    ```sudo apt-get install csh tcsh```
    
    after the installation.
    
    > A license key ([link](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)) is also needed before using FreeSurfer.

3.  download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (FMRIB Software Library).

    > **How to install**: FSL is installed using the *fsl_installer.py* downloaded from [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation). You need to register your personal information to the FSL site before download. After download you need to run the installer script and wait for the installation to finish.
    >
    > **Verify your install**: when the installation finished, type in
    > ```
    > bet -h
    > ```
    > in your console. If no error occurs then everything is OK! :)

4.  (optional) verify your install:
    1) activate your virtual environment
    2) enter Python by typing and running:
    
    ```bash
    python
    ```

    3) then, enter & run the following script line by line:

    ```python
    from deepwmh.main.integrity_check import check_system_integrity
    check_system_integrity()
    ```

    4) if something is missing, you will see some error messages popping up. Some tips about how to fix the
       errors are also given. You can follow the tips to fix those problems and repeat Step 4 to verify your
       install until no error occurs (as shown below).
        
       ![1](https://user-images.githubusercontent.com/18594210/196351063-732bfe8e-14d6-4fbc-8311-1e5ebbeeca04.png)


5.  here we provided two examples of using a public dataset ([OASIS-3](https://www.oasis-brains.org/)) 
    to train a model from scratch, see 
    
    ```
    experiments/010_OASIS3/run_Siemens_Biograph_mMR.py
    experiments/010_OASIS3/run_Siemens_TrioTim.py
    ```
    
    you can also run these two examples if you organized the dataset structure correctly:
    
    ```bash
    cd .../experiments/010_OASIS3/
    python run_Siemens_Biograph_mMR.py
    ```
    ```bash
    cd .../experiments/010_OASIS3/
    python run_Siemens_TrioTim.py
    ```
    
    
    
