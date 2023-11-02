# Paper

Chenghao Liu, Zhizheng Zhuo, Liying Qu, Ying Jin, Tiantian Hua, Jun Xu, Guirong Tan, Yuna Li, Yunyun Duan, Tingting Wang,
Zaiqiang zhang, Yanling zhang, Rui Chen, Pinnan Yu, Peixin Zhang, Yulu Shi, Jianguo Zhang, Decai Tian, Runzhi Li, Xinghu Zhang,
Fudong Shi, Yanli Wang, Jiwei Jiang, Aaron Carass, Yaou Liu, Chuyang Ye. "<i><b>DeepWMH: a deep learning tool for accurate white matter
hyperintensity segmentation without requiring manual annotations for training</i></b>". Science Bulletin, 2023.

<i><b>The paper is currently under review. The complete code and pretrained model are now uploaded for the review process.</b></i>

<p align="center">
  <img src="https://github.com/lchdl/DeepWMH/blob/develop/images/segmentation.png" width="350" />
  <img src="https://github.com/lchdl/DeepWMH/blob/develop/images/segmentation2.png" width="460" /> 
</p>


# Table of contents

- [About DeepWMH](#about-deepwmh)
- [Python Requirement](#python-requirement)
- [Quick start: how to use our pretrained model (only for Linux-based systems)](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems)
- [Advanced: how to train a model using data of my own? (only for Linux-based systems)](#advanced-how-to-train-a-model-using-data-of-my-own-only-for-linux-based-systems)

## About DeepWMH

+ <i><b>If you have any questions or find any bugs in the code, please open an issue or create a pull request!</b></i>

DeepWMH is an annotation-free white matter hyperintensities (WMH) segmentation tool based on deep learning,
designed for accurately segmenting WMH lesions using T2FLAIR images without labeled data. An overview of the 
whole processing pipeline is shown below.

![DeepWMH lesion segmentation pipeline overview.](https://github.com/lchdl/DeepWMH/blob/develop/images/pipeline.png)

The figure below shows a more detailed version of the processing pipeline. Please refer to the supplementary 
materials for more information.

![Method details.](https://github.com/lchdl/DeepWMH/blob/develop/images/method.png)

## Python Requirement

Before installation, you may need to update your Python version if it is lower than "<b>3.7.1</b>".
Also, this tool is based on Python 3, Python 2 is deprecated and should no longer be used anymore.

## Quick start: how to use our pretrained model (only for Linux-based systems)

The fastest way of applying our tool to your research is by <i><b>using our pre-trained model</i></b> directly.
To use our pre-trained model for inference, please carefully follow the steps below:

1.  Update your Python environment. Then, create a new virtual environment using the following commands:
    ```bash
    pip install -U pip                         # update pip
    pip install -U setuptools                  # update setuptools
    pip install wheel                          # install wheel
    python -m pip install --user virtualenv    # install virtualenv
    python -m venv <path_to_your_virtual_env>  # create a virtual environment under <path_to_your_virtual_env>
    ```
    for more detailed information about how to create a Python 3 virtual environment, please
    refer to this [link](https://docs.python.org/3/library/venv.html). 

2.  Activate the virtual environment you just created:

    ```bash
    source <path_to_your_virtual_env>/bin/activate
    ```
    
    NOTE: the virtual environment should <b><i>ALWAYS</i></b> be activated during the following steps. 

3.  Install PyTorch in your virtual environment. See https://pytorch.org/ for more info.

4.  Download nnU-Net source code from "https://github.com/lchdl/nnUNet_for_DeepWMH".
    > PLEASE download the forked version above, DO NOT download from
    > https://github.com/MIC-DKFZ/nnUNet as the forked version has 
    > made some necessary changes.

    Unzip the code, "cd" into the directory where setup.py is located, then execute the following command:

    ```bash
    pip install -e .
    ```

    This will install customized nnU-Net and all its dependencies into your environment.
    > Please install PyTorch <i><b>BEFORE</i></b> nnU-Net as suggested in https://github.com/MIC-DKFZ/nnUNet,
    > and make sure your CUDA/cuDNN version is compatible with your GPU driver version.

5.  Download DeepWMH from "https://github.com/lchdl/DeepWMH".
    Unzip the code, "cd" into the directory where setup.py is located, then execute the following command:
    ```bash
    pip install -e .
    ```

6.  Download and unzip ROBEX ([Robust Brain Extraction](https://ieeexplore.ieee.org/abstract/document/5742706)) from "https://www.nitrc.org/projects/robex", then add:

    ```bash
    export ROBEX_DIR="/path/to/your/unzipped/ROBEX/dir/"
    ```
    
    in your ~/.bashrc, make sure "runROBEX.sh" is in this directory, then 

    ```bash
    source ~/.bashrc
    ```

    to update the change.

7.  <b>(Optional)</b> compile & install ANTs toolkit from "https://github.com/ANTsX/ANTs", or download the
    pre-compiled binaries [here](https://github.com/ANTsX/ANTs/releases/). This is mainly for intensity correction.
    You can also skip this step if you don't want to install it. However, the segmentation performance can be seriously
    affected if the image is corrupted by strong intensity bias due to magnetic field inhomogeneity.
    For optimal performance I strongly recommend you to install ANTs.

    > **Verify your install**: to see whether ANTs is installed correctly in your system, after the installation you need to type in
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

8.  <b>(Optional)</b> verify your install:
    1) activate your virtual environment
    2) enter Python by typing and running:
    
    ```bash
    python
    ```

    3) then, enter & run the following script line by line:

    ```python
    from deepwmh.main.integrity_check import check_system_integrity
    check_system_integrity(verbose=True, ignore_ANTs=True, ignore_FreeSurfer=True, ignore_FSL=True)
    ```

    4) if something is missing, you will see some error messages popping up. Some tips about how to fix the
       errors are also given. You can follow the tips to fix those problems and repeat Step 8 to verify your
       install until no error occurs.

9.  After installation, run
    
    ```bash
    DeepWMH_predict -h
    ```

    if no error occurs, then the installation is complete! Now you are ready to use our pretrained model for
    segmentation.

10. Download our pre-trained model (~200 MB) from 

    1) "https://drive.google.com/drive/folders/1CDJkY5F95sW638UGjohWDqXvPtBTI1w3?usp=share_link" or
    2) "https://pan.baidu.com/s/1j7aESa4NEcu95gsHLR9BqQ?pwd=yr3o"
    
    Then use

    ```bash
    DeepWMH_install -m <tar_gz_file> -o <model_install_dir>
    ```
    
    to install model (as indicated by <tar_gz_file>) to a specific location (as indicated by <model_install_dir>).

11. Using pre-trained model to segment WMH lesions from FLAIR images with the following command:

    ```bash
    DeepWMH_predict -i <input_images> -n <subject_names> -m <model_install_dir> -o <output_folder> -g <gpu_id>
    ```
    
    > if you don't have ANTs toolkit installed in your machine (see Step 7 for how to install ANTs), you 
    > need to add "--skip-bfc" to the end of the command, such as:
    > <pre>DeepWMH_predict -i <...> -n <...> -m <...> -o <...> -g <...> <b>--skip-bfc</b></pre>
    > the segmentation performance can be seriously affected if the image is corrupted by strong intensity 
    > bias due to magnetic field inhomogeneity

    NOTE: you can specify multiple input images. The following command gives a complete example of this:

    ```bash
    DeepWMH_predict \
    -i /path/to/FLAIR_1.nii.gz /path/to/FLAIR_2.nii.gz /path/to/FLAIR_3.nii.gz \ # specify three FLAIR images
    -n subject_1               subject_2               subject_3 \               # then give three subject names
    -m /path/to/your/model/dir/ \                                                # model installation directory
    -o /path/to/your/output/dir/ \                                               # output directory
    -g 0                                                                         # gpu index
    ```

## Advanced: how to train a model using data of my own? (only for Linux-based systems)

1.  Follow the Steps 1--7 in [Quick start](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems) section. Note that if you want to train a custom model, you <b><i>must</i></b> 
    download and compile ANTs toolkit, Step 7 in [Quick start](#quick-start-how-to-use-our-pretrained-model-only-for-linux-based-systems) section is no longer optional.

2.  Download and install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/). Note that you may also need to 
    install "csh" and "tcsh" shell by running 
    
    ```sudo apt-get install csh tcsh```
    
    after the installation.
    
    > A license key ([link](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall)) is also needed before using FreeSurfer.

3.  Download and install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki) (FMRIB Software Library).

    > **How to install**: FSL is installed using the *fsl_installer.py* downloaded from [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation). You need to register your personal information to the FSL site before download. After download you need to run the installer script and wait for the installation to finish.
    >
    > **Verify your install**: when the installation finished, type in
    > ```
    > bet -h
    > ```
    > in your console. If no error occurs then everything is OK! :)

4.  <b>(Optional)</b> verify your install:
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
        
       ![integrity check](https://github.com/lchdl/DeepWMH/blob/develop/images/integrity_check.png)


5.  Here we provided two examples of using a public dataset ([OASIS-3](https://www.oasis-brains.org/)) 
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
    
    
    
