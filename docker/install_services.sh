#!/bin/bash

# Navigate to the /opt directory
cd opt/

# Download and install nnUNet_for_DeepWMH
wget -O nnUNet_for_DeepWMH.zip https://github.com/lchdl/nnUNet_for_DeepWMH/archive/refs/heads/develop.zip
unzip nnUNet_for_DeepWMH.zip && rm nnUNet_for_DeepWMH.zip
cd nnUNet_for_DeepWMH-develop/
python3 -m pip install -e .

# Download and install DeepWMH
cd ..
wget -O DeepWMH.zip https://github.com/lchdl/DeepWMH/archive/refs/heads/develop.zip
unzip DeepWMH.zip && rm DeepWMH.zip
cd DeepWMH-develop/
python3 -m pip install -e .

# Download and set up ROBEX
cd ..
wget -O ROBEXv12.linux64.tar.gz "https://www.nitrc.org/frs/download.php/5994/ROBEXv12.linux64.tar.gz?i_agree=1&download_now=1"
tar -xzf ROBEXv12.linux64.tar.gz && rm ROBEXv12.linux64.tar.gz

# Return to the root directory
cd /