# Use a lightweight Python 3.9 base image
FROM python:3.9-slim

# metadata
# LABEL maintainer="Jaya Chandra Terli <terlijayachandra@gmail.com>" \
#       version="1.0.1" \
#       description="Dockerfile for DeepWMH tool with PyTorch and ANTs"

# Combine system dependencies installation and cleanup
RUN apt-get update && apt-get install -y \
    wget unzip bc python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and install wheel in a single command
RUN python3 -m pip install -U pip setuptools wheel --no-cache-dir

# Set a working directory
# WORKDIR /

# Use COPY instead of wget if the ANTs zip file is local
# COPY ./ants-2.5.4-ubuntu-22.04-X64-gcc.zip /tmp/ants.zip
RUN wget https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip && \
    unzip ants-2.5.4-ubuntu-22.04-X64-gcc.zip -d /opt && \
    rm ants-2.5.4-ubuntu-22.04-X64-gcc.zip
ENV PATH="$PATH:/opt/ants-2.5.4/bin"
ENV ANTS_RANDOM_SEED=0

# Install PyTorch and related libraries with --no-cache-dir
RUN python3 -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

# Copy and run the install_services.sh script
COPY ./install_services.sh /install_services.sh
RUN /install_services.sh

# Set the ROBEX directory environment variable
ENV ROBEX_DIR="/opt/ROBEX"

# Copy and install the DeepWMH model
RUN mkdir model
COPY ./model_v1.0.tar.gz /model/model_v1.0.tar.gz
RUN DeepWMH_install -m /model/model_v1.0.tar.gz -o /model/ -f

# Set the default entrypoint to DeepWMH_predict
ENTRYPOINT [ "DeepWMH_predict" ]
