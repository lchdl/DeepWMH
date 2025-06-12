#!/bin/bash

# Define variables for subject ID, data directory, output directory, and input image
export subj_id="0xxxxxxx"
export data_dir="/path/to/your/datafolder"
export working_dir="/path/to/store/your/ouptuts"
export flair_img="flair_0xxxxxxx.nii"

cd $working_dir
# Ensure the logs and output directory exists
mkdir -p $working_dir/logs
mkdir -p $working_dir/output

# Run the Docker container with the DeepWMH_predict tool
docker run --rm --gpus all \
    -v $data_dir:/data \
    -v $working_dir/output:/output \
    #change to apropriate image
    deepwmh:v.1.0.1 \
    -i /data/$flair_img \
    -n $subj_id \
    -m /model \
    -o /output/$subj_id \
    -g 0 > $working_dir/logs/${subj_id}.log 2>&1 &