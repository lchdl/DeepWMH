### Dockerized setup for running the **DeepWMH** tool

---
1. pull the root repositrory locally and cd to the docker dir
```bash
cd docker
```

2. build the deepwmh image

Configure the dockerfile (optional) and run the command below
```bash
docker build -t <image_name> .
```

### 3. Running Segmentation Tool
run the tool using one of the following methods:
1. Custom script run_deepwmh.sh that process through images and generate logs:
    ```bash
    ./run_deepwmh.sh
    ```
- Logs are saved in the `logs` directory.
- Output files are saved in the `output` directory.
2. or just run the following command directly in the terminal:
    ```bash
    docker run --rm --gpus all \
        -v /path/to/data:/data \
        -v /path/to/output:/output \
        # image name
        deepwmh:v1.0.1 \
        -i /data/<flair_image> \
        -n <subject_id> \
        -m /model \
        -o /output/<subject_id> \
        -g 0 &
    ```
Replace `/path/to/data`, `/path/to/output`, `<flair_image>`, `<subject_id>`, and `/model` with the appropriate paths and values.

---

### 4. Debugging
For debugging purposes, built a new image with entrypoint set to '/bin/bash' and execute the following

start a container from commandline,
```bash
docker run --gpus all --restart always -dit --name deepwmh deepwmh:shell
```
then the container can be accessed using
```bash
docker exec -it deepwmh /bin/bash
```
---

## Notes
- Make sure NVIDIA drivers and CUDA are installed locally.
- The install_services.sh script is used to setup and download services.
- The model zip file must be placed in the doker dir, it is currently being copied from the host to the container during build process.