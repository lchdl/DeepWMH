This repository provides a Dockerized setup for running the **DeepWMH** tool available here at <https://github.com/lchdl/DeepWMH>.

---

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository_url>
cd docker-deepwmh
```

### 2. Build the Docker Image
```bash
docker build -t <image_name> .
```
or pull the prebuilt docker image (recommended to build the image localy)
```bash
docker pull deepwmh:v1.0.1
```

### 3. Running Segmentation Tool
You can run the tool using one of the following methods:
1. Run the provided script run_deepwmh.sh to initialize the container:
    ```bash
    ./run_deepwmh.sh
    ```
2. or just run the following command directly in the terminal:
    ```bash
    docker run --rm --gpus all \
        -v /path/to/data:/data \
        -v /path/to/output:/output \
        deepwmh:v1.0.1 \ # image name
        -i /data/<flair_image> \
        -n <subject_id> \
        -m /model \
        -o /output/<subject_id> \
        -g 0
    ```
Replace `/path/to/data`, `/path/to/output`, `<flair_image>`, `<subject_id>`, and `/model` with the appropriate paths and values.

---

### 4. Debugging (server)
For debugging purposes we need a container built with image `deepwmh:shell`

To create one, run the following command
```bash
docker run --gpus all --restart always -dit --name deepwmh deepwmh:shell
```
then the container can be accessed using
```bash
docker exec -it deepwmh /bin/bash
```

## Logs and Output
- Logs are saved in the `logs` directory.
- Output files are saved in the `output` directory.

---

## Notes
- Ensure that NVIDIA drivers and CUDA are installed for GPU support.
- The install_services.sh script sets up all required dependencies.
- The model file must be present in the source directory, as it is copied from the host to the container during execution.