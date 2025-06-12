#### Dockerized Setup for DeepWMH Segmentation Tool

This guide provides instructions to build and run the DeepWMH brain segmentation tool inside a Docker container.

---

#### 1. Clone Repository and Navigate to Docker Directory

Clone the main repository locally and move into the Docker setup folder:

```bash
cd docker
```

---

#### 2. Build the Docker Image

Optionally configure the `Dockerfile`, then build the Docker image using:

```bash
docker build -t <image_name> .
```

Replace `<image_name>` with your preferred image tag (e.g., `deepwmh:predict`).

---

#### 3. Run the Segmentation Tool

There are two ways to run the segmentation pipeline:

##### Option A: Using the `run_deepwmh.sh` Script

This script processes images and generates logs automatically.

```bash
./run_deepwmh.sh
```

- Logs will be saved to the `logs/` directory.
- Output will be stored in the `output/` directory.

##### Option B: Run Docker Manually

You can also run the tool directly from the command line:

```bash
docker run --rm --gpus all \
    -v /path/to/data:/data \
    -v /path/to/output:/output \
    deepwmh:predict \
    -i /data/<flair_image> \
    -n <subject_id> \
    -m /model \
    -o /output/<subject_id> \
    -g 0 &
```

Make sure to replace:
- `/path/to/data` with the path to your FLAIR input images.
- `/path/to/output` with the desired output directory.
- `<flair_image>`, `<subject_id>`, and `/model` with your specific filenames/paths.

---

#### 4. Debugging Mode

If you need to troubleshoot or access the container interactively:

##### Build a Debug Image with Bash Entrypoint

```bash
docker build -t deepwmh:shell --build-arg ENTRYPOINT="/bin/bash" .
```

##### Start the Container

```bash
docker run --gpus all --restart always -dit --name deepwmh deepwmh:shell
```

##### Access the Container Shell

```bash
docker exec -it deepwmh /bin/bash
```

---

#### Notes

- Make sure NVIDIA drivers and CUDA are correctly installed on your host machine.
- Use `install_services.sh` to set up and download required services.
- The model `.zip` file should be placed in the `docker/` directory before building the image. It will be copied into the container during the build process.

