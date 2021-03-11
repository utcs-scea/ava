# AvA Docker build images
This directory provides Docker images for building various AvA specs.

## Building these images
Build the docker images with the Makefile. To build an individual image type `make {image_name}` where image_name comes from the Dockerfile.{image_name}. Some images have extra dependencies, such as the CUDA images, which rely on the `nvidia-docker` apt package.

## Extra dependencies
CUDA images require nvidia-docker to be installed. If building from an Ubuntu host, this can be installed with the following command:
```
sudo apt install nvidia-docker -y
```

## Using these images
The `run_docker.sh` script launches a docker image with the environment properly configured to build AvA. This script can be used to run a single command, or to get an interactive shell. The first argument to this script is the docker image that should be used, for example `ava-cuda-10.1`.

Single command: `./run_docker.sh ava-cuda-10.1 bash -c "complex command && other command && echo \"done\""`
Interactive shell: `./run_docker.sh ava-cuda-10.1`
