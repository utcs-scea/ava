#!/bin/bash

# TODO: input versions via arguments
VERSION_FULL="7.6.5.32"
VERSION="${VERSION_FULL%.*}"
CUDA_VERSION="10.1"
OS_ARCH="linux-x64"

# Get the PPA repository driver
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
# sudo apt-get update

 # Install CUDA driver
# sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-drivers

# Download cuDNN v7.6
CUDNN_TAR_FILE="cudnn-${CUDA_VERSION}-${OS_ARCH}-v${VERSION_FULL}.tgz"
CUDNN_URL="https://developer.download.nvidia.com/compute/redist/cudnn/v${VERSION}/${CUDNN_TAR_FILE}"
wget -c ${CUDNN_URL} -P ~

pushd "${PWD}" || exit

tar -xzvf ${CUDNN_TAR_FILE}

# Copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.1/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/
sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*

popd || exit