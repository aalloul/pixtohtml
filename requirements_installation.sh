#!/bin/bash

# Ubuntu 16.04 GPU-instance with Nvidia driver properly installed
# by Jihong
#
# ============================================================ #

echo "Checking for CUDA and installing."
if ! dpkg-query -W cuda; then
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda cuda-8–0
fi

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

echo "Install CUDNN"
cd $HOME
tar xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
rm -rf ~/cuda
rm cudnn-8.0-linux-x64-v5.1.tgz

echo "Checking for Docker and installing"
if ! dpkg-query -W docker; then
    # Add docker repository
    sudo apt-get update
    sudo apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
    sudo apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main'
    sudo apt-get update
    # Install
    sudo apt-get install -y docker-engine
    # add user to docker group
    sudo usermod -aG docker $USER
fi

echo "Checking for NVIDIA Docker and installing"
if ! dpkg-query -W nividia-docker; then
    # install nvidia-docker
    wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
fi

echo "Checking for VIM and installing"
if ! dpkg-query -W vim-gnome; then
    sudo apt-get install vim-gnome
    # Customized step: Use Ultimate vimrc
    git clone https://github.com/amix/vimrc.git ~/.vim_runtime
    sh ~/.vim_runtime/install_awesome_vimrc.sh
    # Customized step: Use YouCompleteMe (YCM)
    git clone https://github.com/Valloric/YouCompleteMe.git ~/.vim_runtime/sources_non_forked/YouCompleteMe
fi

echo "Checking for python-pip and installing"
if ! dpkg-query -W python-pip*; then
    sudo apt-get install python-pip
fi