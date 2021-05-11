#!/bin/bash

if [[ $(lsb_release -rs) != "18.04" ]]; then
  echo "The support of $(lsb_release -ds) is untested. Continue (y/n)?"
  read -r yn_value
  if [[ ${yn_value} != "y" ]]; then
    echo "Dependency installation cancelled"
    exit 0
  fi
fi

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt update
sudo apt purge --auto-remove cmake
sudo apt install cmake cmake-curses-gui
sudo apt install git build-essential python3 python3-pip libglib2.0-dev clang-7 libclang-7-dev libboost-all-dev libconfig++-dev indent
python3 -m pip install pip
python3 -m pip install toposort astor 'numpy==1.15.0' blessings
