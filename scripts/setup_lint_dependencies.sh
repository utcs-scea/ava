#!/bin/bash

sudo apt install python3 python3-pip
python3 -m pip install --upgrade pip
python3 -m pip install setuptools
python3 -m pip install 'clang-format==9.0.0' 'black==21.5b0' 'pylint==2.8.2' 'shellcheck-py==0.7.2.1'