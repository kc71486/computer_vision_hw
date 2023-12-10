#!/bin/bash

#prelimary:
sudo apt install python3 python3-pip python3.10-venv

python3 -m venv hw1_env

source hw1_env/bin/activate
pip install matplotlib
pip install pyqt5
pip install opencv-contrib-python
pip install torch
pip install torchvision
pip install torchinfo
deactivate
