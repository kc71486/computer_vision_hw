#!/bin/bash

#prelimary:
sudo apt install python3 python3-pip python3.10-venv
sudo apt install ffmpeg

python3 -m venv hw2_env

source hw2_env/bin/activate
pip install matplotlib
pip install pyqt5
pip install opencv-contrib-python
pip install scikit-learn
pip install ffmpeg-python
pip install torch
pip install torchvision
pip install torchinfo
deactivate
