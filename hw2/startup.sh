#!/bin/bash

pac_list=("python3.10" "python3-pip" "python3.10-venv", "ffmpeg")
env_name="hw2_env"

for pac in ${pac_list[@]}
do
    if [ "$(dpkg -l | awk '/'$pac'/ {print }' | wc -l)" -eq 0 ]; then
        sudo apt install $pac
    fi
done

if [ ! -d $env_name ]
then
    python3 -m venv $env_name
    echo "created environment"
fi

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
