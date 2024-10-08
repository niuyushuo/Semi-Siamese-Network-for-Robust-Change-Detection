# Semi-Siamese-Network-for-Robust-Change-Detection
Official Implementation of "Semi-Siamese Network for Robust Change Detection Across Different Domains with Applications to 3D Printing"

## Installition
Create a conda environment:
```
conda create -n python3.10_pytorch2.0 python=3.10
conda activate python3.10_pytorch2.0
```

Install pytorch based on your cuda version:
```
nvidia-smi
```

<img src='https://github.com/niuyushuo/Change-Detection-for-filamentous-Carbon/blob/main/images/smi.png' width="500" height="400">

<img src='https://github.com/niuyushuo/Change-Detection-for-filamentous-Carbon/blob/main/images/pytorch.png' width="400" height="200">

Install pytorch:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install the rest packages:
```
conda install matplotlib
conda install esri::einops
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install anaconda::seaborn
conda install anaconda::openpyxl

## For WHU dataset
### Train Unet
