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
```

## For WHU dataset
### Train Unet
1. Download dataset at https://uconn-my.sharepoint.com/:f:/g/personal/yushuo_niu_uconn_edu/ElnZ7CR5w4NPqH96S2NnfF0BRoSKagR1rySANhSbr1KuVg?e=JhTB2L
2. At Unet_train folder, run crop_img.py to get the crop images for WHU dataset
3. Run_cd.py to train Unet.
### Train Semi-Siamese model
