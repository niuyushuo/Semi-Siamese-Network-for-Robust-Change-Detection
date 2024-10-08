# Semi-Siamese-Network-for-Robust-Change-Detection
Official Implementation of "Semi-Siamese Network for Robust Change Detection Across Different Domains with Applications to 3D Printing"

## Network
<img src=https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/model_architecture.png width="500" height="400"> 

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
1. Download dataset at https://uconn-my.sharepoint.com/:f:/g/personal/yushuo_niu_uconn_edu/ElnZ7CR5w4NPqH96S2NnfF0BRoSKagR1rySANhSbr1KuVg?e=JhTB2L.
2. Unzip crop_img.zip and whu_dataset.zip.
3. At Unet_train folder, run crop_img.py to get the crop images for WHU dataset. (Or just use the cropped images at crop_img folder)
4. In train main_cd.py, update the path where save cropped images.
5. Run main_cd.py to train Unet.
6. If need to check the trained Unet model's performance, update the path where save whu_dataset.
7. Run predict_whu.py.
### Train Semi-Siamese model
1. At Sia_train folder, update the path where save the whu_dataset in data_config.py.
2. At models folder, set Semi-Siam (with init), Siamese (with init), and Semi-Siam (without init) to train the models you want at train_sia.py.
3. At Sia_train folder, run main_train.py.
4. For the model with initialization, update the path to the trained Unet model in semi_with_weights.py or siamese_with_weights.py.
6. To test trained performance and make prediction plots, update path of trained model in evaluator_sia.py.
7. At Sia_train folder, run main_pred.py.

## Test the models trained in the paper
1. Download save_model.zip at https://uconn-my.sharepoint.com/:f:/g/personal/yushuo_niu_uconn_edu/ElnZ7CR5w4NPqH96S2NnfF0BRoSKagR1rySANhSbr1KuVg?e=JhTB2L.
2. Unzip save_model.zip.
3. At models folders, update the path to save_model folder in evaluator_sia.py
4. run main_pred.py.
