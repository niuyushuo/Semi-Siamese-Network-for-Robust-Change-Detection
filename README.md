# Semi-Siamese Network for Robust Change Detection
Official Implementation of **"Semi-Siamese Network for Robust Change Detection Across Different Domains with Applications to 3D Printing"**

## Network
<img src='https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/model_architecture.png' width="500" height="400">

## Installition

Create a conda environment:

```bash
conda create -n python3.10_pytorch2.0 python=3.10
conda activate python3.10_pytorch2.0
```

Install PyTorch based on your CUDA version:

```bash
nvidia-smi
```

<img src='https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/smi.png' width="500" height="400">

<img src='https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/pytorch.png' width="400" height="200">

Install PyTorch:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the remaining packages:

```bash
conda install matplotlib
conda install esri::einops
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install anaconda::seaborn
conda install anaconda::openpyxl
```

## For WHU dataset

### Train Unet

1. Download `Whu_dataset.zip` from:
   https://drive.google.com/file/d/1TBLCNBEPVUBkFLaJpt7GhkjIKBnVhZde/view?usp=drive_link
2. Unzip `Whu_dataset.zip`.
3. In the `Unet_train` folder, update the dataset path in `main_cd.py`.
4. Run `main_cd.py` to train the Unet model.
5. To check the trained Unet model's performance, update the dataset path and the trained Unet model path in `predict_whu.py`.
6. Run `predict_whu.py`.

### Train Semi-Siamese model

1. At `Sia_train` folder, update the path where save the `whu_dataset` in `data_config.py`.
2. At `models` folder, set **Semi-Siam (with init)**, **Siamese (with init)**, or **Semi-Siam (without init)** to train the models in `train_sia.py`.
3. For the model with initialization, update the path to the trained Unet model in `semi_with_weights.py` or `siamese_with_weights.py`.
4. At `Sia_train` folder, run `main_train.py`.
5. To test trained performance and make prediction plots, update the path of the trained model in `evaluator_sia.py`.
6. At `Sia_train` folder, run `main_pred.py`.

## Test the models showed in the paper

1. Download `save_model.zip` from:
   https://drive.google.com/file/d/1DXIj8oQ8P4rQ0WYOb25d98Qs00JGIfAp/view?usp=drive_link
2. Unzip `save_model.zip`.
3. At `models` folder, update the path to the `save_model` folder in `evaluator_sia.py`.
4. Run `main_pred.py`.

## Vertical Line Experimental Dataset

Download the Vertical Line Experimental Dataset from:

https://drive.google.com/file/d/1iJTqo5CJ_V6839YWDmoYygEac8OPmWFm/view?usp=drive_link

As described in the paper, the dataset contains a total of **65 schematic images** arranged in sequential order.

- Images **1–41** were used for training.
- Images **42–49** were used for validation.
- Images **50–57** were used for testing.
- Images **58–65** were not included in the quantitative experiments, but model predictions and qualitative visualizations were also generated for these images.

The train, validation, and test splits follow the sequential order of the schematic images, consistent with the experimental setup described in the paper.

The same training and evaluation programs used for the WHU dataset can also be applied to the Vertical Line Experimental Dataset. Simply follow the instructions in the **For WHU dataset** section above and replace the WHU dataset path with the path to the Vertical Line Experimental Dataset in the corresponding configuration and training files.

## Citation

If you find this repository useful in your research or projects, please consider citing our paper. We appreciate your support!

```bibtex
@inproceedings{niu2023semisiamese,
  title={Semi-Siamese Network for Robust Change Detection Across Different Domains with Applications to 3D Printing},
  author={Niu, Yushuo and Chadwick, Edward and Ma, Anson W. and Yang, Qian},
  booktitle={International Conference on Computer Vision Systems (ICVS)},
  pages={183--196},
  year={2023},
  publisher={Springer}
}
```

If you have any questions or suggestions, feel free to open an issue or contact me.
