# Semi-Siamese Network for Robust Change Detection

Official implementation of **"Semi-Siamese Network for Robust Change Detection Across Different Domains with Applications to 3D Printing"**

## Network

<img src="https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/model_architecture.png" width="500" height="400">

## Installation

Create a conda environment:

```bash
conda create -n python3.10_pytorch2.0 python=3.10
conda activate python3.10_pytorch2.0
```

Check your CUDA version:

```bash
nvidia-smi
```

<img src="https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/smi.png" width="500" height="400">

Select the appropriate PyTorch version based on your CUDA version:

<img src="https://github.com/niuyushuo/Semi-Siamese-Network-for-Robust-Change-Detection/blob/main/images/pytorch.png" width="400" height="200">

For example, install PyTorch with CUDA 12.1:

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

## Data and Pretrained Models

The datasets and pretrained models used in this project are archived on Zenodo:

- **Zenodo:** https://zenodo.org/records/22117510

The Zenodo record contains:

- `Whu_dataset.zip` — WHU dataset package used in this implementation
- `Vertical_line.zip` — Vertical Line Experimental Dataset used in the 3D-printing experiments
- `save_model.zip` — trained models reported in the paper

Google Drive download links are also provided below as backup download options.

## WHU Dataset

The WHU Building Change Detection Dataset used in this project is based on the publicly available **WHU Building Dataset** provided by the Group of Photogrammetry and Computer Vision (GPCV) at Wuhan University.

- **Original WHU Building Dataset:** https://gpcv.whu.edu.cn/data/building_dataset.html
- **Dataset used in this implementation (Zenodo):** https://zenodo.org/records/22117510
- **Backup download (Google Drive):** https://drive.google.com/file/d/1TBLCNBEPVUBkFLaJpt7GhkjIKBnVhZde/view?usp=drive_link

Please also cite the original WHU dataset publication when using this dataset:

> S. Ji, S. Wei, and M. Lu, "Fully Convolutional Networks for Multisource Building Extraction from an Open Aerial and Satellite Imagery Dataset," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 1, pp. 108–120, 2019.

### Train U-Net

1. Download `Whu_dataset.zip` from Zenodo:
   https://zenodo.org/records/22117510

   Alternatively, use the Google Drive backup link:
   https://drive.google.com/file/d/1TBLCNBEPVUBkFLaJpt7GhkjIKBnVhZde/view?usp=drive_link

2. Unzip `Whu_dataset.zip`.
3. In the `Unet_train` folder, update the dataset path in `main_cd.py`.
4. Run `main_cd.py` to train the U-Net model.
5. To evaluate the trained U-Net model, update the dataset path and the trained U-Net model path in `predict_whu.py`.
6. Run `predict_whu.py`.

### Train Semi-Siamese Model

1. In the `Sia_train` folder, update the path to the WHU dataset in `data_config.py`.
2. In the `models` folder, select **Semi-Siam (with init)**, **Siamese (with init)**, or **Semi-Siam (without init)** for training in `train_sia.py`.
3. For models with initialization, update the path to the trained U-Net model in `semi_with_weights.py` or `siamese_with_weights.py`.
4. In the `Sia_train` folder, run `main_train.py`.
5. To evaluate the trained model and generate prediction plots, update the path to the trained model in `evaluator_sia.py`.
6. In the `Sia_train` folder, run `main_pred.py`.

## Test the Models Reported in the Paper

The trained models reported in the paper are included in `save_model.zip`.

- **Primary download (Zenodo):** https://zenodo.org/records/22117510
- **Backup download (Google Drive):** https://drive.google.com/file/d/1DXIj8oQ8P4rQ0WYOb25d98Qs00JGIfAp/view?usp=drive_link

To test the trained models:

1. Download `save_model.zip`.
2. Unzip `save_model.zip`.
3. In the `models` folder, update the path to the `save_model` folder in `evaluator_sia.py`.
4. Run `main_pred.py`.

## Vertical Line Experimental Dataset

The **Vertical Line Experimental Dataset** introduced and used in our paper is available on Zenodo, with a Google Drive link provided as a backup download option.

- **Primary download (Zenodo):** https://zenodo.org/records/22117510
- **Backup download (Google Drive):** https://drive.google.com/file/d/1iJTqo5CJ_V6839YWDmoYygEac8OPmWFm/view?usp=drive_link

Download and unzip `Vertical_line.zip`.

As described in the paper, the dataset contains a total of **65 schematic images** arranged in sequential order.

- Images **1–41** were used for training.
- Images **42–49** were used for validation.
- Images **50–57** were used for testing.
- Images **58–65** were not included in the quantitative experiments, but model predictions and qualitative visualizations were also generated for these images.

The training, validation, and test splits follow the sequential order of the schematic images, consistent with the experimental setup described in the paper.

The same training and evaluation programs used for the WHU dataset can also be applied to the Vertical Line Experimental Dataset. Follow the instructions in the **WHU Dataset** section above and replace the WHU dataset path with the path to the Vertical Line Experimental Dataset in the corresponding configuration and training files.

## Citation

If you find this repository useful in your research or projects, please consider citing our paper:

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

## Contact

For questions regarding the code, datasets, or this work, please contact:

- **Yushuo Niu:** niuyy9026@gmail.com
- **Qian Yang:** qyang@uconn.edu

You are also welcome to open an issue in this repository.
