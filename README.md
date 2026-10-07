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

- `Whu_dataset.zip` — the publicly available WHU dataset used in this implementation
- `Vertical_line.zip` — the original Vertical Line Experimental Dataset created by the authors of this work for the 3D-printing experiments
- `save_model.zip` — trained models reported in the paper

Please note that the **WHU dataset is an existing publicly available dataset provided by Wuhan University**, whereas the **Vertical Line Experimental Dataset is original to the authors of this work** and was created for the 3D-printing experiments presented in our paper.

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
3. In this GitHub repository, navigate to the `whu_dataset/Unet_train` folder.
4. Open `main_cd.py` in `whu_dataset/Unet_train` and update the dataset path to point to the downloaded WHU dataset.
5. Run `main_cd.py` to train the U-Net model.
6. To evaluate the trained U-Net model, update the dataset path and the trained U-Net model path in `predict_whu.py`.
7. Run `predict_whu.py`.

### Train Semi-Siamese Model

1. In this GitHub repository, navigate to the `whu_dataset/Sia_train` folder.
2. Update the path to the downloaded WHU dataset in `data_config.py`.
3. In the `models` folder, select **Semi-Siam (with init)**, **Siamese (with init)**, or **Semi-Siam (without init)** for training in `train_sia.py`.
4. For models with initialization, update the path to the trained U-Net model in `semi_with_weights.py` or `siamese_with_weights.py`.
5. Run `main_train.py` from the `whu_dataset/Sia_train` folder.
6. To evaluate the trained model and generate prediction plots, update the path to the trained model in `evaluator_sia.py`.
7. Run `main_pred.py` from the `whu_dataset/Sia_train` folder.

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

The **Vertical Line Experimental Dataset is an original dataset created by the authors of this work** for the 3D-printing experiments presented in our paper. In contrast to the publicly available WHU dataset used for cross-domain evaluation, the Vertical Line Experimental Dataset originates from our own experimental study.

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
