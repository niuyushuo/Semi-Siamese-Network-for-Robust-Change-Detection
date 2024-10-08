import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffectis

from PIL import Image
import re
import time
import gc
from random import shuffle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from datasets.WHU_dataset import CDDataset_WHU
from models import unet
import train_cd
import utils

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('Device:', device)
num_gpu = torch.cuda.device_count()
print('Number of GPUs Available:', num_gpu)

root_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/whu_dataset'  #### Original size image (trained by 32*32)
img_size = 256

vis_dir ='./whu_results'

test_dataset = CDDataset_WHU(root_dir=root_dir, split='test',
                                 img_size=img_size,is_train=False,to_tensor=True)

dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

model = unet.UNet(in_channels=3, out_channels=3, init_features=32)
model = model.to(device)

checkpoint = torch.load('/home/yun13001/code/Change_detection/pytorch_semi_unet/crop_img/save_model/best_model_whu.pth')
model.load_state_dict(checkpoint['model'])


model.eval()

with torch.no_grad():
    batch_id=0
    for sample in dataloader:

        batch_id=batch_id+1

        test_img = sample['A'].to(device)
        labels = (sample['B']).to(device)

        outputs = model(test_img)

        m = nn.Sigmoid()
        outputs = m(outputs)

        pred = utils.make_numpy_grid(outputs)
        rgb = utils.make_numpy_grid(test_img)
        label = utils.make_numpy_grid(labels)
        label_thres = utils.make_numpy_grid(torch.round(labels))

        vis = np.concatenate([pred, rgb, label,label_thres], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(vis_dir, 'eval_' + str(batch_id)+'.jpg')
        plt.imsave(file_name, vis)
