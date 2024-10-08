import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from PIL import Image
import re
import time
import gc
from random import shuffle
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from datasets.WHU_dataset import CDDataset_WHU
from models import unet
from models.losses import StyleLoss, content_loss
import train_cd

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
num_gpu = torch.cuda.device_count()
print('Number of GPUs Available:', num_gpu)


root_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/crop_img'
img_size = 32

train_dataset = CDDataset_WHU(root_dir=root_dir, split='train',
                                 img_size=img_size,is_train=True,to_tensor=True)

val_dataset = CDDataset_WHU(root_dir=root_dir, split='val',
                                 img_size=img_size,is_train=False,to_tensor=True)

datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders = {x: DataLoader(datasets[x], batch_size=8,     
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

print(len(dataloaders))

model = unet.UNet(in_channels=3, out_channels=3, init_features=32)
model = model.to(device)


learning_rate =0.1 * 1e-4 
weight_decay = 5e-4 
momentum = 0.999
num_epochs = 10 #200
patience = 100 

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
#optimizer = optim.AdamW(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), weight_decay=0.01)

def lambda_rule(epoch):
    lr_l = 1.0 - epoch / float(num_epochs + 1)
    return lr_l
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, verbose=True)

#### Differernt loss function
#criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
#criterion = StyleLoss
#criterion = nn.KLDivLoss(reduction="mean",log_target=True)  ###batchmean
#criterion = content_loss


tic = time.perf_counter()

best_model, val_acc, val_loss, train_acc, train_loss, epoch_list = train_cd.train_model(model, dataloaders, criterion, optimizer, scheduler, patience, device, num_epochs=num_epochs)
torch.save(best_model.state_dict(), './save_model/best_model_whu.pkl')

toc = time.perf_counter()
print(f"training time :{toc - tic:0.4f} seconds")

np.save('./mse_loss/epoch_list_whu.npy', epoch_list)
np.save('./mse_loss/train_loss_whu.npy', train_loss)
np.save('./mse_loss/train_acc_whu.npy', train_acc)
np.save('./mse_loss/val_loss_whu.npy', val_loss)
np.save('./mse_loss/val_acc_whu.npy', val_acc)

