import numpy as np # linear algebra

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from models import unet

class Semi_siamese(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(Semi_siamese, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(self.device)

        self.unet1 = unet.UNet(in_channels=3, out_channels=3, init_features=32)
        self.unet1.to(self.device)

        self.unet2 = unet.UNet(in_channels=3, out_channels=3, init_features=32)
        self.unet2.to(self.device)


        self.checkpoint = torch.load('/home/yun13001/code/change_detection/whu_dataset/Unet_train/save_model/best_model_whu.pth', map_location=torch.device(self.device),weights_only=True)

        self.unet1.load_state_dict(self.checkpoint['model'])
        self.unet2.load_state_dict(self.checkpoint['model'])


        self.enc1_1 = self.unet1.encoder1
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_2 = self.unet1.encoder2
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_3 = self.unet1.encoder3
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1_4 = self.unet1.encoder4
        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_1 = self.unet1.bottleneck

        self.enc2_1 = self.unet2.encoder1
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_2 = self.unet2.encoder2
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_3 = self.unet2.encoder3
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2_4 = self.unet2.encoder4
        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_2 = self.unet2.bottleneck        


        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.Conv2d(16, 2, 3, stride=1, padding=1),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 2, 3, stride=1, padding=1),
        )

    def encoder1(self,x):
        enc1 = self.enc1_1(x)
        enc2 = self.enc1_2(self.pool1_1(enc1))
        enc3 = self.enc1_3(self.pool1_2(enc2))
        enc4 = self.enc1_4(self.pool1_3(enc3))
        bottleneck = self.bottleneck_1(self.pool1_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def encoder2(self,x):
        enc1 = self.enc2_1(x)
        enc2 = self.enc2_2(self.pool2_1(enc1))
        enc3 = self.enc2_3(self.pool2_2(enc2))
        enc4 = self.enc2_4(self.pool2_3(enc3))
        bottleneck = self.bottleneck_2(self.pool2_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def decoder(self,enc1,enc2,enc3,enc4,bottleneck):

        dec4 = self.unet1.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.unet1.decoder4(dec4)
        dec3 = self.unet1.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.unet1.decoder3(dec3)
        dec2 = self.unet1.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.unet1.decoder2(dec2)
        dec1 = self.unet1.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.unet1.decoder1(dec1)
        return torch.sigmoid(self.unet1.conv(dec1))

    def forward(self,x1,x2):

        x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle = self.encoder1(x1)
        x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle = self.encoder2(x2)

        out1 = self.decoder(x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle)
        out2 = self.decoder(x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle)

        dis = out1 - out2
        out = self.conv_layer(dis)

        return out


