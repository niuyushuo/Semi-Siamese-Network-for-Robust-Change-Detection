import numpy as np # linear algebra
from collections import OrderedDict

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


class Semi_siamese_(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(Semi_siamese_, self).__init__()

        features = init_features
        self.encoder1_1 = Semi_siamese_._block(in_channels, features, name="enc1_1")
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1_2 = Semi_siamese_._block(features, features * 2, name="enc1_2")
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1_3 = Semi_siamese_._block(features * 2, features * 4, name="enc1_3")
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1_4 = Semi_siamese_._block(features * 4, features * 8, name="enc1_4")
        self.pool1_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_1 = Semi_siamese_._block(features * 8, features * 16, name="bottleneck_1")

        self.encoder2_1 = Semi_siamese_._block(in_channels, features, name="enc2_1")
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_2 = Semi_siamese_._block(features, features * 2, name="enc2_2")
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_3 = Semi_siamese_._block(features * 2, features * 4, name="enc2_3")
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2_4 = Semi_siamese_._block(features * 4, features * 8, name="enc2_4")
        self.pool2_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck_2 = Semi_siamese_._block(features * 8, features * 16, name="bottleneck_2")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Semi_siamese_._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Semi_siamese_._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Semi_siamese_._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Semi_siamese_._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )


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
        enc1 = self.encoder1_1(x)
        enc2 = self.encoder1_2(self.pool1_1(enc1))
        enc3 = self.encoder1_3(self.pool1_2(enc2))
        enc4 = self.encoder1_4(self.pool1_3(enc3))
        bottleneck = self.bottleneck_1(self.pool1_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def encoder2(self,x):
        enc1 = self.encoder2_1(x)
        enc2 = self.encoder2_2(self.pool2_1(enc1))
        enc3 = self.encoder2_3(self.pool2_2(enc2))
        enc4 = self.encoder2_4(self.pool2_3(enc3))
        bottleneck = self.bottleneck_2(self.pool2_4(enc4))
        return enc1,enc2,enc3,enc4,bottleneck

    def decoder(self,enc1,enc2,enc3,enc4,bottleneck):

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))


    def forward(self,x1,x2):

        x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle = self.encoder1(x1)
        x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle = self.encoder2(x2)

        out1 = self.decoder(x1_enc1,x1_enc2,x1_enc3,x1_enc4,x1_bottle)
        out2 = self.decoder(x2_enc1,x2_enc2,x2_enc3,x2_enc4,x2_bottle)

        dis = out1 - out2
        out = self.conv_layer(dis)

        return out


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
                                                            

