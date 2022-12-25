import torch
import torch.nn as nn
import torch.nn.functional as F
import os
#import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



def recover(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class YANTI(Dataset):
    """Kaiti Caligraphy Dataset."""

    def __init__(self, root_dir:str, train:bool=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'YANTI')
        self.transform = transform
        self.train = train

    def __len__(self):
        if self.train:
            return 857
        else:
            return 157

    def to_name(self, idx):
        if self.train:
            return(str(idx).zfill(3))
        else:
            return(str(2175+idx))


    def __getitem__(self, idx:int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        name = self.to_name(idx)

        std_name = os.path.join(self.img_dir, 'standard',
                                name+'.jpg')
        stds = io.imread(std_name)

        cali_name = os.path.join(self.img_dir, 'yanti',
                                name+'.jpg')
        calis = io.imread(cali_name)

        if self.transform:
            stds = self.transform(stds)
            calis = self.transform(calis)

        return stds, calis


class CaliTransform(nn.Module):

    def __init__(self, ngpu=1, nc=1, ndf=64, ngf=64):
        super(CaliTransform, self).__init__()
        self.ngpu = ngpu
        
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        input = input.view(input.size(0), 1, 128, 128)
        output = self.encoder(input)
        output = self.decoder(output)
        return(output)