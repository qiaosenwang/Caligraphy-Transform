from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils import *


FIGS_DIR = './figs/'
SAVE_DIR = './save/'
DATA_DIR = '/home/wangqs/Data/'

use_cuda = False
batch_size = 8

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5),   
                                         std=(0.5))])

test_dataset = YANTI(root_dir=DATA_DIR, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


use_cuda = use_cuda and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CaliTransform().to(device)
criterion = nn.MSELoss()
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'model.ckpt')))
model.eval()

test_loss = 0
num_test = 0
num_steps = len(test_loader)
for i, data in enumerate(test_loader):

    pre_src, pre_tgt = data

    src = pre_src.to(device)
    tgt = pre_tgt.to(device)
    
    output = model(src)
    loss = criterion(output.view(batch_size,-1), tgt.view(batch_size, -1))
    test_loss += loss.item() * pre_src.size(0)
    num_test += pre_src.size(0)

    if (i+1) == num_steps - 1:
        output = output.view(output.size(0), 1, 128, 128)
        save_image(recover(output.data), os.path.join(FIGS_DIR, 'test_generated_characters.png'))
        tgt = tgt.view(tgt.size(0), 1, 128, 128)
        save_image(recover(tgt.data), os.path.join(FIGS_DIR, 'test_target_characters.png'))
        src = src.view(src.size(0), 1, 128, 128)
        save_image(recover(src.data), os.path.join(FIGS_DIR, 'test_source_characters.png'))
    
test_loss /= num_test
print('test loss: {:.4f}' .format(test_loss))