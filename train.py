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
import pylab


FIGS_DIR = './figs/'
SAVE_DIR = './save/'
DATA_DIR = '/home/wangqs/Data/'

use_cuda = True
batch_size = 16
num_epochs = 300

lr = 1e-1
weight_decay = 1e-3
gamma = 0.95

if not os.path.exists(FIGS_DIR):
    os.makedirs(FIGS_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5),   
                                         std=(0.5))])


train_dataset = YANTI(root_dir=DATA_DIR, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = YANTI(root_dir=DATA_DIR, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


use_cuda = use_cuda and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = CaliTransform().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=gamma)

num_train_steps = len(train_loader)
num_test_steps = len(test_loader)
train_losses = []
test_losses = []

train_loss = 0
num_train = 0
for epoch in range(num_epochs):
    model.train()
    for i, data in enumerate(train_loader):

        pre_src, pre_tgt = data

        if pre_src.size(0) < batch_size:
            continue

        src = pre_src.to(device)
        tgt = pre_tgt.to(device)
        
        output = model(src)
        loss = criterion(output.view(batch_size,-1), tgt.view(batch_size, -1))
        train_loss += loss.item() * pre_src.size(0)
        num_train += pre_src.size(0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) == num_train_steps - 1:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, num_train_steps, loss.data.item()))
        
    scheduler.step()

    if (epoch + 1) % 50 == 0:
        output = output.view(output.size(0), 1, 128, 128)
        save_image(recover(output.data), os.path.join(FIGS_DIR, 'generated_characters-{}.png'.format(epoch+1)))
        tgt = tgt.view(tgt.size(0), 1, 128, 128)
        save_image(recover(tgt.data), os.path.join(FIGS_DIR, 'target_characters-{}.png'.format(epoch+1)))
        src = src.view(src.size(0), 1, 128, 128)
        save_image(recover(src.data), os.path.join(FIGS_DIR, 'source_characters-{}.png'.format(epoch+1)))

    train_loss /= num_train
    train_losses.append(train_loss)

    model.eval()
    test_loss = 0
    num_test = 0
    for i, data in enumerate(test_loader):

        pre_src, pre_tgt = data

        src = pre_src.to(device)
        tgt = pre_tgt.to(device)
        
        output = model(src)
        loss = criterion(output.view(batch_size,-1), tgt.view(batch_size, -1))
        test_loss += loss.item() * pre_src.size(0)
        num_test += pre_src.size(0)
        
    test_loss /= num_test
    print('Epoch [{}/{}], test loss: {:.4f}' .format(epoch+1, num_epochs, test_loss))

    test_losses.append(test_loss)

    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    plt.plot(range(1, epoch + 2), train_losses, label='train loss')
    plt.plot(range(1, epoch + 2), test_losses, label='test loss')    
    plt.legend()
    plt.savefig(os.path.join(FIGS_DIR, 'loss.png'))
    plt.close()


torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'model.ckpt'))