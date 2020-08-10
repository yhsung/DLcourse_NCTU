#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm
# Exercise: tensorboard setup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/test')

log_filename = datetime.datetime.now().strftime("log/tk%Y-%m-%d_%H_%M_%S.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M:%S',
            filename=log_filename)
# 定義 handler 輸出 sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 設定輸出格式
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# handler 設定輸出格式
console.setFormatter(formatter)
# 加入 hander 到 root logger
logging.getLogger('').addHandler(console)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 164)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)

# Dataloader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True,num_workers = 2)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./MNIST_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True,num_workers = 2)

#Define Network, we implement LeNet here
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5),stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5,5),stride=1, padding=0)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1) #flatten
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = Net()
print(model)
summary(model, (1, 28, 28))

# Exercise: adding 3x3 convolution networks
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.net= nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=(5,5),stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(6, 6, kernel_size=(3,3),stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(6, 6, kernel_size=(3,3),stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(6, 16, kernel_size=(5,5),stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16*4*4, 120),
                nn.Linear(120, 84),
                nn.Linear(84, 10))

    def forward(self, x):
        out = self.net(x)
        return out

model = Net2()
print(model)
summary(model, (1, 28, 28))

# Exercise: 2x wider network
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.net= nn.Sequential(
                nn.Conv2d(1, 12, kernel_size=(5,5),stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(12, 12, kernel_size=(3,3),stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 12, kernel_size=(3,3),stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(12, 32, kernel_size=(5,5),stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32*4*4, 120),
                nn.Linear(120, 84),
                nn.Linear(84, 10))

    def forward(self, x):
        out = self.net(x)
        return out

model = Net3()
print(model)
summary(model, (1, 28, 28))

#define optimizer/loss function
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# Exercise: different optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

#learning rate scheduling
def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
       lr = 0.01
    elif epoch < 15:
       lr = 0.001
    else:
       lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Exercise: scheduler for learning rate
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
#training function
def train(epoch):
    model.train()
    #adjust_learning_rate(optimizer, epoch)

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
    #for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = Loss(output, target)
        loss.backward()
        optimizer.step()
        logging.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        with torch.no_grad():
                output = model(data)
        test_loss += Loss(output, target).item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#run and save model
if args.cuda:
    model.to(device)
for epoch in tqdm(range(args.epochs)):
    train(epoch)
    test(epoch)
    scheduler.step()
    savefilename = 'LeNet_'+str(epoch)+'.tar'
    torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, savefilename)

# Exercise: write to tensorboard
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
# show images
matplotlib_imshow(img_grid, one_channel=True)
# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)
