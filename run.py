import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.dirichlet import Dirichlet

from utils import *
from models import *
import pytorch_classification
from pytorch_classification.models.cifar.resnet import *
from pytorch_classification.models.cifar.preresnet import *
from pytorch_classification.models.cifar.densenet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    dataloader = torchvision.datasets.CIFAR10
    num_classes = 10
else:
    dataloader = torchvision.datasets.CIFAR100
    num_classes = 100

full_train_set = dataloader('/data', train=True, download=True, transform=transform_train)
torch.manual_seed(42)
train_set, val_set = torch.utils.data.random_split(full_train_set, [45000, 5000])
torch.manual_seed(torch.initial_seed())

batch_size = 64
train_gen = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True,shuffle=True)
val_gen = torch.utils.data.DataLoader(val_set, batch_size=batch_size, drop_last=True,shuffle=True)

test_gen = torch.utils.data.DataLoader(
  dataloader('/data', train=False, download=True, transform=transform),
                              batch_size=batch_size, drop_last=True, shuffle=True)
l = len(train_gen)*batch_size
l_val = len(val_gen)*batch_size
l_test = len(test_gen)*batch_size

def train_heads(net_, lr=0.01, start_epoch=0, epochs=1, optimizer=None, scheduler=None):
  net_.train()
  criterion = nn.CrossEntropyLoss(reduction='none')  
  optimizer = torch.optim.SGD(net_.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
  # scheduler = ReduceLROnPlateau(optimizer, patience=8, verbose=True)
  sample_wts = net_.sample_wts

  for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
  	start = time.time()
    running_loss = np.zeros(net_.n_heads)
    correct = np.zeros(net_.n_heads)
    batch_no = 0
    for inputs, labels in train_gen:
      # get the inputs
      inputs = inputs.to(device)
      labels = labels.to(device)

      # forward + backward + optimize
      output = net_(inputs.float())
      optimizer.zero_grad()
      for i in range(net_.n_heads):
        output[i] = output[i].to(device)
        loss = criterion(output[i], labels)
        loss = (loss * sample_wts[i] / sample_wts[i].sum()).sum()
        loss.backward(retain_graph=True)

        # print statistics
        running_loss[i] += loss.item()
        _, predicted = torch.max(output[i].data, 1)
        predicted = predicted.to(device)
        correct[i] += (predicted == labels).sum().item()

      optimizer.step()
      if batch_no%20 == 0:
        print('.', end='')
      batch_no +=1
      
    wandb({"time": time.time()-start}, epoch)
    val_loss, val_acc = eval_heads(net_, val_gen)
    print('[%d]' %(epoch+1), end=' ')
    avg_loss = 0.0
    avg_acc = 0.0
    loss_head = []
    train_acc_head = []
    for i in range(net_.n_heads):
      print('{%d} loss: %.6f train acc = %.3f' %(i, running_loss[i] / l, 100 * correct[i] / l), end='\t')
      loss_head.append(running_loss[i] / l)
      train_acc_head.append(100 * correct[i] / l)
      wandb.log({"loss head": loss_head, "train acc head": train_acc_head}, step=epoch)
      avg_loss += running_loss[i] / l
      avg_acc += 100 * correct[i] / l
    avg_loss = avg_loss/net_.n_heads
    avg_acc = avg_acc/net_.n_heads
    print('     avg loss = %.6f   avg acc = %.3f   val loss = %.6f   val acc = %.3f' %(avg_loss, avg_acc, val_loss, val_acc))
    wandb.log({"avg loss": avg_loss, "avg acc": avg_acc, "val loss": val_loss, "val acc": val_acc}, step=epoch)
    # learning rate adjustments
    if scheduler!=None:
    	scheduler.step(avg_loss)
  
  print('Finished Training')
  
  
def eval_heads(net_, gen):
  net_.eval()
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
      correct = 0
      total = 0
      running_loss = 0.0
      for inputs, labels in gen:
        labels = labels.to(device)
        inputs = inputs.to(device)
        x = net_(inputs.float())
        outputs_sum = x[0].to(device)
        for i in range(1, net_.n_heads):
          outputs = x[i].to(device)
          outputs_sum = torch.add(outputs_sum, outputs)
        outputs = outputs_sum/net_.n_heads
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.to(device)
        correct += (predicted == labels).sum().item()

      net_.train()
      return running_loss/(len(gen)*batch_size), 100 * correct / (len(gen)*batch_size)