from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import random
from munch import Munch
import tqdm
import pandas as pd 
import cv2 as cv
import numpy as np 
from PIL import Image
from skimage import io, transform
import time
import os
import copy

config_dict = {
    "batch_size": 64,
    "lr": 0.01,
    "weight_decay": 0,
    "step_size": 20,
    "gamma_lr_scheduler": .8,
    "num_epochs": 200,
    "bidirectional": True,
    "num_layers": 1,
    "num_frames_val": "50 [0: 50, 45: 95]",
    "num_frames_train": "50 [0: 50]",
    "TORCH_SEED": 5000,
    "NUMPY_SEED": 5000+42,
    "TORCH_CUDA_SEED": 5000,
}
config = Munch(config_dict)

random.seed(config.NUMPY_SEED)
np.random.seed(config.NUMPY_SEED)
torch.manual_seed(config.TORCH_SEED)
torch.cuda.manual_seed_all(config.TORCH_CUDA_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dict_vector = torch.load("train_vector.pth") 
val_dict_vector = torch.load("val_vector.pth")
class EyegazeTimeData(Dataset):
    def __init__(self, direction_dict, phase= "train"):
        self.direction_dict = direction_dict
        self.phase = phase
        self.keys = list(direction_dict.keys())
        self.num_keys = len(self.keys)

    def __len__(self):
        if self.phase == "train":
            return len(self.direction_dict)*2
        else: 
            return len(self.direction_dict)
    def __getitem__(self, idx):
        if self.phase == "train":
            flag_second = False
            if idx >= self.num_keys:
                idx = idx - self.num_keys
                flag_second = True
            key = self.keys[idx]
            value = list(self.direction_dict[key].values())
            with open(f"labels/{self.phase}/" + key + ".txt") as f: 
                lines = f.readlines()
            seq_pred = torch.zeros(size= (5, 3),dtype = torch.float)
            if flag_second:
                seq_given = torch.tensor(value[45:95])
                for i in range(5):
                    seq_pred[i] = torch.tensor(list(map(float, lines[95+i][:-1].split(",")[1:])))
            else: 
                seq_given = torch.tensor(value[0:50])
                for i in range(5):
                    seq_pred[i] = torch.tensor(list(map(float, lines[50+i][:-1].split(",")[1:])))

        elif self.phase == "val": 
            key = self.keys[idx]
            value = list(self.direction_dict[key].values())
            seq_given = torch.tensor(value[0:50])
            with open(f"labels/{self.phase}/" + key + ".txt") as f: 
                lines = f.readlines()
            seq_pred = torch.zeros(size= (5, 3),dtype = torch.float)
            for i in range(5):
                seq_pred[i] = torch.tensor(list(map(float, lines[50+i][:-1].split(",")[1:])))
        return seq_given, seq_pred


train_ds = EyegazeTimeData(train_dict_vector, phase= "train")
val_ds = EyegazeTimeData(val_dict_vector, phase= "val")

trainloader = DataLoader(train_ds, batch_size= config.batch_size, shuffle= True, num_workers=0)
valloader = DataLoader(val_ds, batch_size= config.batch_size, shuffle= False, num_workers=0)

dataloaders = {"train":trainloader , 
               "val": valloader}
dataset_size = {"train": len(train_ds),
                "val": len(val_ds)}

def acos_loss(pred, true):
    """
    Expected shape for true and pred is (m, 3)
    where m is the batchsize and 3 are the x, y, z coordinates
    """
    bs = pred.size(0)
    batch_dot_product = torch.bmm(pred.view(bs, 1, 3), true.view(bs, 3, 1)).reshape(bs, )
    norm_true= torch.norm(true, p= 2, dim= -1)
    norm_pred= torch.norm(pred, p= 2, dim= -1)

    loss = torch.mean(torch.acos(batch_dot_product/(norm_true*norm_pred)))
    return loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TimeModel(nn.Module):
    def __init__(self, num_layers= 1, bidirectional= True, batch_first =True):
        super(TimeModel, self).__init__()
        self.lstm = nn.LSTM(3, 32, num_layers, bidirectional = bidirectional, batch_first=batch_first)
        self.linear1 = nn.Linear(32, 16)
        self.linear2 = nn.Linear(16, 3)
        # self.linear3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.sum(h, dim= 0).view(-1, 32)
        h = self.relu(self.linear1(h))
        h = self.linear2(h)
        # h = self.linear3(h)
        h_norm = torch.norm(h, dim=-1).unsqueeze(-1)
        out = h/(h_norm + 1e-16)
        return out

model = TimeModel(num_layers=config.num_layers, bidirectional = config.bidirectional).to(device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma= config.gamma_lr_scheduler)

since = time.time()
epoch_time = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 99999

step_loss_log = {"train" : [], "val" : []}
epoch_loss_log = {"train" : [], "val" : []}

for epoch in range(config.num_epochs):
    print('Epoch {}/{}'.format(epoch, config.num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_size = 0
        steps = 0
        
        # Iterate over data.
        for inputs, labels in (dataloaders[phase]):
            inputs = inputs.to(device)
            # inputs = inputs.to(device).view(-1, 45*3)
            labels = labels.to(device).view(-1, 3)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs).unsqueeze(1)
                outputs = outputs.expand(-1, 5, 3).reshape(-1, 3)
                loss = acos_loss(outputs.type(torch.DoubleTensor), labels.type(torch.DoubleTensor))
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_size += inputs.size(0)
            steps += 1
            if steps%50 == 0:
                print('\t\t Step No. {} of {} phase Epoch: {} Loss: {:.4f}'.format(steps, phase, epoch, running_loss/running_size))

            step_loss_log[phase].append(loss.item())
            

        epoch_loss = running_loss / dataset_size[phase]
        epoch_loss_log[phase].append(epoch_loss)

        if phase == 'val' and scheduler != None:
            scheduler.step()

        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

torch.save(model.state_dict(), "last_lstm_wts.pth")