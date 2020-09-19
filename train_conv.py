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
import tqdm
import pandas as pd 
import cv2 as cv
import numpy as np 
from PIL import Image
from skimage import io, transform
import time
import os
import copy

from munch import Munch
CONFIGS_DICT = {
    "BATCH_SIZE": 32,
    "NUM_WORKERS" : 32,
    "WEIGHT_DECAY": 0.0005,
    "LEARNING_RATE": 0.0001,
    "STEP_SIZE": 5,
    "NUM_EPOCHS": 50,
    "GAMMA_LR_SCHEDULE": .5,
    "TORCH_SEED": 42,
    "NUMPY_SEED": 2020,
    "TORCH_CUDA_SEED": 40
}

configs = Munch(CONFIGS_DICT)

random.seed(configs.NUMPY_SEED)
np.random.seed(configs.NUMPY_SEED)
torch.manual_seed(configs.TORCH_SEED)
torch.cuda.manual_seed_all(configs.TORCH_CUDA_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train = pd.read_csv("labels_train.csv")
val = pd.read_csv("labels_val.csv")

class EyeGazeDataset(Dataset):
    def __init__(self, root_dir, df, transform= None):
        self.root_dir = root_dir 
        self.df = df.sample(frac= 1)
        self.transform = transform
    def __getitem__(self, idx):
        vector  = np.array(self.df.iloc[idx, 1:] , dtype= np.float64)
        img_folder, image_name = self.df.iloc[idx, 0].split("_")
        image_name = image_name + ".png"
        img_path = os.path.join(self.root_dir, img_folder, image_name)
        # print(img_path)
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(vector)
    def __len__(self):
        return len(self.df)


train_transform = transforms.Compose([
    transforms.Resize(size=(320, 200)), 
    transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])


val_transform = transforms.Compose([
    transforms.Resize(size=(320, 200)), 
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])


train_dataset= EyeGazeDataset("images/train/", train, train_transform)
val_dataset= EyeGazeDataset("images/val/", val, val_transform)

image_datasets = {"train": train_dataset ,
                  "val":val_dataset }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=configs.BATCH_SIZE,
                                             shuffle=True if x == "train" else False, num_workers=configs.NUM_WORKERS, drop_last=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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
model= torchvision.models.resnet18(True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE, weight_decay=configs.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs.STEP_SIZE, gamma= configs.GAMMA_LR_SCHEDULE)

since = time.time()
epoch_time = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 99999

step_loss_log = {"train" : [], "val" : []}
epoch_loss_log = {"train" : [], "val" : []}

for epoch in range(configs.NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch, configs.NUM_EPOCHS - 1))
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
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                outputs_norm = torch.norm(outputs, p= 2, dim= -1).view(configs.BATCH_SIZE, 1)
                outputs = outputs/(outputs_norm + 1e-16)
                loss = acos_loss(outputs.type(torch.DoubleTensor), labels.type(torch.DoubleTensor))
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_size += inputs.size(0)
            steps += 1
            if steps%200 == 0:
                print('\t\t Step No. {} of {} phase Epoch: {} Loss: {:.4f}'.format(steps, phase, epoch, running_loss/running_size))
            if steps%1000 == 0:
                print("\n")
                print("\t\t\t Labels", labels[:4, :])
                print("\t\t\t Outputs", outputs[:4, :])
                print("\n")
            # log 
            step_loss_log[phase].append(loss.item())
            

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_loss_log[phase].append(epoch_loss)

        if phase == 'val' and scheduler != None:
            scheduler.step()

        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if phase == "val":
            print("_"*20)
            print("Created a snapshot at Epoch No. {} with the Best Acos-Loss as {:.4f}".format( epoch, best_loss)) 
            print("_"*20)

            history = {"steps": step_loss_log,
           "epoch": epoch_loss_log}

            torch.save({
            "Epoch Number": epoch+1,
            "Model Name": "resnet18",
            "config": CONFIGS_DICT, 
            "history": history,
            "best_acos_loss":best_loss,
            "best_model_wts": best_model_wts,
            "model": model.state_dict(), 
            "optimizer":optimizer.state_dict() ,
            "scheduler": scheduler.state_dict(),
        }, "best_conv_wts.pth", )
            
    time_elapsed = time.time() - epoch_time
    print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    epoch_time = time.time()
    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Loss: {:4f}'.format(best_loss))