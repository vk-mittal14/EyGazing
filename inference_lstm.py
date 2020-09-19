from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

import random
import json
from munch import Munch
from tqdm import tqdm
import cv2 as cv
import numpy as np 


config_dict = {
    "bidirectional": True,
    "num_layers": 1,
    "TORCH_SEED": 5000,
    "NUMPY_SEED": 5000+42,
    "TORCH_CUDA_SEED": 5000,
    "USE_GPU": torch.cuda.is_available(),
    "CHK_PT_PATH": "./last_lstm_wts.pth", 
    "SPLIT": "test"
}
config = Munch(config_dict)

random.seed(config.NUMPY_SEED)
np.random.seed(config.NUMPY_SEED)
torch.manual_seed(config.TORCH_SEED)
torch.cuda.manual_seed_all(config.TORCH_CUDA_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class EyegazeTimeTestData(Dataset):
    def __init__(self, direction_dict):
        self.direction_dict = direction_dict
        self.keys = list(direction_dict.keys())
    def __len__(self):
        return len(self.direction_dict)
    def __getitem__(self, idx):
        key = self.keys[idx]
        value = list(self.direction_dict[key].values())
        value = torch.tensor(value)[0:50]
        return value, key

test_dict_vector = torch.load(f"{config.SPLIT}_vector.pth")
test_ds = EyegazeTimeTestData(test_dict_vector)
testloader = DataLoader(test_ds, batch_size = 1)


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

model = TimeModel(num_layers=config.num_layers, bidirectional = config.bidirectional)
if config.USE_GPU:
    model = model.cuda()
model.eval()

if config.USE_GPU:
    loaded = torch.load(config.CHK_PT_PATH)
else: 
    loaded = torch.load(config.CHK_PT_PATH, map_location=torch.device('cpu'))

model.load_state_dict(loaded)

out_file = {}

for inputs, seq_name in tqdm(testloader):

    if config.USE_GPU: 
        inputs = inputs.cuda()
        
    with torch.no_grad():
        outputs = model(inputs).unsqueeze(1).expand(-1, 5, 3).reshape(-1, 3)

    out_file[seq_name[0]] = {}
    i = 0
    for ii in range(50, 55):
        out_file[seq_name[0]][str(ii)] = list(map(float, list(outputs[i].detach().cpu().numpy())))
        i += 1

with open(f"{config.SPLIT}_predictions.json", "w") as f:
    json.dump(out_file, f)
    print(f"Saved {config.SPLIT}_predictions.json")