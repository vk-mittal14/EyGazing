"""
folder structure

images
    train
    val
    test
        0000
            000.png 
            001.png
            ..
            0049.png
        0001
        0002
        ..
        6399
"""

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import torchvision
import cv2 as cv 
import random
from munch import Munch 
from PIL import Image
from tqdm import tqdm
import glob
import os 

CONFIGS_DICT = {
    "BATCH_SIZE": 32,
    "TORCH_SEED": 42,
    "NUMPY_SEED": 2020,
    "TORCH_CUDA_SEED": 40,
    "torch_backends_cudnn_deterministic" : True, 
    "torch_backends_cudnn_benchmark" : False, 
    "USE_GPU": torch.cuda.is_available(),
    "CHK_PT_PATH": "./best_conv_wts.pth", 
    "IMAGES_PATH": "./images/",
    # "SPLIT": "test",
}

configs = Munch(CONFIGS_DICT)

random.seed(configs.NUMPY_SEED)
np.random.seed(configs.NUMPY_SEED)
torch.manual_seed(configs.TORCH_SEED)
torch.cuda.manual_seed_all(configs.TORCH_CUDA_SEED)
torch.backends.cudnn.deterministic = configs.torch_backends_cudnn_deterministic
torch.backends.cudnn.benchmark = configs.torch_backends_cudnn_benchmark


# ResNet model (trained on Tesla P100)
model= torchvision.models.resnet18(False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
if configs.USE_GPU: 
    model.cuda()
model.eval()

if configs.USE_GPU: 
    loaded = torch.load(configs.CHK_PT_PATH)
else: 
    loaded = torch.load(configs.CHK_PT_PATH, map_location=torch.device('cpu'))

model.load_state_dict(loaded["best_model_wts"])

# images dataset 
class EyeGazeTestDataset(Dataset):
    def __init__(self, images_path, transform= None):
        self.images_path = images_path 
        self.transform = transform
    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        img_path_split = img_path.split("/")
        img_folder, image_name =  img_path_split[-2], img_path_split[-1]
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if img.size != (320, 200):
            img = img.resize((320, 200))
            # ensures the size of image is 320, 200
        assert img.size == (320, 200)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_folder, image_name
    def __len__(self):
        return len(self.images_path)

def run_split(split): 
    images_path = sorted(glob.glob(configs.IMAGES_PATH +f"{split}/**/*.png"))
    all_seq =sorted(os.listdir(configs.IMAGES_PATH + f"{split}/"))
    # print(all_seq)
    test_dict_vector = {}
    for i in all_seq: 
        test_dict_vector[i] = {}
    
    composed_transform = transforms.Compose([
        transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    
    testds = EyeGazeTestDataset(images_path, composed_transform)
    testloader =torch.utils.data.DataLoader(testds, batch_size=configs.BATCH_SIZE)
    test_size = len(testds)
    print(f"Size of {split} dataset: ", test_size)
    
    # ResNet predictions 
    for inputs, seq, name in tqdm(testloader):
        if configs.USE_GPU: 
            inputs = inputs.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            outputs_norm = torch.norm(outputs, p= 2, dim= -1).view(configs.BATCH_SIZE, 1)
            outputs = outputs/(outputs_norm + 1e-16)
    
        if configs.USE_GPU:
            outputs = outputs.detach().cpu().numpy()
        else: 
            outputs = outputs.detach().numpy()
    
        for ii in range(configs.BATCH_SIZE):
            # print(seq[ii], name[ii], outputs[ii])
            test_dict_vector[seq[ii]][name[ii]] = outputs[ii]
    
    torch.save(test_dict_vector, f"{split}_vector.pth")
    print(f"Saved the {split} images vector predictions from ResNet18 as {split}_vector.pth")


print("Extracting vectors for train set")
run_split("train")
print("Extracting vectors for val set")
run_split("val")
print("Extracting vectors for test set")
run_split("test")