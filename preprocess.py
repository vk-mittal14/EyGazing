import os 
import shutil 
import pandas as pd 
from tqdm import tqdm

label_dir = "labels/val"
df = pd.DataFrame(columns=["image_id", "x", "y", "z"])
labels_l = os.listdir(label_dir)
for label in tqdm(labels_l): 
    with open(os.path.join(label_dir, label)) as f:
        lines = f.readlines()
    for line in lines:
        line_split = line[:-1].split(",")
        img_name = label[:-4] + "_" + line_split[0]
        line_split_xyz = list(map(float, line_split[1:]))
        df = df.append({"image_id": img_name, "x":line_split_xyz[0], "y":line_split_xyz[1], "z":line_split_xyz[2]}, ignore_index=True)


df.to_csv("labels_val.csv", index= False)

label_dir = "labels/train"
df = pd.DataFrame(columns=["image_id", "x", "y", "z"])
labels_l = os.listdir(label_dir)
for label in tqdm(labels_l): 
    with open(os.path.join(label_dir, label)) as f:
        lines = f.readlines()
    for line in lines:
        line_split = line[:-1].split(",")
        img_name = label[:-4] + "_" + line_split[0]
        line_split_xyz = list(map(float, line_split[1:]))
        df = df.append({"image_id": img_name, "x":line_split_xyz[0], "y":line_split_xyz[1], "z":line_split_xyz[2]}, ignore_index=True)


df.to_csv("labels_train.csv", index= False)