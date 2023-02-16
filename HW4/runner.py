from coco_dataloader import COCO_loader
from models import HW4Net1, HW4Net2, HW4Net3
from dataset import MyDataset
from train_val import train, validate_and_conf_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(60146)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from tqdm import tqdm

# set categories and data path
categories = ["airplane", "bus", "cat", "dog", "pizza"]
data_path = "/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom"
train_size, val_size = 1500, 500
batch_size = 16  # change this as needed

# build the required size of the data
cl = COCO_loader()
for category in tqdm(categories):
    cl.save_images_to_folder(
        category=category, train_size=train_size, val_size=val_size
    )

# get the dataset
train_dataset = MyDataset(root=data_path, categories=categories, split="train")
val_dataset = MyDataset(root=data_path, categories=categories, split="val")

# plotter for sample images from each class
num_images = 3
all_images = []
for category in categories:
    cat_images = []
    path = os.path.join(data_path, category, "train")
    image_list = os.listdir(path)[:num_images]
    # print(image_list)
    for img in image_list:
        img = cv2.imread(os.path.join(path, img))
        cat_images.append(img)
    tot_images = np.concatenate(cat_images, axis=1)

    all_images.append(tot_images)

all_images = np.concatenate(all_images, axis=0)

cv2.imwrite("results/sampled_images.png", all_images)


# build data loader from dataset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# train all 3 models
net1 = HW4Net1()
train(net=net1, train_dataloader=train_dataloader, epochs=10, net_name="Net1")
net2 = HW4Net2()
train(net=net2, train_dataloader=train_dataloader, epochs=10, net_name="Net2")
net3 = HW4Net3()
train(net3, train_dataloader=train_dataloader, epochs=10, net_name="Net3")

# load the best model
net1.load_state_dict(torch.load("results/Net1.pt"))
net2.load_state_dict(torch.load("results/Net2.pt"))
net3.load_state_dict(torch.load("results/Net3.pt"))

# use the loaded model for validation and build confusion matrix along with accuracy
validate_and_conf_matrix(net1, val_dataset, categories, name="Net1")
validate_and_conf_matrix(net2, val_dataset, categories, name="Net2")
validate_and_conf_matrix(net3, val_dataset, categories, name="Net3")

# plot training losses with saved csv of running training loss
t_1 = pd.read_csv("results/Net1.csv", header=None)
t_2 = pd.read_csv("results/Net2.csv", header=None)
t_3 = pd.read_csv("results/Net3.csv", header=None)
plt.plot(t_1[2], label="Net1")
plt.plot(t_2[2], label="Net2")
plt.plot(t_3[2], label="Net3")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Running average loss")
plt.title("Training Loss for all three models")
plt.savefig("results/training_loss.png")
