from model import ViT
from dataset import MyDataset
from train_val import train, test

import torch
from torch.utils.data import DataLoader

import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

img_size = 64
patch_size = 16
num_classes = 5
embedding_size = 256
max_seq_length = (img_size // patch_size) ** 2 + 1
num_encoder_blocks = 4
num_atten_heads = 8
epochs = 10
# I set einops_usage to True and False in ViT() below to test both versions
einops_usage = False

vit_embeddings = ViT(
    img_size=img_size,
    patch_size=patch_size,
    num_classes=num_classes,
    embedding_size=embedding_size,
    num_heads=num_atten_heads,
    num_encoders=num_encoder_blocks,
    max_seq_length=max_seq_length,
    einops_usage=einops_usage,
)


# root = "/home/akshita/Documents/data/coco_custom"
root = "/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom"
categories = ["airplane", "bus", "cat", "dog", "pizza"]
train_dataset = MyDataset(root=root, categories=categories, split="train")
batch_size = 8
name = f"ein{str(einops_usage)}_b{batch_size}_es{embedding_size}_enc{num_encoder_blocks}_att{num_atten_heads}"
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

# try train, if error write error to log file
try:
    train(vit_embeddings, train_loader, epochs=epochs, name=name)
except Exception as e:
    with open("error_log.txt", "w") as f:
        f.write(str(e))
    print("Error written to error_log.txt")

# try test, if error write error to log file
try:
    vit_embeddings.load_state_dict(torch.load(f"{name}_epoch_{epochs}.pt"))
    val_dataset = MyDataset(root=root, categories=categories, split="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    cm, acc = test(vit_embeddings, val_loader, name=name)
    with open(f"{name}_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion Matrix: {cm}\n")
except Exception as e:
    with open(f"{name}_error_log.txt", "w") as f:
        f.write(str(e))
    print("Error written to error_log.txt")

sol = "./solutions"
file_list = os.listdir(sol)
categories = ["airplane", "bus", "cat", "dog", "pizza"]

for i in file_list:
    if ".csv" in i and "train" not in i and ".png" not in i:
        df = pd.read_csv(os.path.join(sol, i))
        t = df["targets"].tolist()
        p = df["predictions"].tolist()
        cm = confusion_matrix(t, p)
        plt.figure()
        sns.heatmap(
            cm, annot=cm, xticklabels=categories, yticklabels=categories, fmt="g"
        )
        plt.title(f"Confusion matrix , accuracy={accuracy_score(t, p)}")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.savefig(f"solutions/cm_{i}.png")

for i in file_list:
    if "train.csv" in i and ".png" not in i:
        df = pd.read_csv(os.path.join(sol, i), header=None)
        # print(df.head())
        train = df[2].tolist()
        plt.figure()
        plt.plot(train)
        plt.title(f"Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(f"solutions/loss_{i}.png")
