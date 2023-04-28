from model import ViT
from dataset import MyDataset
from train_val import train, test

import torch
from torch.utils.data import DataLoader



img_size = 64
patch_size = 16
num_classes = 5
embedding_size = 128
max_seq_length = (img_size // patch_size) ** 2 + 1
num_encoder_blocks = 3
num_atten_heads = 4
epochs = 5

vit_embeddings = ViT(
    img_size=img_size,
    patch_size=patch_size,
    num_classes=num_classes,
    embedding_size=embedding_size,
    num_heads=num_atten_heads,
    num_encoders=num_encoder_blocks,
    max_seq_length=max_seq_length,
)


#root = "/home/akshita/Documents/data/coco_custom"
root = "/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom"
categories = ["airplane", "bus", "cat", "dog", "pizza"]
train_dataset = MyDataset(root=root, categories=categories, split="train")
name = 'b16_lr1e3'
batch_size = 16 
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
    cm, acc =  test(vit_embeddings, val_loader, name=name)
    with open(f"{name}_results.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion Matrix: {cm}\n")
except Exception as e:
    with open(f"{name}_error_log.txt", "w") as f:
        f.write(str(e))
    print("Error written to error_log.txt")
