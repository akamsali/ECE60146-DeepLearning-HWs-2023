from dataset import MyDataset
from model import NetForYolo
from train_val import train, validation

from torch.utils.data import DataLoader


categories = ["bus", "cat", "pizza"]

batch_size = 4
train_data = MyDataset(
    categories=categories, split="train", manifest_path="./manifests", mac=True
)
train_data = MyDataset(
    categories=categories, split="train", manifest_path="./manifests"
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

ynet = NetForYolo(depth=2)

train(train_loader, ynet)
val_data = MyDataset(
    categories=categories, split="val", manifest_path="./manifests", mac=True
)
val_data = MyDataset(categories=categories, split="val", manifest_path="./manifests")
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
validation(val_loader, ynet)