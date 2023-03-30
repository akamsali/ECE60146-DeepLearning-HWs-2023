from dataloader import MyDataset
from train import train_gan

from torch.utils.data import DataLoader
import torch


train_dataset = MyDataset()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_gan(train_loader, epochs=10, name="gan", device=device)
