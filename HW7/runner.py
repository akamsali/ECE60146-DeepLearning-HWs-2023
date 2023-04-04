from dataloader import MyDataset
from train import train_gan, train_wgan
from utils import generate_fake_images, get_FID_score

from torch.utils.data import DataLoader
import torch



batch_size = 8
epochs = 50
gan_name = "gan"
wgan_name = "wgan"

train_dataset = MyDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_gan(train_loader, epochs=epochs, name=gan_name, device=device)
train_wgan(train_loader, epochs=epochs, name=wgan_name, device=device, batch_size=batch_size)

generate_fake_images(model_path=f"./solutions/{gan_name}.pt",
                     device=device)

generate_fake_images(model_path=f"./solutions/{wgan_name}.pt",
                     device=device)