from dataloader import MyDataset
from train import train_gan, train_wgan
from utils import generate_fake_images, get_FID_score, plot_grid

from torch.utils.data import DataLoader
import torch

import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


batch_size = 8
epochs = 50
gan_name = "gan"

train_dataset = MyDataset()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_gan(train_loader, epochs=epochs, name=gan_name, device=device)
generate_fake_images(model_path=f"./solutions/{gan_name}.pt", device=device)


batch_size = 16
epochs = 50
wgan_name = "wgan"
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_wgan(
    train_loader, epochs=epochs, name=wgan_name, device=device, batch_size=batch_size
)
generate_fake_images(model_path=f"./solutions/{wgan_name}.pt", device=device)


def load_pkl_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# load and plot GAN stuff

# gan_img_data_path = '/home/akshita/Documents/Acads/HW7/gan_final_img_list.pkl'
gan_loss_G_path = "/home/akshita/Documents/Acads/HW7/gan_final_G_loss.pkl"
gan_loss_D_path = "/home/akshita/Documents/Acads/HW7/gan_final_D_loss.pkl"
gan_final_fake_data_path = "/home/akshita/Documents/Acads/HW7/gan_final_fake_data"

# gan_img_data = load_pkl_data(gan_img_data_path)
gan_loss_G = load_pkl_data(gan_loss_G_path)
gan_loss_D = load_pkl_data(gan_loss_D_path)
plt.figure()
plt.plot(gan_loss_D, label="Discriminator")
plt.plot(gan_loss_G, label="Generator")
plt.legend()
plt.title("GAN Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("./solutions/gan_loss.png")

# images = []
# for imgobj in gan_img_data:
#     img = tvtF.to_pil_image(imgobj)
#     images.append(img)
# imageio.mimsave("./solutions/gan_generation_animation.gif", images, fps=5)

# load and plot GAN stuff


# gan_img_data_path = '/home/akshita/Documents/Acads/HW7/wgan_final_img_list.pkl'
wgan_loss_G_path = "/home/akshita/Documents/Acads/HW7/wgan_final_G_loss.pkl"
wgan_loss_D_path = "/home/akshita/Documents/Acads/HW7/wgan_final_C_loss.pkl"
wgan_final_fake_data_path = "/home/akshita/Documents/Acads/HW7/wgan_final_fake_data"

# gan_img_data = load_pkl_data(gan_img_data_path)

wgan_loss_G = load_pkl_data(wgan_loss_G_path)
wgan_loss_D = load_pkl_data(wgan_loss_D_path)
plt.figure()
plt.plot(wgan_loss_D, label="Critic")
plt.plot(wgan_loss_G, label="Generator")
plt.legend()
plt.title("WGAN Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
# plt.show()
plt.savefig("./solutions/wgan_loss.png")


eval_path = "/home/akshita/Documents/data/pizzas/eval"

real_image_files = [
    os.path.join(eval_path, f) for f in os.listdir(eval_path) if f.endswith(".jpg")
]
fake_image_files = [
    os.path.join(gan_final_fake_data_path, f)
    for f in os.listdir(gan_final_fake_data_path)
    if f.endswith(".jpg")
]
wfake_image_files = [
    os.path.join(wgan_final_fake_data_path, f)
    for f in os.listdir(wgan_final_fake_data_path)
    if f.endswith(".jpg")
]

# print(len(fake_image_files), len(real_image_files))
real_image_list = [cv2.imread(f) for f in real_image_files[:16]]
fake_image_list = [cv2.imread(f) for f in fake_image_files[:16]]
wfake_image_list = [cv2.imread(f) for f in wfake_image_files[:16]]


# print(len(real_image_list), len(fake_image_list))
real_grid = plot_grid(4, 4, real_image_list)
cv2.imwrite(f"./solutions/real_images.jpg", real_grid)
fake_grid = plot_grid(4, 4, fake_image_list)
cv2.imwrite(f"./solutions/fake_images.jpg", fake_grid)
wfake_grid = plot_grid(4, 4, wfake_image_list)
cv2.imwrite(f"./solutions/fake_images.jpg", wfake_grid)

fid_value = get_FID_score(real_image_files, fake_image_files)
w_fid_value = get_FID_score(real_image_files, wfake_image_files)
print(f"FID score for BCE-GAN and WGAN are: {fid_value}, {w_fid_value}")
