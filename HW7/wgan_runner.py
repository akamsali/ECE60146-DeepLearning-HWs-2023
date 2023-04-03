import torch 
from utils import weights_init
import model

import torch.nn as nn
import torchvision

import pickle
import csv


def train_wgan(train_loader, 
                noise_dim=100, 
                batch_size=4, 
                device=torch.device("cpu"), 
                lr=0.0002, 
                betas=(0.5, 0.999), 
                epochs=1,
                clipping_thresh = 0.01,
                name="wgan"):

    netC = model.CriticCG1().to(device)
    netG = model.Generator().to(device)
    netC.apply(weights_init)
    netG.apply(weights_init)

    fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    # instead of real and fake labels, we use 1 and -1
    one = torch.FloatTensor([1]).to(device)
    minus_one = torch.FloatTensor([-1]).to(device)
    # real_label = 1
    # fake_label = 0

    optimizerC = torch.optim.Adam(netC.parameters(), lr=lr, betas=betas)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=betas)

    C_losses = []
    G_losses = []                               
    img_list = []
    iters = 0
    gen_iterations = 0   
    loss_G_flag = 100000
    loss_D_flag = 100000

    for epoch in range(epochs):
        data_iterator = iter(train_loader)
        i = 0
        n_critic = 5
        
        while i < len(train_loader):
            
            for param in netC.parameters():
                param.requires_grad = True
            
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_critic = 100
            
            ic = 0
        
            while ic < n_critic and i < len(train_loader):
                ic += 1
                for p in netC.parameters():
                    p.data.clamp_(-clipping_thresh, clipping_thresh)
                
                netC.zero_grad()
                real_images = data_iterator.next().to(device)
                i += 1
                b_size = real_images.size(0)

                critic_for_real = netC(real_images)
                critic_for_real.backward(minus_one)

                noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
                fake = netG(noise)
                critic_for_fake = netC(fake)
                critic_for_fake.backward(one)

                wasserstein_distance = critic_for_real - critic_for_fake
                critic_loss = critic_for_fake - critic_for_real

                optimizerC.step()
            
            # now we come to generator, for which we don't need to update critic
            # so we freeze the critic parameters
            for p in netC.parameters():
                p.requires_grad = False
            # we need to update generator
            netG.zero_grad()
            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake = netG(noise)
            critic_for_fake = netC(fake)
            gen_loss = critic_for_fake
            critic_for_fake.backward(minus_one)

            # update generator
            optimizerG.step()
            gen_iterations += 1
            if i % (n_critic*20) == 0:
                save_vals = [epoch, epochs, i, len(train_loader), 
                             critic_loss.data[0], gen_loss.data[0],
                             wasserstein_distance.data[0]]

                logger = open(f"./solutions/{name}.csv", "a", newline="")
                with logger:
                    write = csv.writer(logger)
                    write.writerow(save_vals)

                # if running_G_loss < loss_G_flag:
                #     loss_G_flag = running_G_loss
                #     torch.save(netG.state_dict(), "./solutions/gen_" + name + ".pt")
                # if running_D_loss < loss_D_flag:
                #     loss_D_flag = running_D_loss
                #     torch.save(netD.state_dict(), "./solutions/dis_" + name + ".pt")

                # running_D_loss = 0.0
                # running_G_loss = 0.0

            C_losses.append(critic_loss.data[0])
            G_losses.append(gen_loss.data[0])
            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))
            iters += 1


                


        
    with open(f"./solutions/{name}_C_loss.pkl", "wb") as logger:
        pickle.dump(C_losses, logger)
    logger.close()
    with open(f"./solutions/{name}_G_loss.pkl", "wb") as logger:
        pickle.dump(G_losses, logger)
    logger.close()
    with open(f"./solutions/{name}_img_list.pkl", "wb") as logger:
        pickle.dump(img_list, logger)
    logger.close()

from dataloader import MyDataset
# from train import train_gan

from torch.utils.data import DataLoader
import torch

data_path = '/home/akshita/Documents/data/pizzas'


train_dataset = MyDataset(data_path)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_wgan(train_loader, epochs=1, name="wgan", device=device)