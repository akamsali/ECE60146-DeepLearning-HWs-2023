
import torch 
from utils import weights_init
import model

import torch.nn as nn
import torchvision

import pickle
import csv


def train_gan(train_loader, 
                noise_dim=100, 
                batch_size=4, 
                device=torch.device("cpu"), 
                lr=0.0002, 
                betas=(0.5, 0.999), 
                epochs=1, 
                name="gan"):

    netD = model.DiscriminatorDG1().to(device)
    netG = model.Generator().to(device)
    netD.apply(weights_init)
    netG.apply(weights_init)

    fixed_noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=betas)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=betas)

    criterion = nn.BCELoss()
    G_losses = []                               
    D_losses = []
    img_list = []
    iters = 0   
    loss_G_flag = 100000
    loss_D_flag = 100000
    for epoch in range(epochs):
        running_D_loss = 0.0
        running_G_loss = 0.0
        for i, data in enumerate(train_loader):
            # Discriminator
            netD.zero_grad()
            # get
            real_images = data.to(device)
            b_size = real_images.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # print(label.shape)
            output = netD(real_images).view(-1)
            # print(output.shape)
            D_real_err = criterion(output, label)
            D_real_err.backward()

            # D_x = output.mean().item()

            # generate noise
            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            # detach to avoid backprop through Discriminator
            output = netD(fake.detach()).view(-1)
            D_fake_err = criterion(output, label)
            D_fake_err.backward()
            # D_G_z1 = output.mean().item()
            
            D_total_err = D_real_err + D_fake_err
            # print(D_total_err.item())
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            G_err = criterion(output, label)
            G_err.backward()
            # D_G_z2 = output.mean().item()
            optimizerG.step()

            D_losses.append(D_total_err.item())
            G_losses.append(G_err.item())
            running_D_loss += D_total_err.item()
            running_G_loss += G_err.item()


            if i % 100 == 0:
                save_vals = [epoch, epochs, i, len(train_loader), 
                            D_total_err.item(), G_err.item(), 
                            running_D_loss/50, running_G_loss/50]

                logger = open(f"./solutions/{name}.csv", "a", newline="")
                with logger:
                    write = csv.writer(logger)
                    write.writerow(save_vals)

                if running_G_loss < loss_G_flag:
                    loss_G_flag = running_G_loss
                    torch.save(netG.state_dict(), "./solutions/gen_" + name + ".pt")
                if running_D_loss < loss_D_flag:
                    loss_D_flag = running_D_loss
                    torch.save(netD.state_dict(), "./solutions/dis_" + name + ".pt")

                running_D_loss = 0.0
                running_G_loss = 0.0

            if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(train_loader)-1)):   
                        with torch.no_grad():             
                            fake = netG(fixed_noise).detach().cpu()  ## detach() removes the fake from comp. graph. 
                                                                     ## for creating its CPU compatible version
                        img_list.append(torchvision.utils.make_grid(fake, padding=1, pad_value=1, normalize=True))
            iters += 1

    with open(f"./solutions/{name}_D_loss.pkl", "wb") as logger:
        pickle.dump(D_losses, logger)
    logger.close()
    with open(f"./solutions/{name}_G_loss.pkl", "wb") as logger:
        pickle.dump(G_losses, logger)
    logger.close()
    with open(f"./solutions/{name}_img_list.pkl", "wb") as logger:
        pickle.dump(img_list, logger)
    logger.close()