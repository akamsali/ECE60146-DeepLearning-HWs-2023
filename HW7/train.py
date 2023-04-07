import torch
from utils import weights_init
import model

import torch.nn as nn
import torchvision

import pickle
import csv


def train_gan(
    train_loader,
    noise_dim=100,
    batch_size=4,
    device=torch.device("cpu"),
    lr=0.0002,
    betas=(0.5, 0.999),
    epochs=1,
    name="gan",
):

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
                save_vals = [
                    epoch,
                    epochs,
                    i,
                    len(train_loader),
                    D_total_err.item(),
                    G_err.item(),
                    running_D_loss / 50,
                    running_G_loss / 50,
                ]

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

            if (iters % 500 == 0) or (
                (epoch == epochs - 1) and (i == len(train_loader) - 1)
            ):
                with torch.no_grad():
                    fake = (
                        netG(fixed_noise).detach().cpu()
                    )  ## detach() removes the fake from comp. graph.
                    ## for creating its CPU compatible version
                img_list.append(
                    torchvision.utils.make_grid(
                        fake, padding=1, pad_value=1, normalize=True
                    )
                )
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


def calc_gradient_penalty(netC, real_data, fake_data, LAMBDA=10):
    """
    Implementation by Marvin Cao: https://github.com/caogang/wgan-gp
    Marvin Cao's code is a PyTorch version of the Tensorflow based implementation provided by
    the authors of the paper "Improved Training of Wasserstein GANs" by Gulrajani, Ahmed,
    Arjovsky, Dumouli,  and Courville.
    """
    # BATCH_SIZE = self.dlstudio.batch_size
    # LAMBDA = self.adversarial.LAMBDA
    epsilon = torch.rand(1).cuda()
    interpolates = epsilon * real_data + ((1 - epsilon) * fake_data)
    interpolates = interpolates.requires_grad_(True).cuda()
    critic_interpolates = netC(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def train_wgan(
    train_loader,
    noise_dim=100,
    batch_size=4,
    device=torch.device("cpu"),
    lr=0.0002,
    betas=(0.5, 0.999),
    epochs=1,
    LAMBDA=10,
    name="wgan",
):

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
        print(len(train_loader))
        data_iterator = iter(train_loader)
        i = 0
        n_critic = 5
        while i < len(train_loader):
            for param in netC.parameters():
                param.requires_grad = True

            ic = 0
            # print("entering loop: ", i, n_critic, gen_iterations)
            # while ic < n_critic and i < len(train_loader):
            # for _ in range(n_critic):
            while ic < n_critic and i < len(train_loader):
                # if i >= len(train_loader):
                # break

                # for p in netC.parameters():
                # p.data.clamp_(-clipping_thresh, clipping_thresh)

                netC.zero_grad()
                # real_images = data_iterator.next().to(device)
                real_images = next(data_iterator).to(device)
                i += 1
                b_size = real_images.size(0)

                critic_for_real = netC(real_images)
                critic_for_real.backward(minus_one)

                noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
                fake = netG(noise)
                critic_for_fake.backward(one)

                gradient_penalty = calc_gradient_penalty(
                    netC, real_images, fake, LAMBDA
                )
                gradient_penalty.backward()
                critic_loss = critic_for_fake - critic_for_real + gradient_penalty
                wasserstein_distance = critic_for_real - critic_for_fake

                optimizerC.step()

            # now we come to generator, for which we don't need to update critic
            # so we freeze the critic parameters
            for p in netC.parameters():
                p.requires_grad = False
            # we need to update generator
            # print("entering generator")
            netG.zero_grad()
            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake = netG(noise)
            critic_for_fake = netC(fake)
            gen_loss = critic_for_fake
            critic_for_fake.backward(minus_one)

            # update generator
            optimizerG.step()
            gen_iterations += 1

            c_loss_val = critic_loss.data[0].item()
            g_loss_val = gen_loss.data[0].item()
            if i % 200 == 0:
                save_vals = [
                    epoch,
                    epochs,
                    i,
                    len(train_loader),
                    c_loss_val,
                    g_loss_val,
                    wasserstein_distance.data[0].item(),
                ]

                logger = open(f"./solutions/{name}.csv", "a", newline="")
                with logger:
                    write = csv.writer(logger)
                    write.writerow(save_vals)

                if g_loss_val < loss_G_flag:
                    loss_G_flag = g_loss_val
                    torch.save(netG.state_dict(), "./solutions/gen_" + name + ".pt")
                if c_loss_val < loss_D_flag:
                    loss_D_flag = c_loss_val
                    torch.save(netC.state_dict(), "./solutions/critic_" + name + ".pt")

                # running_D_loss = 0.0
                # running_G_loss = 0.0

            C_losses.append(critic_loss.data[0].item())
            G_losses.append(g_loss_val)
            if (iters % 500 == 0) or (
                (epoch == epochs - 1) and (i == len(train_loader) - 1)
            ):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(
                    torchvision.utils.make_grid(fake, padding=2, normalize=True)
                )
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
