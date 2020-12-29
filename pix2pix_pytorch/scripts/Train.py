import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from datasets import ImageDataset
import itertools
import glob
from utils import ReplayBuffer
from Models import Discriminator
from utils import weights_init_normal
from utils import LambdaLR
import csv
import numpy as np
from torch import autograd


class PixelToPixel(object):
    """docstring for pixel2pixel"""

    def __init__(self, Unet):
        if Unet:
            from Models import Generator_resnet_unet as Generator
        else:
            from Models import Generator_resnet as Generator

        self.generator = Generator

    def dataloader(self, opt):
        if not os.path.exists('output'):
            os.makedirs('output')

        transforms1 = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                       transforms.RandomCrop(opt.size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]

        dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms1, unaligned=True),
                                batch_size=opt.batchSize, shuffle=True, num_workers=0)

        return dataloader

    def calc_gradient_penalty(self, netD, real_data, fake_data, batch_size):
        DIM = 256
        LAMBDA = 10
        bh = fake_data.shape[0]
        alpha = torch.rand(bh, 1)
        alpha = alpha.expand(bh, int(real_data.nelement() / bh)).contiguous()
        alpha = alpha.view(bh, 3, DIM, DIM)
        alpha = alpha.cuda()

        fake_data = fake_data.view(bh, 3, DIM, DIM)
        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def train(self, opt, dataloader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("you are using", device)
        assert device != 'cpu', "Unable to find GPU! Stop running now!"
        # Networks
        netG_A2B = self.generator(opt.input_nc, opt.output_nc)
        netD_B = Discriminator(opt.output_nc)

        netG_A2B.to(device)
        netD_B.to(device)

        netG_A2B.apply(weights_init_normal)
        netD_B.apply(weights_init_normal)

        # Lossess
        criterion_GAN = torch.nn.MSELoss()
        criterion_identity = torch.nn.L1Loss()

        ### optimizers

        optimizer_G = torch.optim.Adam(netG_A2B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                               opt.decay_epoch).step)

        Tensor = torch.cuda.FloatTensor
        target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

        fake_B_buffer = ReplayBuffer()

        x = []
        Loss_Generator = []
        Loss_Identity = []
        Loss_GANs = []

        for epoch in range(opt.n_epochs):

            print("training in", epoch + 1, "'s epoch", "/", opt.n_epochs)

            for i, batch in enumerate(dataloader):
                gen_train = 1
                if ((i % gen_train) == 0):
                    for p in netD_B.parameters():
                        p.requires_grad_(False)

                    if i % 100 == 99:
                        print(i + 1, "/", len(dataloader))

                real_A = batch['A'].to(device)
                real_B = batch['B'].to(device)
                ### generate two fake images
                optimizer_G.zero_grad()
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B) * 5.044

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                loss_G = loss_identity_B + loss_GAN_A2B
                loss_G.backward()

                optimizer_G.step()

                ###### Discriminator B ######
                for p in  netD_B.parameters():
                    p.requires_grad_(True)
                # print(gradient_penalty)
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                gradient_penalty = self.calc_gradient_penalty(netD_B, real_B, fake_B, opt.batchSize)
                # print(loss_D_real,loss_D_fake,gradient_penalty)
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5 + gradient_penalty*.5
                loss_D_B.backward(retain_graph=True)
                # loss_D_B.backward()

                optimizer_D_B.step()

            Loss_Generator.append(loss_G.detach().cpu().numpy())
            Loss_Identity.append(loss_identity_B.detach().cpu().numpy())
            Loss_GANs.append(loss_GAN_A2B.detach().cpu().numpy())

            x.append(epoch)

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_AtoB.pth')

        return Loss_Generator, Loss_Identity, Loss_GANs, x

    def PlotandSave(self, Loss_Generator, Loss_Identity, Loss_GANs, x):

        NL = ['The Loss of Generator.csv', 'Identity Loss.csv', 'The Loss of GAN.csv']
        VAL = [Loss_Generator, Loss_Identity, Loss_GANs]

        for i in (range(len(NL))):

            csvfile = NL[i]
            Val = VAL[i]

            # Assuming res is a flat list
            with open(csvfile, "w") as output:
                writer = csv.writer(output, lineterminator='\n')
                for val in Val:
                    writer.writerow([val])

        plt.subplot(3, 1, 1)
        plt.plot(x, Loss_Generator)
        plt.xlabel('num of epoch')
        plt.ylabel('The Loss of Generator')

        plt.subplot(3, 1, 2)
        plt.plot(x, Loss_Identity)
        plt.xlabel('num of epoch')
        plt.ylabel('Identity Loss')

        plt.subplot(3, 1, 3)
        plt.plot(x, Loss_GANs)
        plt.xlabel('num of epoch')
        plt.ylabel('The Loss of GAN')

        plt.show()
