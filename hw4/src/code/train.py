#%matplotlib inline
import argparse
import os
import sys
import ipdb
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython import embed

from models import Generator, Discriminator
from data import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Set random seem for reproducibility
manualSeed = 999 # random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if USE_CUDA:
    torch.cuda.manual_seed_all(manualSeed)


class GANtrainer(object):
    """docstring for trainer"""
    def __init__(self, args, dataset):
        super(GANtrainer, self).__init__()
        self.dataroot = "./data/selected_cartoonset100k"
        self.outroot = "./out"
        self.modelroot = "./model"
        self.workers = 2
        self.batch_size = 32
        self.image_size = 64

        self.nc = 3 # Number of channels in the training images. For color images this is 3
        self.nz = 100 # Size of z latent vector (i.e. size of generator input)
        self.ngf = 64 # Size of feature maps in generator
        self.ndf = 64 # Size of feature maps in discriminator
        self.num_epochs = 5 # Number of training epochs
        self.lr = 2e-5 # Learning rate for optimizers
        self.beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
        self.ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=self.workers)

    def plot_test(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imsave('./out/test.png', np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


    def init_model(self):
        self.netG = Generator(self.ngpu, self.nz, self.ngf, self.nc).to(device)
        if (device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
        print(self.netG)

        self.netD = Discriminator(self.ngpu, self.nc, self.ndf).to(device)
        if (device.type == 'cuda') and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
        print(self.netD)

        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=device)

        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    def save_models(self, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth'.format(self.modelroot, epoch))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth'.format(self.modelroot, epoch))


    def train(self):
        # Training Loop
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                img, feat = data['image'], data['feature']
                embed()
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data['image'].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, device=device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print("")
                print('\r[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, self.num_epochs, i, len(self.dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), end="")

                # Save Losses for plotting later
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1


def main(args):
    dataset = Cartoonset100kDataset(attr_txt="./data/selected_cartoonset100k/cartoon_attr.txt",
                                    root_dir="./data/selected_cartoonset100k/images/",
                                    transform=transforms.Compose([
                                              # transforms.Resize(64),
                                              # transforms.CenterCrop(64),
                                              # transforms.Rescale(256),
                                              # transforms.RandomCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    trainer = GANtrainer(args, dataset)
    trainer.init_model()
    trainer.plot_test()
    trainer.train()


if __name__ == '__main__':
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        print(device)
        parser = argparse.ArgumentParser()
        parser.add_argument('-dp', '--data_path', type=str, default='../data')
        parser.add_argument('-tp', '--test_path', type=str, default='../data/test.csv')
        parser.add_argument('-e', '--epochs', type=int, default=10)
        parser.add_argument('-b', '--batch_size', type=int, default=16)
        parser.add_argument('-hn', '--hidden_size', type=int, default=512)
        parser.add_argument('-lr', '--lr_rate', type=float, default=1e-5)
        parser.add_argument('-dr', '--drop_p', type=float, default=0.5)
        parser.add_argument('-md', '--model_dump', type=str, default='./bert_model_ver3.tar')
        parser.add_argument('-ml', '--model_load', type=str, default=None, help='Model Load')
        parser.add_argument('-p', '--print_iter', type=int, default=271, help='Print every p iterations')
        parser.add_argument('-mc', '--max_count', type=int, default=40)
        parser.add_argument('-tr', '--train', type=int, default=1)
        parser.add_argument('-o', '--output_csv', type=str, default="out.csv")
        parser.add_argument('-l', '--max_length', type=int, default=64, help='Max sequence length')
        parser.add_argument('-bert', '--bert_type', type=str, default='bert-large-uncased', help='Select Bert Type')
        args = parser.parse_args()
        main(args)
