#%matplotlib inline
import os
import random
import itertools

import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.autograd as autograd
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
# import matplotlib.animation as animation
# from IPython.display import HTML

from argument import USE_CUDA, device
from models import Generator, Discriminator

class GANtrainer(object):
    def __init__(self, args):
        super(GANtrainer, self).__init__()
        self.dataroot = "./data/selected_cartoonset100k"
        self.outroot = "./out"
        self.modelroot = "./model"
        self.model_load = "./model/models_ep65.tar"
        self.workers = 2
        self.batch_size = 16
        self.image_size = 64
        self.n_class = 15
        self.LAMBDA = 10

        self.nc = 3 # Number of channels in the training images. For color images this is 3
        self.nz = 200 # Size of z latent vector (i.e. size of generator input)
        self.ngf = 64 # Size of feature maps in generator
        self.ndf = 64 # Size of feature maps in discriminator
        self.num_epochs = 100 # Number of training epochs
        self.lr = 2e-4 # Learning rate for optimizers
        self.beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
        self.ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.

    def generating_conditions(self):
        num_feat = [6, 4, 3, 2]
        combin = None
        for n in num_feat:
            if combin:
                combin = list(itertools.product(combin, np.eye(n).tolist()))
                combin = [a+b for a,b in combin]
            else:
                combin = np.eye(n).tolist()
        self.combin = combin

    def plot_test(self):
        real_batch = next(iter(self.dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imsave('./out/test.png', np.transpose(vutils.make_grid(real_batch['image'].to(device)[:64], padding=2, normalize=False).cpu(),(1,2,0)))

    def init_dataset(self, dataset):
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=self.workers)

    def init_model(self):
        self.netG = Generator(self.ngpu, self.nz, self.ngf, self.nc).to(device)
        if (device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
        print(self.netG)

        self.netD = Discriminator(self.ngpu, self.nc, self.ndf, self.n_class).to(device)
        if (device.type == 'cuda') and (self.ngpu > 1):
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
        print(self.netD)

        # self.criterion = nn.BCELoss()
        self.bce = nn.BCELoss().cuda()
        self.cep = nn.BCEWithLogitsLoss().cuda()

        self.fixed_noise = torch.randn(1, self.nz, device=device)

        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        if not os.path.exists('images/fake'):
            os.makedirs('images/fake')
        if not os.path.exists('images/real'):
            os.makedirs('images/real')
        if not os.path.exists('test_output'):
            os.makedirs('test_output')

    def denorm(self, x):
        return x * 0.5 + 0.5

    def save_models(self, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth'.format(self.modelroot, epoch))
        torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth'.format(self.modelroot, epoch))

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'netG': self.netG.state_dict(),
            'netD': self.netD.state_dict(),
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
        }, os.path.join(self.modelroot, f"models_ep{epoch}.tar"))

    def load_model(self, ckptname):
        print("> Loading..")
        ckpt = torch.load(ckptname)
        self.netG.load_state_dict(ckpt['netG'])
        self.netD.load_state_dict(ckpt['netD'])

    def test(self, predict, labels):
        pred = predict.data.max(1)[1]
        correct = pred.eq(labels.data).cpu().sum()
        return correct, len(labels.data)

    def calc_gradient_penalty(self, real_data, fake_data, real_label):
        batch_size = real_data.size()[0]
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous().view(batch_size, 3, 128, 128)
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, real_label)
        _disc_interpolates = disc_interpolates[0]

        gradients = autograd.grad(outputs=_disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(_disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def train(self):
        # Training Loop
        self.netG.train()
        self.netD.train()
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        iters = 0

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            # self.save_model(epoch)
            for i, data in enumerate(self.dataloader):
                image, label = data['image'], data['feature']
                if image.size()[0] != self.batch_size:
                    continue
                real_input = Variable(image).cuda()
                real_label = Variable(label).float().cuda()
                real_ = Variable(torch.ones(real_label.size()[0])).cuda()

                noise = Variable(torch.Tensor(self.batch_size, self.nz).normal_(0, 1)).cuda()
                fake_label = real_label
                fake_ = Variable(torch.zeros(fake_label.size()[0])).cuda()

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                fake_input = self.netG(noise, fake_label)

                real_pred, real_cls = self.netD(real_input, real_label)
                fake_pred, fake_cls = self.netD(fake_input.detach(), fake_label)
                correct, length = self.test(real_cls, real_label.max(1)[1])

                real_loss = self.bce(nn.Sigmoid()(real_pred), real_) + self.cep(real_cls, real_label)
                fake_loss = self.bce(nn.Sigmoid()(fake_pred), fake_) + self.cep(fake_cls, fake_label)

                gradient_penalty = self.calc_gradient_penalty(real_input.data, fake_input.data, real_label)

                self.optimizerD.zero_grad()
                d_loss = real_loss + fake_loss + gradient_penalty
                d_loss.backward()
                self.optimizerD.step()
                D_x = real_pred.data.mean()
                D_G_z1 = fake_pred.data.mean()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.optimizerG.zero_grad()
                fake_pred, fake_cls = self.netD(fake_input, fake_label)
                real_ = Variable(torch.ones(fake_label.size()[0])).cuda()
                g_loss = self.bce(nn.Sigmoid()(fake_pred), real_) + self.cep(fake_cls, fake_label)
                g_loss.backward()
                self.optimizerG.step()
                D_G_z2 = fake_pred.data.mean()

                # # Output training stats
                if i % 3000 == 0:
                    print("")
                print('\r [{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f} / {:.4f}, Accuracy: {}/{} = {:.2f}%'.format(
                    epoch+1, self.num_epochs, i+1, len(self.dataloader), d_loss.data, g_loss.data,
                    D_x, D_G_z1, D_G_z2, correct, length, 100.* correct / length), end=""
                )

                # Save Losses for plotting later
                self.G_losses.append(g_loss.data)
                self.D_losses.append(d_loss.data)

                # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i >= len(self.dataloader)-3)):
                #     with torch.no_grad():
                #         fake = self.netG(self.fixed_noise).detach().cpu()
                #     self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            vutils.save_image(self.denorm(fake_input.data), f'images/fake/fake_{epoch:03d}.jpg')
            vutils.save_image(self.denorm(real_input.data), f'images/real/real_{epoch:03d}.jpg')

            if (epoch+1) % 5 == 0:
                self.save_model(epoch+1)

    def gen_output(self, iters=5):
        self.load_model(self.model_load)
        self.generating_conditions()
        output_image = []
        count = 0
        with torch.no_grad():
            self.netG.eval()
            for _ in range(iters):
                for label in self.combin:
                    fixed_noise = torch.randn(1, self.nz, device=device)
                    label = Variable(torch.Tensor(label)).float().unsqueeze(0).to(device)
                    out = self.netG(fixed_noise, label)
                    vutils.save_image(self.denorm(out.data), f'test_output/{count}.png')
                    count+=1
                    if count >= 5000:
                        exit()

