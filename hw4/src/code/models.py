import torch
import torch.nn as nn
from torch.autograd import Variable
from IPython import embed

from argument import USE_CUDA, device
from spectral_normalization import SpectralNorm

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_sn(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight_bar.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.ngf = ngf
        self.proj = nn.Linear(nz, ngf*4*4)
        self.label_proj = nn.Linear(15, ngf*8)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf*8, 4, 2, 1),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8*2, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf*1, 4, 2, 1),
            nn.BatchNorm2d(ngf*1),
            nn.ReLU(True),
        )

        self.conv5 = nn.ConvTranspose2d(ngf*1, nc, 4, 2, 1)

        self.tanh = nn.Tanh()

        self.apply(weights_init)


    def forward(self, input, label):
        batch_size = input.size(0)

        x = self.proj(input)
        x = x.view(batch_size, -1, 4, 4)
        label = self.label_proj(label)
        label = label.unsqueeze(2).expand(batch_size, self.ngf*8, 64).view(batch_size, self.ngf*8, 8, 8)

        x = self.conv1(x)
        x = torch.cat([x, label], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        output = self.tanh(x)

        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, num_classes=15):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.label_proj = nn.Linear(15, ndf*8)
        self.proj = nn.Linear(ndf*4*4, ndf)


        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf*8*2, ndf*1, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(weights_init)

    def forward(self, input, label):
        batch_size = input.size(0)

        label = self.label_proj(label)
        label = label.unsqueeze(2).expand(batch_size, self.ndf*8, 64).view(batch_size, self.ndf*8, 8, 8)

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.cat([x, label], dim=1)
        x = self.conv5(x)

        x = x.view(batch_size, -1)
        x = self.proj(x)
        s = self.gan_linear(x)
        c = self.aux_linear(x)

        return s.squeeze(1), c

class SNDiscriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf, num_classes=15):
        super(SNDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.label_proj = nn.Linear(15, ndf*8)
        self.proj = nn.Linear(ndf*4*4, ndf)

        self.conv1 = SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False))
        self.conv3 = SpectralNorm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False))
        self.conv4 = SpectralNorm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False))
        self.conv5 = SpectralNorm(nn.Conv2d(ndf*8*2, ndf*1, 4, 2, 1, bias=False))

        self.gan_linear = nn.Linear(ndf * 1, 1)
        self.aux_linear = nn.Linear(ndf * 1, num_classes)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.apply(weights_init_sn)

    def forward(self, input, label):
        batch_size = input.size(0)

        label = self.label_proj(label)
        label = label.unsqueeze(2).expand(batch_size, self.ndf*8, 64).view(batch_size, self.ndf*8, 8, 8)

        x = self.lrelu(self.conv1(input))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.cat([x, label], dim=1)
        x = self.conv5(x)

        x = x.view(batch_size, -1)
        x = self.proj(x)
        s = self.gan_linear(x)
        c = self.aux_linear(x)

        return s.squeeze(1), c

if __name__ == '__main__':
    inp = Variable(torch.Tensor(32, 3, 128, 128).normal_(0, 1))
    noise = Variable(torch.Tensor(32, 64, 4, 4).normal_(0, 1))
    netG = Generator(1, 100, 64, 3)
    netD = Discriminator(1, 3, 64)
    embed()