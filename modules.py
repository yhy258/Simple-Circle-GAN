import torch
import torch.nn as nn

class ResidualBlock(nn.Module):  # First or non first
    def __init__(self, dim, stride=2, upsample=False, downsample=False, is_first=False):  # pooling : stride
        super().__init__()
        self.pooling = None
        self.first = None

        if is_first:
            self.main = nn.Sequential(
                nn.Conv2d(3, dim, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, 1, 1),
            )
            self.first = nn.Conv2d(3, dim, 3, 1, 1)
            self.pooling = nn.AvgPool2d(kernel_size= 3, stride =2, padding =1)
        else:
            # self.main = nn.Sequential(
            #     nn.BatchNorm2d(dim),
            #     nn.ReLU(),
            #     nn.Conv2d(dim, dim, 3, 1, 1),
            #     nn.BatchNorm2d(dim),
            #     nn.ReLU(),
            #     nn.Conv2d(dim, dim, 3, 1, 1),
            # )
            self.main = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, 1, 1),
            )
            if upsample:
                self.pooling = nn.Upsample(scale_factor=stride, mode='bilinear')
            if downsample:
                self.pooling = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        shortcut = x
        if self.first:
            shortcut = self.first(shortcut)
        x = self.main(x)
        if self.pooling:
            x = self.pooling(x)
            shortcut = self.pooling(shortcut)
        x += shortcut
        x = self.dropout(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, dim, DEVICE):
        super(Discriminator, self).__init__()
        downsamples = [True, True, True, False]
        self.module_list = nn.ModuleList()
        layers = self.make_block(dim, downsamples)
        for layer in layers:
            self.module_list.append(layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pivotal = torch.empty(1,dim).clone().detach().requires_grad_(True).to(DEVICE)

        torch.nn.init.orthogonal_(self.pivotal)

    def forward(self, x):
        bs = x.size(0)
        for module in self.module_list:
            x = module(x)
        x = self.avgpool(x)
        x = x.view(bs, -1)
        return x, self.pivotal

    def make_block(self, dim, downsamples):
        firsts = [True, False, False, False]
        layers = []
        for downsample, first in zip(downsamples, firsts):
            layers.append(ResidualBlock(dim, 2, upsample=False, downsample=downsample, is_first=first))
        return layers


class Generator(nn.Module):
    def __init__(self, hid_dim, dim):
        super(Generator, self).__init__()
        self.linear = nn.Linear(hid_dim, 4096)

        upsamples = [True, True, True]

        self.module_list = nn.ModuleList()
        layers = self.make_block(dim, upsamples)
        for layer in layers:
            self.module_list.append(layer)
        self.last = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        bs = x.size(0)
        x = self.linear(x)
        x = x.view(bs, -1, 4, 4)
        for module in self.module_list:
            x = module(x)
        x = self.last(x)

        return x

    def make_block(self, dim, upsamples):
        layers = []
        for upsample in upsamples:
            layers.append(ResidualBlock(dim, 2, upsample=upsample, downsample=False, is_first=False))

        return layers


