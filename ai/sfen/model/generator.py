import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding

LRELU_SLOPE = 0.1

class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv2d(h.latent_space_dim, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock2  

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.conv_strides, h.conv_kernels)):
            self.ups.append(weight_norm(
                nn.ConvTranspose2d(h.conv_filters[0]//(2**i), h.conv_filters[0]//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.conv_filters[0]//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.upsample_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(nn.Conv2d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        print('---------------------------------Generator layers---------------------------------')
        for i in self.ups:
            print(i)
        print()
        print(self.conv_pre)
        print(self.conv_post)
        print("------------------------------------------------------------------------------------")

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 1)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                   padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(nn.Conv2d(channels, channels, kernel_size, 1, dilation=dilation[1],
                                   padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            diff = x.size(2) - xt.size(2)
            if diff > 0:
                # Add padding to the end of xt to match the size of x
                xt = F.pad(xt, (0, diff))
            elif diff < 0:
                # Trim x to match the size of xt
                x = x[:, :, :xt.size(2)]
            x = xt + x
        return x


    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
