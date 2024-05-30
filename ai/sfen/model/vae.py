import torch
import torch.nn as nn
import torch.nn.functional as F

from generator import Generator
class VAE(nn.Module):
    def __init__(self, h):
        super(VAE, self).__init__()

        # TODO: Add class attributes to h
        self.input_shape = h.input_shape
        self.conv_filters = h.conv_filters
        self.conv_kernels = h.conv_kernels
        self.conv_strides = h.conv_strides
        self.latent_space_dim = h.latent_space_dim
        self.reconstruction_loss_weight = h.lambda_sf

        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.encoder_conv_output_size, h.latent_space_dim)
        self.fc_logvar = nn.Linear(self.encoder_conv_output_size, h.latent_space_dim)
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        layers = []
        in_channels = self.input_shape[0]
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.encoder_conv_output_size = self._get_conv_output_size(layers)
        return nn.Sequential(*layers)

    def _build_decoder(self):
        return Generator(h=self)

    def _get_conv_output_size(self, layers):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            dummy_output = nn.Sequential(*layers)(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def encode(self, x):
        conv_out = self.encoder(x)
        mu = self.fc_mu(conv_out)
        logvar = self.fc_logvar(conv_out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
