import torch
import torch.nn as nn
import torch.nn.functional as F
from model.generator import Generator
class VAE(nn.Module):
    def __init__(self, h):
        super(VAE, self).__init__()

        self.input_shape = [1, h.num_mels, 32]
        self.conv_filters = h.conv_filters
        self.conv_kernels = h.conv_kernels
        self.conv_strides = h.conv_strides
        self.latent_space_dim = h.latent_space_dim
        self.reconstruction_loss_weight = h.lambda_sf

        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.encoder_conv_output_size, h.latent_space_dim)
        self.fc_logvar = nn.Linear(self.encoder_conv_output_size, h.latent_space_dim)
        self.decoder = self._build_decoder(h)

    def _build_encoder(self):
        layers = []
        in_channels = self.input_shape[0]  # Should be 1 for grayscale input
        print(in_channels, "input channels")
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())

        print('---------------------------------Encoder layers---------------------------------')
        for i in layers:
            print(i)
        print()

        self.encoder_conv_output_size = self._get_conv_output_size(layers)
        return nn.Sequential(*layers)

    def _build_decoder(self, h):
        return Generator(h=h)

    def _get_conv_output_size(self, layers):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)  # [1, 1, 80, 29]
            print("encoder convolution input shape:", dummy_input.size())
            dummy_output = nn.Sequential(*layers)(dummy_input)
            print("encoder convolution output shape:", dummy_output.size(), end='\n\n')
            return dummy_output.view(1, -1).size(1)

    def encode(self, x):
        print("input shape:", x.size())
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
        # Add a channel dimension
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, num_mels, spec_frames]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
