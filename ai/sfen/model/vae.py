import torch
import torch.nn as nn
import torch.nn.functional as F
from model.generator import Generator


class VAE(nn.Module):
    def __init__(self, h):
        super(VAE, self).__init__()

        self.input_shape = [1, h.num_mels, h.spec_split * h.shape]
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
        in_channels = self.input_shape[1]  # num_mels
        print(in_channels, "input channels")
        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        layers.append(nn.Flatten())

        print('---------------------------------Encoder layers---------------------------------')
        for i in layers:
            print(i)
        print()

        self.encoder_conv_output_size = self._get_conv_output_size(layers)
        print("encoder_conv_output_size:", self.encoder_conv_output_size)
        return nn.Sequential(*layers)

    def _build_decoder(self, h):
        return Generator(h=h)

    def _get_conv_output_size(self, layers):
        with torch.no_grad():
            dummy_input = torch.zeros(*self.input_shape)
            print("encoder convolution input shape:", dummy_input.size())
            dummy_output = nn.Sequential(*layers)(dummy_input)
            print("encoder convolution output shape:", dummy_output.size(), end='\n\n')
            return dummy_output.view(1, -1).size(1)
        # encoder convolution input shape: torch.Size([1, 80, 128])
        # encoder convolution output shape: torch.Size([1, 1024])

    def encode(self, x):
        print("input shape:", x.size())
        conv_out = self.encoder(x)
        print("conv_out shape:", conv_out.size())
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
        x = x.squeeze(1)  # Shape: [batch_size, num_mels, spec_frames]
        print(x.size(), "input shape")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = z.unsqueeze(2)  # Add dimension for Conv1d input [batch_size, latent_dim, 1]
        x_recon = self.decode(z)
        return x_recon, mu, logvar
