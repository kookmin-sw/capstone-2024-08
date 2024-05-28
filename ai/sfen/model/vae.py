import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        layers = []
        in_channels = self.input_shape[0]  # num_mels

        for out_channels, kernel_size, stride in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2))
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

    def _build_decoder(self):
        layers = []

        # Initial dense layer
        layers.append(nn.Linear(self.latent_space_dim, self.encoder_conv_output_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(self.encoder_conv_output_size))
        layers.append(nn.Unflatten(1, (self.conv_filters[-1], 10, 4)))

        # Transposed Conv2D layers
        layers.append(nn.ConvTranspose2d(self.conv_filters[-1], self.conv_filters[-2], self.conv_kernels[-1]+1, self.conv_strides[0], padding=self.conv_kernels[-1]//2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(self.conv_filters[-2]))

        layers.append(nn.ConvTranspose2d(self.conv_filters[-2], self.conv_filters[-3], self.conv_kernels[-2], self.conv_strides[1], padding=self.conv_kernels[-2]//2, output_padding=(1, 0)))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(self.conv_filters[-3]))

        layers.append(nn.ConvTranspose2d(self.conv_filters[-3], self.conv_filters[-4], self.conv_kernels[-3], self.conv_strides[2], padding=self.conv_kernels[-3]//2, output_padding=(1, 0)))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(self.conv_filters[-4]))

        # Last ConvTranspose2D layer with adjusted output_padding
        layers.append(nn.ConvTranspose2d(self.conv_filters[-4], self.input_shape[0], self.conv_kernels[-4], self.conv_strides[3], padding=self.conv_kernels[-4]//2))
        layers.append(nn.Sigmoid())

        print('---------------------------------Decoder layers---------------------------------')
        for i in layers:
            print(i)
        print()

        return nn.Sequential(*layers)

    def _get_conv_output_size(self, layers):
        with torch.no_grad():
            dummy_input = torch.zeros(*self.input_shape)
            dummy_input = dummy_input.unsqueeze(0)
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
        # Shape: [batch_size, num_mels, spec_frames]
        print(x.size(), "input shape")
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        print(x_recon.size(), "output shape")
        return x_recon, mu, logvar
