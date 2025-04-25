"""Variational AutoEncoder (VAE) model for MNIST dataset.

This module defines a VAE with a simple encoder-decoder architecture using fully connected layers.
The encoder maps input images to a latent Gaussian distribution, and the decoder reconstructs
images from latent samples.
"""

import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    """Variational AutoEncoder for generative modeling.

    Args:
        input_dim (int): Dimension of the input (e.g., 784 for flattened 28x28 MNIST images).
        hidden_dim (int): Dimension of the hidden layer (default: 200).
        output_dim (int): Dimension of the latent space (default: 20).
    """
    def __init__(self, input_dim, hidden_dim=200, output_dim=20):
        super().__init__()
        # Encoder layers
        self.img_2hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, output_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, output_dim)

        # Decoder layers
        self.z_2hid = nn.Linear(output_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        """Encode input to latent distribution parameters.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            tuple: (mu, sigma), where mu and sigma are tensors of shape [batch_size, output_dim]
                   representing the mean and standard deviation of the latent distribution.
        """
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        return mu, sigma

    def decode(self, z):
        """Decode latent samples to reconstructed inputs.

        Args:
            z (torch.Tensor): Latent tensor of shape [batch_size, output_dim].

        Returns:
            torch.Tensor: Reconstructed tensor of shape [batch_size, input_dim] with values in [0,1].
        """
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        """Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            tuple: (x_reconstructed, mu, sigma), where:
                - x_reconstructed: Reconstructed input, shape [batch_size, input_dim].
                - mu: Latent mean, shape [batch_size, output_dim].
                - sigma: Latent standard deviation, shape [batch_size, output_dim].
        """
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
