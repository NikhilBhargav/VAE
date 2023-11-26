"""
" Code to implement Variational Auto Encoder from scratch using Pytorch
"""
import torch
import torch.nn.functional as F
from torch import nn


class VariationalAutoEncoder(nn.Module):
    """
    VAE Class

    input_dim-->latent_dim--> N(mean, std) -->Parametrization Trick -->Decoder --> Sample Output image
    """
    def __init__(self, input_dim, hidden_dim=200, z_dim=20):
        super(VariationalAutoEncoder, self).__init__()
        ''' Encoder '''
        self.img_2_hidden = nn.Linear(input_dim, hidden_dim)
        # Standard Guassian
        self.hidden_2_mu = nn.Linear(hidden_dim, z_dim)
        self.hidden_2_sigma = nn.Linear(hidden_dim, z_dim)

        ''' Decoder '''
        self.z_2_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_2_image = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        """
        Encodes the given input x to latent variable z

        :param x:
        :return:
        """
        # q_phi(z|x)
        h = self.relu(self.img_2_hidden(x))
        mu = self.hidden_2_sigma(h)
        sigma = self.hidden_2_mu(h)

        return mu, sigma

    def decode(self, z):
        """
        Reconstruct x from to given latent variable z

        :param z:
        :return:
        """
        # p_theta(x|z)
        h = self.relu(self.z_2_hidden(z))
        img = self.hidden_2_image(h)

        # For MNIST normalize images b/w 0 and 1
        return torch.sigmoid(img)

    def forward(self, x):
        """
        Forward propagation to first encode the image to latent and
        then decode the latent to reconstruct the image

        :param x:
        :return:
        """
        mu, sigma = self.encode(x)

        # Sample standard normal distribution
        epsilon = torch.randn_like(sigma)

        # Re-parametrize z so that it is Lipschitz continuous
        z_repr = mu + sigma * epsilon
        x_reconstructed = self.decode(z_repr)

        # We need reconstructed image for reconstruction loss and mu, sigma for KL-Divergence
        return x_reconstructed, mu, sigma


def main():
    """
    Main entry point

    :return:
    """
    x = torch.randn(4, 28*28)
    vae = VariationalAutoEncoder(input_dim=784)
    x_prime, mu, sigma = vae(x)

    # Check the shape
    print(x_prime.shape, mu.shape, sigma.shape)


if __name__ == '__main__':
    main()
