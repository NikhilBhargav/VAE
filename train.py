"""
Code to train my Variational Auto Encoder
on MNIST image dataset (greyscale digit images)
using Pytorch
"""

""" System Module """
import torch

# For Standard datasets
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
import os
# For easier DS management
from torch.utils.data import DataLoader

""" User module """
from model import VariationalAutoEncoder


class Config:
    """ Configuration """

    def __init__(self, input_dim, h_dim, z_dim, num_epochs, batch_size, lr):
        """
        Configure my module with these param/hyp
        :param input_dim:
        :param h_dim:
        :param z_dim:
        :param num_epochs:
        :param batch_size:
        :param lr:
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_dim = input_dim
        self._h_dim = h_dim
        self._z_dim = z_dim
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        # Andrew Karpathy constant
        self._lr = lr

    @property
    def h_dim(self):
        return self._h_dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def z_dim(self):
        return self._z_dim

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def lr(self):
        return self._lr

    @property
    def num_epochs(self):
        return self._num_epochs


def inference(model, dataset, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0

    # Get all images
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []

    # Get mu and sigma
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]

    # Generate a new image
    tmp = os.path.join(Path(__file__).resolve().parent, "output")
    os.makedirs(tmp, exist_ok=True)

    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        name = f"generated_{digit}_ex{example}.png"
        save_image(out, os.path.join(tmp, name))


def main():
    myconfig = Config(input_dim=784, h_dim=200, z_dim=20, num_epochs=100, batch_size=32, lr=3e-4)

    # Dataset Loading
    dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=dataset, batch_size=myconfig.batch_size, num_workers=2, shuffle=True)
    model = VariationalAutoEncoder(myconfig.input_dim, myconfig.h_dim, myconfig.z_dim).to(myconfig.device)

    optimizer = optim.Adam(model.parameters(), lr=myconfig.lr)

    # Try MCE loss or MSE but here y's (pixel) are just 0 and 1
    # l(x,y) = SUM{ y_n*log(x_n) + (1-y_n)*(log(1-x_n)) }
    loss_fn = nn.BCELoss(reduction="sum")

    # Start training
    for epoch in range(myconfig.num_epochs):
        loop = tqdm(enumerate(train_loader))
        for idx, (x, _) in loop:

            # Forward pass (Keep number of points same as batch size and then flatten all
            x = x.to(myconfig.device).view(x.shape[0], myconfig.input_dim)
            x_reconstructed, mu, sigma = model(x)

            # Compute both Losses
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = - 1 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Back Propagate
            # alpha is lagrange multiplier. You can vary it to weight both losses
            alpha = 0.5
            loss = alpha * reconstruction_loss + (1.0 - alpha) * kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # Do Inference
    model = model.to("cpu")
    for idx in range(10):
        inference(model, dataset, idx, num_examples=5)


if __name__ == '__main__':
    main()
