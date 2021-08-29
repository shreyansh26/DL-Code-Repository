import argparse
import os

import torch
import torch.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image

from model import Discriminator, Generator

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda') if CUDA_AVAILABLE else torch.device('cpu')


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
configs = parser.parse_args()
print(configs)

img_shape = (configs.channels, configs.img_size, configs.img_size)

# Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discrimiator
generator = Generator(configs, img_shape)
discriminator = Discriminator(configs, img_shape)

generator.to(DEVICE)
discriminator.to(DEVICE)
adversarial_loss.to(DEVICE)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(configs.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=configs.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=configs.lr, betas=(configs.b1, configs.b2))

Tensor = torch.cuda.FloatTensor if CUDA_AVAILABLE else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(configs.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        real = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as input to generator
        z = Variable(Tensor(torch.randn((imgs.shape[0], configs.latent_dim), device=DEVICE)))

        # Generate a batch of imgs
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), real)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated images
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)  # gen_imgs is detached to preveent training generator
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, configs.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % configs.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)