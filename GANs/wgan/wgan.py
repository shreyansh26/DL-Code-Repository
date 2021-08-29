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
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
configs = parser.parse_args()
print(configs)

img_shape = (configs.channels, configs.img_size, configs.img_size)


# Configure data loader
os.makedirs("../data/mnist", exist_ok=True)
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


# Initialize generator and discriminator
generator = Generator(configs, img_shape)
discriminator = Discriminator(configs, img_shape)

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=configs.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=configs.lr)

Tensor = torch.cuda.FloatTensor if CUDA_AVAILABLE else torch.FloatTensor

# ----------
#  Training
# ----------
batches_done = 0

for epoch in range(configs.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as input to generator
        z = Variable(Tensor(torch.randn((imgs.shape[0], configs.latent_dim), device=DEVICE)))

        # Generate fake images
        gen_imgs = generator(z).detach()

        # Measure discriminator's ability to classify real from generated images
        # Adversarial loss
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(gen_imgs))

        d_loss.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-configs.clip_value, configs.clip_value)

        
        # Train the generator every n_critic iterations
        if i % configs.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of imgs
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            # Adversarial loss
            g_loss = -torch.mean(discriminator(gen_imgs))

            g_loss.backward()
            optimizer_G.step()

        
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, configs.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        if batches_done % configs.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        
        batches_done += 1