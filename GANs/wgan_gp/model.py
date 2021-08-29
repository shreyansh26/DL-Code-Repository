import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, configs, img_shape):
        super().__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers += [nn.BatchNorm1d(out_features, momentum=0.1)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            return layers

        self.model = nn.Sequential(
            *block(configs.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img


class Discriminator(nn.Module):
    def __init__(self, configs, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flattened = img.view(img.size(0), -1)
        score = self.model(img_flattened)

        return score