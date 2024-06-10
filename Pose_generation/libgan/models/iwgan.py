import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class iwgan:
    def __init__(self, latent_dim, feature_dim, transformation_dim, seq_length):
        super(iwgan, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.transformation_dim = transformation_dim

        self.generator = Generator(latent_dim=self.latent_dim, feature_dim=self.feature_dim, transformation_dim=self.transformation_dim, seq_length=self.seq_length)
        self.discriminator = Discriminator(feature_dim=self.feature_dim + self.transformation_dim, seq_length=self.seq_length)


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim, transformation_dim, seq_length):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.transformation_dim = transformation_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.transformation_model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, int(transformation_dim * seq_length)),
            nn.Sigmoid()
        )

        self.codes_model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, int(feature_dim * seq_length)),
        )

    def forward(self, z):
        codes = self.codes_model(z)
        codes = codes.view(codes.shape[0], *(1, self.feature_dim, self.seq_length))
        codes = F.softmax(codes, dim=2)

        transformations = self.transformation_model(z)
        transformations = transformations.view(transformations.shape[0], *(1, self.transformation_dim, self.seq_length))
        img = torch.cat((codes, transformations), 2)

        return img


class Discriminator(nn.Module):
    def __init__(self, feature_dim, seq_length):
        super(Discriminator, self).__init__()

        self.feature_dim = feature_dim
        self.seq_length = seq_length

        self.model = nn.Sequential(
            nn.Linear(int((self.feature_dim*self.seq_length)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, codes):
        codes_flat = codes.view(codes.shape[0], -1)
        validity = self.model(codes_flat)

        return validity
