import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class con_iwgan:
    def __init__(self, n_classes, latent_dim, feature_dim, transformation_dim, seq_length):
        super(con_iwgan, self).__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.seq_length = seq_length
        self.transformation_dim = transformation_dim

        self.generator = Generator(n_classes=self.n_classes, latent_dim=self.latent_dim, feature_dim=self.feature_dim,
                                   transformation_dim=self.transformation_dim, seq_length=self.seq_length)

        self.discriminator = Discriminator(n_classes= self.n_classes, feature_dim=self.feature_dim + self.transformation_dim,
                                           seq_length=self.seq_length)


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim, feature_dim, transformation_dim, seq_length):
        super(Generator, self).__init__()

        self.n_classes = n_classes
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

        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        self.transformation_model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, int(transformation_dim * seq_length)),
            nn.Sigmoid()
        )

        self.codes_model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            nn.Linear(256, int(self.feature_dim * self.seq_length)),
        )

    def forward(self, z, labels):
        aa = self.label_emb(labels)
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        codes = self.codes_model(gen_input)
        codes = codes.view(codes.shape[0], *(1, self.feature_dim, self.seq_length))
        codes = F.softmax(codes, dim=2)

        transformations = self.transformation_model(gen_input)
        transformations = transformations.view(transformations.shape[0], *(1, self.transformation_dim, self.seq_length))
        img = torch.cat((codes, transformations), 2)

        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, feature_dim, seq_length):
        super(Discriminator, self).__init__()

        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.seq_length = seq_length

        self.model1 = nn.Sequential(
            nn.Linear(int((self.feature_dim*self.seq_length)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

        self.model2 = nn.Sequential(
            nn.Linear(int((self.feature_dim*self.seq_length)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.n_classes)
        )

        self.linear1 = nn.Linear(256, 1)
        self.linear2 = nn.Linear(256, self.n_classes)

    def forward(self, codes):
        # Concatenate label embedding and image to produce input
        codes_flat = codes.view(codes.size(0), -1)
        validity = self.model1(codes_flat)

        #validity = self.linear1(output)
        #validity = validity.view(-1)
        gen_labels = self.model2(codes_flat)

        return validity, gen_labels
