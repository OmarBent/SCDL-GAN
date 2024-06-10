import argparse
import numpy as np
import torch
import torch.autograd as autograd
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
#from libgan.models.__init__ import get_model
from .__init__ import get_model
import torch.nn as nn
import os
import scipy.io as sio
from tensorboardX import SummaryWriter


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples, labels = None):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(FloatTensor(real_samples.shape[0], real_samples.shape[1]).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates[0],
        inputs=(interpolates),
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Data(Dataset):
    def __init__(self, data, data_labels=None, transform=None):
        self.transform = transform
        self.train_data = data
        self.train_labels = data_labels

    def __getitem__(self, index):
        img = self.train_data[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.train_labels is not None:
            label = self.train_labels[index]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.train_data)


def train(opt, codes, labels):

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Configure data loader
    nb_sequences = len(codes)
    features_tensor = torch.empty(nb_sequences, 1, opt.feature_dim + opt.transformation_dim, opt.seq_length)
    for i in range(0, len(codes)):
        features_tensor[i, 0, :, :] = torch.from_numpy(codes[i])

    labels_tensor = torch.from_numpy(labels)

    dataset = Data(features_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.PG_batch_size, shuffle=True)

    # Setup Model
    model = get_model(opt)

    if cuda:
        model.generator.cuda()
        model.discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = model.generator(z)

            # Real images
            real_validity = model.discriminator(real_imgs)
            # Fake images
            fake_validity = model.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(model.discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = model.generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = model.discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # Logger
                #num_batches = len(dataloader) / opt.PG_batch_size
                #logger.log(d_loss, g_loss, epoch, opt.PG_batch_size, num_batches)

                # batches_done = epoch * len(dataloader) + i
                if epoch == opt.n_epochs-1:
                    z_gen = Variable(Tensor(np.random.normal(0, 1, (opt.n_gen, opt.latent_dim))))
                    fake_seq = model.generator(z_gen)
                    for j in range(opt.n_gen):
                        filename = opt.output_dir+'/sequence_{}.txt'.format(str(j+1))
                        if cuda:
                            fake_data = fake_seq.data[j, 0, :, :].cpu().numpy()
                            np.savetxt(filename, fake_data, fmt='%.4f')
                batches_done += opt.n_critic

    #sio.savemat(opt.output_dir + '/sequences.mat', {'sequences': fake_seq})

    return model


def con_train(opt, codes, labels):

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Configure data loader
    nb_sequences = len(codes)
    features_tensor = torch.empty(nb_sequences, 1, opt.feature_dim + opt.transformation_dim, opt.seq_length)
    for i in range(0, len(codes)):
        features_tensor[i, 0, :, :] = torch.from_numpy(codes[i])

    labels_tensor = torch.from_numpy(labels)

    dataset = Data(features_tensor, labels_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.PG_batch_size, shuffle=True)

    # Setup Model
    model = get_model(opt)

    if cuda:
        model.generator.cuda()
        model.discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # ----------
    #  Training
    # ----------

    # Logging
    writer = SummaryWriter()

    batches_done = 0
    for epoch in range(opt.n_epochs):
        print('-' * 115)

        for i, (imgs, real_labels) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            # We add - if labels start from 1
            real_labels = Variable(real_labels.type(LongTensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            fake_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, imgs.shape[0])))

            # Generate a batch of images
            fake_imgs = model.generator(z, real_labels)

            # Train with real images
            real_validity, gen_labels = model.discriminator(real_imgs)
            aux_error_real = criterion(gen_labels, real_labels).mean()

            # Train with fake images
            fake_validity, fake_lab = model.discriminator(fake_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(model.discriminator, real_imgs.data, fake_imgs.data)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty \
                     + opt.C_SCALE * aux_error_real

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------
                # Generate a batch of images
                fake_imgs = model.generator(z, fake_labels)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity, gen_labels = model.discriminator(fake_imgs)
                aux_error_fake = criterion(gen_labels, fake_labels).mean()

                g_loss = -torch.mean(fake_validity) + opt.G_SCALE * aux_error_fake

                # Total number of labels
                total = fake_labels.size(0)

                # Total correct predictions

                correct = (torch.argmax(gen_labels, dim=1) == fake_labels).sum()

                # Calculate Accuracy
                accuracy = int(correct) / total

                g_loss.backward()
                optimizer_G.step()

                log = "[Epoch {:6d}/{:6d}] [Batch {:2d}/{:2d}] [D loss: {:6.3f}] [G loss: {:6.3f}] " \
                      "[Test accuracy: {:6.3f}] [CGAN g loss: {:6.3f}]".format(epoch, opt.n_epochs, i, len(dataloader),
                                                                               d_loss.item(), g_loss.item(), accuracy,
                                                                               aux_error_fake)

                print(log)

                # Logging
                writer.add_scalar('data/d_loss', d_loss, epoch)
                writer.add_scalar('data/g_loss', g_loss, epoch)
                writer.add_scalar('data/cgan_d_loss', aux_error_real, epoch)
                writer.add_scalar('data/cgan_g_loss', aux_error_fake, epoch)
                writer.add_scalar('data/gradient_penalty', gradient_penalty, epoch)
                writer.add_scalar('data/accuracy', accuracy, epoch)

                # torch.save(model.generator, opt.checkpoint + "generator_latest.pt")
                # torch.save(model.discriminator, opt.checkpoint + "discriminator_latest.pt")

                # ----------------
                #  Save data
                # ----------------

                if epoch == opt.n_epochs-1:
                    z_gen = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_gen, opt.latent_dim))))
                    # ----------------
                    #  Save data
                    # ----------------

                    if epoch == opt.n_epochs - 1:
                        z_gen = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_gen * opt.n_classes, opt.latent_dim))))
                        fake_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, opt.n_gen*opt.n_classes)))
                        idx = 0
                        for j in range(opt.n_classes):
                            for h in range(opt.n_gen):
                                fake_labels[idx] = j
                                idx += 1

                        fake_seq = model.generator(z_gen, fake_labels)



                    # save fake sequences
                    for j in range(opt.n_gen):
                        filename = opt.output_dir+'/sequences/sequence_{}.txt'.format(str(j+1))
                        if cuda:
                            fake_data = fake_seq.data[j, 0, :, :].cpu().numpy()
                            np.savetxt(filename, fake_data, fmt='%.4f')

                    # save fake labels
                    np.savetxt(opt.output_dir+'/fake_labels.txt', fake_labels.data.cpu().numpy())

                batches_done += opt.n_critic

    return model
if __name__ == '__main__':
    # Configure data loader
    os.makedirs('Results/taichi_SC', exist_ok=True)
    # Logger
    #logger = Logger(model_name='IWGAN_Generation_sc', data_name='ucf')

    # Configure data loader
    # ----------------- data loader -----------------
    features = sio.loadmat('taichi_codes_SC.mat')['taichi_codes_SC']
    nb_sequences = len(features)

    # Keep the last sequence for test
    codes = features[:nb_sequences].squeeze()
    dim = codes[0].shape

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', nargs='?', type=str, default='IWGAN',
                        help='Architecture to use [\'IWGAN, iwcgan \']')
    parser.add_argument("--output_dir", type=str, default='/generated_sequences', help="save directory")
    parser.add_argument("--n_epochs", type=int, default=50000, help="number of epochs of training")
    parser.add_argument("--n_gen", type=int, default=500, help="number of sequences to generate at last epoch")
    parser.add_argument("--PG_batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--feature_dim", type=int, default=dim[0], help="size of each image dimension")
    parser.add_argument("--seq_length", type=int, default=dim[1], help="size of each image dimension")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)

    train(opt, codes)
