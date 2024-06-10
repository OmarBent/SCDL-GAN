# Import
import os
import pylab
import argparse
import numpy as np
from .scdl import scdl
import scipy.io as sio
from .Tools.tsne import tsne
from .resampling import resampling
from .loadfeatures import loadfeatures
from .libgan.models.train_gan import con_train, train
from .Tools.data_utils import reconstruction
from .Tools.data_utils import tsne_data_format
from .Tools.visualize_data import visualize_sequence


def gen_pose(opt):

    # Create files to save SC-DL outputs
    os.makedirs(opt.file_to_save, exist_ok=True)
    os.makedirs(opt.file_to_save+'/codes', exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.output_dir+'/sequences', exist_ok=True)

    # Load data
    sequences = sio.loadmat(opt.dataroot + '/sequences')['sequences'].squeeze() # sequence of shape (nb_frames, nb_landmarks, dim)
    labels = sio.loadmat(opt.dataroot + '/labels')['labels'].squeeze()

    # Re-sample sequences to same length
    s = np.arange(0, 1, 1 / opt.seq_length)
    res_sequences = []
    res_transf_sequences = []
    for i in range(len(sequences)):
        resampled_sequence, resampled_transform = resampling(sequences[i], s)
        res_sequences.append(resampled_sequence)
        # Normalize vectors in the range [0, 1]
        for j in range(np.asarray(resampled_transform[0]).shape[0]):
            min_value = min(np.asarray(resampled_transform[0][j, :]))
            max_value = max(np.asarray(resampled_transform[0][j, :]))
            resampled_transform[0][j, :] = np.true_divide(np.asarray(resampled_transform[0][j, :]) - min_value, max_value - min_value)

        res_transf_sequences.append(resampled_transform)

    # Run SC-DL
    #scdl(np.asarray(res_sequences), opt)

    # Load data and concatenate sparse codes with corresponding filtered transformations
    data = np.zeros((len(sequences), opt.feature_dim + opt.transformation_dim, opt.seq_length))
    data[:, :opt.feature_dim, :] = np.asarray(loadfeatures(opt.file_to_save + '/codes'))
    for i in range(len(sequences)):
        data[i, opt.feature_dim: opt.feature_dim + opt.transformation_dim, :] = res_transf_sequences[i][0]

    # Run iwGAN for skeletal sequence generation
    # TODO save model
    if opt.arch == 'IWGAN':
        model = train(opt, data, labels)
    elif opt.arch == 'IWCGAN':
        model = con_train(opt, data, labels)

    # Apply t_sne for visualization
    codes = loadfeatures(opt.output_dir+'\sequences')
    x, y = tsne_data_format(codes, data, opt)
    Y = tsne(x, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, y)
    pylab.show()

    # Shape reconstruction of generated codes
    dictionary = sio.loadmat(opt.file_to_save+'/Dictionary.mat')['dictionary']
    gen_pose_seq = reconstruction(codes, dictionary, opt, linear=True)
    sio.savemat(opt.output_dir + '/gen_pose_seq.mat', {'gen_pose_seq': gen_pose_seq})


if __name__ == "__main__":
    sequences = sio.loadmat('C:/Users/BENTANFOUS/Desktop/Video-generation-framework/Data_process/tes_taichi_keypoints')[
        'tes_taichi_keypoints'].squeeze()
    # Re-sample sequences to same length
    s = np.arange(0, 1, 1/20)
    for i in range(len(sequences)):
        sequences[i] = resampling(sequences[i], s)
    print("hello")
