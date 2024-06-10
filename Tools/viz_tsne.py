import pylab
import numpy as np
import scipy.io as sio
from Pose_generation.resampling import resampling
from Pose_generation.Tools.tsne import tsne
from options.train_options import TrainOptions
from Pose_generation.Tools.data_utils import tsne_data_format_exp
from Pose_generation.loadfeatures import loadfeatures


def viz_tsne(opt):
    codes = sio.loadmat(opt.dataroot + '/sequences')[
        'sequences'].squeeze()  # sequence of shape (nb_frames, nb_landmarks, dim)
    labels = sio.loadmat(opt.dataroot + '/labels')['labels'].squeeze()
    # Re-sample sequences to same length
    s = np.arange(0, 1, 1 / opt.seq_length)
    res_sequences = []
    res_transf_sequences = []
    for i in range(len(codes)):
        resampled_sequence, resampled_transform = resampling(codes[i], s)
        res_sequences.append(resampled_sequence)
    # Apply t_sne for visualization
    x, y = tsne_data_format_exp(res_sequences, labels, opt)
    Y = tsne(x, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, y)
    pylab.show()


if __name__ == "__main__":
    # Load args
    train_opt = TrainOptions().parse()

    viz_tsne(train_opt)