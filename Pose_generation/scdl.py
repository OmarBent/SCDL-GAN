from .sparse_coding import sparse_coding_parallel
from .kmeans import kmeans
from .Tools.centeredscaled import centeredscaled
import scipy.io as sio
from .Tools.visualize_data import visualize_sequence
from .kmeans import kmeans_kendall
from .loadfeatures import loadfeatures
import os
from joblib import Parallel, delayed
import multiprocessing


def scdl(sequences, opt):
    # Prepare training data
    train_frames = []
    for i in range(len(sequences)):
        for j in range(int(sequences[i].shape[1]/opt.step)):
            skeleton = centeredscaled(sequences[i][:, j*opt.step, :])[0]
            train_frames.append(skeleton)

    # Construct dictionary
    print("======== K-means to build dictionary ========")
    dictionary = kmeans_kendall(opt.feature_dim, train_frames)
    sio.savemat(opt.file_to_save+'/Dictionary.mat', {'dictionary': dictionary})
    #dictionary = sio.loadmat(opt.file_to_save+'/Dictionary.mat')['dictionary']
    #visualize_sequence(dictionary)

    # Sparse coding
    print("======== Kendall sparse coding ========")
    inputs = range(len(sequences))
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(sparse_coding_parallel)(sequences, dictionary, opt.sparsity, opt.file_to_save, i) for i in inputs)


if __name__ == '__main__':

    # Load data
    sequences = sio.loadmat('data/sequences_florence')['sequences'].squeeze()
    subjects = sio.loadmat('data/subjects_florence')['subject_labels'].squeeze()

    # Parameters
    k = 5 # size of dictionary
    step = 3 #
    sparsity = 0.1
    train_subjects = [1]#, 2, 3, 4, 5]
    file_to_save = 'Results_test'
    os.makedirs(file_to_save, exist_ok=True)
    os.makedirs(file_to_save+'/codes', exist_ok=True)

    # Run SCDL
    scdl(sequences, k, sparsity, file_to_save, step)

