import scipy.io as sio
from Tools.coding import coding
from Tools.bi_lstm import bi_lstm
from Tools.loadfeatures import loadfeatures
from Pose_generation.gen_pose import gen_pose
from options.train_options import TrainOptions
from joblib import Parallel, delayed
import multiprocessing
from Pose_generation.sparse_coding import sparse_coding_parallel


if __name__ == '__main__':
    # ... run your application ...

    #### Load options ###
    train_opt = TrainOptions().parse()

    ### SCDL + Training Pose-GAN ###
    gen_pose(train_opt)

    ### Action classification using Bi-LSTM on sparse code sequences ###

    # Loading data
    action_labels = sio.loadmat(train_opt.dataroot + '/action_labels')['labels'].squeeze()  # Load training action labels
    subject_labels = sio.loadmat(train_opt.dataroot + '/subject_labels')['subjects'].squeeze()
    sequences = sio.loadmat(train_opt.dataroot + '/subject_labels')['subjects'].squeeze()
    dictionary = sio.loadmat(train_opt.file_to_save + '/Dictionary.mat')['dictionary']

    # Sparse coding of train and test sequences
    coding(sequences, dictionary, train_opt.sparsity, train_opt.file_codes_classif, train_opt.per_class)

    inputs = range(len(sequences))
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(sparse_coding_parallel)(sequences, dictionary, train_opt.sparsity, train_opt.file_codes_classif, i) for i in inputs)

    original_data = loadfeatures(train_opt.file_codes_classif)

    # Training Bi-LSTM
    if train_opt.experience == 'original':
        print("======== Training Bi-LSTM  with original data ========")
        accuracy, scores = bi_lstm(train_opt, original_data, subject_labels, action_labels)

    elif train_opt.experience == 'augmented':
        print("======== Training Bi-LSTM  with original data + generated_data ========")
        generated_data = loadfeatures(train_opt.output_dir + '/sequences')
        generated_labels = tuple(open(train_opt.output_dir + '/fake_labels.txt', 'r'))
        data = original_data + generated_data
        for i in range(train_opt.n_gen):
            subject_labels[i + len(sequences)] = train_opt.train_subjects[0]
            action_labels[i + len(sequences)] = int(generated_labels[i])

        accuracy, scores = bi_lstm(train_opt, data, subject_labels, action_labels)







