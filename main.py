import os
import numpy as np
import scipy.io as sio
import multiprocessing
from joblib import Parallel, delayed

from models.lstm import bi_lstm
from utils.data_utils import loadfeatures
from utils.kendall_ops import centeredscaled
from config.train_options import TrainOptions
from models.kendall_sparse_coding import sparse_coding_parallel, sc_global, sc_class


def coding(sequences, dictionary, sparsity_par, file_to_save, per_class):
    """
    Perform Kendall sparse coding for a list of sequences using either global or class-specific dictionaries.

    Args:
        sequences (list): A list of shape sequences.
        dictionary (list or nested list): Dictionary of atoms.
        sparsity_par (float): Sparsity regularization parameter.
        file_to_save (str): Directory path to save the sparse codes.
        per_class (bool): Whether to use a per-class dictionary or a global one.
    """
    nb_atoms = len(dictionary) * len(dictionary[0]) if per_class == 1 else len(dictionary)
    nb_sequences = len(sequences)

    for i in range(nb_sequences):
        if i % 50 == 0:
            print(f"[INFO] Sparse coding sequence {i + 1}/{nb_sequences}")

        nbframes = len(sequences[i][1])
        w = np.zeros((nb_atoms, nbframes))

        for j in range(nbframes):
            x = centeredscaled(np.asarray(sequences[i][:, j, :]))
            h = sc_class(x, dictionary, sparsity_par) if per_class == 1 else sc_global(x, dictionary, sparsity_par)
            w[:, j] = h

        # Save sparse codes
        os.makedirs(os.path.join(file_to_save, 'codes'), exist_ok=True)
        filename = os.path.join(file_to_save, 'codes', f'sequence_{i + 1}.txt')
        np.savetxt(filename, w, fmt='%.4f')


def main():
    """
    Main function to run Kendall sparse coding and train a Bi-LSTM classifier.
    """
    # Load training configuration
    train_opt = TrainOptions().parse()

    print("[INFO] Loading labels and sequences...")
    action_labels = sio.loadmat(os.path.join(train_opt.dataroot, 'action_labels'))['labels'].squeeze()
    subject_labels = sio.loadmat(os.path.join(train_opt.dataroot, 'subject_labels'))['subjects'].squeeze()
    sequences = sio.loadmat(os.path.join(train_opt.dataroot, 'subject_labels'))['subjects'].squeeze()

    print("[INFO] Loading dictionary...")
    dictionary = sio.loadmat(os.path.join(train_opt.file_to_save, 'Dictionary.mat'))['dictionary']

    print("[INFO] Performing sparse coding...")
    coding(sequences, dictionary, train_opt.sparsity, train_opt.file_codes_classif, train_opt.per_class)

    print("[INFO] Performing parallel sparse coding...")
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(sparse_coding_parallel)(
            sequences, dictionary, train_opt.sparsity, train_opt.file_codes_classif, i
        ) for i in range(len(sequences))
    )

    print("[INFO] Loading sparse-coded features...")
    original_data = loadfeatures(train_opt.file_codes_classif)

    print("======== Training Bi-LSTM with original data ========")
    accuracy, scores = bi_lstm(train_opt, original_data, subject_labels, action_labels)

    # Output final results
    print(f"\n[RESULT] Accuracy: {accuracy:.4f}")
    print(f"[RESULT] Scores: {scores}")


if __name__ == '__main__':
    main()
