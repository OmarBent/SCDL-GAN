from .Tools.centeredscaled import centeredscaled
from .sparse_coding_kendall import sparse_coding_kendall
import numpy as np


def sparse_coding(sequences, dictionary, sparsity_par, file_to_save):

    nb_atoms = len(dictionary)
    nb_sequences = len(sequences)
    for i in range(nb_sequences):
        print("Sparse coding of sequence %d " % (i+1))
        nbframes = len(sequences[i][1])
        w = np.zeros((nb_atoms, nbframes))
        for j in range(nbframes):
            x = centeredscaled(np.asarray(sequences[i][:, j, :]))[0]
            h = sparse_coding_kendall(x, dictionary, sparsity_par)
            w[:, j] = h

        # save results
        filename = file_to_save + '/codes' + '/sequence_%s.txt' % str(i+1)
        np.savetxt(filename, w, fmt='%.4f')


def sparse_coding_parallel(sequences, dictionary, sparsity_par, file_to_save, i):
    nb_atoms = len(dictionary)
    #print("Sparse coding of sequence %d " % (i+1))
    nbframes = len(sequences[i][1])
    w = np.zeros((nb_atoms, nbframes))
    for j in range(nbframes):
        x = centeredscaled(np.asarray(sequences[i][:, j, :]))[0]
        h = sparse_coding_kendall(x, dictionary, sparsity_par)
        w[:, j] = h

    # save results
    filename = file_to_save + '/codes' + '/sequence_%s.txt' % str(i+1)
    np.savetxt(filename, w, fmt='%.4f')
