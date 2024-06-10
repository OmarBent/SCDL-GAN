import numpy as np
from .centeredscaled import centeredscaled
from .sparse_coding_kendall import sc_global
from .sparse_coding_kendall import sc_class


def coding(sequences, dictionary, sparsity_par, file_to_save, per_class):

    if per_class == 1:
        nb_atoms = len(dictionary) * len(dictionary[0])
    else:
        nb_atoms = len(dictionary)

    nb_sequences = len(sequences)

    for i in range(nb_sequences):
        if i % 50 == 0:
            print("Sparse coding of sequence %d/%d " % (i+1, nb_sequences))
        nbframes = len(sequences[i][1])
        w = np.zeros((nb_atoms, nbframes))
        for j in range(nbframes):
            x = centeredscaled(np.asarray(sequences[i][:, j, :]))
            if per_class == 1:
                h = sc_class(x, dictionary, sparsity_par)
            else:
                h = sc_global(x, dictionary, sparsity_par)
            w[:, j] = h

        # save results
        filename = file_to_save + '/codes' + '/sequence_%s.txt' % str(i+1)
        np.savetxt(filename, w, fmt='%.4f')




