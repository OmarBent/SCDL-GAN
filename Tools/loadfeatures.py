import os
import numpy as np


def loadfeatures(data_dir):

    nbfiles = len(os.listdir(data_dir))
    data = []
    for i in range(nbfiles):
            filename = '{0}/sequence_{1}.txt'.format(data_dir, str(i+1))
            sequence = np.loadtxt(filename)
            data.append(np.array(sequence))
    return data
