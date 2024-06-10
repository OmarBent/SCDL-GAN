import numpy as np
import scipy.io as sio
from .Tools.visualize_data import reconstruction


def get_generated_data(data_dir):
    indices = np.arange(400000, 500000, 2000)
    data = []

    for i in indices :
        for j in range(2):
            filename = '{0}/codes_{1}.txt'.format(data_dir, str(i+j))
            sequence = np.loadtxt(filename)
            data.append(np.array(sequence))
    return data





if __name__ == '__main__':
    codes_wave1 = get_generated_data('Results/codes_wave1')
    dictionary_wave1 = sio.loadmat('data/dictionary_wave1.mat')['dictionary_class9'][0]

    sequences_wave1 = reconstruction(codes_wave1, dictionary_wave1)
    sio.savemat('data/sequences_wave1.mat', {'sequences_wave1': sequences_wave1})

    codes_wave2 = get_generated_data('Results/codes_wave2')
    dictionary_wave2 = sio.loadmat('data/dictionary_wave2.mat')['final_means'][0]

    sequences_wave2 = reconstruction(codes_wave2, dictionary_wave2)
    sio.savemat('data/sequences_wave2.mat', {'sequences_wave2': sequences_wave2})





