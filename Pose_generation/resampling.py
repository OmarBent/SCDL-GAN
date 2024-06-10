import numpy as np
from math import sin
from math import acos
import scipy.io as sio
from .Tools.procrustes import procrustes
from .Tools.centeredscaled import centeredscaled
from .Tools.visualize_data import visualize_sequence


def resampling(original_sequence, s):
    n1, m1, k1 = np.asarray(original_sequence).shape

    tau = np.zeros((n1, n1))

    sequence = np.zeros((m1, n1, k1))
    transform_sequence = np.zeros((k1+1, n1))
    for i in range(n1):
        sequence[:, i, :], transform_sequence[:, i] = centeredscaled(original_sequence[i, :, :])
        tau[i, 0] = (i+1)/(n1-1)-1/(n1-1)

    # Re-sampling
    trans_sequence_res = linear_resampling(transform_sequence, s)

    sequence_res = np.zeros((m1, len(s), k1))

    for i in range(len(s)):
        k = np.where(s[i] <= tau[:])
        ind1 = k[0][0]-1
        ind2 = k[0][0]
        if ind1 == -1:
            ind1 = 0
            ind2 = 1

        w1 = (s[i]-tau[ind1, 0])/(tau[ind2, 0]-tau[ind1, 0])
        w2 = (tau[ind2, 0] - s[i]) / (tau[ind2, 0] - tau[ind1, 0])

        x_new = sequence[:, ind1, :]
        y_new = sequence[:, ind2, :]

        #if (np.linalg.norm(x_new, ord='fro') != 0) and (np.linalg.norm(y_new, ord='fro') != 0):
         #   _, y_new, _, _, _ = procrustes(x_new, y_new)
          #  y_new = centeredscaled(y_new)

        theta = float(acos(np.matrix.trace(np.matmul(x_new, y_new.T))))
        fr, __ = centeredscaled((1 / sin(theta)) * (sin(w2 * theta) * x_new + sin(w1 * theta) * y_new))
        sequence_res[:, i, :] = fr

    return sequence_res, trans_sequence_res


def linear_resampling(original_sequence, s):
    n1, m1 = np.asarray(original_sequence).shape

    tau = np.zeros((m1, m1))

    sequence = original_sequence
    for i in range(m1):
        #sequence[i, :] = original_sequence[i, :]
        tau[i, 0] = (i+1)/(m1-1)-1/(m1-1)

    # Re-sampling
    sequence_res = np.zeros((n1, len(s)))
    sequence_res_diff = np.zeros((n1, len(s)))

    for i in range(len(s)):
        k = np.where(s[i] <= tau[:])
        ind1 = k[0][0]-1
        ind2 = k[0][0]
        if ind1 == -1:
            ind1 = 0
            ind2 = 1

        w1 = (s[i]-tau[ind1, 0])/(tau[ind2, 0]-tau[ind1, 0])
        w2 = (tau[ind2, 0] - s[i]) / (tau[ind2, 0] - tau[ind1, 0])

        x_new = sequence[:, ind1]
        y_new = sequence[:, ind2]

        theta = np.linalg.norm(x_new - y_new)
        sequence_res[:, i] = (1 / theta) * (np.linalg.norm(w2 * theta) * x_new + np.linalg.norm(w1 * theta) * y_new)
        if i > 0:
            sequence_res_diff[:, i] = ((1 / theta) * (np.linalg.norm(w2 * theta) * x_new + np.linalg.norm(w1 * theta) * y_new)) - sequence_res[:, i-1]

    return sequence_res, sequence_res_diff


if __name__ == "__main__":

    sequence = sio.loadmat('data/sequence1.mat')['sequence1']
    # visualize_sequence(sequence)
    s = np.arange(0, 1, 1/30)
    sequences_n = resampling(sequence, s)
    visualize_sequence(sequences_n)
    #sio.savemat('sequences_n.mat', {'sequences_n': sequences_n})
