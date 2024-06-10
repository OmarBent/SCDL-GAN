import numpy as np
import scipy.io as sio
from .procrustes import procrustes
from geomstats.hypersphere import HypersphereMetric
import time
from .centeredscaled import centeredscaled


def weighted_karcher(original_code, original_dictionary):
    # Initialization
    eps = 0.0001
    eps1 = 0.1
    E = []
    code = []
    dictionary = []

    # Keep only codes greater than 0.01
    # Returns a list of codes greater than 0.01 and the corresponding dictionary elements (atoms)
    for i in range(original_dictionary.shape[0]):
        if (original_code[i] > 0.01):
            code.append(original_code[i])
            dictionary.append(original_dictionary[i])

    # Convert list to array
    code = np.asarray(code)
    dictionary = np.asarray(dictionary)

    # m : number of remaining atoms
    # n : number of joint points
    # dim : dimension of joint points
    m, n, dim = dictionary.shape

    # Y : maximum value of codes
    # I : indice of the maximum value
    Y, I = np.amax(code), np.argmax(code)

    # M: dictionary element (atom) corresponding to the maximum value of codes (the initial mean)
    M = dictionary[I]

    # (n,dim) : (15,3) - Initialization
    MuV = np.ones((n, dim))
    V = np.zeros((m, n*dim))

    # print(np.linalg.norm(MuV, ord='fro'))

    # HyperSphereMetric
    metric = HypersphereMetric(dimension= n*dim-1 )
    nb_iteration = 0


    while (np.linalg.norm(MuV, 'fro') > eps):
        nb_iteration +=1

        for i in range(m):
            # Procrustes
            X = dictionary[i]

            if (np.linalg.norm(X, ord='fro') != 0) and (np.linalg.norm(M, ord='fro') != 0):
                _, X, _, _, _ = procrustes(M, X)
                #X = np.matmul(X, tr.transpose())

            # Compute the direction VI \in T_X(C)
            X = centeredscaled(X)
            M = centeredscaled(M)
            V[i, :] = metric.log(point=X.reshape(1, n*dim), base_point=M.reshape(1,n*dim))


        # Compute the average direction
        MuV = np.zeros((n, dim))
        for i in range(m):
            MuV += code[i]*V[i, :].reshape(n, dim)

        MuV = (1/n) * MuV
        M = metric.exp(tangent_vec=eps1*MuV.reshape(1, n*dim), base_point=M.reshape(1, n*dim)).reshape(n, dim)
        E.append(np.linalg.norm(MuV, ord='fro'))
        # print(np.linalg.norm(MuV, ord='fro'))

    return M


# ---------------
#  Main
# ---------------


if __name__ == '__main__' :
    start = time.time()
    print(start)
    codes = sio.loadmat('Data/codes_florence_per_class')['codes'][0]
    dictionary = sio.loadmat('Data/dictionary_per_class')['dictionary_florence'][0]

    M = weighted_karcher(codes[0].squeeze()[0].transpose()[0], dictionary[0].squeeze())
    end = time.time()
    print(end)
    print(end-start)