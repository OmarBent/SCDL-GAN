import numpy as np
import scipy.io as sio
from Tools.centeredscaled import centeredscaled
from Tools.procrustes import procrustes
from geomstats.hypersphere import HypersphereMetric
#from Tools.visualize_data import visualize_skeleton


def karcher_mean(data):
    eps = 0.01
    # Convert list to array
    data = np.asarray(data)

    # m : number of datapoints
    # n : number of joints
    # dim : dimension of joint
    m, n, dim = data.shape

    # The initial mean
    mean = centeredscaled(data[0, :, :])
    MuV = np.ones((n, dim))
    v = np.zeros((m, n*dim))

    # HyperSphereMetric
    metric = HypersphereMetric(dimension=n*dim-1)

    while np.linalg.norm(MuV, 'fro') > eps:
        for i in range(m):
            # Procrustes
            x = data[i, :, :]
            #if (np.linalg.norm(x, ord='fro') != 0) and (np.linalg.norm(mean, ord='fro') != 0):
                #_, x, _, _, _ = procrustes(mean, x)

            # Compute the direction VI \in T_X(C)
            x = centeredscaled(x)
            v[i, :] = metric.log(point=x.reshape(1, n*dim), base_point=mean.reshape(1, n*dim))

        # Compute the average direction
        MuV = np.zeros((n, dim))
        for i in range(m):
            MuV += v[i, :].reshape(n, dim)

        MuV = (1/n) * MuV
        mean = centeredscaled(metric.exp(tangent_vec=eps*MuV.reshape(1, n*dim), base_point=mean.reshape(1, n*dim)).reshape(n, dim))
        #print(np.linalg.norm(MuV, ord='fro'))
    return mean


# ---------------
#  Main
# ---------------


if __name__ == '__main__':
    dictionary = sio.loadmat('Dictionary')['dictionary']
    M = karcher_mean(dictionary)
    #visualize_skeleton(M)




