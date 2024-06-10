import numpy as np


def centeredscaled(x):
    muX = x.mean(0)
    X0 = x - muX

    # Vector of filtered translation and scale
    transform = np.zeros(muX.shape[0] + 1)
    transform[:muX.shape[0]] = muX

    ssX = (X0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)

    # scale to equal (unit) norm
    X0 = X0 / normX

    transform[muX.shape[0]] = normX

    return X0, transform.T
