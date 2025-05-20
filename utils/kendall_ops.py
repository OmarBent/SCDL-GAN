import numpy as np
from math import sin, acos
from geomstats.hypersphere import HypersphereMetric


def centeredscaled(x):
    """
    Center and normalize a matrix to unit Frobenius norm.

    Args:
        x (np.ndarray): Input matrix of shape (n_joints, dim).

    Returns:
        np.ndarray: Centered and scaled version of input.
    """
    mu = x.mean(axis=0)
    x_centered = x - mu
    norm = np.linalg.norm(x_centered, 'fro')
    return x_centered / norm


def karcher_mean(data, epsilon=0.01):
    """
    Compute the Karcher mean on the hypersphere.

    Args:
        data (np.ndarray): Array of shape (m, n_joints, dim) where `m` is the number of samples.
        epsilon (float): Convergence threshold.

    Returns:
        np.ndarray: The Karcher mean.
    """
    data = np.asarray(data)
    m, n, dim = data.shape

    mean = centeredscaled(data[0])
    metric = HypersphereMetric(dimension=n * dim - 1)

    while True:
        tangent_vectors = np.zeros((m, n * dim))

        for i in range(m):
            x = centeredscaled(data[i])
            tangent_vectors[i] = metric.log(point=x.reshape(1, -1), base_point=mean.reshape(1, -1))

        # Average tangent direction
        avg_tangent = np.mean(tangent_vectors, axis=0).reshape(1, -1)

        if np.linalg.norm(avg_tangent) < epsilon:
            break

        # Update mean
        mean = metric.exp(tangent_vec=epsilon * avg_tangent, base_point=mean.reshape(1, -1)).reshape(n, dim)
        mean = centeredscaled(mean)

    return mean


def procrustes(X, Y, compute_optimal_scale=False):
    """
    Perform Procrustes analysis to align Y to X.

    Args:
        X (np.ndarray): Target matrix of shape (N, M).
        Y (np.ndarray): Input matrix of shape (N, M).
        compute_optimal_scale (bool): Whether to compute optimal scaling.

    Returns:
        tuple: (d, Z, T, b, c) where:
            d: residual error
            Z: transformed Y
            T: rotation matrix
            b: scale factor
            c: translation vector
    """
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY

    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)

    X0 /= normX
    Y0 /= normY

    A = X0.T @ Y0
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T

    T = V @ U.T
    if np.linalg.det(T) < 0:
        V[:, -1] *= -1
        s[-1] *= -1
        T = V @ U.T

    traceTA = s.sum()
    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX * traceTA * Y0 @ T + muX
    else:
        b = 1
        d = 1 + (normY**2 / normX**2) - 2 * traceTA * (normY / normX)
        Z = normY * Y0 @ T + muX

    c = muX - b * muY @ T

    return d, Z, T, b, c


def resampling(original_sequence, s):
    """
    Geodesic resampling of a sequence using spherical interpolation.

    Args:
        original_sequence (np.ndarray): Sequence of shape (n_frames, n_joints, dim).
        s (list or np.ndarray): Sample points for interpolation in [0, 1].

    Returns:
        tuple: (resampled sequence, transform sequence)
    """
    n_frames, n_joints, dim = original_sequence.shape
    sequence = np.zeros((n_joints, n_frames, dim))
    transform_sequence = np.zeros((dim + 1, n_frames))
    tau = np.array([(i + 1) / (n_frames - 1) - 1 / (n_frames - 1) for i in range(n_frames)])

    for i in range(n_frames):
        sequence[:, i, :] = centeredscaled(original_sequence[i])

    resampled_seq = np.zeros((n_joints, len(s), dim))

    for i, si in enumerate(s):
        idx = np.searchsorted(tau, si)
        ind1 = max(idx - 1, 0)
        ind2 = min(idx, n_frames - 1)

        w1 = (si - tau[ind1]) / (tau[ind2] - tau[ind1] + 1e-8)
        w2 = 1 - w1

        x_new = sequence[:, ind1, :]
        y_new = sequence[:, ind2, :]

        theta = acos(np.trace(x_new @ y_new.T))
        interp = (sin(w2 * theta) * x_new + sin(w1 * theta) * y_new) / sin(theta)
        resampled_seq[:, i, :] = centeredscaled(interp)

    return resampled_seq, transform_sequence


def linear_resampling(original_sequence, s):
    """
    Linearly resample a 2D trajectory.

    Args:
        original_sequence (np.ndarray): Original sequence of shape (n_features, n_time_steps).
        s (list or np.ndarray): Sample positions in [0, 1].

    Returns:
        tuple: (resampled sequence, sequence difference)
    """
    n_features, n_steps = original_sequence.shape
    tau = np.array([(i + 1) / (n_steps - 1) - 1 / (n_steps - 1) for i in range(n_steps)])

    sequence_res = np.zeros((n_features, len(s)))
    sequence_diff = np.zeros((n_features, len(s)))

    for i, si in enumerate(s):
        idx = np.searchsorted(tau, si)
        ind1 = max(idx - 1, 0)
        ind2 = min(idx, n_steps - 1)

        w1 = (si - tau[ind1]) / (tau[ind2] - tau[ind1] + 1e-8)
        w2 = 1 - w1

        x_new = original_sequence[:, ind1]
        y_new = original_sequence[:, ind2]

        theta = np.linalg.norm(x_new - y_new)
        interp = (w2 * x_new + w1 * y_new) if theta == 0 else (1 / theta) * (np.linalg.norm(w2 * theta) * x_new + np.linalg.norm(w1 * theta) * y_new)
        sequence_res[:, i] = interp
        if i > 0:
            sequence_diff[:, i] = interp - sequence_res[:, i - 1]

    return sequence_res, sequence_diff
