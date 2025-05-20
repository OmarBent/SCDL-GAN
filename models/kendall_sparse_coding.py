import os
import numpy as np
import cvxpy as cp
from .geomstats.hypersphere import HypersphereMetric
from utils.kendall_ops import centeredscaled, procrustes


def kendall_sparse_coding(x, dictionary, lam):
    """
    Perform intrinsic sparse coding in Kendall's shape space using convex optimization.

    Args:
        x (ndarray): Shape (N, M) - the input shape/frame.
        dictionary (list of ndarray): List of atoms (each of shape (N, M)).
        lam (float): Sparsity parameter.

    Returns:
        ndarray: Vector of length n (number of atoms), the sparse code.
    """
    n = len(dictionary)
    N, M = dictionary[0].shape
    metric = HypersphereMetric(dimension=N * M - 1)

    f = np.zeros((N * M, n))
    for i, atom in enumerate(dictionary):
        f[:, i] = metric.log(
            point=atom.reshape(1, N * M),
            base_point=x.reshape(1, N * M)
        )

    w = cp.Variable(n)
    objective = cp.Minimize((1 / n) * cp.norm(f @ w, p=2) ** 2 + lam * cp.norm1(w))
    constraints = [cp.sum(w) == 1]
    cp.Problem(objective, constraints).solve()

    return w.value


def sparse_coding(sequences, dictionary, sparsity_par, file_to_save):
    """
    Perform sparse coding on all sequences and save codes per sequence.

    Args:
        sequences (list): List of sequences, each a tuple (metadata, frames).
        dictionary (list): List of atoms (each a shape of size (N, M)).
        sparsity_par (float): Sparsity parameter lambda.
        file_to_save (str): Directory to save output codes.
    """
    os.makedirs(os.path.join(file_to_save, 'codes'), exist_ok=True)
    nb_atoms = len(dictionary)

    for i, (_, frames) in enumerate(sequences):
        print(f"[INFO] Sparse coding of sequence {i + 1}")
        nb_frames = len(frames)
        w = np.zeros((nb_atoms, nb_frames))

        for j in range(nb_frames):
            x = centeredscaled(np.asarray(frames[j]))[0]
            w[:, j] = kendall_sparse_coding(x, dictionary, sparsity_par)

        np.savetxt(
            os.path.join(file_to_save, 'codes', f'sequence_{i + 1}.txt'),
            w, fmt='%.4f'
        )


def sparse_coding_parallel(sequences, dictionary, sparsity_par, file_to_save, i):
    """
    Sparse coding for a single sequence (supports multiprocessing).

    Args:
        sequences (list): List of sequences.
        dictionary (list): List of atoms.
        sparsity_par (float): Sparsity parameter lambda.
        file_to_save (str): Output directory path.
        i (int): Index of the sequence to process.
    """
    os.makedirs(os.path.join(file_to_save, 'codes'), exist_ok=True)
    nb_atoms = len(dictionary)
    _, frames = sequences[i]
    nb_frames = len(frames)
    w = np.zeros((nb_atoms, nb_frames))

    for j in range(nb_frames):
        x = centeredscaled(np.asarray(frames[j]))[0]
        w[:, j] = kendall_sparse_coding(x, dictionary, sparsity_par)

    np.savetxt(
        os.path.join(file_to_save, 'codes', f'sequence_{i + 1}.txt'),
        w, fmt='%.4f'
    )


def sc_global(x, dictionary, lam):
    """
    Intrinsic sparse coding using Procrustes-aligned atoms.

    Args:
        x (ndarray): A shape of size (N, M).
        dictionary (list): List of atoms (each of shape (N, M)).
        lam (float): Sparsity parameter.

    Returns:
        ndarray: Sparse code vector.
    """
    n = len(dictionary)
    N, M = dictionary[0].shape
    metric = HypersphereMetric(dimension=N * M - 1)

    f = np.zeros((N * M, n))
    for i, atom in enumerate(dictionary):
        _, z, *_ = procrustes(x, atom, compute_optimal_scale=False)
        f[:, i] = metric.log(
            point=z.reshape(1, N * M),
            base_point=x.reshape(1, N * M)
        )

    w = cp.Variable(n)
    objective = cp.Minimize((1 / n) * cp.norm(f @ w, p=2) ** 2 + lam * cp.norm1(w))
    constraints = [cp.sum(w) == 1]
    cp.Problem(objective, constraints).solve()

    return w.value


def sc_class(x, dictionary, lam):
    """
    Class-specific sparse coding: compute codes from multiple class-specific dictionaries.

    Args:
        x (ndarray): A shape of size (N, M).
        dictionary (list of list): Each sublist is a class-specific list of atoms.
        lam (float): Sparsity parameter.

    Returns:
        ndarray: Concatenated sparse code over all classes.
    """
    nb_classes = len(dictionary)
    N, M = dictionary[0][0].shape
    metric = HypersphereMetric(dimension=N * M - 1)

    total_atoms = sum(len(class_dict) for class_dict in dictionary)
    final_code = np.zeros(total_atoms)

    if np.linalg.norm(x) > 0:
        codes = []
        for class_dict in dictionary:
            n = len(class_dict)
            f = np.zeros((N * M, n))
            for i, atom in enumerate(class_dict):
                base = centeredscaled(x).reshape(1, N * M)
                point = centeredscaled(atom).reshape(1, N * M)
                f[:, i] = metric.log(point=point, base_point=base)

            w = cp.Variable(n)
            objective = cp.Minimize((1 / n) * cp.norm(f @ w, p=2) ** 2 + lam * cp.norm1(w))
            constraints = [cp.sum(w) == 1]
            cp.Problem(objective, constraints).solve()
            codes.append(w.value)

        # Concatenate class-specific codes
        idx = 0
        for code in codes:
            final_code[idx:idx + len(code)] = code
            idx += len(code)

    return final_code
