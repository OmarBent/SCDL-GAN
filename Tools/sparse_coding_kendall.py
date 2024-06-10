import cvxpy as cp
import numpy as np
from geomstats.hypersphere import HypersphereMetric
from .procrustes import procrustes
from .centeredscaled import centeredscaled

def sc_global(x, dictionary, lam):
    """
    Implementation of intrinsic sparse coding in the Kendall's shape space.
    Args
      x: sequence of NxM shapes (of number t)
      dictionary: tuple of n atoms of size NxM each
      lam: sparsity parameter lambda
    Returns:
      w: nxt vector of weights
    """

    # size of dictionary.
    n = len(dictionary)
    N, M = np.asarray(dictionary[0]).shape

    metric = HypersphereMetric(dimension=N * M - 1)

    f = np.zeros((N * M, n))
    for i in range(0, n):
        atom = np.asarray(dictionary[i])
        d, z, T, b, c = procrustes(x, atom, compute_optimal_scale=False)
        f[:, i] = metric.log(point=z.reshape(1, N * M), base_point=x.reshape(1, N * M))

    # Construct the problem.
    w = cp.Variable(n)
    objective = cp.Minimize(((1/n)*(cp.norm(cp.sum_squares(f*w), p=2))) + lam*cp.norm1(w))
    constraints = [sum(w) == 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.

    return w.value#, result


def sc_class(x, dictionary, lam):
    """
    Implementation of class-specific sparse coding in the Kendall's shape space.
    Args
      x: sequence of NxM shapes (of number t)
      dictionary: tuple of n atoms of size NxM each
      lam: sparsity parameter lambda
    Returns:
      w: nxt vector of weights
    """
    N, M = np.asarray(dictionary[0][0]).shape

    metric = HypersphereMetric(dimension=N * M - 1)
    nb_dictionaries = len(dictionary)
    nb_atoms = len(dictionary[0]) * nb_dictionaries

    codes = []
    final_code = np.zeros(nb_atoms)

    if np.linalg.norm(x) > 0:
        for d in range(nb_dictionaries):
            # size of dictionary.
            n = len(dictionary[d])

            f = np.zeros((N * M, n))
            for i in range(0, n):
                atom = np.asarray(dictionary[d][i])
                #_, z, _, _, _ = procrustes(centeredscaled(x), centeredscaled(atom), compute_optimal_scale=False)
                f[:, i] = metric.log(point=centeredscaled(atom).reshape(1, N * M), base_point=centeredscaled(x).reshape(1, N * M))

            # Construct the problem.
            w = cp.Variable(n)
            objective = cp.Minimize(((1 / n) * (cp.norm(cp.sum_squares(f * w), p=2))) + lam * cp.norm1(w))
            constraints = [sum(w) == 1]
            prob = cp.Problem(objective, constraints)

            # The optimal objective value is returned by `prob.solve()`.
            result = prob.solve()
            # The optimal value for x is stored in `x.value`.
            codes.append(w.value)
            del w
        idx = 0
        for j in range(nb_dictionaries):
            for h in range(len(codes[j])):
                final_code[idx] = codes[j][h]
                idx = idx + 1

    return final_code  # , result