import cvxpy as cp
import numpy as np
from .geomstats.hypersphere import HypersphereMetric



def sparse_coding_kendall(x, dictionary, lam):
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
        #d, z, T, b, c = procrustes(x, atom, compute_optimal_scale=True)
        f[:, i] = metric.log(point=atom.reshape(1, N * M), base_point=x.reshape(1, N * M))

    # Construct the problem.
    w = cp.Variable(n)
    objective = cp.Minimize(((1/n)*(cp.norm(cp.sum_squares(f*w), p=2))) + lam*cp.norm1(w))
    constraints = [sum(w) == 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.

    return w.value#, result
