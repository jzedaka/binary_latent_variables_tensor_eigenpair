import numpy as np


def tensor_power_iteration(T, tol=1e-10, max_iter=10_000):
    """
    """
    n = T.shape[0]
    eigenpairs = []

    # Power iteration for each eigenpair
    for _ in range(T.shape[0]*2):
        # Initialize a random vector
        v = np.random.rand(n)
        v /= np.linalg.norm(v) + 1e-8

        for iteration in range(max_iter):
            # Tensor apply
            Tv = np.einsum('ijk,j,k->i', T, v, v)
            l = np.dot(v, Tv)

            # Normalize the eigenvector
            v_new = Tv / np.linalg.norm(Tv)

            # Check convergence
            if np.linalg.norm(v - v_new) < tol:
                break

            v = v_new

        eigenpairs.append((l, v))

        T = T - l * np.einsum('i,j,k->ijk', v, v, v)

    return eigenpairs
