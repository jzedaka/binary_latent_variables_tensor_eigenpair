import numpy as np
#


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int):

    # Move the specified mode to the first axis
    tensor_transposed = np.moveaxis(tensor, mode, 0)

    # Perform matrix multiplication along the first axis
    result = np.tensordot(matrix, tensor_transposed, axes=(1, 0))

    # Move the result back to the original axis order
    final_result = np.moveaxis(result, 0, mode)

    return final_result

def tensor_power_iteration(T, tol=1e-10, max_iter=100):
    """
    Finds all eigenpairs of a symmetric third-order tensor using power iteration.

    Parameters:
        T (np.ndarray): Symmetric tensor of shape (n, n, n).
        tol (float): Convergence tolerance for eigenvalue updates.
        max_iter (int): Maximum number of iterations.

    Returns:
        eigenpairs (list): A list of tuples (lambda, v) containing the eigenvalue
                           and the corresponding eigenvector.
    """
    n = T.shape[0]
    assert T.shape == (n, n, n), "Input tensor must be third-order and cubic."

    def tensor_vector_product(T, v):
        """Performs tensor-vector product: T[i,j,k] * v[j] * v[k]."""
        return np.einsum('ijk,j,k->i', T, v, v)

    def deflate_tensor(T, lambda_, v):
        """Deflates the tensor by removing the contribution of an eigenpair."""
        return T - lambda_ * np.einsum('i,j,k->ijk', v, v, v)

    # To store all eigenpairs
    eigenpairs = []

    # Power iteration for each eigenpair
    for _ in range(n):
        # Initialize a random vector
        v = np.random.rand(n)
        v /= np.linalg.norm(v)

        for iteration in range(max_iter):
            # Tensor-vector product
            Tv = tensor_vector_product(T, v)

            # Compute the eigenvalue using Rayleigh quotient
            lambda_ = np.dot(v, Tv)

            # Normalize the eigenvector
            v_new = Tv / np.linalg.norm(Tv)

            # Check convergence
            if np.linalg.norm(v - v_new) < tol:
                break

            v = v_new

        # Store the eigenpair
        eigenpairs.append((lambda_, v))

        # Deflate the tensor to find the next eigenpair
        T = deflate_tensor(T, lambda_, v)

    return eigenpairs


def init_model(d: int = 6, m: int = 3, n: int = 100, sigma: int = 0):

    # random h vector
    a = np.random.randn(d)
    h_cov = np.random.randn(d, d)
    h_cov = np.dot(h_cov, h_cov.T)
    h = np.random.multivariate_normal(mean=a, cov=h_cov, size=n).astype(np.int16)  # binary rounding!
    h[h != 0] = 1

    # random W matrix. cloumns are drqwn from the unit sphere!
    W = np.random.randn(d, m)
    # normalize
    W_norms_vec = np.sqrt(np.square(W).sum(axis=1))
    W = W / W_norms_vec[:, None]

    print(W.shape)
    print(np.square(W[0]).sum())
    X = W.T @ h.T

    # add gaussian noise
    X +=  sigma * np.random.randn(*X.shape)

    return X


def estimate_moments(X: np.ndarray):
    M1 = np.mean(X, axis=0)
    M2 = X.T @ X / X.shape[1]

    n_features = X.shape[1]
    M3 = np.zeros((n_features, n_features, n_features))
    for x in X:
        M3 += np.outer(x, np.outer(x, x).flatten()).reshape(n_features, n_features, n_features)
    M3 /= X.shape[1]
    print(X.shape, M1.shape, M2.shape, M3.shape)
    return M1, M2, M3

def denoise_moments(M1, M2, M3, sigma: int):
    M2 -= sigma * sigma * np.eye(M2.shape[0])
    base_vectors = np.eye(M2.shape[0])
    n_features = M2.shape[0]
    for i in range(M2.shape[0]):
        M3 -= sigma * sigma * (np.outer(M1, np.outer(base_vectors[i], base_vectors[i]).flatten()).reshape(n_features, n_features, n_features)
                                + np.outer(base_vectors[i], np.outer(M1, base_vectors[i]).flatten()).reshape(n_features, n_features, n_features)
                                + np.outer(base_vectors[i], np.outer(base_vectors[i], M1).flatten()).reshape(n_features, n_features, n_features))

    return M2, M3

def get_whitning_matrix(M: np.ndarray):
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues[eigenvalues < 0] = 1e-6
    eigenvalues_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    K = eigenvectors @ eigenvalues_inv_sqrt @ eigenvectors.T
    return K

def get_candidates(X: np.ndarray, sigma:int, tau=1e-3):
    print(X, X.shape)
    M1, M2, M3 = estimate_moments(X=X)
    print(M2, X.shape)

    M2, M3 = denoise_moments(M1=M1, M2=M2, M3=M3, sigma=sigma)
    K = get_whitning_matrix(M2)

    W = mode_n_product(M3, K, 0)
    W = mode_n_product(W, K, 1)
    W = mode_n_product(W, K, 2)

    eigenpairs = tensor_power_iteration(W)
    candidates = []
    for l, v in eigenpairs:
        if l - tau >= 1:
            c = np.dot(K, v) / l
            candidates.append(c)
    print(eigenpairs)
    return np.array(candidates  )

def compute_ks_score(candidates: np.ndarray):
    pass


def binary_rounding_fillter(candidates: np.ndarray, test_X):
    # print(candidates.shape, test_X.shape)
    cX = np.dot(candidates, test_X.T)
    cX_round = np.round(cX)
    cX_round[cX >= 1] = 1
    cX_round[cX < 1] = 0
    # print(cX.shape)
    diff_norm = np.sum(np.power((cX - cX_round), 2), axis=1)
    norm = np.sum(np.power(candidates, 2), axis=1)
    # print(diff_norm, norm.shape, diff_norm, diff_norm.shape)
    scores = diff_norm / (norm + 1e-6)
    print(scores, scores.shape)
    return candidates[np.argsort(scores)[-3:]]
    # scores = compute_ks_score(candidates=candidates)


def main():
    print("Main")
    sigma = 0
    X = init_model(sigma=sigma, n=10)
    print(f'X shape= {X.shape}')
    candidates = get_candidates(X=X, sigma=sigma)
    binary_rounding_fillter(candidates, X)

if __name__ == '__main__':
    print("Start")
    main()
