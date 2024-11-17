import numpy as np
from eigenpairs import tensor_power_iteration

class BLVEP:

    def __init__(self, d: int, sigma: int, tau: int = 1e-8):
        self.d = d
        self.sigma = sigma
        self.tau = tau

    @staticmethod
    def _mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int):

        # Move the specified mode to the first axis
        tensor_transposed = np.moveaxis(tensor, mode, 0)

        # Perform matrix multiplication along the first axis
        result = np.tensordot(matrix.T, tensor_transposed, axes=(1, 0))

        # Move the result back to the original axis order
        res = np.moveaxis(result, 0, mode)

        return res

    @staticmethod
    def _estimate_moments(X: np.ndarray):
        M1 = np.mean(X, axis=1)
        M2 = X @ X.T / X.shape[1]

        n_features = X.shape[0]
        M3 = np.zeros((n_features, n_features, n_features))
        for x in X.T:
            M3 += np.einsum('i,j,k->ijk', x, x, x)

        M3 /= X.shape[1]
        return M1, M2, M3

    def _denoise_moments(self, M1, M2, M3):
        M2 -= self.sigma * self.sigma * np.eye(M2.shape[0], dtype=np.float64)
        base_vectors = np.eye(M2.shape[0], dtype=np.float64)
        n_features = M2.shape[0]
        for i in range(M2.shape[0]):
            to_reduce =  np.einsum('i,j,k->ijk', M1, base_vectors[i], base_vectors[i]) +  \
                         np.einsum('i,j,k->ijk', base_vectors[i], M1,  base_vectors[i]) + \
                         np.einsum('i,j,k->ijk', base_vectors[i], base_vectors[i], M1)
            M3 -= self.sigma * self.sigma * to_reduce

        return M2, M3

    def _get_whitning_matrix(self, M: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx[:self.d]]
        eigenvectors = eigenvectors[:, idx[:self.d]]
        eigenvalues[eigenvalues < 0] = 1e-6

        eigenvalues_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        K = eigenvectors @ eigenvalues_inv_sqrt 
        return K

    def _get_candidates(self, X: np.ndarray):
        M1, M2, M3 = self._estimate_moments(X=X)

        M2, M3 = self._denoise_moments(M1=M1, M2=M2, M3=M3)

        K = self._get_whitning_matrix(M=M2)

        W = self._mode_n_product(M3, K, 0)
        W = self._mode_n_product(W, K, 1)
        W = self._mode_n_product(W, K, 2)

        eigenpairs = tensor_power_iteration(W)
        candidates = []
        # print(f'Number of eigenpairs = {len(eigenpairs)}')
        for l, v in eigenpairs:
            if 1 - self.tau <= l:
                c = np.dot(K, v) / l
                candidates.append(c)
        
        # print(f'Number of candidates = {len(candidates)}')
        return np.array(candidates)

    def _binary_rounding_fillter(self, candidates: np.ndarray, test_X: np.ndarray):
        cX = np.dot(candidates, test_X)
        cX_round = np.round(cX)
        cX_round[cX >= 1] = 1
        cX_round[cX < 1] = 0
        # print(cX.shape)
        diff_norm = np.sum(np.power((cX - cX_round), 2), axis=1)
        norm = np.sum(np.power(candidates, 2), axis=1)
        scores = diff_norm / (norm + 1e-6)
        return candidates[np.argsort(scores)[-self.d:]]

    def recover_W(self, X):
        candidates = self._get_candidates(X=X)
        filltered_c = self._binary_rounding_fillter(candidates, X)
        return np.linalg.pinv(filltered_c)
