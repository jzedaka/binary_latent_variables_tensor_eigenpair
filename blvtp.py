import itertools
import numpy as np
from scipy.stats import norm
from typing import AnyStr
from eigenpairs import tensor_power_iteration
import tensorly


class BLVEP:

    def __init__(self,
                 d: int,
                 sigma: float,
                 fillter_method: AnyStr = 'ks',
                 tau: float = 1e-8,
                 wls_K: int = 3,
                 rng: np.random.RandomState = None):
        assert fillter_method in ['br', 'ks']
        self.d = d
        self.sigma = sigma
        self.tau = tau
        self.fillter_method = fillter_method
        self.wls_K = wls_K
        self.rng = rng

    @staticmethod
    def _estimate_moments(X: np.ndarray):
        M1 = np.mean(X, axis=1)
        M2 = X @ X.T / X.shape[1]

        n_features = X.shape[0]
        M3 = np.zeros((n_features, n_features, n_features))
        for x in X.T:
            M3 += np.einsum("i,j,k->ijk", x, x, x)

        M3 /= X.shape[1]
        return M1, M2, M3

    def _denoise_moments(self, M1, M2, M3):
        M2 -= self.sigma * self.sigma * np.eye(M2.shape[0], dtype=np.float64)
        base_vectors = np.eye(M2.shape[0], dtype=np.float64)
        for i in range(M2.shape[0]):
            to_reduce = (
                np.einsum("i,j,k->ijk", M1, base_vectors[i], base_vectors[i])
                + np.einsum("i,j,k->ijk", base_vectors[i], M1, base_vectors[i])
                + np.einsum("i,j,k->ijk", base_vectors[i], base_vectors[i], M1)
            )
            M3 -= self.sigma * self.sigma * to_reduce

        return M2, M3

    def _get_whitning_matrix(self, M: np.ndarray):
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx[: self.d]]
        eigenvectors = eigenvectors[:, idx[: self.d]]
        eigenvalues[eigenvalues < 0] = 1e-8

        eigenvalues_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        K = eigenvectors @ eigenvalues_inv_sqrt
        return K

    def _get_candidates(self, X: np.ndarray):
        M1, M2, M3 = self._estimate_moments(X=X)

        M2, M3 = self._denoise_moments(M1=M1, M2=M2, M3=M3)

        K = self._get_whitning_matrix(M=M2)

        W = tensorly.tenalg.multi_mode_dot(M3, [K, K, K], [0, 1, 2], transpose=True)

        eigenpairs = tensor_power_iteration(T=W, rng=self.rng)
        candidates = []
        lambdas = []
        for l, v in eigenpairs:
            if 1 - self.tau <= l:
                c = np.dot(K, v) / l
                candidates.append(c)
                lambdas.append(l)

        return np.array(candidates), np.array(lambdas)

    def _ks_filtering(self, candidates: np.ndarray, X_fillter: np.ndarray, lambdas: np.ndarray):

        num_candidates = candidates.shape[0]
        ks_scores = np.zeros(num_candidates)

        for i, c in enumerate(candidates):
            p = 1 / lambdas[i] ** 2
            sigma = self.sigma * np.linalg.norm(c)
            cX = np.dot(c, X_fillter)
            G = lambda t: (1 - p) * norm.cdf(t, 0, sigma) + p * norm.cdf(t, 1, sigma)
            
            t_vals = cX
            empirical_cdf = np.searchsorted(np.sort(cX), cX, side='right') / cX.size
            ks_stat = np.max(np.abs(G(t_vals) - empirical_cdf))
            ks_scores[i] = ks_stat

        return candidates[np.argsort(ks_scores)[-self.d :]]

    def _binary_rounding_fillter(self, candidates: np.ndarray, X_fillter: np.ndarray):
        cX = np.dot(candidates, X_fillter)

        cX_round = np.round(cX)
        cX_round[cX_round >= 1] = 1
        cX_round[cX_round < 1] = 0
        diff_norm = np.sum(np.power((cX - cX_round), 2), axis=1)
        norm = np.sum(np.power(candidates, 2), axis=1)
        scores = diff_norm / (X_fillter.shape[1] * norm + 1e-6)
        return candidates[np.argsort(scores)[-self.d :]]

    def recover_W(self, X1: np.ndarray, X2: np.ndarray):
        candidates, lambdas = self._get_candidates(X=X1)
        assert len(candidates) >= self.d, len(candidates)
        if self.fillter_method == 'br' or self.sigma == 0:
            filltered_c = self._binary_rounding_fillter(candidates=candidates,
                                                        X_fillter=X2)
        elif self.fillter_method == 'ks':
            filltered_c = self._ks_filtering(candidates=candidates,
                                             X_fillter=X2,
                                             lambdas=lambdas)
        else:
            raise Exception("invalid fillter_method")

        W_hat = np.linalg.pinv(filltered_c)
        W_wls = self.WLS_step(W=W_hat, X=X2.T)

        return W_hat, W_wls

    def WLS_step(self, W: np.ndarray, X: np.ndarray):

        n_samples = X.shape[0]

        binary_vectors = np.array(list(itertools.product([0, 1], repeat=self.d)))
        h_top_K = np.zeros((n_samples, self.wls_K, self.d))
        Pi = np.zeros((self.wls_K, n_samples))
        
        for j in range(n_samples):
            likelihoods = np.zeros(binary_vectors.shape[0])
            for k, h in enumerate(binary_vectors):
                Wh = np.dot(W, h)
                squared_distance = np.sum((X[j]  - Wh) ** 2)
                liklihood = (1 / np.sqrt(2 * np.pi * self.sigma**2)) * np.exp(-squared_distance / (2 * self.sigma**2))
                likelihoods[k] = liklihood
            
            top_k_indices = np.argsort(likelihoods)[-self.wls_K:] 
            h_top_K[j, ...] = binary_vectors[top_k_indices]
            top_k_likelihoods = likelihoods[top_k_indices]
            
            # Normalize the likelihoods to get the weight matrix Pi
            Pi[:, j] = top_k_likelihoods / np.sum(top_k_likelihoods)


        # S = None
        # H = None
        # for j in range(n_samples):
        #     x_j = X[j]
        #     for k in range(self.wls_K):
        #         h_kj = h_top_K[j, k, :]
        #         if S is None:
        #             S = Pi[k, j] * np.outer(x_j, h_kj)
        #             H = Pi[k, j] * np.outer(h_kj, h_kj)
        #         else:
        #             S = np.concatenate((S, Pi[k, j] * np.outer(x_j, h_kj)), axis=1)
        #             H = np.concatenate((H, Pi[k, j] * np.outer(h_kj, h_kj)), axis=1)

        # H_inv = np.linalg.pinv(H).astype(np.float32)
        # W_wls = S.astype(np.float32) @ H_inv 

        Pi = np.sqrt(Pi)
        H_vec = None
        X_vec = None
        for j in range(n_samples):
            x_j = X[j]
            for k in range(self.wls_K):
                h_kj = h_top_K[j, k, :]
                if X_vec is None:
                    X_vec = Pi[k, j] * x_j
                    H_vec = Pi[k, j] * h_kj
                else:
                    X_vec = np.vstack((X_vec, Pi[k, j] * x_j))
                    H_vec = np.vstack((H_vec,  Pi[k, j] * h_kj))

        W_wls = np.linalg.lstsq(H_vec, X_vec)[0].T

        return W_wls 
