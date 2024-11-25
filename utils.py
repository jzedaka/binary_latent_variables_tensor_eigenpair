import numpy as np
import itertools
from typing import Tuple


def sample_x(h_mean, h_cov, W, n: int, sigma: float):
    h = np.random.multivariate_normal(mean=h_mean, cov=h_cov, size=n)
    # binary rounding!
    h[h >= 0] = 1
    h[h < 0] = 0
    X = W.T @ h.T
    # add gaussian noise
    X += sigma * np.random.randn(*X.shape)

    return X


def init_model(d: int, m: int):

    # random h vector
    h_mean = np.zeros(d)
    h_mean = np.random.randn(d)

    # Generate the covarian
    tmp = np.random.randn(d, d)

    # Perform QR decomposition on tmp
    Q, _ = np.linalg.qr(tmp)
    D = np.diag(np.abs(np.random.randn(d)))
    h_cov = Q @ D @ Q.T

    # random W matrix. cloumns are drqwn from the unit sphere!
    W = np.random.randn(d, m)
    # normalize
    W_norms_vec = np.linalg.norm(W, axis=1)
    W = W / W_norms_vec[:, None]

    return h_mean, h_cov, W


def calc_loss(W, W_hat) -> Tuple[float, np.ndarray]:
    best_perm_loss = None
    best_perm = None
    for perm in itertools.permutations(range(W.shape[1])):
        WP = W_hat.copy()
        for i in perm:
            WP[:, i] = W_hat[:, perm[i]]
        loss = np.linalg.norm(W - WP)
        if best_perm_loss is None:
            best_perm_loss = loss
            best_perm = np.array(perm)
        elif loss < best_perm_loss:
            best_perm_loss = loss
            best_perm = np.array(perm)

    return best_perm_loss, best_perm
