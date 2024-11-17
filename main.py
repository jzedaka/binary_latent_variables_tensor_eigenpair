import numpy as np
from blvep import BLVEP


def sample_x(h_mean, h_cov, W, n: int, sigma: int):
    h = np.random.multivariate_normal(mean=h_mean, cov=h_cov, size=n).astype(np.int16)  # binary rounding!
    h[h != 0] = 1
    X = W.T @ h.T
    # add gaussian noise
    X +=  sigma * np.random.randn(*X.shape)

    return X

def init_model(d: int, m: int):

    # random h vector
    h_mean = np.random.randn(d)
    h_cov = np.random.randn(d, d)
    h_cov = np.dot(h_cov, h_cov.T)

    # random W matrix. cloumns are drqwn from the unit sphere!
    W = np.random.randn(d, m)
    # normalize
    W_norms_vec = np.sqrt(np.square(W).sum(axis=1))
    W = W / W_norms_vec[:, None]

    return h_mean, h_cov, W


def main():

    n_steps = 10
    d = 1
    m = 10

    h_mean, h_cov, W = init_model(d=d, m=m)

    for sigma in [0, 0.1, 0.2, 0.3, 0.4]:
        print("--------------------------")
        blvep_alg = BLVEP(d=d, sigma=sigma)

        for n in np.array([1e2, 1e3, 1e4, 1e5], dtype=np.int32):
            err = 0
            for _ in range(n_steps):
                X = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma)
                W_hat = blvep_alg.recover_W(X=X)
                assert W_hat.shape == W.T.shape, W_hat.shape
                err += np.mean(np.power(W.T - W_hat, 2))

            print(f'sigma = {sigma},  n = {n}, error = {err/n_steps}')

if __name__ == '__main__':
    main()
