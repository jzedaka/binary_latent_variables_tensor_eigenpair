import numpy as np


def init_model(d: int = 6, m: int = 3, n: int = 100, sigma: int = 0):

    # random h vector
    a = np.random.randn(d)
    h_cov = np.random.randn(d, d)
    h_cov = np.dot(h_cov, h_cov.T)
    h = np.random.multivariate_normal(mean=a, cov=h_cov, size=n).astype(np.int16)
    h[h != 0] = 1
    print(h.shape)
    # random W matrix
    W = np.random.randn(d, m)
    W_norms_vec = np.sqrt(np.square(W).sum(axis=1))
    W = W / W_norms_vec[:, None]

    print(W.shape)
    print(np.square(W[0]).sum())
    Z = W.T @ h.T
    return Z


def estimate_moments(Z: np.ndarray):
    M1 = np.mean(Z, axis=0)
    M2 = (Z @ Z.T) / Z.shape[1]
    M3 = None
    return M1, M2, M3

def denoise_moments(M1, M2, M3, sigma: int):
    dM1 = None
    dM2 = None
    return dM1, dM2

def get_candidates(Z: np.ndarray, sigma:int ):
    M1, M2, M3 = estimate_moments(Z=Z)
    dM1, dM2 = denoise_moments(M1=M1, M2=M2, M3=M3, sigma=sigma)


def main():
    print("Main")
    sigma = 0
    Z = init_model(sigma=sigma)
    print(f'Z shape= {Z.shape}')
    candidates = get_candidates(Z=Z, sigma=sigma)


if __name__ == '__main__':
    print("Start")
    main()