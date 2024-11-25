import numpy as np
from blvep import BLVEP
from utils import init_model, sample_x, calc_loss
import matplotlib.pyplot as plt
import seaborn as sns


def perform_experiment(d=3, m=5, n_steps=10):

    results = dict()
    h_mean, h_cov, W = init_model(d=d, m=m)
    print(W.T)
    for sigma in [0.1, 0.4, 0.8]:
        results[sigma] = dict()
        print("--------------------------")
        blvep_alg = BLVEP(d=d, sigma=sigma, tau=0, fillter_method='ks', wls_K=d)
        Ns = np.array([10e3, 10e4], dtype=np.int32)
        for n in Ns:
            errors = []
            sucsess = 0
            for _ in range(n_steps):
                X = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma)
                X_fillter = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma)
                try:
                    W_hat = blvep_alg.recover_W(X=X, X_fillter=X_fillter)
                    sucsess += 1
                    assert W_hat.shape == W.T.shape, W_hat.shape
                    loss, _ = calc_loss(W.T, W_hat)
                    errors.append(loss)
                except:
                    pass
            
            results[sigma][n] = np.array(errors)
            print(f"sigma = {sigma},  n = {n}, error = {np.mean(errors)} +- {np.std(errors)}")

    return results

def main():
    results = dict()
    for d in [1]:
        m = d * 5
        print('^^^^^^^^^^^^^^^^^')
        print(f'd = {d}, m = {m}')
        print('^^^^^^^^^^^^^^^^^')

        r = perform_experiment(d=d, m=m, n_steps=1)
        results[d] = r


if __name__ == "__main__":
    main()
