import numpy as np
import traceback
from blvep import BLVEP
from utils import init_model, sample_x, calc_loss
import matplotlib.pyplot as plt
import seaborn as sns


def perform_experiment(d, m, n_steps, rng):

    results = dict()
    h_mean, h_cov, W = init_model(d=d, m=m, rng=rng)
    print(W.T)
    for sigma in [0.1, 0.2, 0.4, 0.8]:
        results[sigma] = dict()
        print("--------------------------")
        k = d
        blvep_alg = BLVEP(d=d, sigma=sigma, tau=0, fillter_method='ks', wls_K=k, rng=rng)
        Ns = np.array([10e2, 10e3, 10e4], dtype=np.int32)
        for n in Ns:
            errors = []
            errors_wls = []
            sucsess = 0
            while sucsess < n_steps:
                X1 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                X2 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                try:
                    W_hat, W_wls = blvep_alg.recover_W(X1=X1, X2=X2)
                    sucsess += 1
                    assert W_hat.shape == W.T.shape, W_hat.shape
                    loss, _ = calc_loss(W.T, W_hat)
                    loss_wls, _ = calc_loss(W.T, W_wls)
                    errors.append(loss)
                    errors_wls.append(loss_wls)
                except Exception as e:
                    pass
            
            results[sigma][n] = np.array(errors)
            print(f"sigma = {sigma},  n = {n}, error = {np.mean(errors)} +- {np.std(errors)}")
            print(f"WLS - sigma = {sigma},  n = {n}, error = {np.mean(errors_wls)} +- {np.std(errors_wls)}")


    return results

def main():
    seed = 1234
    rng = np.random.RandomState(seed)
    results = dict()
    for d in [1, 2, 3]:
        m = 10
        print('^^^^^^^^^^^^^^^^^')
        print(f'd = {d}, m = {m}')
        print('^^^^^^^^^^^^^^^^^')

        r = perform_experiment(d=d, m=m, n_steps=10, rng=rng)
        results[d] = r


if __name__ == "__main__":
    main()
