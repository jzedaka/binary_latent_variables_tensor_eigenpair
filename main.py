import numpy as np
import pickle
from blvep import BLVEP
from utils import init_model, sample_x, calc_loss
import matplotlib.pyplot as plt
import seaborn as sns


def perform_experiment1(d, m, n_steps, rng):

    results = dict()
    h_mean, h_cov, W = init_model(d=d, m=m, rng=rng)
    print(W.T)
    for sigma in [0.1, 0.2, 0.4, 0.8]:
        results[sigma] = dict()
        print("--------------------------")
        # wls_K=0 -> No WLS step
        blvep_alg = BLVEP(d=d, sigma=sigma, tau=0, fillter_method='ks', wls_K=0, rng=rng)
        Ns = np.array([10e2, 10e3, 10e4, 10e5], dtype=np.int32)
        for n in Ns:
            errors = []
            sucsess = 0
            while sucsess < n_steps:
                X1 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                X2 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                try:
                    W_hat, _ = blvep_alg.recover_W(X1=X1, X2=X2)
                    sucsess += 1
                    assert W_hat.shape == W.T.shape, W_hat.shape
                    loss, _ = calc_loss(W.T, W_hat)
                    errors.append(loss)

                except Exception as e:
                    pass
            
            results[sigma][n] = np.array(errors)
            print(f"sigma = {sigma},  n = {n}, error = {np.mean(errors)} +- {np.std(errors)}")

    return results


def plot_results1(results):
    sigma = 0.2
    plt.figure()
    x = list(results[1][sigma].keys())
    for d in list(results.keys()):
        mean = np.array(list(results[d][0.2].values())).mean(axis=1)
        stds = np.array(list(results[d][0.2].values())).std(axis=1)
        plt.plot(x,mean, '--o', label=f'd = {d}')
        plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.xlabel('n')
    plt.title(f"Sigma = {sigma}")
    plt.ylabel('normalized Frobenius norm')
    plt.tight_layout()
    
    plt.figure()
    n = 10e4
    x = list(results[1].keys())
    for d in list(results.keys()):
        mean = np.array([results[d][sigma][n] for sigma in x]).mean(axis=1)
        stds =  np.array([results[d][sigma][n] for sigma in x]).std(axis=1)
        plt.plot(x,mean, '--o', label=f'd = {d}')
        plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.xlabel('sigma')
    plt.title(f"n = {int(n)}")
    plt.ylabel('normalized Frobenius norm')
    plt.tight_layout()
    plt.show()


def perform_experiment2(d, m, n_steps, rng):

    results = dict(spectral=dict(), spectral_wls=dict())

    print('^^^^^^^^^^^^^^^^^')
    print(f'd = {d}, m = {m}')
    print('^^^^^^^^^^^^^^^^^')
    h_mean, h_cov, W = init_model(d=d, m=m, rng=rng)
    print(W.T)
    for sigma in [0.1, 0.2, 0.4, 0.8]:
        results['spectral'][sigma] = dict()
        results['spectral_wls'][sigma] = dict()
        print("--------------------------")
        blvep_alg = BLVEP(d=d, sigma=sigma, tau=0, fillter_method='ks', wls_K=d, rng=rng)
        Ns = np.array([10e2, 10e3, 10e4], dtype=np.int32)
        for n in Ns:
            errors = []
            wls_errors = []
            sucsess = 0
            while sucsess < n_steps:
                X1 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                X2 = sample_x(h_mean=h_mean, h_cov=h_cov, W=W, n=n, sigma=sigma, rng=rng)
                try:
                    W_hat, W_wls = blvep_alg.recover_W(X1=X1, X2=X2)
                    sucsess += 1
                    assert W_hat.shape == W.T.shape, W_hat.shape
                    loss, _ = calc_loss(W.T, W_hat)
                    wls_loss, _ = calc_loss(W.T, W_wls)
                    errors.append(loss)
                    wls_errors.append(wls_loss)

                except Exception as e:
                    pass
            
            results['spectral'][sigma][n] = np.array(errors)
            results['spectral_wls'][sigma][n] = np.array(wls_errors)

            print(f"sigma = {sigma},  n = {n}, error = {np.mean(errors)} +- {np.std(errors)}")
            print(f"WLS sigma = {sigma},  n = {n}, error = {np.mean(wls_errors)} +- {np.std(wls_errors)}")

    return results


def plot_results2(results):
    sigma = 0.2
    plt.figure()
    x = list(results['spectral'][sigma].keys())
    mean = np.array(list(results['spectral'][0.2].values())).mean(axis=1)
    stds = np.array(list(results['spectral'][0.2].values())).std(axis=1)
    plt.plot(x,mean, '--o', label=f'spectral')
    plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)
    
    mean = np.array(list(results['spectral_wls'][0.2].values())).mean(axis=1)
    stds = np.array(list(results['spectral_wls'][0.2].values())).std(axis=1)
    plt.plot(x,mean, '--o', label=f'spectral + WLS')
    plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.xlabel('n')
    plt.title(f"Sigma = {sigma}")
    plt.ylabel('normalized Frobenius norm')
    plt.tight_layout()
    
    plt.figure()
    n = 10e4
    x = list(results['spectral'].keys())
    mean = np.array([results['spectral'][sigma][n] for sigma in x]).mean(axis=1)
    stds =  np.array([results['spectral'][sigma][n] for sigma in x]).std(axis=1)
    plt.plot(x,mean, '--o', label=f'spectral')
    plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)

    mean = np.array([results['spectral_wls'][sigma][n] for sigma in x]).mean(axis=1)
    stds =  np.array([results['spectral_wls'][sigma][n] for sigma in x]).std(axis=1)
    plt.plot(x,mean, '--o', label=f'spectral + WLS')
    plt.fill_between(x, mean - stds, mean + stds, alpha=0.3)
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.xlabel('sigma')
    plt.title(f"n = {int(n)}")
    plt.ylabel('normalized Frobenius norm')
    plt.tight_layout()
    plt.show()


def main():
    seed = 1234
    rng = np.random.RandomState(seed)
    results = dict()
    for d in [1, 2, 3]:
        m = 10
        print('^^^^^^^^^^^^^^^^^')
        print(f'd = {d}, m = {m}')
        print('^^^^^^^^^^^^^^^^^')

        r = perform_experiment1(d=d, m=m, n_steps=20, rng=rng)
        results[d] = r

    plot_results1(results)
    results = perform_experiment2(d=2, m=10, n_steps=5, rng=rng)

    plot_results2(results)


if __name__ == "__main__":
    main()
