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

def perform_experiment2(d, m, n_steps, rng):

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
    for d in [1, 2]:
        m = 10
        print('^^^^^^^^^^^^^^^^^')
        print(f'd = {d}, m = {m}')
        print('^^^^^^^^^^^^^^^^^')

        r = perform_experiment1(d=d, m=m, n_steps=20, rng=rng)
        results[d] = r
    
    # save results to file
    with open(r'C:\Users\jonathanz\Desktop\blv\saved_results1.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_results():
    with open(r'C:\Users\jonathanz\Desktop\blv\saved_results1.pkl', 'rb') as f:
        results = pickle.load(f)
    sigma = 0.2
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
    
    n = 10e5
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

if __name__ == "__main__":
    # plot_results()
    main()
