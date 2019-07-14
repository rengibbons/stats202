"""Performs analysis for STAT 202 HW 2 P 10, Sum 2019"""
import statistics
import numpy as np
import pandas as pd


def main():
    """Peforms bootstrap to determine standard error estimates."""
    df = pd.read_csv('data/Boston.csv')
    n_obs = len(df)
    np.random.seed(111)

    # Part a
    medv_mean = np.mean(df['medv'])
    print('medv mean = {:.3f}'.format(medv_mean))

    # Part b
    medv_stan_err = statistics.stdev(df['medv']) / np.sqrt(n_obs)
    print('medv standard error = {:.5f}'.format(medv_stan_err))

    # Part c
    n_boot_iters = 10000
    medv_mean_array = np.zeros(n_boot_iters)
    for ii in range(n_boot_iters):
        ind = np.random.choice(n_obs, n_obs, replace=True)
        medv_mean_array[ii] = np.mean(df.loc[ind, 'medv'])

    medv_stan_err_boot = statistics.stdev(medv_mean_array)
    print('medv standard error (bootstrap) = {:.5f}'.format(medv_stan_err_boot))

    # Part d
    ci_95 = [medv_mean - 2 * medv_stan_err,
             medv_mean + 2 * medv_stan_err]
    ci_95_boot = [medv_mean - 2 * medv_stan_err_boot,
                  medv_mean + 2 * medv_stan_err_boot]
    print('95% CI             = [{:.3f}, {:.3f}]'.format(ci_95[0], ci_95[1]))
    print('95% CI (bootstrap) = [{:.3f}, {:.3f}]'.format(ci_95_boot[0], ci_95_boot[1]))

    # Part e
    medv_med = np.median(df['medv'])
    print('medv med = {:.3f}'.format(medv_med))

    # Part f
    medv_med_array = np.zeros(n_boot_iters)
    for ii in range(n_boot_iters):
        ind = np.random.choice(n_obs, n_obs, replace=True)
        medv_med_array[ii] = np.median(df.loc[ind, 'medv'])

    medv_med_stan_err_boot = statistics.stdev(medv_med_array)
    print('medv median standard error (bootstrap) = {:.5f}'.format(medv_med_stan_err_boot))

    # Part g
    medv_10 = np.percentile(df['medv'], 10)
    print('medv 10th percentile = {:.3f}'.format(medv_10))

    # Part f
    medv_10_array = np.zeros(n_boot_iters)
    for ii in range(n_boot_iters):
        ind = np.random.choice(n_obs, n_obs, replace=True)
        medv_10_array[ii] = np.percentile(df.loc[ind, 'medv'], 10)

    medv_10_stan_err_boot = statistics.stdev(medv_10_array)
    print('medv 10th percenile standard error (bootstrap) = {:.5f}'.format(medv_10_stan_err_boot))


if __name__ == '__main__':
    main()
