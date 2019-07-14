"""Performs analysis for STAT 202 HW 2 P 8, Sum 2019"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def boot_fn(df, ind):
    """Returns the logistic regression coefficients for a data sample (Part b)."""
    df_new = df.loc[ind]
    ind_vars = ['income', 'balance']
    Y = df_new['bool default']
    X = sm.add_constant(df_new[ind_vars])
    model = sm.Logit(Y, X).fit()
    return model.params.values


def stan_errs(coeffs):
    """Computes the bootstrap standard errors (Part c)."""
    B, p = coeffs.shape
    standard_errors = np.zeros(p)
    for jj in range(p):
        standard_errors[jj] = np.sqrt(1 / (B - 1) * np.sum((coeffs[:, jj] - \
                              np.mean(coeffs[:, jj])) ** 2.0))
    return standard_errors


def main():
    """Computes various logistic regressions to compute standard errors."""
    dataset = sm.datasets.get_rdataset(dataname='Default', package='ISLR')
    df = dataset.data

    # Part a
    # Fit logistic regression to entire data set.
    ind_vars = ['income', 'balance']
    df['bool default'] = df['default'] == 'Yes'
    Y = df['bool default']
    X = sm.add_constant(df[ind_vars])
    model = sm.Logit(Y, X).fit()
    print(model.summary())
    print('Standard errors')
    print(model.bse)

    ## Part b/c
    n_ind = len(df)
    n_boot_iters = 1000
    np.random.seed(123)

    coefficients = np.zeros([n_boot_iters, len(ind_vars) + 1])
    for ii in range(n_boot_iters):
        ind = np.random.choice(n_ind, n_ind, replace=True)
        coefficients[ii, :] = boot_fn(df, ind)

    standard_errors = stan_errs(coefficients)
    print('            Coefficient  Standard Error')
    print('Intercept:  {:.3e}   {:.3e}'.format(np.mean(coefficients[:, 0]), standard_errors[0]))
    print('income:     {:.3e}   {:.3e}'.format(np.mean(coefficients[:, 1]), standard_errors[1]))
    print('balance:    {:.3e}   {:.3e}'.format(np.mean(coefficients[:, 2]), standard_errors[2]))


if __name__ == '__main__':
    main()
