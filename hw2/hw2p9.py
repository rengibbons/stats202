"""Performs analysis for STAT 202 HW 2 P 9, Sum 2019"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def loocv(df, response_var, pred_vars):
    """Computes the LOOCV error for OLS."""
    MSE = 0
    n = len(df)
    for ii in range(n):
        ind_train = np.setdiff1d(np.arange(n), ii)
        X_train = df.loc[ind_train, pred_vars]
        Y_train = df.loc[ind_train, response_var]
        X_test = df.loc[ii, pred_vars]
        Y_test = df.loc[ii, response_var]
        model = LinearRegression().fit(X_train, Y_train)

        X_test = np.transpose(X_test.values)
        MSE += np.power(Y_test - model.predict([X_test]), 2)

    return MSE[0] / n
 

def main():
    """Performs LOOCV on four least squares models."""
    df = pd.read_csv('data/ch5_q8_simulation.csv')

    # Part b
    plt.figure()
    plt.scatter(df['x'], df['y'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatterplot y vs. x')
    plt.savefig('plots/8a.png')

    # Part c
    response_var = 'y'
    pred_vars_lin = ['x']
    pred_vars_quad = ['x', 'x2']
    pred_vars_cub = ['x', 'x2', 'x3']
    pred_vars_quar = ['x', 'x2', 'x3', 'x4']

    poly_terms = pd.DataFrame({'x2': np.power(df['x'], 2),
                               'x3': np.power(df['x'], 3),
                               'x4': np.power(df['x'], 4)})
    df = pd.concat([df, poly_terms], axis=1)

    CV_error_lin = loocv(df, response_var, pred_vars_lin)
    CV_error_quad = loocv(df, response_var, pred_vars_quad)
    CV_error_cub = loocv(df, response_var, pred_vars_cub)
    CV_error_quar = loocv(df, response_var, pred_vars_quar)

    print('Part c')
    print('CV error (linear)    = {:.3f}'.format(CV_error_lin))
    print('CV error (quadratic) = {:.3f}'.format(CV_error_quad))
    print('CV error (cubic)     = {:.3f}'.format(CV_error_cub))
    print('CV error (quartic)   = {:.3f}'.format(CV_error_quar))

    # Part d
    np.random.seed(801)
    y = np.random.randn(100)
    x = np.random.randn(100)
    y = x - 2 * np.power(x, 2) + np.random.randn(100)

    df = pd.DataFrame({'x': x,
                       'x2': np.power(x, 2),
                       'x3': np.power(x, 3),
                       'x4': np.power(x, 4),
                       'y': y})

    CV_error_lin = loocv(df, response_var, pred_vars_lin)
    CV_error_quad = loocv(df, response_var, pred_vars_quad)
    CV_error_cub = loocv(df, response_var, pred_vars_cub)
    CV_error_quar = loocv(df, response_var, pred_vars_quar)

    print('Part d')
    print('CV error (linear)    = {:.3f}'.format(CV_error_lin))
    print('CV error (quadratic) = {:.3f}'.format(CV_error_quad))
    print('CV error (cubic)     = {:.3f}'.format(CV_error_cub))
    print('CV error (quartic)   = {:.3f}'.format(CV_error_quar))

    # Part f
    model = sm.OLS(df.loc[:, response_var], df.loc[:, pred_vars_quar]).fit()
    print(model.summary())


if __name__ == '__main__':
    main()
