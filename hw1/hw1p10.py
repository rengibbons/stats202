"""Performs analysis for STAT 202 HW 1 P 10, Sum 2019"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    # """"Performs various mult lin reg operaitons on Auto data set."""
    # Part a
    df = pd.read_csv('data/ch3_q14_simulation.csv')

    # Part b
    corr_matrix = df.corr()
    print('PART B - CORRELATION MATRIX')
    print(corr_matrix)

    plt.figure()
    plt.scatter(df['x1'],df['x2'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Features Scatter Plot')
    plt.savefig('plots/10a.png')
    
    # Part c
    model = smf.ols(formula='y ~ x1 + x2', data=df).fit()
    print('PART C - x1 + x2')
    print(model.summary())

    # Part d
    model = smf.ols(formula='y ~ x1', data=df).fit()
    print('PART D - x1')
    print(model.summary())
    
    # Part e 
    model = smf.ols(formula='y ~ x2', data=df).fit()
    print('PART E - x2')
    print(model.summary())

    # Part g
    x1 = np.append(df.filter(items=['x1']).values, 0.1)
    x2 = np.append(df.filter(items=['x2']).values, 0.8)
    y = np.append(df.filter(items=['y']).values, 6.0)

    data_set = {'x1': x1, 'x2': x2, 'y': y}
    df = pd.DataFrame(data=data_set)

    plt.scatter(df['x1'],df['x2'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Features Scatter Plot')
    plt.savefig('plots/10b.png')

    model = smf.ols(formula='y ~ x1 + x2', data=df).fit()
    print('PART G - x1 + x2')
    print(model.summary())
    y_pred = model.fittedvalues
    model_leverage = model.get_influence().hat_matrix_diag
    model_norm_residuals = model.get_influence().resid_studentized_internal

    model = smf.ols(formula='y ~ x1', data=df).fit()
    print('PART G - x1')
    print(model.summary())
    
    model = smf.ols(formula='y ~ x2', data=df).fit()
    print('PART G - x2')

    plt.figure()
    sns_plot = sns.residplot(y_pred, y)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    fig = sns_plot.get_figure()
    fig.savefig('plots/10c.png')

    plt.figure()
    plt.scatter(model_leverage, model_norm_residuals)
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Leverage')
    plt.savefig('plots/10d.png')

    print(model.summary())


if __name__ == '__main__':
    main()
