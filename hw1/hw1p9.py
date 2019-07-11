"""Performs analysis for STAT 202 HW 1 P 9, Sum 2019"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    """"Performs various mult lin reg operaitons on Auto data set."""
    dataset_auto = sm.datasets.get_rdataset(dataname='Auto', package='ISLR')
    df_auto = dataset_auto.data
    df_auto.columns = ['mpg', 'cyl', 'displ', 'HP', 'weight', 'accel', 'year', 'origin', 'name']

    # Part a
    plt.figure()
    pd.plotting.scatter_matrix(df_auto, alpha=0.5, diagonal='kde')
    plt.savefig('plots/9a.png')

    # Part b
    corr_matrix = df_auto.corr()
    print(corr_matrix)

    # Part c
    model = smf.ols(formula='mpg ~ cyl + displ + HP + weight + accel + year + C(origin)',
                    data=df_auto).fit()
    print(model.summary())

    # Part d
    mpg = df_auto.filter(items=['mpg'])
    mpg_pred = model.fittedvalues
    model_leverage = model.get_influence().hat_matrix_diag
    model_norm_residuals = model.get_influence().resid_studentized_internal

    plt.figure()
    sns_plot = sns.residplot(mpg_pred, mpg)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    fig = sns_plot.get_figure()
    fig.savefig('plots/9b.png')

    plt.figure()
    plt.scatter(model_leverage, model_norm_residuals)
    plt.xlabel('Leverage')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Leverage')
    plt.savefig('plots/9c.png')

    # Part e
    model = smf.ols(formula='mpg ~ cyl + displ * HP + HP * weight + accel + year + C(origin)',
                    data=df_auto).fit()
    print(model.summary())

    # Part f
    # mpg square root
    df_auto['mpg'] = np.sqrt(mpg)
    model = smf.ols(formula='mpg ~ cyl + displ + HP + weight + accel + year + C(origin)',
                    data=df_auto).fit()
    print(model.summary())

    # mpg squared
    df_auto['mpg'] = np.square(mpg)
    model = smf.ols(formula='mpg ~ cyl + displ + HP + weight + accel + year + C(origin)',
                    data=df_auto).fit()
    print(model.summary())

if __name__ == '__main__':
    main()
