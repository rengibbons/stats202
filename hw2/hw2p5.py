"""Performs analysis for STAT 202 HW 2 P 5, Sum 2019"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm


def confusion_matrix(model, x_test, y_test):
    """"Prints the confusion matrix."""
    pred = np.array(model.predict(x_test) > 0.5, dtype=float)
    return np.transpose(np.histogram2d(y_test, pred, bins=2)[0])


def main():
    """"Performs Weekly data set."""
    dataset = sm.datasets.get_rdataset(dataname='Weekly', package='ISLR')
    df_weekly = dataset.data

    # Part a
    print("==== Columns of Weekly dataset ====")
    print(df_weekly.columns)
    print('==== Description of Weekly dataset ====')
    print(df_weekly.describe())
    print("==== Correlations of Weekly dataset ====")
    print(df_weekly.corr())

    plt.figure()
    pd.plotting.scatter_matrix(df_weekly, alpha=0.5, diagonal='kde')
    plt.savefig('plots/5a.png')

    ## Part b
    ind_vars = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
    df_weekly['Binary Direction'] = df_weekly['Direction'] == 'Up'
    Y = df_weekly['Binary Direction']
    X = sm.add_constant(df_weekly[ind_vars])
    model = sm.Logit(Y, X).fit()
    print(model.summary())

    ## Part c
    pred_table = np.transpose(model.pred_table())
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)

    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

    ## Part d
    ind_vars = ['Lag2']
    df_weekly_pred = df_weekly.loc[df_weekly['Year'] < 2009]
    df_weekly_test = df_weekly.loc[df_weekly['Year'] >= 2009]
    y_pred = df_weekly_pred['Binary Direction']
    y_test = df_weekly_test['Binary Direction']
    x_pred = sm.add_constant(df_weekly_pred[ind_vars])
    x_test = sm.add_constant(df_weekly_test[ind_vars])

    model = sm.Logit(y_pred, x_pred).fit()
    pred_table = confusion_matrix(model, x_test, y_test)
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)
    print(model.summary())
    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

    ## Part e
    model = LinearDiscriminantAnalysis()
    model.fit(x_pred, y_pred)
    pred_table = confusion_matrix(model, x_test, y_test)
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)
    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

    ## Part f
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_pred, y_pred)
    pred_table = confusion_matrix(model, x_test, y_test)
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)
    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

    ## Part g
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(x_pred, y_pred)
    pred_table = confusion_matrix(model, x_test, y_test)
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)
    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

    ## Part i
    ind_vars = ['Lag2', 'Year', 'Volume', 'Today']
    x_pred = sm.add_constant(df_weekly_pred[ind_vars])
    x_test = sm.add_constant(df_weekly_test[ind_vars])

    model = LinearDiscriminantAnalysis()
    model.fit(x_pred, y_pred)
    pred_table = confusion_matrix(model, x_test, y_test)
    fraction_correct = np.trace(pred_table) / np.sum(pred_table)
    print('====== LDA ======')
    print(ind_vars)
    print(pred_table)
    print('fraction correct = {:.3f}'.format(fraction_correct))

if __name__ == '__main__':
    main()
