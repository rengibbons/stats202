"""Performs analysis for STAT 202 HW 2 P 7, Sum 2019"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_error_rate(df, ind_vars, seed):
    """Computes the logistic regression error rate on a test set."""
    (n, p) = df.shape
    n_train = int(0.5 * n)

    # Split data into train and test sets.
    # Task (i)
    np.random.seed(seed)
    ind_train = np.sort(np.random.choice(n, n_train, replace=False))
    ind_test = np.setdiff1d(np.arange(n), ind_train)
    df_train = df.loc[ind_train]
    df_test = df.loc[ind_test]

    y_train = df_train['bool default']
    y_test = df_test['bool default']
    x_train = sm.add_constant(df_train[ind_vars])
    x_test = sm.add_constant(df_test[ind_vars])

    # Fit logistic regression model and predict error rate.
    # Task (ii)
    model = sm.Logit(y_train, x_train).fit()

    # Task (iii)
    pred_table = confusion_matrix(model, x_test, y_test)

    # Task (iv)
    error_rate = 1 - np.trace(pred_table) / np.sum(pred_table)
    return error_rate


def confusion_matrix(model, x_test, y_test):
    """"Prints the confusion matrix."""
    pred = np.array(model.predict(x_test) > 0.5, dtype=float)
    return np.transpose(np.histogram2d(y_test, pred, bins=2)[0])


def main():
    """Computes various logistic regression using validation set method."""
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

    # Part b
    # Compute error rate using validation set approach three times.
    seed = np.array([801, 859, 989, 123])
    error_rate = np.zeros(len(seed))
    for ii in range(len(seed)):
        error_rate[ii] = compute_error_rate(df, ind_vars, seed[ii])

    print('1st error rate = {:.5f}'.format(error_rate[0]))
    print('2nd error rate = {:.5f}'.format(error_rate[1]))
    print('3rd error rate = {:.5f}'.format(error_rate[2]))
    print('4th error rate = {:.5f}'.format(error_rate[3]))

    # Part d
    # Add student as a dummy variable to dataframe.
    dummy_student = pd.get_dummies(df['student'], prefix='student')
    df = df.join(dummy_student.loc[:, 'student_Yes'])
    ind_vars = ['income', 'balance', 'student_Yes']

    # Compute error rate using validation set approach three times (with student).
    error_rate = np.zeros(len(seed))
    for ii in range(len(seed)):
        error_rate[ii] = compute_error_rate(df, ind_vars, seed[ii])

    print('1st error rate (w/ student dummy) = {:.5f}'.format(error_rate[0]))
    print('2nd error rate (w/ student dummy) = {:.5f}'.format(error_rate[1]))
    print('3rd error rate (w/ student dummy) = {:.5f}'.format(error_rate[2]))
    print('4th error rate (w/ student dummy) = {:.5f}'.format(error_rate[3]))


if __name__ == '__main__':
    main()
