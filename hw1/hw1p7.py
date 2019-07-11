"""Performs analysis for STAT 202 HW 1 P 7, Sum 2019"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.cluster.hierarchy as shc

def main():
    """"Performs clustering on USArrests dataset."""
    dataset_arrest = sm.datasets.get_rdataset(dataname='USArrests', package='datasets')
    df_arrest = dataset_arrest.data

    df_data = df_arrest.values
    df_labels = df_arrest.index.values

    # Part a
    link = shc.linkage(df_data, method='complete', metric='euclidean')
    plt.figure()
    shc.dendrogram(link, labels=df_labels)
    plt.title("US Arrests Clustering")
    plt.savefig('plots/7a.png')

    # Part b
    cut = shc.fcluster(link, 3, criterion='maxclust')
    for cluster in enumerate(zip(df_labels, cut)):
        print(cluster)

    # Part c
    df_data_norm = (df_data - np.mean(df_data, axis=0)) / np.std(df_data, axis=0)
    link_norm = shc.linkage(df_data_norm, method='complete', metric='euclidean')
    plt.figure()
    shc.dendrogram(link_norm, labels=df_labels)
    plt.title("US Arrests Clustering Normalized")
    plt.savefig('plots/7b.png')

if __name__ == '__main__':
    main()
