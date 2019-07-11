"""Performs analysis for STAT 202 HW 2 P 6, Sum 2019"""
import numpy as np
from matplotlib import pyplot as plt

def bootstrap_probability(n):
    """Computes probability jth observation is in bootstrap sample."""
    return 1 - (1 - 1 / n) ** n

def main():
    """"Performs simple bootstrap probabilities."""
    # Part d
    n = 5
    print('Pr(in|n={:.0f}) = {:.2f}'.format(n, bootstrap_probability(n)))

    # Part e 
    n = 100
    print('Pr(in|n={:.0f}) = {:.2f}'.format(n, bootstrap_probability(n)))

    # Part f 
    n = 10000
    print('Pr(in|n={:.0f}) = {:.2f}'.format(n, bootstrap_probability(n)))

    # Part g
    N = 100000
    n = np.arange(N) + 1
    pr = np.zeros(N)
    for ii in range(N):
        pr[ii] = bootstrap_probability(ii+1)
        
    plt.figure()
    plt.scatter(n, pr)
    plt.xlabel('Observation Set Size')
    plt.ylabel('Probability')
    plt.title('Probability jth Sample in Bootstrap Sample')
    plt.savefig('plots/6a.png') 

    # Part h
    n_iters = 10000
    n = 100
    j = 4
    
    count = 0
    for ii in range(n_iters): 
        count += np.sum(np.random.choice(np.arange(n)+1, size=n, replace=True) == j) > 0

    pr = count / n_iters
    print('Pr(j={:.0f} in bootstrap) = {:.4f}'.format(j, pr))


if __name__ == '__main__':
    main()
