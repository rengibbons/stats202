import numpy as np

def main():
    pt = np.array([0, 0, 0])

    obs = np.array([[0, 3, 0],
                    [2, 0, 0],
                    [0, 1, 3],
                    [0, 1, 2],
                    [-1, 0, 1],
                    [1, 1, 1]])

    dist = np.zeros(obs.shape[0])
    
    for i in range(obs.shape[0]):
        dist[i] = pow(pow(pt[0] - obs[i,0],2) + pow(pt[1] - obs[i,1],2) + pow(pt[2] - obs[i,2],2), 0.5)
 
    print(dist)


if __name__ == '__main__':
    main()
