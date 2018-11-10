import numpy as np

if __name__ == '__main__':
    n= 1000
    M = np.load('matrix.raw')
    M2 = M[:n]
    M2.dump('sample.raw')
