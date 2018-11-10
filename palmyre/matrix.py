import numpy as np
import os


def process():
    """
    Dump a NumPy array to a file 'matrix.npy' 
    with as many lines as samples,
    and two columns:
    - couples (parent, child)
    - lines of behavior_sequence files.

    The file can then be loaded with np.load('matrix.npy').
    """
    n = 200000
    matrix = np.empty((n, 2), dtype=object)
    training_dir = '/home/cloud/hackathon/training_dataset/'

    for i in range(1,n):
        index = str(i).zfill(6)
        with open(training_dir + 'training_{}_process_generation.txt'.format(index), 'r') as f:
            processes = []
            for line in f:
                spawn = line.split(' -> ')
                processes.append((spawn[0], spawn[1]))
        with open(training_dir + 'training_{}_behavior_sequence.txt'.format(index), 'r') as f:
            lines = [line for line in f]
        matrix[i][0] = processes
        matrix[i][1] = lines
    ndarray = np.array(matrix)
    np.save('matrix.npy', ndarray)
    
    

if __name__ == '__main__':
    process()