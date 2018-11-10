import numpy as np
import os


def process():
    """
    Dump a NumPy array to a file 'matrix_validation.npy' 
    with as many lines as samples,
    and two columns:
    - couples (parent, child)
    - lines of behavior_sequence files.

    The file can then be loaded with np.load('matrix_validation.npy').
    """
    n = 40000
    matrix = np.empty((n, 2), dtype=object)
    training_dir = '/home/cloud/hackathon/validation_dataset/'

    for i in range(1,n):
        index = str(i).zfill(6)
        with open(training_dir + 'validation_{}_process_generation.txt'.format(index), 'r') as f:
            processes = []
            for line in f:
                spawn = line.split(' -> ')
                processes.append((spawn[0], spawn[1]))
        with open(training_dir + 'validation_{}_behavior_sequence.txt'.format(index), 'r') as f:
            lines = [line for line in f]
        matrix[i][0] = processes
        matrix[i][1] = lines
        if i % 100 == 0:
            print(i)
    ndarray = np.array(matrix)
    np.save('matrix_validation.npy', ndarray)
    
    

if __name__ == '__main__':
    process()