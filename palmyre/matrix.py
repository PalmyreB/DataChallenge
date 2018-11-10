import numpy as np
import os
from tqdm import tqdm


def process():
    """
    Dump a NumPy array to a file 'matrix_training.npy' 
    with as many lines as samples,
    and two columns:
    - couples (parent, child)
    - lines of behavior_sequence files.

    The file can then be loaded with np.load('matrix_training.npy').
    """
    n = 200000
    max_len = n/10
    matrix = np.empty((max_len, 2), dtype=object)
    training_dir = '/home/cloud/hackathon/training_dataset/'

    for i in tqdm(range(3*max_len+1,n)):
        index = str(i).zfill(6)
        j = i % max_len
        with open(training_dir + 'training_{}_process_generation.txt'.format(index), 'r') as f:
            processes = []
            for line in f:
                spawn = line.split(' -> ')
                processes.append((spawn[0], spawn[1]))
        with open(training_dir + 'training_{}_behavior_sequence.txt'.format(index), 'r') as f:
            lines = f.readlines()
        matrix[j][0] = processes
        matrix[j][1] = lines

        # Create a new .npy file every max_len iterations.
        if j == 0:
            ndarray = np.array(matrix)
            np.save('matrix_training_{}.npy'.format(int(i/max_len)), ndarray)
    
    # Concatenate all .npy files.
    concatArray = np.ndarray('matrix_training_1.npy')
    for i in range(2, int(n / max_len)):
        f = 'matrix_training_{}.npy'.format(i)
        concatArray = np.concatenate((concatArray, np.load(f)), axis=0)
        os.remove(f)
    np.save('matrix_training.npy', concatArray)
    
    

if __name__ == '__main__':
    process()