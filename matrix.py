# from itertools import (takewhile,repeat)
import numpy as np
import os
import re


# def count_lines(f):
#     """
#     Fast line counter.
#     f has to be opened as a binary ('rb').
#     """
#     bufgen = takewhile(lambda x: x, (f.read(1024*1024) for _ in repeat(None)))
#     return sum( buf.count(b'\n') for buf in bufgen )

def process():
    """
    Dump a NumPy array to a file 'matrix.raw' 
    with as many lines as samples,
    and two columns:
    - couples (parent, child)
    - lines of behavior_sequence files.

    The file can then be loaded with numpy.load(file).
    """
    folder = '/home/cloud/hackathon/'
    training_dir = folder + 'training_dataset/'
    matrix_dir = folder + 'palmyre/'
    matrix = []

    for i in range (1, 20000):
        index = str(i).zfill(6)
        with open(training_dir + 'training_{}_process_generation.txt'.format(index), 'r') as f:
            processes = []
            for line in f:
                spawn = re.search(u'(.*) -> (.*)', line)
                if  spawn:
                    processes.append((spawn.group(1), spawn.group(2)))
        with open(training_dir + 'training_{}_behavior_sequence.txt'.format(index), 'r') as f:
            lines = [line for line in f]
        matrix.append([processes, lines])
    ndarray = np.array(matrix)
    ndarray.dump('matrix.raw')

if __name__ == '__main__':
    process()