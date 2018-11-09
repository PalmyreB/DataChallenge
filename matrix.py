# from itertools import (takewhile,repeat)
import re

folder = '/home/palmyre/Documents/Huawei Data Challenge/sub_training_set/'

# def count_lines(f):
#     bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
#     return sum( buf.count(b'\n') for buf in bufgen )

def process(i_beg = 1, i_end = 6):
    """
    Create a file `matrix.txt` with number of processes and number of nodes (lines).

    i_beg : index of first sample
    i_end : index+1 of last sample
    """
    matrix = []
    for i in range (i_beg, i_end):
        index = str(i).zfill(6)
        with open(folder + 'training_{}_process_generation.txt'.format(index), 'r') as f:
            processes = set()
            for line in f:
                child = re.search(u'(?<=-> )\w+\.exe', line)
                processes.add(child.group(0))
            nb_processes = len(processes)
        with open(folder + 'training_{}_behavior_sequence.txt'.format(index), 'r') as f:
            nb_lines =  sum(1 for line in f)
        matrix.append([nb_processes, nb_lines])
    with open(folder + 'matrix.txt', 'w') as m:
        m.write(str(matrix))

if __name__ == '__main__':
    process()