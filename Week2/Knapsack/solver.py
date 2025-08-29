#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import time
from operator import itemgetter

Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    init_time = time.time()

    ## reverse ordering the items reduce the operations in DP and
    ## enables redefining the equations as O(k,j) = max(O(k,j-1),Vj+O(k-Wj,j-1)) in all cases.
    items.sort(reverse=True, key=itemgetter(2))

    print(items)

    ##O(k,j) are computed by observing only current and previos column.
    origin_row = np.zeros((capacity + 1, 1), dtype=np.int32)
    destination_row = np.zeros((capacity + 1, 1), dtype=np.int32)

    ##A boolean matrix is used to preserve the relationship between consecutives columns.
    back_tracing = np.matrix(np.zeros((capacity + 1, item_count + 1), dtype=bool))

    for j in range(1, item_count + 1):
        destination_row[items[j - 1][2]:] = np.maximum(origin_row[:][items[j - 1][2]:], (
                    np.roll(origin_row[:], items[j - 1][2], axis=0)[items[j - 1][2]:] + items[j - 1][1]))
        back_tracing[:, j] = destination_row[:] == origin_row[:]
        origin_row[:] = destination_row[:]

    ##backtracing is performed from the bottom-right corner and observing the entries in the
    ##boolean matrix.
    pointer = (capacity, item_count)
    taken = [0] * len(items)
    for i in range(item_count - 1, -1, -1):
        if back_tracing[pointer] == True:
            pointer = (pointer[0], pointer[1] - 1)
        else:
            pointer = (pointer[0] - items[pointer[1] - 1][2], pointer[1] - 1)
            taken[i] = 1

    ##selected items are reordered to preserve the initial order.
    indexes = [it[0] for it in items]
    taken = [i for _, i in sorted(zip(indexes, taken))]

    value = destination_row[-1][0]

    ##Prepare the solution in the specified output format.
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

