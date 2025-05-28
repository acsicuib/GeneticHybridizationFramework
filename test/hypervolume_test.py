from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv  import HV

import numpy as np
import pickle
import math
import time
import random as rnd
import sys
import logging



OBJ_LIST_LEN = 7

def get_ref_dirs_len(dim, partitions):
    # Combinations
    numerator = 1
    for n in range(partitions + 1, dim + partitions):
        numerator *= n
    return numerator / math.factorial(dim - 1)

ref_dirs_len_matrix = np.array([[
        get_ref_dirs_len(d,p)
        for p in range(1, 101)
    ] for d in range(3, OBJ_LIST_LEN+1)], dtype=np.uint32)

def get_partitions_from_population(dim, population):
    if   dim == 1: return 0
    elif dim == 2: return population - 1

    part = 1
    while ref_dirs_len_matrix[dim-3][part-1] <= population:
        part += 1
    return part - 1


"""
P = (2, 0.0033597946166992188)
Q = (3, 0.011887311935424805)
R = (4, 0.04232287406921387)
S = (5, 0.1781768798828125)
T = (6, 1.2251098155975342)
U = (7, 1.701786756515503)
"""


if __name__ == '__main__':
    logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(message)s')
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pop_size = 200
    executions = 5

    print('p_size\tn_obj\ttime\n')
    for pop_size in range(50, 1001, 50):
        for n_obj in range(2, OBJ_LIST_LEN+1):
            n_partitions = get_partitions_from_population(
                    n_obj, pop_size
                ) + 1

            ref_dirs = get_reference_directions(
                    'das-dennis',
                    n_obj,
                    n_partitions=n_partitions
                ) / 2

            ind = HV(np.ones((n_obj)))

            curr_ex = 0
            mean_time = 0.
            try:
                while curr_ex < executions:
                    logging.debug('pop_size = {}, n_obj = {}, execution = {}'.format(
                            pop_size, n_obj, curr_ex + 1))
                    subset = ref_dirs[rnd.sample(range(0, len(ref_dirs)), pop_size)]

                    start = time.time()
                    solution = ind(subset)
                    end = time.time()

                    mean_time += end - start
                    curr_ex += 1

            except KeyboardInterrupt:
                if curr_ex == 0: curr_ex += 1 # avoid division by zero

            mean_time /= curr_ex

            print('{}\t{}\t{}'.format(pop_size, n_obj, mean_time))

        print()

