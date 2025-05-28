from pymoo.core.sampling   import Sampling
from pymoo.core.crossover  import Crossover
from pymoo.core.mutation   import Mutation
from pymoo.core.duplicate  import ElementwiseDuplicateElimination
from pymoo.core.repair     import Repair
from pymoo.core.callback   import Callback
from pymoo.core.population import Population

import numpy as np
import random
import datetime

# ==============================================================================
# SAMPLINGS
# ==============================================================================
class MySampling_v1(Sampling):
    """
    Ignore replicas. Start with 1 replica for each service.
    """
    def __init__(self, n_replicas=1, n_algs=1, alg_idx=0, n_steps=0):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_algs = n_algs
        self.alg_idx = alg_idx
        self.n_steps = 1 if n_steps == 0 else n_steps

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1 if self.n_algs == 1 else 2),  None, dtype=object)
        for i in range(n_samples):
            matrix = np.zeros((problem.N_TASKS, problem.N_NODES), np.uint8)
            for row in range(problem.N_TASKS):
                col = random.randrange(problem.N_NODES)
                matrix[row, col] = 1

            X[i,0] = matrix

            if self.n_algs > 1:
                X[i,1] = np.zeros((self.n_steps, self.n_algs))
                X[i,1][:, self.alg_idx] = 1.

        return X

class MySampling_v2(Sampling):
    """
    Distribute a random amount of replicas so that 1 <= replicas <= N_REPLICAS
    for each row.
    """
    def __init__(self, n_replicas=1, n_algs=1, alg_idx=0, n_steps=0):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_algs = n_algs
        self.alg_idx = alg_idx
        self.n_steps = 1 if n_steps == 0 else n_steps

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1 if self.n_algs == 1 else 2),  None, dtype=object)
        for i in range(n_samples):
            matrix = np.zeros((problem.N_TASKS, problem.N_NODES), np.uint8)
            for row in range(problem.N_TASKS):
                col_list = random.sample(
                        range(problem.N_NODES),
                        random.randint(1, self.n_replicas)
                    )
                for col in col_list:
                    matrix[row, col] = 1

            X[i,0] = matrix

            if self.n_algs > 1:
                X[i,1] = np.zeros((self.n_steps, self.n_algs))
                X[i,1][:, self.alg_idx] = 1.

        return X

class MySampling_v3(Sampling):
    """
    Distribute replicas along selected nodes.
    """
    def __init__(self, n_replicas=1, n_algs=1, alg_idx=0, n_steps=0):
        super().__init__()
        self.n_replicas = n_replicas
        self.n_algs = n_algs
        self.alg_idx = alg_idx
        self.n_steps = 1 if n_steps == 0 else n_steps

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1 if self.n_algs == 1 else 2),  None, dtype=object)
        for i in range(n_samples):
            matrix = np.zeros((problem.N_TASKS, problem.N_NODES), np.uint8)
            sel_nodes = random.sample(
                    range(problem.N_NODES),
                    random.randint(self.n_replicas, problem.N_NODES)
                )
            for row in range(problem.N_TASKS):
                col_list = random.sample(
                        sel_nodes,
                        random.randint(1, self.n_replicas)
                    )
                for col in col_list:
                    matrix[row, col] = 1

            X[i,0] = matrix

            if self.n_algs > 1:
                X[i,1] = np.zeros((self.n_steps, self.n_algs))
                X[i,1][:, self.alg_idx] = 1.

        return X

# ==============================================================================
# CROSSOVERS
# ==============================================================================
class MyCrossover_v1(Crossover):
    """
    Do not ensure anything. Repair operator will take effect.
    """

    def __init__(self, n_replicas=1):
        super().__init__(n_parents = 2, n_offsprings = 2)
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape[:3]
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]

            # Cross 2D (rows and cols)
            Y[0, k, 0] = np.zeros((problem.N_TASKS, problem.N_NODES), np.uint8)
            Y[1, k, 0] = np.zeros((problem.N_TASKS, problem.N_NODES), np.uint8)

            for i in range(problem.N_TASKS):
                for j in range(problem.N_NODES):
                    if random.random() < 0.5:
                        Y[0, k, 0][i, j] = p1[i, j]
                        Y[1, k, 0][i, j] = p2[i, j]
                    else: 
                        Y[0, k, 0][i, j] = p2[i, j]
                        Y[1, k, 0][i, j] = p1[i, j]

            # Genetic load
            if n_var > 1:
                p1_inh, p2_inh = X[0, k, 1], X[1, k, 1]
                ch_inh = (p1_inh + p2_inh) / 2
                Y[0, k, 1], Y[1, k, 1] = ch_inh, ch_inh

        return Y

class MyCrossover_v2(Crossover):
    """
    Ensure that offsprings keep the restriction 1 <= replicas <= N_REPLICAS
    Offsprings are not codependent.
    """
    # How do we ensure that offsprings keep this restriction for each row?
    #
    #   1 <= #1 <= N_REPLICAS
    #
    # Given two parents:
    #
    #   0  1  1  0  0       0  1  1  0  1 
    #   1  0  1  1  0       1  1  0  1  1 
    #   0  1  0  1  1       0  0  1  1  0 
    #   1  0  1  1  0       1  0  1  0  1 
    #   0  0  1  0  0       0  1  1  0  0 
    #
    # Find their differences:
    #
    #   0  1  1  0 [0]      0  1  1  0 [1]
    #   1 [0][1] 1 [0]      1 [1][0] 1 [1]
    #   0 [1][0] 1 [1]      0 [0][1] 1 [0]
    #   1  0  1 [1][0]      1  0  1 [0][1]
    #   0 [0] 1  0  0       0 [1] 1  0  0 
    #
    # These are the cells that may change during crossover on each offspring.
    #
    # For each row of each offspring, ensure that:
    #
    #   1 <= #1 + #(0 -> 1) - #(1 -> 0) <= N_REPLICAS
    #
    # We can only choose values for #(0 -> 1) and #(1 -> 0):
    #
    #   1 - #1 <= #(0 -> 1) - #(1 -> 0) <= N_REPLICAS - #1
    #
    # So we have:
    #
    #   #0->1 \ #1->0 |  0   1   2   3   4   5 ...
    #   --------------+---------------------------
    #               0 |[ 0][-1] -2  -3  -4  -5  
    #               1 |[ 1][ 0][-1] -2  -3  -4  
    #               2 |[ 2][ 1][ 0][-1] -2  -3  
    #               3 |  3 [ 2][ 1][ 0][-1] -2  
    #               4 |  4   3 [ 2][ 1][ 0][-1] 
    #               5 |  5   4   3 [ 2][ 1][ 0] 
    #             ... | 
    #
    # Notice how the two constraints bound the space of solutions diagonally.
    #
    # Second offspring does not depend on first offspring. Both offsprings are
    # made the same way.

    def __init__(self, n_replicas=1):
        super().__init__(n_parents = 2, n_offsprings = 2)
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape[:3]
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]

            if self.n_replicas == 1:
                # Cross rows
                off1, off2 = [], []
                for i in range(problem.N_TASKS):
                    if random.random() < 0.5:
                        off1.append(p1[i])
                        off2.append(p2[i])
                    else:
                        off1.append(p2[i])
                        off2.append(p1[i])
                Y[0, k, 0], Y[1, k, 0] = np.array(off1, np.uint8), np.array(off2, np.uint8)
            else:
                # Cross 2D (rows and cols)
                Y[0, k, 0], Y[1, k, 0] = p1.copy(), p1.copy()

                p_diff = np.subtract(p1, p2, dtype=np.int8)
                    #  0: 0->0 or 1->1
                    # -1: 0->1
                    #  1: 1->0 

                for i in range(problem.N_TASKS):
                    if not p_diff[i].any(): continue

                    arr01 = np.flatnonzero(p_diff[i] == -1)
                    arr10 = np.flatnonzero(p_diff[i] ==  1)

                    max01 = arr01.size
                    max10 = arr10.size

                    n1p1 = np.flatnonzero(p1[i]).size

                    lbound = 1 - n1p1
                    ubound = self.n_replicas

                    for o in range(2):
                        n01, n10 = random.choice([
                                (i,j)
                                for i in range(max01 + 1)
                                for j in range(max10 + 1)
                                if lbound <= i - j <= ubound
                            ])

                        j01 = random.sample(arr01.tolist(), n01)
                        j10 = random.sample(arr10.tolist(), n10)

                        for j in j01:
                            Y[o, k, 0][i, j] = 1

                        for j in j10:
                            Y[o, k, 0][i, j] = 0

            # Genetic load
            if n_var > 1:
                p1_inh, p2_inh = X[0, k, 1], X[1, k, 1]
                ch_inh = (p1_inh + p2_inh) / 2
                Y[0, k, 1], Y[1, k, 1] = ch_inh, ch_inh

        return Y


class MyCrossover_v3(Crossover):
    """
    Ensure that offsprings keep the restriction 1 <= replicas <= N_REPLICAS
    Offsprings are codependent.
    """

    # How do we ensure that offsprings keep this restriction for each row?
    #
    #   1 <= #1 <= N_REPLICAS
    #
    # Given two parents:
    #
    #   0  1  1  0  0       0  1  1  0  1 
    #   1  0  1  1  0       1  1  0  1  1 
    #   0  1  0  1  1       0  0  1  1  0 
    #   1  0  1  1  0       1  0  1  0  1 
    #   0  0  1  0  0       0  1  1  0  0 
    #
    # Find their differences:
    #
    #   0  1  1  0 [0]      0  1  1  0 [1]
    #   1 [0][1] 1 [0]      1 [1][0] 1 [1]
    #   0 [1][0] 1 [1]      0 [0][1] 1 [0]
    #   1  0  1 [1][0]      1  0  1 [0][1]
    #   0 [0] 1  0  0       0 [1] 1  0  0 
    #
    # These are the cells that may change during crossover on each offspring.
    #
    # For each row of first offspring, ensure that:
    #
    #   1 <= #1 + #(0 -> 1) - #(1 -> 0) <= N_REPLICAS
    #
    # We can only choose values for #(0 -> 1) and #(1 -> 0):
    #
    #   1 - #1 <= #(0 -> 1) - #(1 -> 0) <= N_REPLICAS - #1
    #
    # So we have:
    #
    #   #0->1 \ #1->0 |  0   1   2   3   4   5 ...
    #   --------------+---------------------------
    #               0 |[ 0][-1] -2  -3  -4  -5  
    #               1 |[ 1][ 0][-1] -2  -3  -4  
    #               2 |[ 2][ 1][ 0][-1] -2  -3  
    #               3 |  3 [ 2][ 1][ 0][-1] -2  
    #               4 |  4   3 [ 2][ 1][ 0][-1] 
    #               5 |  5   4   3 [ 2][ 1][ 0] 
    #             ... | 
    #
    # Notice how the two constraints bound the space of solutions diagonally.
    #
    # For second offspring relative to first, we have that #(0 -> 1) becomes
    # #(1 -> 0) and #(1 -> 0) becomes #(0 -> 1), so we have two restrictions:
    #
    #   1 - #1_o1 <= #(0 -> 1) - #(1 -> 0) <= N_REPLICAS - #1_o1
    #   1 - #1_o2 <= #(1 -> 0) - #(0 -> 1) <= N_REPLICAS - #1_o2
    #
    # Notice that #1 is different for both offsprings.
    #
    # With the resulting table for second restriction (notice that signs have
    # changed):
    #
    #   #0->1 \ #1->0 |  0   1   2   3   4   5 ...
    #   --------------+---------------------------
    #               0 |[ 0]  1   2   3   4   5  
    #               1 |[-1][ 0]  1   2   3   4  
    #               2 |[-2][-1][ 0]  1   2   3  
    #               3 |[-3][-2][-1][ 0]  1   2  
    #               4 | -4 [-3][-2][-1][ 0]  1  
    #               5 | -5  -4 [-3][-2][-1][ 0] 
    #             ... | 
    #
    # Notice that second restriction can be rewriten as:
    #
    #           1 - #1_o2   <=    #(1 -> 0) - #(0 -> 1)   <=    N_REPLICAS - #1_o2
    #                                           ^
    #                                           |
    #                                           v
    #        -( 1 - #1_o2 ) >= -( #(1 -> 0) - #(0 -> 1) ) >= -( N_REPLICAS - #1_o2 )
    #                                           ^
    #                                           |
    #                                           v
    #           #1_o2 - 1   >=    #(0 -> 1) - #(1 -> 0)   >=    #1_o2 - N_REPLICAS 
    #                                           ^
    #                                           |
    #                                           v
    #   #1_o2 - N_REPLICAS  <=    #(0 -> 1) - #(1 -> 0)   <=    #1_o2 - 1  
    #
    # Mixing both restrictions, we have:
    #
    #           max( 1 - #1_o1 , #1_o2 - N_REPLICAS ) <=
    #        <=          #(0 -> 1) - #(1 -> 0)        <=
    #        <= min( #1_o2 - 1 , N_REPLICAS - #1_o1 )
    #
    #   #0->1 \ #1->0 |  0   1   2   3   4   5 ...
    #   --------------+---------------------------
    #               0 |[ 0][-1] -2  -3  -4  -5  
    #               1 |  1 [ 0][-1] -2  -3  -4  
    #               2 |  2   1 [ 0][-1] -2  -3  
    #               3 |  3   2   1 [ 0][-1] -2  
    #               4 |  4   3   2   1 [ 0][-1] 
    #               5 |  5   4   3   2   1 [ 0] 
    #             ... | 

    def __init__(self, n_replicas=1):
        super().__init__(n_parents = 2, n_offsprings = 2)
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape[:3]
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]

            if self.n_replicas == 1:
                # Cross rows
                off1, off2 = [], []
                for i in range(problem.N_TASKS):
                    if random.random() < 0.5:
                        off1.append(p1[i])
                        off2.append(p2[i])
                    else:
                        off1.append(p2[i])
                        off2.append(p1[i])
                Y[0, k, 0], Y[1, k, 0] = np.array(off1, np.uint8), np.array(off2, np.uint8)
            else:
                # Cross 2D (rows and cols)
                Y[0, k, 0], Y[1, k, 0] = p1.copy(), p2.copy()

                p_diff = np.subtract(p1, p2, dtype=np.int8)
                    #  0: 0->0 or 1->1
                    # -1: 0->1
                    #  1: 1->0 

                for i in range(problem.N_TASKS):
                    if not p_diff[i].any(): continue

                    arr01 = np.flatnonzero(p_diff[i] == -1)
                    arr10 = np.flatnonzero(p_diff[i] ==  1)

                    max01 = arr01.size
                    max10 = arr10.size

                    n1p1 = np.flatnonzero(p1[i]).size
                    n1p2 = np.flatnonzero(p2[i]).size

                    lbound = max(1 - n1p1, n1p2 - self.n_replicas)
                    ubound = min(n1p2 - 1, self.n_replicas - n1p1)

                    n01, n10 = random.choice([
                            (i,j)
                            for i in range(max01 + 1)
                            for j in range(max10 + 1)
                            if lbound <= i - j <= ubound
                        ])

                    j01 = random.sample(arr01.tolist(), n01)
                    j10 = random.sample(arr10.tolist(), n10)

                    for j in j01:
                        Y[0, k, 0][i, j] = 1
                        Y[1, k, 0][i, j] = 0

                    for j in j10:
                        Y[0, k, 0][i, j] = 0
                        Y[1, k, 0][i, j] = 1

            # Genetic load
            if n_var > 1:
                p1_inh, p2_inh = X[0, k, 1], X[1, k, 1]
                ch_inh = (p1_inh + p2_inh) / 2
                Y[0, k, 1], Y[1, k, 1] = ch_inh, ch_inh

        return Y

# ==============================================================================
# REPAIRS
# ==============================================================================
class MyRepair(Repair):
    # 1) Delete leftover replicas, giving priority to those that are in memory
    #    overflowing devices.
    # 2) Add needed replicas on nodes that can contain it.
    # 4) Randomly fix services' replicas that overflow device memory.
    def __init__(self, n_replicas=1):
        super().__init__()
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        task_memory = problem.network.getTaskMemoryArray()
        tid_array = np.arange(problem.N_TASKS)
        for i in range(len(X)):
            # 1) Delete leftover replicas, giving priority to those that are in
            #    memory overflowing devices.
            np.random.shuffle(tid_array)
            for tid in tid_array[np.sum(X[i,0], axis=1)[tid_array] > self.n_replicas]:
                available = problem.network.getNodeAvailableMemoryArray(X[i,0])

                node_idxs_1 = np.flatnonzero(X[i,0][tid])

                node_idxs_1_filter_1 = node_idxs_1[available[node_idxs_1] <  0.]
                node_idxs_1_filter_2 = node_idxs_1[available[node_idxs_1] >= 0.]

                np.random.shuffle(node_idxs_1_filter_1)
                np.random.shuffle(node_idxs_1_filter_2)

                node_idxs_1_priority = np.concatenate(
                        (node_idxs_1_filter_1, node_idxs_1_filter_2))

                to_remove = node_idxs_1.size - self.n_replicas

                for nid in node_idxs_1_priority[:to_remove]:
                    X[i,0][tid,nid] = 0

            # 2) Add needed replicas on nodes that can contain it.
            np.random.shuffle(tid_array)
            for tid in tid_array[np.sum(X[i,0], axis=1)[tid_array] == 0]:
                tmem = task_memory[tid]
                available = problem.network.getNodeAvailableMemoryArray(X[i,0])

                node_idxs_0 = np.flatnonzero(X[i,0][tid]-1)
                node_idxs_0_filter = node_idxs_0[available[node_idxs_0] - tmem > 0.]

                if node_idxs_0_filter.size > 0:
                    nid = np.random.choice(node_idxs_0_filter)
                    X[i,0][tid,nid] = 1
                
            # 3) Randomly fix services' replicas that overflow device memory.
            np.random.shuffle(tid_array)
            for tid in tid_array:
                tmem = task_memory[tid]
                available = problem.network.getNodeAvailableMemoryArray(X[i,0])

                node_idxs_1 = np.flatnonzero(X[i, 0][tid])
                node_idxs_0 = np.setdiff1d(np.arange(problem.N_NODES), node_idxs_1)

                # From all ones, select those who surpass the available memory
                node_idxs_1_filter = node_idxs_1[available[node_idxs_1] < 0.]
                len1 = len(node_idxs_1_filter)

                # From all zeros, select those who have enough available memory
                # to hold this specific task
                node_idxs_0_filter = node_idxs_0[available[node_idxs_0] - tmem > 0.]
                len0 = len(node_idxs_0_filter)

                if len0 != 0 :
                    if len0 < len1:
                        # Remove overflowing replicas
                        for nid in node_idxs_1_filter[len0:]:
                            X[i,0][tid,nid] = 0

                        node_idxs_1_filter = node_idxs_1_filter[:len0]
                        len1 = len0

                    # Move to make nodes more compact
                    n0_sorted = node_idxs_0_filter[
                            np.lexsort((available[node_idxs_0_filter],))
                        ]

                    for j in range(len1):
                        orig = node_idxs_1_filter[j]
                        dest = n0_sorted[j]
                        X[i,0][tid,orig] = 0
                        X[i,0][tid,dest] = 1

        return X

# ==============================================================================
# MUTATIONS
# ==============================================================================
class MyMutation_v1(Mutation):
    """
        Do not ensure any constraint.
        Rows with more replicas have more probability to move one of its
        replicas.
        A cell is randomly selected to switch its value from 0 to 1 or 1 to 0
        with equal probability, meaning that the more zeros there are in the
        matrix, the more probable is to change a 0 to 1 and vice versa.
    """
    def __init__(self, p_move=0.1, p_change=0.1, p_binomial=0.05, n_replicas=1):
        super().__init__()
        self.p_move   = p_move   # Change position of 1
        self.p_change = p_change # Change 1 to 0 or 0 to 1
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            rnd = random.random()
            if   rnd < self.p_move:
                pairs = np.transpose(np.nonzero(X[i,0]))
                tid, nid_src = pairs[random.randrange(pairs.shape[0])]
                zeros = np.flatnonzero(X[i,0][tid] - 1)
                nid_dst = zeros[random.randrange(zeros.size)]
                X[i,0][tid,nid_src] = 0
                X[i,0][tid,nid_dst] = 1
            elif rnd < self.p_move + self.p_change:
                tid = random.randrange(problem.N_TASKS)
                nid = random.randrange(problem.N_NODES)
                if X[i,0][tid,nid] == 0:
                    X[i,0][tid,nid] = 1
                else:
                    X[i,0][tid,nid] = 0
        return X

# TODO: Nueva versión que haga más de una mutación sobre un mismo individuo
class MyMutation_v2(Mutation):
    """
        Ensure constraints.
        Rows with more replicas have more probability to move one of its
        replicas.
        A cell is randomly selected to switch its value from 0 to 1 or 1 to 0
        with equal probability, meaning that the more zeros there are in the
        matrix, the more probable is to change a 0 to 1 and vice versa.
    """
    def __init__(self, p_move=0.1, p_change=0.1, p_binomial=0.05, n_replicas=1):
        super().__init__()
        self.p_move   = p_move   # Change position of 1
        self.p_change = p_change # Change 1 to 0 or 0 to 1
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            rnd = random.random()
            if   rnd < self.p_move:
                # Get a random one on the matrix
                pairs = np.transpose(np.nonzero(X[i,0]))
                tid, nid_src = pairs[random.randrange(pairs.shape[0])]

                # Restrictions
                tmem = problem.network.getTaskMemoryArray()[tid]
                available = problem.network.getNodeAvailableMemoryArray(X[i,0])

                # Get a random zero on that row that fit restrictions
                zeros = np.flatnonzero(X[i,0][tid] - 1)
                zeros_filter = zeros[available[zeros] >= tmem]
                if zeros_filter.size > 0:
                    nid_dst = zeros_filter[random.randrange(zeros_filter.size)]

                    # Move replica
                    X[i,0][tid,nid_src] = 0
                    X[i,0][tid,nid_dst] = 1

            elif rnd < self.p_move + self.p_change:
                # Changing 1 -> 0 can be done if:
                #   - Task's replicas > 1
                # Changing 0 -> 1 can be done if:
                #   - Task's replicas < N_REPLICAS
                #   - Node's available memory >= Task memory
                available = problem.network.getNodeAvailableMemoryArray(X[i,0])
                cell_array = []
                for tid in range(problem.N_TASKS):
                    tmem = problem.network.getTaskMemoryArray()[tid]
                    trep = np.sum(X[i,0][tid])
                    for nid in range(problem.N_NODES):
                        if X[i,0][tid,nid] == 1 and \
                                trep > 1:
                            cell_array.append((tid,nid))
                        elif X[i,0][tid,nid] == 0 and \
                                trep < self.n_replicas and \
                                tmem <= available[nid]:
                            cell_array.append((tid,nid))

                if cell_array:
                    # Randomly choose a cell with equal probability
                    tid, nid = random.choice(cell_array)
                    if X[i,0][tid,nid] == 0:
                        X[i,0][tid,nid] = 1
                    else:
                        X[i,0][tid,nid] = 0
        return X

class MyMutation_v3(Mutation):
    """
        Ensure constraints.
        All tasks have equal probability to move one of its replicas.
        Changing cell from 0 to 1 has the same probability as changing cell from
        1 to 0.
    """
    def __init__(self, p_move=0.1, p_change=0.1, p_binomial=0.05, n_replicas=1):
        super().__init__()
        self.p_move   = p_move   # Change position of 1
        self.p_change = p_change # Change 1 to 0 or 0 to 1
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            rnd = random.random()

            if rnd < self.p_move:
                # First type: move assigned task to a different node
                row = random.randrange(problem.N_TASKS)

                nid_array_1 = np.flatnonzero(X[i,0][row])
                nid_array_0 = np.setdiff1d(
                        np.arange(problem.N_NODES),
                        nid_array_1
                    )

                if nid_array_1.size > 0:
                    # Ensure that selected node can hold that task
                    tmem = problem.network.getTaskMemoryArray()[row]
                    available = problem.network.getNodeAvailableMemoryArray(X[i,0])
                    nid_array_0_filter = nid_array_0[available[nid_array_0] >= tmem]

                    if nid_array_0_filter.size > 0:
                        col_src  = np.random.choice(nid_array_1)
                        col_dest = np.random.choice(nid_array_0_filter)
                        X[i,0][row, col_src ] = 0
                        X[i,0][row, col_dest] = 1

            elif self.n_replicas > 1 and rnd < self.p_move + self.p_change:
                # Second type: change task assignment state within a node
                row = random.randrange(problem.N_TASKS)
                nid_array_1 = np.flatnonzero(X[i,0][row])

                # Update rnd to fit between 0 and 1
                rnd = (rnd - self.p_move) / self.p_change

                if nid_array_1.size >= self.n_replicas:
                    # Change 1 to 0 with 100% probability
                    ch_01 = False
                elif nid_array_1.size <= 1:
                    # Change 0 to 1 with 100% probability
                    ch_01 = True
                elif rnd < 0.5:
                    # Change 1 to 0 with 50% probability
                    ch_01 = False
                else:
                    # Change 0 to 1 with 50% probability
                    ch_01 = True

                if ch_01:
                    nid_array_0 = np.setdiff1d(
                            np.arange(problem.N_NODES),
                            nid_array_1
                        )

                    # Ensure that selected node can hold that task
                    tmem = problem.network.getTaskMemoryArray()[row]
                    available = problem.network.getNodeAvailableMemoryArray(X[i,0])
                    nid_array_0_filter = nid_array_0[available[nid_array_0] >= tmem]

                    if nid_array_0_filter.size > 0:
                        col = np.random.choice(nid_array_0)
                        X[i,0][row,col] = 1
                    
                else:
                    col = np.random.choice(nid_array_1)
                    X[i,0][row,col] = 0

        return X

class MyMutation_v4(Mutation):
    """
        Ensure constraints.
        Rows with more replicas have more probability to move one of its
        replicas.
        A cell is randomly selected to switch its value from 0 to 1 or 1 to 0
        with equal probability, meaning that the more zeros there are in the
        matrix, the more probable is to change a 0 to 1 and vice versa.

        A new probability parameter. Depending on the size of the problem, for
        each individual, apply M mutations given by a binomial distribution.
    """
    def __init__(self, p_move=0.1, p_change=0.1, p_binomial=0.05, n_replicas=1):
        super().__init__()
        self.p_move   = p_move   # Change position of 1
        self.p_change = p_change # Change 1 to 0 or 0 to 1
        self.p_binomial = p_binomial # Amount of mutations on a single individual
        self.n_replicas = n_replicas

    def _do(self, problem, X, **kwargs):
        n_mutation_array = np.random.binomial(
                problem.N_TASKS - 1, self.p_binomial, size=len(X)) + 1
        for i in range(len(X)):
            n_mutations = n_mutation_array[i]
            if n_mutations == 0:
                continue

            rnd = random.random()
            if   rnd < self.p_move:
                # Get a list of random ones in the matrix
                pairs = np.transpose(np.nonzero(X[i,0]))
                pairs_mut = pairs[random.sample(range(pairs.shape[0]), n_mutations)]
                for tid, nid_src in pairs_mut:
                    # Restrictions
                    tmem = problem.network.getTaskMemoryArray()[tid]
                    available = problem.network.getNodeAvailableMemoryArray(X[i,0])

                    # Get a random zero on that row that fit restrictions
                    zeros = np.flatnonzero(X[i,0][tid] - 1)
                    zeros_filter = zeros[available[zeros] >= tmem]
                    if zeros_filter.size > 0:
                        nid_dst = zeros_filter[random.randrange(zeros_filter.size)]

                        # Move replica
                        X[i,0][tid,nid_src] = 0
                        X[i,0][tid,nid_dst] = 1

            elif rnd < self.p_move + self.p_change:
                # Changing 1 -> 0 can be done if:
                #   - Task's replicas > 1
                # Changing 0 -> 1 can be done if:
                #   - Task's replicas < N_REPLICAS
                #   - Node's available memory >= Task memory
                cell_array = [
                        (tid, nid)
                        for tid in range(problem.N_TASKS)
                        for nid in range(problem.N_NODES)
                    ]

                random.shuffle(cell_array)

                mutated = 0
                for tid, nid in cell_array:
                    if mutated == n_mutations:
                        break

                    available = problem.network.getNodeAvailableMemoryArray(X[i,0])
                    tmem = problem.network.getTaskMemoryArray()[tid]
                    trep = np.sum(X[i,0][tid])

                    if X[i,0][tid,nid] == 1 and \
                            trep > 1:
                        X[i,0][tid,nid] = 0
                        mutated += 1

                    elif X[i,0][tid,nid] == 0 and \
                            trep < self.n_replicas and \
                            tmem <= available[nid]:
                        X[i,0][tid,nid] = 1
                        mutated += 1

        return X



class MyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(a.X[0], b.X[0])

class MyCallback(Callback):
    def __init__(self, save_history=False):
        super().__init__()
        self.save_history = save_history
        if save_history:
            self.string_history = ""
        else:
            self.string_solution = ""

    def notify(self, algorithm):
        f = algorithm.pop.get('F_original')

        # Save current solution or append to history
        curr_sol = algorithm.opt.get('F_original')

        if self.save_history:
            dt_now = datetime.datetime.now()
        else:
            self.string_solution = ""

        for o in curr_sol:
            if self.save_history:
                self.string_history += "{} {} ".format( dt_now, algorithm.n_gen)

                for o_n in o:
                    self.string_history += "{} ".format(o_n)

                self.string_history += '\n'

            else:
                for o_n in o:
                    self.string_solution += "{} ".format(o_n)

                self.string_solution += '\n'

class MyHybridCallback(Callback):
    def __init__(self, idx, gen_steps, sol_dumping, queues, barrier, save_history=False, algorithms=[], alg_name=''):
        super().__init__()
        self.idx = idx
        self.gen_steps = gen_steps
        self.sol_dumping = sol_dumping
        self.queues = queues
        self.barrier = barrier
        self.save_history = save_history

        # Multiple algorithms
        # TODO: Check whether you need to pass alg_name or can deduce from algorithms[idx]
        self.n_algs = len(queues)
        self.alg_name = alg_name
        self.algorithms = algorithms

        # String history
        self.string_history = ""
        self.string_gen_load_orig_hist = ""
        self.string_gen_load_last_hist = ""

    def notify(self, algorithm):
        x_ind = algorithm.pop

        # Save current solution or append to history
        curr_sol = algorithm.opt

        dt_now = datetime.datetime.now()

        for e in x_ind:
            self.string_history += "{} {} {} ".format(
                    dt_now,
                    1 if e in curr_sol else 0,
                    algorithm.n_gen
                )

            # TODO: Check if algorithm.n_obj exists
            for o in e.F:
                self.string_history += "{} ".format(o)

            for x_row in e.X[1]:
                for x_elem in x_row:
                    self.string_history += "{} ".format(x_elem)

            self.string_history += "\n"

        # Check generation step
        if algorithm.n_gen in self.gen_steps:
            print('  {} arrived at generation step {}'.format(
                self.alg_name, algorithm.n_gen))

            x = algorithm.opt

            step_idx = self.gen_steps.index(algorithm.n_gen)
            sol_dumping_step = self.sol_dumping[step_idx]

            # Change genetic load from population of current step
            for sol_idx in range(x_ind.shape[0]):
                for alg_idx in range(self.n_algs):
                    x_ind[sol_idx].X[1][step_idx, alg_idx] = 1. if alg_idx == self.idx else 0.

            # Dump solutions to other algorithms
            for i in range(self.n_algs):
                if i == self.idx:
                    # Skip itself
                    continue

                if sol_dumping_step[self.idx][i] == 1:
                    dump_alg_name = self.algorithms[i]
                    print(f'<-- {self.alg_name} is sending data to {dump_alg_name}')
                    self.queues[i].put(x_ind)

            # Wait for other algorithms to dump their solutions
            x_received = []
            for i in range(self.n_algs):
                if i == self.idx:
                    # Skip itself
                    continue

                if sol_dumping_step[i][self.idx] == 1:
                    dump_alg_name = self.algorithms[i]
                    print(f' ?  {self.alg_name} is waiting data from {dump_alg_name}')
                    r = self.queues[self.idx].get()

                    x_received.append(r)
                    print(f'--> {self.alg_name} is receiving data from {dump_alg_name}')

            print(f'  {self.alg_name} is merging the solutions...')
            # TODO: fix CTAEA (and test RVEA)
            if len(x_received) > 0:
                pop = Population(x_ind)
                for x in x_received:
                    x_pop = Population(x)
                    pop = Population.merge(pop, x_pop)

                da = algorithm.da if hasattr(algorithm.__class__, 'da') else None
                algorithm.pop = algorithm.survival.do(
                        algorithm.problem,
                        pop,
                        n_survive=algorithm.pop_size,
                        algorithm=algorithm,
                        da=da
                    )

            # Sync with all other processes
            self.barrier.wait()



if __name__ == '__main__':
    from parameters import configs
    from network import Network
    from problems import Problem01v3
    import pickle

    random.seed(1)
    np.random.seed(1)

    ntw = pickle.load(configs.input)

    problem = Problem01v3(ntw)
    X = MySampling_v3(n_replicas=configs.n_replicas)._do(problem, n_samples=configs.pop_size)

    p1arr = []
    p2arr = []

    idxs = np.arange(configs.pop_size)
    np.random.shuffle(idxs)
    for i,j in zip(idxs[:configs.pop_size//2], idxs[configs.pop_size//2:]):
        p1arr.append(X[i])
        p2arr.append(X[j])

    X_p = np.array([p1arr,p2arr])
    print(X_p[0,0,0])
    print(X_p[1,0,0])
    print()

    X_c = MyCrossover_v2(n_replicas=configs.n_replicas)._do(problem, X_p)
    print(X_c[0,0,0])
    print(problem.network.getNodeAvailableMemoryArray(X_c[0,0,0]))
    print()

    X_ir = np.array([X_c[0,0], X_c[1,0]])
    X_r = MyRepair(n_replicas=configs.n_replicas)._do(problem, X_ir)
    print(X_r[0,0])
    print(problem.network.getNodeAvailableMemoryArray(X_r[0,0]))
    print()

    X_m = X_r
    for _ in range(100):
        X_m = MyMutation_v2(p_move=configs.mutation_prob_move,
                p_change=configs.mutation_prob_change,
                n_replicas=configs.n_replicas)._do(problem, X_m)
        print(X_m[0,0])
        print(np.sum(X_m[0,0],axis=1))
        print(problem.network.getNodeAvailableMemoryArray(X_m[0,0]))

