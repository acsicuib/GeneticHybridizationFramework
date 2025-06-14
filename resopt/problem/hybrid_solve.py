#from solve_problem import algdict, sampling_v, crossover_v, mutation_vs, solve
#        get_ref_dirs_len, ref_dirs_len_matrix, get_partitions_from_population, 

from pymoo.termination import get_termination
from pymoo.optimize import minimize

from resopt.problem.solve_problem import get_genetic_algorithm
from resopt.problem.problems import Problem01v4

from multiprocessing import Process, Queue, Barrier
import time
import copy
import random as rnd
from argparse import Namespace

def solve_p(idx, ntw, queues, barrier, configs):
    n_algs = len(configs.algorithms)
    alg_name = configs.algorithms[idx]
    n_gen_steps = len(configs.gen_steps)

    conf_copy = Namespace(**vars(configs))
    conf_copy.algorithm = alg_name
    # Convert file objects to paths for the child process
    if hasattr(conf_copy, 'output') and conf_copy.output:
        conf_copy.output = [f.name if hasattr(f, 'name') else f for f in conf_copy.output]

    algorithm = get_genetic_algorithm(
            conf_copy,
            hybrid=True,
            idx = idx,
            queues = queues,
            barrier = barrier,
            n_gen_steps = n_gen_steps
        )

    problem = Problem01v4(ntw, configs.objectives, hybrid=True)
    termination = get_termination(configs.termination_type, configs.n_gen)

    print(f'{alg_name} start')
    res = minimize(
            problem,
            algorithm,
            termination=termination,
            seed=configs.seed,
            verbose=configs.verbose,
            copy_algorithm=False
        )
    print(f'{alg_name} done')

    # Solution of optimization output
    if len(configs.output) > 0 and len(configs.output) == n_algs:
        if configs.save_history:
            string = res.algorithm.callback.string_history
        else:
            string = res.algorithm.callback.string_solution
        
        # Open the file in the child process
        with open(configs.output[idx], 'w') as f:
            f.write(string)
    else:
        print("ERROR: Number of output files should match the number of " + 
                "algorithms used", file=sys.stderr)

def hybrid_solve(ntw, configs):
    alg_list = configs.algorithms
    n_algs = len(alg_list)

    # Create a copy of configs with file objects converted to paths
    configs_copy = Namespace(**vars(configs))
    # Convert all file objects to paths
    for attr_name in dir(configs_copy):
        if not attr_name.startswith('_'):  # Skip private attributes
            attr_value = getattr(configs_copy, attr_name)
            if hasattr(attr_value, 'name'):  # If it's a file object
                setattr(configs_copy, attr_name, attr_value.name)
            elif isinstance(attr_value, list) and attr_value and hasattr(attr_value[0], 'name'):
                # If it's a list of file objects
                setattr(configs_copy, attr_name, [f.name if hasattr(f, 'name') else f for f in attr_value])

    
    # Process array
    p_array = []

    # Queue array
    queues = [Queue() for _ in configs.algorithms]

    # Sync barrier
    barrier = Barrier(n_algs)

    for i in range(n_algs):
        p = Process(target=solve_p, args=(i, ntw, queues, barrier, configs_copy,))
        p.start()
        p_array.append(p)

    for p in p_array:
        p.join()
    pass



# Concurrency simulation
TIMES = [0.1, 0.2, 0.3]

def test_p(idx, ntw, queues, barrier, configs):
    n_algs = len(configs.algorithms)
    alg_name = configs.algorithms[idx]

    print(f'{alg_name} start')
    prev_step = 1
    steps = configs.gen_steps + [configs.n_gen]
    n_steps = len(steps)

    # For each step interval
    for step_idx in range(n_steps):
        step = steps[step_idx]
        print(f'    {alg_name} executing interval {prev_step}-{step}')

        # For each generation in this interval (this represents algorithm execution)
        for gen in range(prev_step, step):
            #print(f'    {alg_name} gen {gen}')
            time.sleep(TIMES[idx])

        prev_step = step

        if step != configs.n_gen:
            sol_dumping_step = configs.sol_dumping[step_idx]

            # Dump solutions to other algorithms
            for i in range(n_algs):
                if i == idx:
                    # Skip itself
                    continue

                if sol_dumping_step[idx][i] == 1:
                    dump_alg_name = configs.algorithms[i]
                    print(f'<-- {alg_name} is sending data to {dump_alg_name}')
                    queues[i].put(f'{alg_name}\'s solutions')


            # Wait for other algorithms to dump their solutions
            for i in range(n_algs):
                if i == idx:
                    # Skip itself
                    continue

                if sol_dumping_step[i][idx] == 1:
                    dump_alg_name = configs.algorithms[i]
                    print(f' ?  {alg_name} is waiting data from {dump_alg_name}')
                    queues[idx].get()
                    print(f'--> {alg_name} is receiving data from {dump_alg_name}')

            # Sync with all other processes
            barrier.wait()

    print(f'{alg_name} done')

def concurrency_test(ntw, configs):
    alg_list = configs.algorithms
    n_algs = len(alg_list)

    # Process array
    p_array = []

    # Queue array
    queues = [Queue() for _ in configs.algorithms]

    # Sync barrier
    barrier = Barrier(n_algs)

    for i in range(n_algs):
        p = Process(target=test_p, args=(i, queues, barrier, configs,))
        p.start()
        p_array.append(p)

    for p in p_array:
        p.join()
    pass

if __name__ == '__main__':
    from parameters import configs

    import pickle
    ntw = pickle.load(configs.input)

    hybrid_solve(ntw, configs)
