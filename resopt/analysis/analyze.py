import random
import datetime
import time

import numpy as np
import pandas as pd

from resopt.files.file_utils import parse_file, get_solution_array
from resopt.param.default import POP_SIZE, ALGORITHM, SEED, TS_FORMAT_STR

from pymoo.indicators.gd  import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv  import HV

#from analyze_functions import S, STE

def S(pf=None):
    def indicator(solution):
        n = len(solution)
        if n < 2:
            return 0.
        d_arr = np.array([
                min([
                        np.linalg.norm(x-y)
                        for x in solution
                        if not np.array_equal(x, y)
                    ])
                for y in solution
            ])

        d_mean = np.average(d_arr)

        return np.sqrt(sum([(d_arr[i]-d_mean)**2 for i in range(n)]) / n)
    
    # Just as a way to give same format as other indicator functions
    return indicator

def STE(pf=None):
    def indicator(solution):
        # Spacing
        n = len(solution)
        if n < 2:
            return 0.
        d_arr = np.array([
                min([
                        np.linalg.norm(x-y)
                        for x in solution
                        if not np.array_equal(x, y)
                    ])
                for y in solution
            ])

        d_mean = np.average(d_arr)

        spacing = sum([(d_arr[i] - d_mean)**2 for i in range(n)]) / (n-1)

        # Extent
        f_min = np.array([
                min([
                        solution[i,j]
                        for i in range(solution.shape[0])
                    ])
                for j in range(solution.shape[1])
            ])

        f_max = np.array([
                max([
                        solution[i,j]
                        for i in range(solution.shape[0])
                    ])
                for j in range(solution.shape[1])
            ])

        extent = np.sum(np.abs(f_max - f_min))

        return spacing / extent

    # Just as a way to give same format as other indicator functions
    return indicator

def mean_time(date_a, time_a):
    FORMAT_STR = '%Y-%m-%d %H:%M:%S.%f'
    acc = 0.
    next_dt = datetime.datetime.strptime(
            '{} {}'.format(date_a[0], time_a[0]),
            FORMAT_STR
        )
    for date, time in zip(date_a, time_a):
        prev_dt = next_dt
        next_dt = datetime.datetime.strptime(
                '{} {}'.format(date,time),
                FORMAT_STR
            )
        acc += (next_dt - prev_dt).total_seconds()

    return acc / (len(date_a) - 1)

def solutions():
    pass

INDICATORS = {
    'GD': GD,
    'IGD': IGD,
    'HV': HV,
    'S': S,
    'STE': STE
}

def generate_csv(configs):
    n_obj = len(configs.objectives)
    n_algs = configs.n_algorithms
    is_pop = configs.entire_population
    n_steps = len(configs.gen_steps)

    alg_ts_dates = []
    alg_ts_times = []

    alg_solutions = []
    alg_population = []
    alg_is_solution = []

    alg_gl_pop = []
    alg_gl_sol = []

    for f in configs.input:
        parsed = get_solution_array(
                f,
                n_obj=n_obj,
                n_algs=n_algs,
                n_steps=n_steps,
                timestamps=True,
                population=is_pop,
                return_dict=True
            )
        ts_date_a = parsed.get('date', [])
        ts_time_a = parsed.get('time', [])
        solutions = parsed.get('solutions', [])
        population = parsed.get('population', solutions)
        gl_sol = parsed.get('genetic_load_sol', [[]] * n_steps)
        gl_pop = parsed.get('genetic_load_pop', [[]] * n_steps)
        generations = len(population)

        # Filter repeated results
        curr_alg_population = []
        curr_alg_solutions = []
        curr_alg_gl_sol = [[] for _ in configs.gen_steps]
        curr_alg_gl_pop = [[] for _ in configs.gen_steps]
        for g in range(generations):
            p = population[g]
            s = solutions[g]
            p_uq_elem, p_uq_idx = np.unique(p, return_index=True, axis=0)
            s_uq_elem, s_uq_idx = np.unique(s, return_index=True, axis=0)

            is_solution = np.all(np.isin(p_uq_elem, s_uq_elem), axis=1)

            curr_alg_population.append(p_uq_elem)
            curr_alg_solutions.append(s_uq_elem)

            if n_algs > 0 and n_steps > 0:
                for step in range(n_steps):
                    gls = gl_sol[step][g]
                    glp = gl_pop[step][g]

                    curr_alg_gl_pop[step].append(glp[p_uq_idx])
                    curr_alg_gl_sol[step].append(gls[s_uq_idx])
            else:
                curr_alg_gl_pop.append(np.array([[] for _ in range(n_steps)]))
                curr_alg_gl_sol.append(np.array([[] for _ in range(n_steps)]))

        alg_population.append(curr_alg_population)
        alg_solutions.append(curr_alg_solutions)
        alg_gl_pop.append(curr_alg_gl_pop)
        alg_gl_sol.append(curr_alg_gl_sol)

        alg_ts_dates.append(ts_date_a)
        alg_ts_times.append(ts_time_a)

    alg_names   = configs.alg_names   
    seed_values = configs.seed_values 
    pop_values  = configs.pop_values  
    gen_values  = configs.gen_values
    len_gen_values = len(gen_values)
    
    # Preparing all combinations of values for the rows on the table
    alg_name_rows   = [
            alg 
            for alg in alg_names 
            for _   in seed_values
            for _   in pop_values 
            for _   in gen_values 
        ]

    execution_rows = [
            ex
            for _   in alg_names 
            for ex  in seed_values
            for _   in pop_values 
            for _   in gen_values 
        ]

    pop_value_rows = [
            pop 
            for _   in alg_names 
            for _   in seed_values
            for pop in pop_values 
            for _   in gen_values 
        ]

    gen_value_rows = [
            gen 
            for _   in alg_names 
            for _   in seed_values
            for _   in pop_values 
            for gen in gen_values 
        ]

    if configs.ref_points is not None:
        pf = np.unique(np.array(configs.ref_points), axis=0)
    else:
        pf = np.zeros((1, n_obj))

    # Values needed for normalization
    if not configs.network:
        o_min = np.array([
                min([
                    np.min(g[:,o])
                    for s in alg_solutions
                    for g in s
                ])
                for o in range(n_obj)
            ])

        o_max = np.array([
                max([
                    np.max(g[:,o])
                    for s in alg_solutions
                    for g in s
                ])
                for o in range(n_obj)
            ])

    else:
        import pickle
        ntw = pickle.load(configs.network)

        o_min_lst, o_max_lst = [], []
        for o in range(n_obj):
            o_min_aux, o_max_aux = ntw.getObjectiveBounds(configs.objectives[o])
            o_min_lst.append(o_min_aux)
            o_max_lst.append(o_max_aux)

        o_min, o_max = np.array(o_min_lst), np.array(o_max_lst)

    # Normalization of everything
    for s in range(len(alg_solutions)): # for each algorithm given
        for g in range(len(alg_solutions[s])): # for each generation
            for o in range(n_obj):
                alg_solutions[s][g][:,o] = \
                        (alg_solutions[s][g][:,o] - o_min[o]) / (o_max[o] - o_min[o])
            #print(g)
            #print(alg_solutions[s][g])
            #print()

    for o in range(n_obj):
        pf[:,o] = (pf[:,o] - o_min[o]) / (o_max[o] - o_min[o])


    # Metrics
    td_list = []
    n_sol_list = []
    gl_pop_list = [[] for _ in range(n_steps)]
    gl_sol_list = [[] for _ in range(n_steps)]
    metrics_dict = {k: [] for k in INDICATORS.keys()}

    for i in range(len(alg_solutions)):
        alg  = alg_name_rows[i*len_gen_values]
        seed = execution_rows[i*len_gen_values]
        pop  = pop_value_rows[i*len_gen_values]

        prev_gen = 1
        for gen in gen_values:
            if gen <= 0: gen = len(alg_solutions[i])

            # Load solutions
            sols = alg_solutions[i][gen-1]

            # Time elapsed
            ts_date_start = alg_ts_dates[i][prev_gen-1]
            ts_time_start = alg_ts_times[i][prev_gen-1]
            dt_start = datetime.datetime.strptime(
                    '{} {}'.format(ts_date_start, ts_time_start),
                    TS_FORMAT_STR
                )


            ts_date_end = alg_ts_dates[i][gen-1]
            ts_time_end = alg_ts_times[i][gen-1]
            dt_end = datetime.datetime.strptime(
                    '{} {}'.format(ts_date_end, ts_time_end),
                    TS_FORMAT_STR
                )

            tdelta = (dt_end - dt_start).total_seconds()

            # Genetic Load
            print("HERE")
            print(gl_pop_list)
            print("."*10)
            if n_algs > 0:
                print("GEN ", gen)
                print("N_ALGS ", n_algs)
                print("N_steps ", list(range(n_steps)))
                for step in range(n_steps):
                    print("\t",step)
                    gl_pop_acc = np.average(alg_gl_pop[i][step][gen-1], axis = 0)
                    gl_sol_acc = np.average(alg_gl_sol[i][step][gen-1], axis = 0)
                    print("\t",gl_pop_acc)
                    gl_pop_list[i].append(gl_pop_acc)
                    gl_sol_list[i].append(gl_sol_acc)

            prev_gen = gen

            # Append metrics
            td_list.append(tdelta)
            n_sol_list.append(len(sols))

            # Performance indicators
            for name, ind_c in INDICATORS.items():
                if name == 'HV':
                    ind = ind_c(np.ones((n_obj)))
                else:
                    # GD and IGD uses Pareto front
                    ind = ind_c(pf)
                solution = ind(sols)
                metrics_dict[name].append(solution)

    # Create pandas dataframe
    df_dict = {
            'Algorithm': alg_name_rows,
            'Seed': execution_rows,
            'Population': pop_value_rows,
            'Generation': gen_value_rows,
            'Solutions': n_sol_list,
            'TimeDelta': td_list}
    
    if n_algs > 0:
        for step in range(n_steps):
            df_dict['GLPop{}'.format(step+1)] = gl_pop_list[step]
            df_dict['GLSol{}'.format(step+1)] = gl_sol_list[step]

    df_dict.update(metrics_dict)

    df = pd.DataFrame(df_dict)

    df.to_csv(configs.output, index=False)
    configs.output.close()

if __name__ == '__main__':
    from parameters import configs

    df = generate_csv(configs)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4)

    df.query('Algorithm == "NSGA2" and Population == 200').boxplot('GD', by='Generation', ax=axs[0])
    df.query('Algorithm == "NSGA2" and Population == 200').boxplot('HV', by='Generation', ax=axs[1])
    df.query('Algorithm == "NSGA2" and Population == 200').boxplot('Solutions', by='Generation', ax=axs[2])
    df.query('Algorithm == "NSGA2" and Population == 200').boxplot('TimeDelta', by='Generation', ax=axs[3])

    plt.show()


