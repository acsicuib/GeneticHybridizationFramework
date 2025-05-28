import sys
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from resopt.files.file_utils import get_solution_array

# <ts_date> <ts_time> <isSolution> <generation> <o1> ... <on> <a1> ... <am>

FORMAT_STR = '%Y-%m-%d %H:%M:%S.%f'

gen_steps  = [25,    50,    75,    100,   125,   150,   175]
# plot_steps = [True] + [False for _ in gen_steps][:-1]
plot_steps = [True  for _ in gen_steps]

n_gen = 200
n_objs = 3
n_algs = 2
n_steps = len(gen_steps)
obj_column = 4
alg_column = obj_column + n_objs
columns = alg_column + n_algs * n_steps

obj_keys  = ['Objective{}'.format(i+1) for i in range(n_objs)]
alg_keys = [
        'GLStep{}_{}'.format(step+1, i+1)
        for step in range(n_steps)
        for i in range(n_algs)
    ]

print("HERE")
print(alg_keys)

# alg_list = ['NSGA2', 'NSGA3', 'UNSGA3', 'SMSEMOA']
alg_list = ['NSGA2', 'NSGA3']
alg_key_remap = {alg_col: alg_name for alg_col, alg_name in zip(alg_keys, alg_list * n_steps)}


def getplotsegments(gen_steps, plot_steps):
    steps = np.arange(n_steps)[plot_steps]
    plot_steps_cpy = [step for step in plot_steps]
    plot_steps_cpy[0] = False
    plot_segments = np.array(gen_steps)[plot_steps_cpy]
    plot_segments = np.concatenate(([0], plot_segments, [n_gen]))
    return plot_segments, steps

def normalplot(df, ax, alg_name = '', only_solutions = False, from_last = False):
    dfg = df.copy(deep=True)

    if only_solutions:
        dfg = dfg.query('IsSolution')
        only_solutions_str = 'only solutions'
    else:
        only_solutions_str = 'entire population'

    full_title = '{}, {}'.format(alg_name, only_solutions_str)
        
    plot_segments, steps = getplotsegments(gen_steps, plot_steps)
    for i in range(len(plot_segments)-1):
        gen_start = plot_segments[i]
        gen_end   = plot_segments[i+1]
        qry = f"{gen_start} < Generation <= {gen_end}"

        for j in range(n_algs):
            s_read = dfg.query(qry)[alg_keys[n_algs * steps[i] + j]]
            dfg.loc[dfg.query(qry).index, alg_keys[j]] = s_read

    dfg = pd.DataFrame(dfg.groupby(['Generation'])[alg_keys[:n_algs]].mean()).reset_index()

    dfg = dfg.rename(columns=alg_key_remap)
    dfg.plot(x='Generation', title=full_title, ax=ax)

def stackplot(df, ax, alg_name = '', only_solutions = False, from_last = False):
    if from_last:
        # alg_keys = ["GLStep1_1"]
        # alg_key_remap = "GLStep1_1"
        from_last_str = 'From last dump'
    else:
        # alg_keys = "GLStep1_1"
        # alg_key_remap = "GLStep1_1"
        # # alg_keys = alg1_keys
        # alg_key_remap = alg1_key_remap
        from_last_str = 'From origin'

    dfg = df.copy(deep=True)

    if only_solutions:
        dfg = dfg.query('IsSolution')
        only_solutions_str = 'only solutions'
    else:
        only_solutions_str = 'entire population'

    full_title = '{}, {}'.format(alg_name, from_last_str, only_solutions_str)
        
    dfg = pd.DataFrame(dfg.groupby(['Generation'])[alg_keys].mean()).reset_index()
    dfg = dfg.rename(columns=alg_key_remap)
    dfg.plot.area(x='Generation', title=full_title, ax=ax, linewidth=0.)

def varplot(df_list, ax, alg_name = '', only_solutions = False, from_last = False):
    if from_last:
        alg_keys = alg2_keys
        alg_key_remap = alg2_key_remap
        from_last_str = 'From last dump'
    else:
        alg_keys = alg1_keys
        alg_key_remap = alg1_key_remap
        from_last_str = 'From origin'

    if only_solutions:
        only_solutions_str = 'only solutions'
    else:
        only_solutions_str = 'entire population'

    for df in df_list:
        dfg = df.copy(deep=True)
        if only_solutions:
            dfg = dfg.query('IsSolution')

        full_title = '{}, {}'.format(alg_name, from_last_str, only_solutions_str)
            
        dfg = pd.DataFrame(dfg.groupby(['Generation'])[alg_keys].var()).reset_index()
        dfg = dfg.rename(columns=alg_key_remap)
        dfg.plot(x='Generation', title=full_title, ax=ax)

def boxplot(df, ax, alg_name = '', only_solutions = False, from_last = False):
    if from_last:
        alg_keys = alg2_keys
        alg_key_remap = alg2_key_remap
        from_last_str = 'from last dump'
    else:
        alg_keys = alg1_keys
        alg_key_remap = alg1_key_remap
        from_last_str = 'from origin'

    dfg = df.copy(deep=True)

    if only_solutions:
        dfg = dfg.query('IsSolution')
        only_solutions_str = 'only solutions'
    else:
        only_solutions_str = 'entire population'

    full_title = '{}, {}, {}'.format(alg_name, from_last_str, only_solutions_str)
        
    gen_list = dfg['Generation'].tolist()
    gen_groups = ["{}-{}".format((g-1)//5*5+1, (g-1)//5*5+5) for g in gen_list]
    dfg.insert(3, "GenGroups", gen_groups, True)
    dfg = dfg.rename(columns=alg_key_remap)

    dd = pd.melt(dfg, id_vars=['GenGroups'], value_vars=alg_list, var_name='Algorithm')
    #dd = dd[(dd.GenGroups == '26-30').idxmax():]

    sns.violinplot(x="GenGroups", y='value', data=dd, hue='Algorithm', ax=ax,
            inner=None, linewidth=0., cut=0, density_norm='count').set_title(full_title)
    #sns.boxplot(x="GenGroups", y='value', data=dd, hue='Algorithm', ax=ax,
    #        flierprops={'marker': '.', 'markersize': 1}).set_title(full_title)

def txt_to_csv(files):
    df_list = []
    for f in files:
        # DATAFRAME GENERATION
        data = f.read().split()
        # f.close()
        # print("COL ",columns)
        # print(data[0::columns][:10])
        # print(data[1::columns][:10])
        # for d, t in zip(data[0::columns], data[1::columns]):
        #     print(d)
                        
        timestamp = [   








            
                dt.datetime.strptime(
                        '{} {}'.format(d, t), FORMAT_STR
                    ) for d, t in zip(data[0::columns], data[1::columns])
            ]

        isSolution = [bool(int(d)) for d in data[2::columns]]
        generation = [int(d) for d in data[3::columns]]
        
        d = {'Timestamp': timestamp, 'IsSolution': isSolution, 'Generation': generation}
        for i in range(n_objs):
            key_name = 'Objective{}'.format(i + 1)
            d[key_name] = [float(d) for d in data[obj_column + i::columns]]

        for step in range(n_steps):
            for i in range(n_algs):
                key_name = 'GLStep{}_{}'.format(step + 1, i + 1)
                d[key_name] = [float(d) for d in data[alg_column + i + n_algs * step::columns]]


        df_list.append(pd.DataFrame(d))
        # print("JERE")
        # print(type(df_list[0]))
        # print(df_list[0].head(4))

    return df_list


