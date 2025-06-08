import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from resopt.analysis.analyze import S, STE
import numpy as np
from datetime import datetime
import sys
import pickle

def load_data_hybrids(path_exp,file_hybrids,algorithm):
    pairs = [f"tt{i}" for i in range(97)] ## FIX WITH the number of steps
    columns = ["date", "time", "pf","generation"] 
    columns += [f"o{i+1}" for i,_ in enumerate(range(3))]
    columns += pairs    
 
    df = pd.read_csv(path_exp + file_hybrids.format(algorithm=algorithm),sep=" ",header=None)
    df.columns = columns
    return df

path_exp_hybrids = "data_individualexp/hybrids/"
file_hybrids = "{algorithm}_1_100-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
based_hybrid = ['NSGA2',"NSGA3","UNSGA3","SMSEMOA"]
# based_hybrid = ['NSGA2']


INDICATORS = {
    'GD': GD,
    'IGD': IGD,
    'HV': HV,
    'S': S,
    'STE': STE
}
generations = 600
n_obj = 3

def normalize_pf():
    pf = np.zeros((1, 3))
        # Normalization of everything
   
    ntw = pickle.load(open("data_individualexp/ntw_722_050-050-025_C", "rb"))

    obj_list = ['distance', 'occ_variance', 'pw_consumption']
    o_min_lst, o_max_lst = [], []
    for o in range(n_obj):
        o_min_aux, o_max_aux = ntw.getObjectiveBounds(obj_list[o])
        o_min_lst.append(o_min_aux)
        o_max_lst.append(o_max_aux)

    o_min, o_max = np.array(o_min_lst), np.array(o_max_lst)

    for o in range(n_obj):
        pf[:,o] = (pf[:,o] - o_min[o]) / (o_max[o] - o_min[o])

    return pf
if __name__ == "__main__":

    alg_name_rows = []
    execution_rows = []
    pop_value_rows = []
    gen_value_rows = []
    n_sol_list = []
    td_list = []
    metrics_dict = {k: [] for k in INDICATORS.keys()}
    
    pf = normalize_pf()


    df_all = pd.DataFrame()
    for algorithm in based_hybrid:
        df = load_data_hybrids(path_exp_hybrids,file_hybrids,algorithm)
        df["datatime"] = df.apply(lambda x: datetime.strptime(x["date"] + " " + x["time"], '%Y-%m-%d %H:%M:%S.%f'), axis=1)
            # datetime.strptime(df["date"] + " " + df["time"], '%Y-%m-%d %H:%M:%S')
        # print(df.head())
        df_all = pd.concat([df_all, df])
       
    seq = list(range(0,601,5))
    seq[0] = 1
    algorithm = "Hybrids"
    for gen in seq:
        print("\t Analyzing generation: ", gen)
        pre_gen = gen-1
        if pre_gen <= 0:#Merging all algorithm this non have sense
            start_time = df_all["datatime"].min()
        else:
            start_time = df_all.loc[df_all['generation'] == pre_gen, 'datatime'].min()
        
        pop_value_rows.append(len(df_all.loc[df_all['generation'] == gen]))
        gen_value_rows.append(gen)  
        alg_name_rows.append(algorithm)
        execution_rows.append(1)
        dft = df_all[(df_all['pf'] == 1) & (df_all['generation'] == gen)]
        
        delta = dft['datatime'].max() - start_time

        td_list.append(delta.total_seconds())
        n_sol_list.append(len(dft))

        
        # Performance indicators
        for name, ind_c in INDICATORS.items():
            if name == 'HV':
                ind = ind_c(np.ones((3)))
            else:
                # GD and IGD uses Pareto front
                ind = ind_c(pf)
            solution = ind(dft[['o1', 'o2', 'o3']].values)
            # print(solution)
            metrics_dict[name].append(solution)

    df_dict = {
        # 'Algorithm': alg_name_rows,
        'Seed': execution_rows,
        'Population': pop_value_rows,
        'Generation': gen_value_rows,
        'Solutions': n_sol_list,
        'TimeDelta': td_list}

    df_dict.update(metrics_dict)

    do = pd.DataFrame(df_dict)
    do.to_csv(path_exp_hybrids+"table_merge.csv", index=False)
