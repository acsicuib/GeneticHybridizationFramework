import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from resopt.analysis.analyze import S, STE
import numpy as np
from utils import load_bash_config, load_data_normalized
from datetime import datetime
import sys
import pickle
import os
from tqdm import tqdm

if __name__ == "__main__":
    config = load_bash_config('script_constants.sh') #Dont detect all the variables
    middle_path_sol_exp = "results/ga_singles/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/"
    file = "{algorithm}_{seed}_{POP_SIZE}-{N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
    output_path = "results/"

    POP_SIZE = config['POP_SIZE']
    N_GEN = config['N_GEN']
    HYBRID_N_GEN = config['HYBRID_N_GEN']
    N_EXECUTIONS = config['N_EXECUTIONS']
    N_OBJECTIVES = len(config['OBJECTIVES'])
    ALGORITHMS = config['ALGORITHMS']
    CUT_GENERATION = 500

    INDICATORS = {
        'GD': GD,
        'IGD': IGD,
        'HV': HV,
        'S': S,
        'STE': STE
    }
    
    # Load Pareto front
    if os.path.exists(output_path+"pareto_front_exp.txt"):
        reference_points = pd.read_csv(output_path+"pareto_front_exp.txt", sep="\t")
        reference_points = reference_points.loc[:, [f"o{i+1}" for i in range(N_OBJECTIVES)]]
        reference_points = reference_points.values
    else:
        sys.exit("Pareto front file not found")
    print("Length of Pareto front: ", len(reference_points))
    print(reference_points)


    #Clean the output file
    if os.path.exists(output_path+f"table_standard_{POP_SIZE}_{N_GEN}.csv"):
        os.remove(output_path+f"table_standard_{POP_SIZE}_{N_GEN}.csv")

    for algorithm in ALGORITHMS:
        for replica in tqdm(range(1,N_EXECUTIONS+1)):
            tqdm.write(f"Processing {algorithm} {replica}")
            input_file = middle_path_sol_exp + file.format(algorithm=algorithm,seed=replica,POP_SIZE=POP_SIZE,N_GEN=N_GEN)
            df = load_data_normalized(input_file,config)
            df["datatime"] = df.apply(lambda x: datetime.strptime(x["date"] + " " + x["time"], '%Y-%m-%d %H:%M:%S.%f'), axis=1)
            # df_all = pd.concat([df_all, df])
 
            alg_name_rows = []
            execution_rows = []
            pop_value_rows = []
            gen_value_rows = []
            n_sol_list = []
            td_list = []
            metrics_dict = {k: [] for k in INDICATORS.keys()}
               
            seq = list(range(0,CUT_GENERATION+1,5))
            seq[0] = 1

            for gen in seq:
                pre_gen = gen-1
                if pre_gen <= 0:#Merging all algorithm this non have sense
                    start_time = df.loc[df['generation'] == 1, 'datatime'].min()
                else:
                    start_time = df.loc[df['generation'] == pre_gen, 'datatime'].min()
                
                pop_value_rows.append(len(df.loc[df['generation'] == gen]))
                gen_value_rows.append(gen)  
                alg_name_rows.append(algorithm)
                execution_rows.append(replica)
                dft = df[(df['generation'] == gen)]
                
                delta = dft['datatime'].max() - start_time

                td_list.append(delta.total_seconds())
                n_sol_list.append(len(dft))

            
                # Performance indicators
                for name, ind_c in INDICATORS.items():
                    if name == 'HV':
                        # For HV, we need a reference point that dominates all solutions
                        # We'll use the maximum values from the Pareto front plus a small margin
                        # ref_point = np.max(reference_points, axis=0) + 0.1
                        ref_point = np.ones(reference_points.shape[1]) + 0.1
                        ind = ind_c(ref_point)
                    else:
                        # GD and IGD uses Pareto front
                        ind = ind_c(reference_points)
                    solution = ind(dft[['o1', 'o2', 'o3']].values)
                    # print(solution)
                    metrics_dict[name].append(solution)

                df_dict = {
                    'Algorithm': alg_name_rows,
                    'Seed': execution_rows,
                    'Population': pop_value_rows,
                    'Generation': gen_value_rows,
                    'Solutions': n_sol_list,
                    'TimeDelta': td_list}

                df_dict.update(metrics_dict)

                do = pd.DataFrame(df_dict)
                # Check if file exists to determine if we need to write header
                file_exists = os.path.exists(output_path+f"table_standard_{POP_SIZE}_{N_GEN}.csv")
                do.to_csv(output_path+f"table_standard_{POP_SIZE}_{N_GEN}.csv", 
                         mode='a',  # append mode
                         header=not file_exists,  # write header only if file doesn't exist
                         index=False)
                
