import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from resopt.analysis.analyze import S, STE
import numpy as np
from utils import load_bash_config, load_data_merged_hybrids
from datetime import datetime
import sys
import pickle
import os
from tqdm import tqdm
# def normalize_pf():
#     pf = np.zeros((1, 3))
#         # Normalization of everything
   
#     ntw = pickle.load(open("data_individualexp/ntw_722_050-050-025_C", "rb"))

#     obj_list = ['distance', 'occ_variance', 'pw_consumption']
#     o_min_lst, o_max_lst = [], []
#     for o in range(n_obj):
#         o_min_aux, o_max_aux = ntw.getObjectiveBounds(obj_list[o])
#         o_min_lst.append(o_min_aux)
#         o_max_lst.append(o_max_aux)

#     o_min, o_max = np.array(o_min_lst), np.array(o_max_lst)

#     for o in range(n_obj):
#         pf[:,o] = (pf[:,o] - o_min[o]) / (o_max[o] - o_min[o])

#     return pf

if __name__ == "__main__":
    config = load_bash_config('script_constants.sh') #Dont detect all the variables
    HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
    CUT_GENERATION = 500
   
    HYBRID_N_GEN = config['HYBRID_N_GEN']
    N_EXECUTIONS = config['N_EXECUTIONS']
    N_OBJECTIVES = len(config['OBJECTIVES'])
    ALGORITHMS = ["Hybrid"]
    SEQ_HYBRIDS = 77 #TODO: change this considering the number of sequence exchange in the hybridization
    
    
    ## TMP
    SEQ_HYBRIDS = 17 #TODO: change this considering the number of sequence exchange in the hybridization
    results_path = "results_100/"
    N_EXECUTIONS  = 1
    

    SEQ_HYBRIDS = 24 #TODO: change this considering the number of sequence exchange in the hybridization
    results_path = "results_imperium/"
    N_EXECUTIONS  = 1
    

    SEQ_HYBRIDS = 17 #TODO: change this considering the number of sequence exchange in the hybridization
    results_path = "results_hybrid_400_500/"
    N_EXECUTIONS  = 3
    



    INDICATORS = {
        'GD': GD,
        'IGD': IGD,
        'HV': HV,
        'S': S,
        'STE': STE
    }
    path_exp_hybrids = results_path+"hybridization/"
    file_hybrids = path_exp_hybrids+"{algorithm}_{replica}_400_{HYBRID_N_GEN}.txt"
    based_hybrid = config['HYBRID_ALGORITHMS']

    # Load Reference points 
    output_path = results_path+"reference_points.txt"
    if os.path.exists(output_path):
        reference_points = pd.read_csv(output_path, sep="\t")
        reference_points = reference_points.loc[:, [f"o{i+1}" for i in range(N_OBJECTIVES)]]
        reference_points = reference_points.values
    else:
        sys.exit("Reference points file not found")
    print("Length of Reference points: ", len(reference_points))
    # print(reference_points)

    #Clean the output file
    if os.path.exists(results_path+f"table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv"):
        os.remove(results_path+f"table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv")
    
    for replica in tqdm(range(1,N_EXECUTIONS+1)):
        print(f"Processing replica {replica}")
        df_all = pd.DataFrame()
        for algorithm in ALGORITHMS:
            # print(f"\tProcessing {algorithm}")
            input_file = file_hybrids.format(algorithm=algorithm,replica=replica,HYBRID_POP_SIZE=HYBRID_POP_SIZE,HYBRID_N_GEN=HYBRID_N_GEN)
            try:
                df = load_data_merged_hybrids(input_file)
                df_all = pd.concat([df_all, df])
            except:
                print(f"\t\tFile does not exist: {input_file}")
                df_all = pd.DataFrame()
                continue
                
        if len(df_all) > 0:
            alg_name_rows = []
            execution_rows = []
            pop_value_rows = []
            gen_value_rows = []
            n_sol_list = []

            metrics_dict = {k: [] for k in INDICATORS.keys()}
        
                    
            seq = list(range(0,CUT_GENERATION+1,5))
            seq[0] = 1
            algorithm = "Hybrids"
            for gen in seq:
                
                pop_value_rows.append(len(df_all.loc[df_all['generation'] == gen]))
                gen_value_rows.append(gen)  
                alg_name_rows.append(algorithm)
                execution_rows.append(replica)
                dft = df_all[(df_all['generation'] == gen)]
            
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

            # When we analyse the whole file, we store the results
            df_dict = {
                'Algorithm': alg_name_rows,
                'Seed': execution_rows,
                'Population': pop_value_rows,
                'Generation': gen_value_rows,
                'Solutions': n_sol_list
            }

            df_dict.update(metrics_dict)
            do = pd.DataFrame(df_dict)
            file_exists = os.path.exists(results_path+f"table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv")
            do.to_csv(results_path+f"table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv", 
                    mode='a',  # append mode
                    header=not file_exists,  # write header only if file doesn't exist
                    index=False)
            



    print(f"End algorithm {algorithm}")
