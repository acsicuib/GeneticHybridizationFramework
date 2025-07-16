from utils import load_bash_config, load_data_hybrids
from compute_unified_PF import find_pareto_efficient_points
from datetime import datetime
import pandas as pd
import os


if __name__ == "__main__":
    config = load_bash_config('script_constants.sh') #Dont detect all the variables
    HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
    CUT_GENERATION = 500
   
    HYBRID_N_GEN = config['HYBRID_N_GEN']
    N_EXECUTIONS = config['N_EXECUTIONS']
    N_OBJECTIVES = len(config['OBJECTIVES'])
    ALGORITHMS = config['ALGORITHMS']
    SEQ_HYBRIDS = 77 #TODO: change this considering the number of sequence exchange in the hybridization
    
    ## TMP
    # SEQ_HYBRIDS = 17
    # results_path = "results_100/"
    # N_EXECUTIONS = 1        
    # SEQ_HYBRIDS = 37
    SEQ_HYBRIDS = 17
    results_path = "results_hybrid_400_500/"
    N_EXECUTIONS = 3      

    path_exp_hybrids = results_path+"hybridization/"
    file_hybrids = path_exp_hybrids+"{algorithm}_{replica}_{HYBRID_POP_SIZE}-{HYBRID_N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
    output_path = results_path+"hybridization/"

    for replica in range(1,N_EXECUTIONS+1):
        print(f"Processing replica {replica}")
        df_all = pd.DataFrame()
        file_output = path_exp_hybrids+f"Hybrid_{replica}_{HYBRID_POP_SIZE*len(ALGORITHMS)}_{CUT_GENERATION}.txt"
        if os.path.exists(file_output):
            os.remove(file_output)

        for algorithm in ALGORITHMS:
            input_file = file_hybrids.format(algorithm=algorithm,replica=replica,HYBRID_POP_SIZE=HYBRID_POP_SIZE,HYBRID_N_GEN=HYBRID_N_GEN)
            try:
                df = load_data_hybrids(input_file,config,seq=SEQ_HYBRIDS)
                df["datatime"] = df.apply(lambda x: datetime.strptime(x["date"] + " " + x["time"], '%Y-%m-%d %H:%M:%S.%f'), axis=1)
                df_all = pd.concat([df_all, df])
            except Exception as e:
                print(f"Error loading data from {input_file}: {e}")
                df_all = pd.DataFrame()
                continue
        
        if not df_all.empty:
            for gen in range(1,CUT_GENERATION+1):
                # Create a DataFrame with the objective values and add an algorithm column
                df_objectives = df_all[df_all['generation'] == gen][['o1', 'o2', 'o3']].copy()
                df_objectives['algorithm'] = 'Hybrid'  # Add algorithm column as required by the function
                pareto_front = find_pareto_efficient_points([df_objectives])
                # print(f"Generation {gen}: {len(df_objectives)} -> {len(pareto_front)}")
                df = pd.DataFrame(pareto_front)
                df.drop(columns=['algorithm'],inplace=True)
                df['generation'] = gen
               
                df.to_csv(file_output, 
                        mode='a',  # append mode
                        header=not os.path.exists(file_output),  # write header only if file doesn't exist
                        index=False)
               

    
