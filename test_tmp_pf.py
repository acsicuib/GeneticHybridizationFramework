from utils import load_bash_config, load_data_hybrids, load_data_normalized
from compute_unified_PF import find_pareto_efficient_points
from datetime import datetime
import pandas as pd
import os

config = load_bash_config('script_constants.sh') #Dont detect all the variables

middle_path_sol_exp = "results/ga_singles/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/"
file = "{algorithm}_{seed}_400-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
algorithm = "NSGA2"
replica = 3


for replica in range(1,10):
    input_file = middle_path_sol_exp + file.format(algorithm=algorithm,seed=replica)
    df = load_data_normalized(input_file,config)

    # Get the points and add algorithm column as expected by the function
    points_df = df[df["generation"] == 500].iloc[:,-3:].copy()
    points_df["algorithm"] = algorithm  # Add algorithm column as expected by the function
    

    pareto_front = find_pareto_efficient_points([points_df])  # Pass as list of DataFrames

    print(len(points_df))
    print(len(pareto_front))
    if len(points_df) != len(pareto_front):
        print(f"Replica {replica} has a different number of points than the pareto front")
        break
    print("*"*50)
