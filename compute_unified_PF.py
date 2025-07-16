import pandas as pd
import numpy as np
from utils import load_bash_config, load_data_normalized, load_data_hybrids
from tqdm import tqdm
import os
import argparse





def find_pareto_efficient_points(all_points):
    """
    Find all Pareto efficient points from a list of DataFrames containing objective values.
    
    Args:
        all_points: List of pandas DataFrames, each containing objective values and algorithm column
        
    Returns:
        pandas DataFrame: All Pareto efficient points with algorithm information preserved
    """
    # Combine all DataFrames into one
    if not all_points:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_points, ignore_index=True)
    
    # Remove duplicate points (including algorithm column)
    combined_df = combined_df.drop_duplicates()
    
    if combined_df.empty:
        return combined_df
    
    # Get objective column names
    obj_columns = [col for col in combined_df.columns if col.startswith('o')]
    
    # # Get algorithm column name (should be the first column that's not an objective)
    # algorithm_column = [col for col in combined_df.columns if not col.startswith('o')][0]
    
    # Convert to numpy array for faster computation (only objectives)
    points = combined_df[obj_columns].values
    
    # Initialize mask for Pareto efficient points
    pareto_mask = np.ones(len(points), dtype=bool)
    
    # Check each point against all others
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                # Check if point j dominates point i
                # A point dominates another if it's better or equal in all objectives
                # and strictly better in at least one objective
                if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                    pareto_mask[i] = False
                    break
    
    # Return only Pareto efficient points (including algorithm column)
    pareto_points = combined_df[pareto_mask].reset_index(drop=True)
    
    return pareto_points


if __name__ == "__main__":
    # Example usage
    config = load_bash_config('script_constants.sh') #Dont detect all the variables
    # network_file = "data/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path
    #TODO: change this to the correct path
    experiment_path = "results"
    exp_singles = experiment_path + "/ga_singles"
    exp_hybridization = experiment_path + "/hybridization"

    middle_path__exp = exp_singles + "/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics"
    network_file = exp_singles + "/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path

    ## TMP
    # experiment_path2 = "results_100"
    # exp_hybridization  = "results_100/hybridization"

    # experiment_path2 = "results_imperium"
    # exp_hybridization = "results_imperium/hybridization"
    
    # experiment_path2 = "results_hybrid_400_500"
    # exp_hybridization = "results_hybrid_400_500/hybridization"

    # single_output_file = experiment_path2 + "/reference_points_single_algorithms.txt"
    # hybrid_output_file = exp_hybridization + "/reference_points_hybrids.txt"
    final_output_file = experiment_path + "/reference_points.txt"  

    # Control variable: "singles", "hybrids", "merge"
    # Get this value from argparse
    parser = argparse.ArgumentParser(description='Compute unified Pareto front')
    parser.add_argument('--control', type=str, default='singles', help='Control which phases to run: singles, hybrids, merge')
    args = parser.parse_args()
    
    control = args.control

    # # Confirm the computation if file exits to avoid recomputing, 
    # if control == "singles" and os.path.exists(exp_singles + "/pareto_front_single_algorithms.txt"):
    #     print("Pareto front from single algorithms already computed")
    #     exit()
    # if control == "hybrids" and os.path.exists(exp_hybridization + "/pareto_front_hybrids.txt"):
    #     print("Pareto front from hybrids already computed")
    #     exit()
    # if control == "merge" and os.path.exists(experiment_path + "/pareto_front_final.txt"):
    #     print("Pareto front FINAL already computed!")
    #     exit()

    log_file = open("log_compute_unified_PF.log", "w")

    N_EXECUTIONS = config['N_EXECUTIONS']
    # ALGORITHMS = config['ALGORITHMS']
    ALGORITHMS = ["NSGA2","NSGA3","UNSGA3","SMSEMOA"]
    CUT_GENERATION = 500
    POP_SIZE = config['POP_SIZE']
    HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
    HYBRID_N_GEN = config['HYBRID_N_GEN']
    N_GEN = config['N_GEN']
    # TMP
    

    SEQ_HYBRIDS = 77 #TODO: change this considering the number of sequence exchange in the hybridization
    

        
    ## TEMPORAL
    
    SEQ_HYBRIDS = 17 #TMP

    # SEQ_HYBRIDS = 37
    N_EXECUTIONS = 30 
    ##

    # Initialize variables
    single_pareto_front = pd.DataFrame()
    
    # Phase 1: Compute Pareto front from single algorithms only
    if control in ["singles"]:
        print("=== PHASE 1: Computing Pareto front from single algorithms ===")
        log_file.write("=== PHASE 1: Computing Pareto front from single algorithms ===\n")
        
        single_algorithm_points = []
        for algorithm in ALGORITHMS:
            for seed2 in range(1,N_EXECUTIONS+1):
                  print(f"Loading {algorithm}_{seed2}")
                  log_file.write(f"Loading {algorithm}_{seed2}\n")
                  input_file = f"{middle_path__exp}/{algorithm}_{seed2}_{POP_SIZE}-{N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
                  df = load_data_normalized(input_file,config)
                  df = df[df["generation"] == CUT_GENERATION].loc[:, [f"o{i+1}" for i in range(len(config['OBJECTIVES']))]]
                  df["algorithm"] = algorithm
                  df["seed"] = seed2
                  print("\tAlgorithm: ",algorithm, "PF points: ", len(df))
                  log_file.write(f"\tAlgorithm: {algorithm}_{seed2} PF points: {len(df)}\n")
                  single_algorithm_points.append(df) 

        # Find Pareto front from single algorithms
        print("Joining single algorithm points")
        log_file.write("Joining single algorithm points\n")
        single_pareto_front = find_pareto_efficient_points(single_algorithm_points)
        
        print(f"Single algorithm points processed: {sum(len(df) for df in single_algorithm_points)}")
        log_file.write(f"Single algorithm points processed: {sum(len(df) for df in single_algorithm_points)}\n")
        print(f"Single algorithm Pareto efficient points found: {len(single_pareto_front)}")
        log_file.write(f"Single algorithm Pareto efficient points found: {len(single_pareto_front)}\n")
        
        # Save single algorithm Pareto front
        if not single_pareto_front.empty:
           
            single_pareto_front.to_csv(single_output_file, index=False, sep='\t')
            log_file.write(f"Single algorithm Pareto front saved to: {single_output_file}\n")
            print(f"Single algorithm Pareto front saved to: {single_output_file}")
        else:
            print("No single algorithm Pareto efficient points found.")
            log_file.write("No single algorithm Pareto efficient points found.\n")
    
    # Phase 2: Compute Pareto front from hybrids and join with single algorithms
    if control in ["hybrids"]:
        hybrid_points = []
        for algorithm in config['HYBRID_ALGORITHMS']:
            for seed2 in range(1,N_EXECUTIONS+1):
                #check if file exists
                if not os.path.exists(f"{exp_hybridization}/{algorithm}_{seed2}_{HYBRID_POP_SIZE}-{HYBRID_N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"):
                    print(f"File does not exist: {exp_hybridization}/{algorithm}_{seed2}_{HYBRID_POP_SIZE}-{HYBRID_N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt")
                    continue
                
                print(f"Processing Hybrid {algorithm}_{seed2}")
                input_file = f"{exp_hybridization}/{algorithm}_{seed2}_{HYBRID_POP_SIZE}-{HYBRID_N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
                df = load_data_hybrids(input_file,config,seq=SEQ_HYBRIDS)
                df = df[(df["generation"] == CUT_GENERATION) & (df["pf"] == 1)].loc[:,[f"o{i+1}" for i in range(len(config['OBJECTIVES']))]]
                df["algorithm"] = f"h_{algorithm}"
                df["seed"] = seed2
                print("\tBase Algorithm: ",algorithm, "Hybrid  PF points: ", len(df))
                log_file.write(f"\tBase Algorithm: {algorithm}_{seed2} Hybrid PF points: {len(df)}\n")
                hybrid_points.append(df)
                
        
        # Find Pareto front from hybrids
        print("Joining hybrids points")
        log_file.write("Joining hybrids points\n")
        hybrid_pareto_front = find_pareto_efficient_points(hybrid_points)
        
        print(f"Hybrid points processed: {sum(len(df) for df in hybrid_points)}")
        log_file.write(f"Hybrid points processed: {sum(len(df) for df in hybrid_points)}\n")
        print(f"Hybrid Pareto efficient points found: {len(hybrid_pareto_front)}")
        log_file.write(f"Hybrid Pareto efficient points found: {len(hybrid_pareto_front)}\n")
        
        # Save hybrid Pareto front
        if not hybrid_pareto_front.empty:
            
            hybrid_pareto_front.to_csv(hybrid_output_file, index=False, sep='\t')
            log_file.write(f"Hybrid Pareto front saved to: {hybrid_output_file}\n")
            print(f"Hybrid Pareto front saved to: {hybrid_output_file}")
        else:
            print("No single algorithm Pareto efficient points found.")
            log_file.write("No single algorithm Pareto efficient points found.\n")


    if control in ["merge"]:

        print("\n=== PHASE merge: Computing Pareto front from hybrids and joining with single algorithms ===")
        log_file.write("\n=== PHASE merge: Computing Pareto front from hybrids and joining with single algorithms ===\n")
        
        # If running only phase2, load the single algorithm Pareto front
        single_pareto_front = pd.read_csv(single_output_file, sep='\t')
        hybrid_points = pd.read_csv(hybrid_output_file, sep='\t')
        

        # Combine single algorithm Pareto front with hybrid points
        all_points_phase2 = [single_pareto_front] + [hybrid_points]
            
        # Find final Pareto front from combined data
        final_pareto_front = find_pareto_efficient_points(all_points_phase2)
        
        print(f"Hybrid points processed: {sum(len(df) for df in hybrid_points)}")
        log_file.write(f"Hybrid points processed: {sum(len(df) for df in hybrid_points)}\n")
        print(f"Total points in Phase 2: {len(single_pareto_front) + sum(len(df) for df in hybrid_points)}")
        log_file.write(f"Total points in Phase 2: {len(single_pareto_front) + sum(len(df) for df in hybrid_points)}\n")
        print(f"Final Pareto efficient points found: {len(final_pareto_front)}")
        log_file.write(f"Final Pareto efficient points found: {len(final_pareto_front)}\n")
        
        # Save final Pareto front
        if not final_pareto_front.empty:
            
            final_pareto_front.to_csv(final_output_file, index=False, sep='\t')
            log_file.write(f"Final Pareto front saved to: {final_output_file}\n")
            print(f"Final Pareto front saved to: {final_output_file}")

    log_file.close()
    
    
    
    
          