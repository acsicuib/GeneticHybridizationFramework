import pandas as pd
import numpy as np
from utils import load_bash_config, load_data_normalized, load_data_hybrids


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
    
    # Get algorithm column name (should be the first column that's not an objective)
    algorithm_column = [col for col in combined_df.columns if not col.startswith('o')][0]
    
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
    network_file = "data/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path


    log_file = open("log_compute_unified_PF.log", "w")

    N_EXECUTIONS = config['N_EXECUTIONS']
    ALGORITHMS = config['ALGORITHMS']
    LAST_GENERATION = config['N_GEN']
    POP_SIZE = config['POP_SIZE']
    HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
    HYBRID_N_GEN = config['HYBRID_N_GEN']
    N_GEN = config['N_GEN']
    
    
    
    all_points = []
    for algorithm in ALGORITHMS:
        for seed2 in range(1,N_EXECUTIONS+1):
              print(f"Processing {algorithm}_{seed2}")
              log_file.write(f"Processing {algorithm}_{seed2}\n")
              input_file = f"data_individualexp/{algorithm}_{seed2}_{POP_SIZE}-{N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
              df = load_data_normalized(input_file,config)
              df = df[df["generation"] == LAST_GENERATION].loc[:, [algorithm] + [f"o{i+1}" for i in range(len(config['OBJECTIVES']))]]
              print("\tAlgorithm: ",algorithm, "PF points: ", len(df))
              log_file.write(f"\tAlgorithm: {algorithm}_{seed2} PF points: {len(df)}\n")
              all_points.append(df) 

    for algorithm in config['HYBRID_ALGORITHMS']:
        for seed2 in range(1,N_EXECUTIONS+1):
            print(f"Processing Hybrid {algorithm}_{seed2}")
            input_file = f"data_individualexp/hybrids/{algorithm}_{seed2}_{HYBRID_POP_SIZE}-{HYBRID_N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
            df = load_data_hybrids(input_file,config)
            df = df[(df["generation"] == LAST_GENERATION) & (df["pf"] == 1)].loc[:, [algorithm] + [f"o{i+1}" for i in range(len(config['OBJECTIVES']))]]
            print("\tBase Algorithm: ",algorithm, "Hybrid  PF points: ", len(df))
            log_file.write(f"\tBase Algorithm: {algorithm}_{seed2} Hybrid PF points: {len(df)}\n")
            all_points.append(df)

    # Find all Pareto efficient points
    pareto_front = find_pareto_efficient_points(all_points)
    
    print(f"Total points processed: {sum(len(df) for df in all_points)}")
    log_file.write(f"Total points processed: {sum(len(df) for df in all_points)}\n")
    print(f"Pareto efficient points found: {len(pareto_front)}")
    log_file.write(f"Pareto efficient points found: {len(pareto_front)}\n")
    
    if not pareto_front.empty:
        # Save Pareto front to file
        output_file = "pareto_front.txt"
        pareto_front.to_csv(output_file, index=False, sep='\t')
        log_file.write(f"\nPareto front saved to: {output_file}\n")
        print(f"\nPareto front saved to: {output_file}")
    else:
        print("No Pareto efficient points found.")
        log_file.write("No Pareto efficient points found.\n")

    log_file.close()
    
    
    
    
          