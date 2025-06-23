import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os
from utils import load_bash_config

def normalize_objectives(input_file, network_file, output_file):
    """
    Normalize the objectives of the solutions to the network bounds 
    Used in GA standalone 

        input_file: file with the solutions
    network_file: file with the network
    output_file: file with the normalized solutions
    """
    # Load the network to get objective bounds
    with open(network_file, 'rb') as f:
        network = pickle.load(f)
    
    # Get objective bounds from network
    obj_list = OBJECTIVES  # These are the objectives in your file
    f_min_list = []
    f_max_list = []
    for obj in obj_list:
        f_min, f_max = network.getObjectiveBounds(obj)
        if f_min == f_max:
            f_max += 1  # Avoid division by zero
        f_min_list.append(f_min)
        f_max_list.append(f_max)
    
    # Read the data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure we have all required fields
                date = parts[0]
                time = parts[1]
                generation = int(parts[2])
                obj1 = float(parts[3]) #TODO: change this
                obj2 = float(parts[4]) #TODO: change this 
                obj3 = float(parts[5]) #TODO: change this
                
                data.append([date, time, generation, obj1, obj2, obj3])
    
    # Convert to numpy array for easier processing
    data = np.array(data)
    
    # Normalize the objectives using the network bounds
    normalized_data = []
    for row in data:
        date, time, gen_num = row[0], row[1], row[2]
        obj_norm = np.zeros(len(obj_list))
        for i, (obj, f_min, f_max) in enumerate(zip(obj_list, f_min_list, f_max_list)):
            obj_norm[i] = (float(row[3+i]) - f_min) / (f_max - f_min)
        normalized_data.append([date, time, gen_num, *obj_norm])
    
    # Write normalized data to output file
    with open(output_file, 'w') as f:
        for row in normalized_data:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}\n") #TODO: change this

if __name__ == "__main__":
    # Example usage
    config = load_bash_config('script_constants.sh') #Dont detect all the variables
    N_EXECUTIONS = config['N_EXECUTIONS']
    ALGORITHMS = config['ALGORITHMS']
    OBJECTIVES = config['OBJECTIVES']
    POP_SIZE = config['POP_SIZE']
    N_GEN = config['N_GEN']
    CUT_GENERATION = config['CUT_GENERATION']
    # ALGORITHMS = ['SMSEMOA']

    middle_path__exp = "solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics"
    network_file = "results/ga_singles/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path
    

    for algorithm in ALGORITHMS:
        for seed2 in range(1,N_EXECUTIONS+1):
            input_file = f"results/ga_singles/{middle_path__exp}/{algorithm}_{seed2}_{POP_SIZE}-{N_GEN}_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
            output_file = str(Path(input_file).with_suffix('.normalized.txt'))
            normalize_objectives(input_file, network_file, output_file)
            print(f"Normalized data written to: {output_file}") 