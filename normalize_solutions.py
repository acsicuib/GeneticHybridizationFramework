import numpy as np
import pandas as pd
from pathlib import Path
import pickle

def normalize_objectives(input_file, network_file, output_file):
    # Load the network to get objective bounds
    with open(network_file, 'rb') as f:
        network = pickle.load(f)
    
    # Get objective bounds from network
    obj_list = ['distance', 'occ_variance', 'pw_consumption']  # These are the objectives in your file
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
                obj1 = float(parts[3])
                obj2 = float(parts[4])
                obj3 = float(parts[5])
                data.append([date, time, generation, obj1, obj2, obj3])
    
    # Convert to numpy array for easier processing
    data = np.array(data)
    
    # Normalize the objectives using the network bounds
    normalized_data = []
    for row in data:
        date, time, gen_num = row[0], row[1], row[2]
        obj1_norm = (float(row[3]) - f_min_list[0]) / (f_max_list[0] - f_min_list[0])
        obj2_norm = (float(row[4]) - f_min_list[1]) / (f_max_list[1] - f_min_list[1])
        obj3_norm = (float(row[5]) - f_min_list[2]) / (f_max_list[2] - f_min_list[2])
        
        normalized_data.append([date, time, gen_num, obj1_norm, obj2_norm, obj3_norm])
    
    # Write normalized data to output file
    with open(output_file, 'w') as f:
        for row in normalized_data:
            f.write(f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}\n")

if __name__ == "__main__":
    # Example usage
    # input_file = "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/NSGA2_1_100-100_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
    network_file = "data/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path
    
    for algorithm in ["NSGA2", "NSGA3", "UNSGA3", "SMSEMOA"]:
        input_file = f"data_individualexp/{algorithm}_1_400-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
        output_file = str(Path(input_file).with_suffix('.normalized.txt'))
        normalize_objectives(input_file, network_file, output_file)
        print(f"Normalized data written to: {output_file}") 