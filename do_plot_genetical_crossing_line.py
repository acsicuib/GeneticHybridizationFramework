import pandas as pd
import numpy as np
import ast
from utils import load_bash_config
import matplotlib.pyplot as plt
import seaborn as sns


config = load_bash_config('script_constants.sh') 
# middle_path_sol_exp = "results/ga_singles/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/exp_single"
# middle_path_ana_exp = "results/ga_singles/analysis/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/exp_single"
# network_file = "results/ga_singles/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path
results_path = "results/"
# results_path = "results_longest/"

# N_EXECUTIONS = config['N_EXECUTIONS']
N_EXECUTIONS = 30
ALGORITHMS = config['ALGORITHMS']
POP_SIZE = config['POP_SIZE']
HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
HYBRID_N_GEN = config['HYBRID_N_GEN']
N_GEN = config['N_GEN']
CUT_GENERATION = 500
SEEDS = range(1,N_EXECUTIONS+1)
 
SEQ_HYBRIDS = 17
path_exp = results_path+"hybridization/"

replicas = 30
file = "{algorithm}_{replica}_400-500_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"

# col_dumps = np.array(ast.literal_eval(config['HYBRID_GEN_STEPS'])).ravel()
col_dumps = config['HYBRID_GEN_STEPS']
n_alg = len(config['HYBRID_ALGORITHMS'])
n_dumps = np.repeat(col_dumps,len(config['HYBRID_ALGORITHMS']))
print(n_dumps)
xlines = np.unique(n_dumps)
print(xlines)
print(len(n_dumps))
n_dumps== len(config['HYBRID_GEN_STEPS'])*len(config['HYBRID_ALGORITHMS'])

pairs = []
for i,n in enumerate(n_dumps):
    ixa = i%n_alg
    pairs.extend([f'A{ixa}_{n}'])
    # for i in range(len(col_dumps)):
    #     pairs.extend([f'D{n}{i}'])

columns = ["algorithm", "date", "time", "isPF", "Generation"] 
columns += [f"O{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
t = len(columns)
columns += pairs

print(columns)

# Load all replicas for each algorithm

df = pd.DataFrame()
for algorithm in config['HYBRID_ALGORITHMS']:
    for replica in range(1, replicas + 1):
        print(f"Loading {algorithm} replica {replica}")
        path = path_exp + file.format(algorithm=algorithm, replica=replica)
        dft = pd.read_csv(path, sep=" ", header=None)
        dft.drop(columns=[len(dft.columns)-1], inplace=True)
        dft.insert(loc=0, column='algorithm', value=algorithm)
        dft['replica'] = replica  # Add replica column
        df = pd.concat([df, dft])

# Update columns to include 'replica'
df.columns = columns + ['replica']

# print(df.head())

filter = ["A0_100","A1_100","A2_100","A3_100"]
rename_filter = dict(zip(filter,config['HYBRID_ALGORITHMS']))
filter.append("Generation")
# print(filter)



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8),
                        layout="constrained")

for ixa,algorithm in enumerate(config['HYBRID_ALGORITHMS']):
    axt = axs[ixa//2,ixa%2]
    axt.set_title(algorithm)
    axt.set_ylim(0,1)
    
    dt = df[df['algorithm'] == algorithm].loc[:,filter + ['replica']]
    dt.rename(columns=rename_filter,inplace=True)

    # Group by Generation and compute mean and std for each algorithm column
    dg_mean = dt.groupby("Generation")[config['HYBRID_ALGORITHMS']].mean()
    dg_std = dt.groupby("Generation")[config['HYBRID_ALGORITHMS']].std()

    # Plot mean and fill std
    line_objs = {}
    for col in config['HYBRID_ALGORITHMS']:
        line, = axt.plot(dg_mean.index, dg_mean[col], label=col, linewidth=2)
        # axt.fill_between(dg_mean.index, 
        #                  dg_mean[col] - dg_std[col], 
        #                  dg_mean[col] + dg_std[col], 
        #                  color=line.get_color(), alpha=0.2)
        line_objs[col] = line
    for x in xlines:
        axt.vlines(x, 0, 1, colors='black', linestyles='dashed')
        # Add X-mark at the first generation greater than the vline
        for col in config['HYBRID_ALGORITHMS']:
            # Find the first index in dg_mean.index greater than x
            next_indices = np.where(dg_mean.index > x)[0]
            if len(next_indices) == 0:
                continue  # No next value exists
            next_idx = next_indices[0]
            x_val = dg_mean.index[next_idx]
            y_val = dg_mean.iloc[next_idx][col]
            axt.plot(x_val, y_val, marker='o', color=line_objs[col].get_color(), markersize=5, markeredgewidth=2)

    # Annotate with percentage text at specific indices: 101, 201, 301, 401
    target_indices = [101, 201, 301, 401]
    algs = config['HYBRID_ALGORITHMS']
    y_offset = 0.04  # vertical offset between annotations
    x_offset = -34

    for idx in target_indices:
        if idx not in dg_mean.index:
            continue  # Skip if this generation is not present
        total = dg_mean.loc[idx].sum()
        if total == 0:
            continue
        for j, col in enumerate(algs):
            percent = dg_mean.loc[idx][col] / total * 100
            if percent < 0.1:
                continue
            y = dg_mean.loc[idx][col]
            color = line_objs[col].get_color()
            # Determine annotation position and alignment (reuse your logic if needed)
            if col.upper() == 'NSGA2':
                y = y + y_offset
                va = 'bottom'
                if ixa == 0 and idx < 200:
                    y = y - y_offset * 3.
                if ixa == 0 and idx >= 200:
                    y = y + y_offset * 1
                if ixa == 3:
                    y = y - y_offset * 2.8
                if ixa == 1 or ixa == 2:
                    y = y - y_offset
            elif col.upper() == 'SMSEMOA':
                y = y + y_offset
                va = 'top'
                if ixa == 0 and idx <= 200:
                    y = y - y_offset * 2.2
                if ixa == 3:
                    y = y - y_offset *1.1

            elif col.upper() in ['UNSGA3']:
                y = y - y_offset
                va = 'top'
                if col.upper() == 'UNSGA3' and ixa == 2 and idx >= 100 and idx <= 200:
                    y = y - y_offset * 1.2
                if ixa == 0 and idx < 200:
                    y = y - y_offset * 0.7
                if ixa == 3:
                    y = y - y_offset * 0.1 
            elif col.upper() in ['NSGA3']:
                y = y + y_offset
                va = 'bottom'
                # if ixa == 1 and idx < 100:
                #     y = y - y_offset
                if ixa == 1:
                    y = y + y_offset * 0.5
                if ixa == 2:
                    y = y + y_offset * 0.5
            else:
                va = 'bottom'
            axt.text(idx - x_offset, y, f"{percent:.1f}%", ha='center', va=va, fontsize=8, color=color, weight='bold')

# Remove per-axes legend
# Add a single legend for the whole figure, outside the rightmost subplot
handles, labels = axs[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=14)
fig.savefig(f'{results_path}plots/genetic_crossing_lines.png', dpi=300, bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory







