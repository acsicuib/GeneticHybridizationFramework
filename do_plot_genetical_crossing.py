import pandas as pd
import numpy as np
import ast
from utils import load_bash_config
import matplotlib.pyplot as plt
import seaborn as sns


COLORS = dict(zip("NSGA2 NSGA3 UNSGA3 SMSEMOA Hybrid".split(),['#ff8500', '#FF595E', '#1982C4', '#6A4C93' ,'#8AC926']))

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

columns = ["algorithm", "replica", "date", "time", "isPF", "Generation"] 
columns += [f"O{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
t = len(columns)
columns += pairs

print(columns)

df = pd.DataFrame()
for algorithm in config['HYBRID_ALGORITHMS']:
    for replica in range(1, replicas + 1):
        try:
            dft = pd.read_csv(path_exp + file.format(algorithm=algorithm, replica=replica), sep=" ", header=None)
            dft.drop(columns=[len(dft.columns)-1], inplace=True)
            dft.insert(loc=0, column='algorithm', value=algorithm)
            dft.insert(loc=1, column='replica', value=replica)
            df = pd.concat([df, dft], ignore_index=True)
        except FileNotFoundError:
            print(f"File not found: {path_exp + file.format(algorithm=algorithm, replica=replica)}")
            continue

# Update columns to include 'replica'
columns = ["algorithm", "replica", "date", "time", "isPF", "Generation"]
columns += [f"O{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
t = len(columns)
columns += pairs

df.columns = columns

print(df.head())

filter = ["A0_100","A1_100","A2_100","A3_100"]
rename_filter = dict(zip(filter,config['HYBRID_ALGORITHMS']))
filter.append("Generation")
print(filter)



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8),
                        layout="constrained")

for ixa,algorithm in enumerate(config['HYBRID_ALGORITHMS']):
    axt = axs[ixa//2,ixa%2]
    axt.set_title(algorithm)
    

    dt = df[df['algorithm'] == algorithm].loc[:,filter]
    dt.rename(columns=rename_filter,inplace=True)

    # Group by Generation and replica, then average across replicas
    dg = dt.groupby(["Generation"]).mean().reset_index()
    dg = dg.groupby("Generation").mean().reset_index()
    dg.set_index('Generation', inplace=True)
    # Create stacked area plot
    stack = axt.stackplot(
        dg.index,
        dg.T,
        labels=dg.columns,
        colors=[COLORS.get(col, "#cccccc") for col in dg.columns]
    )
    for x in xlines:
        axt.vlines(x, 0, 1, colors='black', linestyles='dashed')

    # Annotate with percentage text at midpoints between vlines
    y_stack = np.vstack([np.zeros(len(dg)), np.cumsum(dg.T, axis=0)])
    xlines_sorted = np.sort(xlines)
    # Add start and end for intervals
    x_intervals = [dg.index[0]] + list(xlines_sorted) + [dg.index[-1]]
    midpoints = [(x_intervals[i] + x_intervals[i+1]) / 2 for i in range(len(x_intervals)-1)]
    # For each midpoint, find the closest generation in dg.index
    for midpoint in midpoints:
        closest_idx = np.abs(dg.index - midpoint).argmin()
        x = dg.index[closest_idx]
        total = dg.iloc[closest_idx].sum()
        if total == 0:
            continue
        for i, col in enumerate(dg.columns):
            percent = dg.iloc[closest_idx, i] / total * 100
            if percent < 0.1:  # Only annotate if >0.1% to avoid clutter
                 continue
            y_bottom = y_stack[i, closest_idx]
            y_top = y_stack[i+1, closest_idx]
            y_center = (y_bottom + y_top) / 2
            axt.text(x, y_center, f"{percent:.1f}%", ha='center', va='center', fontsize=8, color='white', weight='bold')

# Remove per-axes legend
# Add a single legend for the whole figure, outside the rightmost subplot
handles, labels = axs[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=14)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=10, ncol=len(labels))
fig.savefig(f'{results_path}plots/genetic_crossing.png', dpi=300, bbox_inches='tight')
plt.close(fig)  # Close the figure to free memory







