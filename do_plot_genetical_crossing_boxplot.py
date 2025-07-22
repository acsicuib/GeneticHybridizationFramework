import pandas as pd
import numpy as np
import ast
from utils import load_bash_config
import matplotlib.pyplot as plt
import seaborn as sns

config = load_bash_config('script_constants.sh')
# results_path = "results/"
results_path = "results_longest/"
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

replicas = 2
file = "{algorithm}_{replica}_400-500_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"

col_dumps = config['HYBRID_GEN_STEPS']
n_alg = len(config['HYBRID_ALGORITHMS'])
n_dumps = np.repeat(col_dumps,len(config['HYBRID_ALGORITHMS']))
xlines = np.unique(n_dumps)

pairs = []
for i,n in enumerate(n_dumps):
    ixa = i%n_alg
    pairs.extend([f'A{ixa}_{n}'])

columns = ["algorithm", "date", "time", "isPF", "Generation"] 
columns += [f"O{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
columns += pairs

# Load all replicas for each algorithm
df = pd.DataFrame()
for algorithm in config['HYBRID_ALGORITHMS']:
    for replica in range(1, replicas + 1):
        path = path_exp + file.format(algorithm=algorithm, replica=replica)
        dft = pd.read_csv(path, sep=" ", header=None)
        dft.drop(columns=[len(dft.columns)-1], inplace=True)
        dft.insert(loc=0, column='algorithm', value=algorithm)
        dft['replica'] = replica  # Add replica column
        df = pd.concat([df, dft])

# Update columns to include 'replica'
df.columns = columns + ['replica']

# Prepare for boxplot: melt the relevant columns for each algorithm
filter_cols = [f"A{i}_{col_dumps[-1]}" for i in range(n_alg)]
rename_filter = dict(zip(filter_cols, config['HYBRID_ALGORITHMS']))

# Bin generations according to HYBRID_GEN_STEPS
bins = [0] + list(col_dumps)
labels = [f"1-{col_dumps[0]}"] + [f"{col_dumps[i-1]+1}-{col_dumps[i]}" for i in range(1, len(col_dumps))]
df['GenBin'] = pd.cut(df['Generation'], bins=bins, labels=labels, right=True)

# For each subplot (algorithm), plot boxplots for each generation bin, with boxes for all algorithms
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), layout="constrained")

for ixa, algorithm in enumerate(config['HYBRID_ALGORITHMS']):
    axt = axs[ixa//2, ixa%2]
    axt.set_title(algorithm)
    # Prepare data for this algorithm: melt across all algorithms for each bin
    dt = df[df['algorithm'] == algorithm]
    # For each bin, collect values for all algorithms
    plot_data = []
    for bin_label in labels:
        for alg in config['HYBRID_ALGORITHMS']:
            # Find the column for this algorithm in this bin
            col = f"A{config['HYBRID_ALGORITHMS'].index(alg)}_{col_dumps[labels.index(bin_label)]}" if bin_label in labels else None
            if col and col in dt.columns:
                vals = dt[dt['GenBin'] == bin_label][col].dropna().values
                for v in vals:
                    plot_data.append({'GenBin': bin_label, 'Algorithm': alg, 'Value': v})
    plot_df = pd.DataFrame(plot_data)
    if not plot_df.empty:
        sns.boxplot(data=plot_df, x='GenBin', y='Value', hue='Algorithm', ax=axt)
        axt.set_ylim(0, 1)
        axt.legend(loc='best', fontsize=8)
    else:
        axt.text(0.5, 0.5, 'No data', ha='center', va='center')

fig.suptitle('Boxplots of Algorithm Performance per Generation Bin', fontsize=16)
fig.savefig(f'{results_path}plots/genetic_crossing_boxplots.png', dpi=300, bbox_inches='tight')
plt.close(fig) 