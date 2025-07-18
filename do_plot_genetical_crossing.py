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
file = "{algorithm}_1_400-500_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"

# col_dumps = np.array(ast.literal_eval(config['HYBRID_GEN_STEPS'])).ravel()
col_dumps = config['HYBRID_GEN_STEPS']
n_alg = len(config['HYBRID_ALGORITHMS'])
n_dumps = np.repeat(col_dumps,len(config['HYBRID_ALGORITHMS']))
print(n_dumps)
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

df = pd.DataFrame()
for algorithm in config['HYBRID_ALGORITHMS']:
    dft = pd.read_csv(path_exp + file.format(algorithm=algorithm),sep=" ",header=None)
    dft.drop(columns=[len(dft.columns)-1],inplace=True)
    dft.insert(loc=0, column='algorithm', value=algorithm)
    df = pd.concat([df,dft])

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

    dg = dt.groupby("Generation").mean().reset_index()
    dg.set_index('Generation', inplace=True)
    # Create stacked area plot
    axt.stackplot(dg.index, dg.T, labels=dg.columns)

    fig.savefig(f'{results_path}plots/genetic_crossing.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory







