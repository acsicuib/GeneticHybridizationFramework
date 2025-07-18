import pandas as pd
from utils import load_bash_config
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def save_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """Save the plot to a file with high resolution"""
    # Create plots directory if it doesn't exist
    os.makedirs(f'{results_path}/plots', exist_ok=True)
    # Save the plot
    fig.savefig(f'{results_path}/plots/{filename}.png', dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)  # Close the figure to free memory

def plot_metrics_time_series(file_metrics,file_hybrids=None):
    """
    Plot time series of metrics (GD, IGD, HV, S, STE) across generations for all algorithms.
    Each metric has its own subplot, with different algorithms shown as lines.
    
    Args:
        path_exp (str): Path to the experiment directory
        file_metrics (str): Name of the metrics file
        algorithms (list): List of algorithm names to plot
    """
    # Read the metrics file

    for replica in SEEDS:
        df = pd.read_csv(file_metrics)
        df = df.loc[df["Generation"]<=CUT_GENERATION].copy()
        
        if file_hybrids is not None:
            df_hybrids = pd.read_csv(file_hybrids)
        else:
            df_hybrids = None

    # Create a figure with subplots for each metric
    metrics = ['GD', 'IGD', 'HV', 'S', 'STE']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
    # fig.suptitle('Metrics Evolution Across Generations', fontsize=16, y=0.95)
    
    # Define a color palette for algorithms
    colors = custom_colors
    colors_hybrids = custom_colors_hybrids

    # Create a single legend for all subplots
    handles = []
    labels = []
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot each algorithm
        for j, alg in enumerate(ALGORITHMS):
            # Get data for this algorithm
            alg_data = df[df['Algorithm'] == alg]
            # Group by generation and calculate mean and std
            grouped = alg_data.groupby('Generation')[metric].agg(['mean', 'std'])

            # Plot mean with error bars for original algorithm
            line = ax.plot(grouped.index, grouped['mean'], 
                         label=alg, 
                         color=colors[j],
                         linewidth=2,
                         marker='o',
                         markersize=3,
                         markevery=10)  # Show markers every 10 generations
            
            # Add shaded area for standard deviation for original algorithm
            ax.fill_between(grouped.index, 
                          grouped['mean'] - grouped['std'],
                          grouped['mean'] + grouped['std'],
                          color=colors[j],
                          alpha=0.2)
            
            # Store handle and label for legend
            if i == 0:  # Only need to store once
                handles.extend(line)
                labels.append(alg)
        # end single algorithms
        if file_hybrids is not None:
            grouped_hybrids = df_hybrids.groupby('Generation')[metric].agg(['mean', 'std'])
            # Plot mean with error bars for hybrid
            linehybrids = ax.plot(grouped_hybrids.index, grouped_hybrids['mean'], 
                        label="Hybrid", 
                        color=colors_hybrids[0],
                        linewidth=2,
                        marker='v',
                        markersize=4,
                        markevery=10)  # Show markers every 10 generations
        # Add shaded area for standard deviation for hybrid
            ax.fill_between(grouped_hybrids.index, 
                        grouped_hybrids['mean'] - grouped_hybrids['std'],
                        grouped_hybrids['mean'] + grouped_hybrids['std'],
                        color=colors_hybrids[0],
                        alpha=0.2)

            if i == 0:  # Only need to store once
                handles.extend(linehybrids)
                labels.append("Hybrid")
        
        # Customize subplot
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0 for GD, IGD, and S
        if metric in ['GD', 'IGD', 'S']:
            ax.set_ylim(bottom=0)
        
        # Add minor grid lines
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)
        ax.minorticks_on()
        
        # Customize tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
        
        # Add metric name as text in the top-right corner
        ax.text(0.98, 0.95, metric, 
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right')
    
    # Add a single legend at the top of the figure
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 0.98),
              ncol=len(ALGORITHMS)+1,
              fontsize=10,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the legend
    
    # Save the plot
    save_plot(fig, 'metrics_time_series')
    plt.show()


config = load_bash_config('script_constants.sh') 
# middle_path_sol_exp = "results/ga_singles/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/exp_single"
# middle_path_ana_exp = "results/ga_singles/analysis/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/exp_single"
# network_file = "results/ga_singles/networks/ntw_722_050-050-025_C"  # You'll need to provide the correct network file path
results_path = "results_longest/"

# N_EXECUTIONS = config['N_EXECUTIONS']
N_EXECUTIONS = 30
ALGORITHMS = config['ALGORITHMS']
POP_SIZE = config['POP_SIZE']
HYBRID_POP_SIZE = config['HYBRID_POP_SIZE']
HYBRID_N_GEN = config['HYBRID_N_GEN']
N_GEN = config['N_GEN']
CUT_GENERATION = 500
SEEDS = range(1,N_EXECUTIONS+1)

print("--------------------------------")
print("Available algorithms: ", ALGORITHMS)
print("--------------------------------")

file_metrics = results_path + f"/table_standard_{POP_SIZE}_{N_GEN}.csv" # BASED ON CUT_GENERATION
file_hybrids = results_path + f"/table_hybrids_{HYBRID_POP_SIZE}_{HYBRID_N_GEN}.csv" 

custom_colors = ['#ff8500', '#FF595E', '#1982C4', '#6A4C93']
cmap = ListedColormap(custom_colors)
markers = ['o', 'o', 'o', 'o'] 
based_hybrid = ['Hybrids']
markers_hybrids = ['v']
custom_colors_hybrids = ['#8AC926']

if __name__ == "__main__":

    plot_metrics_time_series(file_metrics, file_hybrids=file_hybrids)