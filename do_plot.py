import pandas as pd
import numpy as np
import ast
from utils import load_bash_config
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

config = load_bash_config('script_constants.sh') #Dont detect all the variables




# # TEST SOME CONFIGS
# print(config['HYBRID_ALGORITHMS'])
# print(config['HYBRID_GEN_STEPS']) 
# print(config['HYBRID_N_GEN']) 
# print(config['HYBRID_POP_SIZE'])
# # print(config['MYHYBRID_SOL_DUMPING'])
# print(config['OBJECTIVES'])
# print(file.format(algorithm=config['HYBRID_ALGORITHMS'][0]))

#check if results file exists

def load_data(path_exp,file):
    if os.path.exists(path_exp + file.format(algorithm=config['HYBRID_ALGORITHMS'][0])):
        print("File exists")
    else:
        print("File does not exist")

# col_dumps = np.array(ast.literal_eval(config['HYBRID_GEN_STEPS'])).ravel()
    col_dumps = config['HYBRID_GEN_STEPS']
    n_alg = len(config['HYBRID_ALGORITHMS'])
    n_dumps = np.repeat(col_dumps,len(config['HYBRID_ALGORITHMS']))
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

    # print(columns)

    df = pd.DataFrame()
    for algorithm in config['HYBRID_ALGORITHMS']:
        dft = pd.read_csv(path_exp + file.format(algorithm=algorithm),sep=" ",header=None)
        dft.drop(columns=[len(dft.columns)-1],inplace=True)
        dft.insert(loc=0, column='algorithm', value=algorithm)
        df = pd.concat([df,dft])

    df.columns = columns
    return df

def save_plot(fig, filename, dpi=300, bbox_inches='tight'):
    """Save the plot to a file with high resolution"""
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    # Save the plot
    fig.savefig(f'plots/{filename}.png', dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)  # Close the figure to free memory

def plot_acc_area(df):
    filter = ["A0_25","A1_25","A2_25","A3_25"]
    rename_filter = dict(zip(filter,config['HYBRID_ALGORITHMS']))
    filter.append("Generation")

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
    
    save_plot(fig, 'accuracy_area_plot')
    plt.show()


def plot_stocked_bar_area(df):
    # Define the columns and generation ranges we want to analyze
    filter_cols = ["A0_25", "A1_25", "A2_25", "A3_25"]
    rename_filter = dict(zip(filter_cols, config['HYBRID_ALGORITHMS']))
    
    # Define generation ranges for averaging
    gen_ranges = [(1, 25),(26,26), (25, 50), (50, 75), (75, 100), (100, 100)]
    gen_labels = ['1-25', '26-26','25-50', '50-75', '75-100', '100']
    
    # Create figure with extra space at the top for legend
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(2, 2, top=0.85)  # Leave space at top for legend
    axs = gs.subplots()

    # Create a list to store handles and labels for the legend
    handles, labels = None, None

    for ixa, algorithm in enumerate(config['HYBRID_ALGORITHMS']):
        axt = axs[ixa//2, ixa%2]
        axt.set_title(algorithm)
        
        # Filter data for current algorithm
        dt = df[df['algorithm'] == algorithm].loc[:, filter_cols + ['Generation']]
        dt.rename(columns=rename_filter, inplace=True)
        
        # Calculate averages for each generation range
        range_data = []
        for start_gen, end_gen in gen_ranges:
            mask = (dt['Generation'] >= start_gen) & (dt['Generation'] <= end_gen)
            range_avg = dt[mask].mean(numeric_only=True)
            range_avg = range_avg[:-1] #Remove Generation column
            range_data.append(range_avg)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame(range_data, index=gen_labels)
        
        # Create stacked bar plot and store handles and labels
        bars = plot_df.plot(kind='bar', stacked=True, ax=axt, legend=False)
        if handles is None:
            handles, labels = axt.get_legend_handles_labels()
        
        axt.set_xlabel('Generation Range')
        axt.set_ylabel('Average Value')
        plt.setp(axt.get_xticklabels(), rotation=45, ha='right')
    
    # Add a single legend at the top center
    fig.legend(handles, labels, title='Algorithms',
              loc='upper center', bbox_to_anchor=(0.5, 1.0),
              ncol=len(handles), frameon=True)
    
    save_plot(fig, 'stacked_bar_area_plot')
    plt.show()

def plot_density_overlay(df):
    filter_cols = ["A0_25", "A1_25", "A2_25", "A3_25"]
    rename_filter = dict(zip(filter_cols, config['HYBRID_ALGORITHMS']))

    # Define generation ranges for averaging
    gen_ranges = [(1, 25),(26,26), (25, 50), (50, 75), (75, 100), (100, 100)]
    gen_labels = ['1-25', '26','25-50', '50-75', '75-100', '100']
    
    # Calculate global min and max for x-axis limits
    x_min = df[filter_cols].min().min()
    x_max = df[filter_cols].max().max()
    
    # Create main figure with adjusted size for more compact layout
    fig = plt.figure(figsize=(20, 6))  # Further reduced height for more compact view
    
    # Create main grid for algorithms (2x2) with increased spacing
    main_gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.4, height_ratios=[1, 1])  # Increased hspace and wspace
    
    # Create a list of colors for different algorithms
    colors = plt.cm.viridis(np.linspace(0, 1, len(config['HYBRID_ALGORITHMS'])))
    
    # Create a single colorbar axis
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    
    # For each algorithm (main grid cell)
    for ixa, algorithm in enumerate(config['HYBRID_ALGORITHMS']):
        # Create subgrid for generation ranges (1x6) within this algorithm's cell
        sub_gs = main_gs[ixa//2, ixa%2].subgridspec(1, len(gen_ranges), wspace=0.1)  # Increased wspace between heatmaps
        sub_axs = sub_gs.subplots()
        
        # Set main title for this algorithm's cell
        sub_axs[0].set_title(f'Algorithm: {algorithm}', pad=10, fontsize=20)  # Increased pad
        x_pos = 0.18 if ixa % 2 == 0 else 0.62  # o ajusta manualmente según layout
        if ixa == 3:
            x_pos = 0.63
        y_pos = 0.94 if ixa < 2 else 0.5      # depende si es fila superior o inferior

        fig.text(x_pos, y_pos, f'Algorithm: {algorithm}', ha='center', va='center', fontsize=12)

        # For each generation range (subplot)
        for idx, ((start_gen, end_gen), gen_label) in enumerate(zip(gen_ranges, gen_labels)):
            ax = sub_axs[idx]
            
            # Filter data for current generation range and algorithm
            mask = (df['Generation'] >= start_gen) & (df['Generation'] <= end_gen) & (df['algorithm'] == algorithm)
            range_data = df[mask][filter_cols].copy()
            
            if not range_data.empty:
                # Rename columns to algorithm names
                range_data.columns = [rename_filter[col] for col in range_data.columns]
                
                # Sort data giving priority to the algorithmn column
                range_data.sort_values(by=algorithm, ascending=True, inplace=True)
               
                
                # Create heatmap
                ax.set_title(f'Algorithm {algorithm}', pad=5, fontsize=8)  # Increased pad
                sns.heatmap(range_data, 
                           cmap="viridis", 
                           cbar=(ixa == 0 and idx == 0),  # Only show colorbar in first subplot
                           cbar_ax=cbar_ax if (ixa == 0 and idx == 0) else None,
                           annot=False,
                           ax=ax,
                           xticklabels=(ixa//2 == 1),  # Only show x-ticklabels in bottom row
                           yticklabels=False)  # Hide row labels
                
                # Only rotate and set x-ticklabels for bottom row
                if ixa//2 == 1:  # Bottom row
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                else:  # Top row
                    ax.set_xticklabels([])  # Remove x-ticklabels
                    ax.set_xticks([])  # Remove x-ticks
            
            # Set subplot title and labels
            ax.set_title(f'Gen {gen_label}', pad=5, fontsize=8)  # Increased pad
            ax.set_xlabel('')
            ax.set_ylabel('')
    
    # Add a title to the colorbar
    cbar_ax.set_title('Value %', pad=10, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    
    # Save the plot
    save_plot(fig, 'density_overlay_plot')
    plt.show()

if __name__ == "__main__":
    ### MANUAL SPC OF FILE CONFIGURATION
    path_exp = "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3/"
    file = "{algorithm}_1_100-100_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"

    path_exp = "data/solutions/ntw_722_050-050-025_C/obj_distance-occ_variance-pw_consumption/Replicas050/Genetics/hybrid_NSGA2-NSGA3-UNSGA3-SMSEMOA/"
    file = "{algorithm}_1_100-100_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"

    df = load_data(path_exp,file)
    print(df.head())


    plot_acc_area(df) 

    plot_stocked_bar_area(df)

    plot_density_overlay(df)

