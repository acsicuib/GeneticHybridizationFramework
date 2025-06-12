import pandas as pd
import numpy as np
import ast
from utils import load_bash_config, load_data_normalized
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
import sys
import imageio.v3 as iio  # Add this import at the top

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

# def load_data(path_exp,file,replica,algorithm):
#     if not os.path.exists(path_exp + file.format(algorithm=algorithm,replica=replica)):
#         print(f"File does not exist: {path_exp + file.format(algorithm=algorithm,replica=replica)}")
#         sys.exit(-1)

#     columns = ["date", "time", "generation"] 
#     columns += [f"o{i+1}" for i,_ in enumerate(config['OBJECTIVES'])]
 
#     df = pd.read_csv(path_exp + file.format(algorithm=algorithm,replica=replica),sep=" ",header=None)
#     # df.drop(columns=[len(df.columns)-1],inplace=True) #WITH NORMALIZED FILES 
#     df.columns = columns
#     return df

# def load_data_hybrids(path_exp,file_hybrids,replica,algorithm):
#     if not os.path.exists(path_exp + file_hybrids.format(algorithm=algorithm,replica=replica)):
#         print(f"File does not exist: {path_exp + file_hybrids.format(algorithm=algorithm,replica=replica)}")
#         sys.exit(-1)


#     pairs = [f"tt{i}" for i in range(97)]
#     columns = ["date", "time", "pf","generation"] 
#     columns += [f"o{i+1}" for i,_ in enumerate(range(3))]
#     columns += pairs    
 
#     df = pd.read_csv(path_exp + file_hybrids.format(algorithm=algorithm,replica=replica),sep=" ",header=None)
#     print(len(df.columns))
#     # df.drop(columns=[len(df.columns)-1],inplace=True) #WITH NORMALIZED FILES 
#     df.columns = columns
#     return df

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

def create_animation_gif(file_out,frames_dir='plots', start_name='movie', duration=0.05):
    """
    Create an animated GIF from a sequence of PNG files.
    
    Args:
        frames_dir (str): Directory containing the frame images
        output_name (str): Name of the output GIF file
        duration (float): Duration of each frame in seconds
    """
    # Get all movie frame files and sort them numerically
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith(start_name) and f.endswith('.png')],
                        key=lambda x: int(x[len(start_name):-4]))  # Extract number from 'movieXXX.png'
    
    if not frame_files:
        print("No movie frames found!")
        return
    
    # Create full paths for all frames
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    
    # Read all frames with error handling
    frames = []
    expected_shape = None
    
    for i, path in enumerate(frame_paths):
        try:
            frame = iio.imread(path)
            
            # Check if this is the first valid frame
            if expected_shape is None:
                expected_shape = frame.shape
                frames.append(frame)
                print(f"Added frame {i+1}/{len(frame_paths)}")
            # Check if frame has the same shape as the first frame
            elif frame.shape == expected_shape:
                frames.append(frame)
                print(f"Added frame {i+1}/{len(frame_paths)}")
            else:
                print(f"Skipping frame {i+1}/{len(frame_paths)} - different dimensions: {frame.shape} vs {expected_shape}")
        except Exception as e:
            print(f"Error reading frame {i+1}/{len(frame_paths)}: {str(e)}")
            continue
    
    if not frames:
        print("No valid frames were found!")
        return
    
    print(f"Successfully processed {len(frames)} out of {len(frame_paths)} frames")
    
    try:
        # Save as GIF
        iio.imwrite(os.path.join(frames_dir, file_out), frames, duration=duration)
        print(f"Animation saved as {file_out}")
    except Exception as e:
        print(f"Error saving GIF: {str(e)}")

def cleanup_movie_frames(frames_dir='plots'):
    """
    Remove all movie frame PNG files after creating the GIF.
    
    Args:
        frames_dir (str): Directory containing the frame images
    """
    # Get all movie frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.startswith('movie') and f.endswith('.png')]
    
    # Remove each file
    for file in frame_files:
        os.remove(os.path.join(frames_dir, file))
    print(f"Cleaned up {len(frame_files)} movie frame files")

def plot_3d_scatter(algorithm,it,file_out,do_gif=False):
    # it = 1
    # algorithm = "NSGA2"
    df = load_data(path_exp,file,it,algorithm)
    print(df.head())

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['o1'], df['o2'], df['o3'], c=df['generation'], cmap='viridis')

    ax.set_xlabel('Distance')
    ax.set_ylabel('Occupation Variance')
    ax.set_zlabel('Power Consumption')

    if do_gif:
    # Generate all frames
        for ii in range(0,360,1):
            ax.view_init(elev=10., azim=ii)
            save_plot(fig, f"movie{ii}")
    else:
        save_plot(fig, '3d_scatter_plot')
        plt.show()
    plt.close(fig)  # Close the figure to free memory
  

def plot_PF_plots(replica=1,last_generation=600,with_hybrids=False,do_gif=False):
    d_tmp = pd.DataFrame()
    dall = pd.DataFrame()
    for algorithm in ALGORITHMS:
        df = load_data_normalized(path_exp,file,replica,algorithm)
        dt = df.loc[df["generation"]==last_generation].copy()
        dt["algorithm"] = algorithm
        dall = pd.concat([dall,dt])

    dall_hybrids = pd.DataFrame()

    if with_hybrids:
        for algorithm in based_hybrid:
            df = load_data_hybrids(path_exp_hybrids,file_hybrids,replica,algorithm)
            dt = df.loc[df["generation"]==last_generation].copy()
            dt["algorithm"] = algorithm
            dall_hybrids = pd.concat([dall_hybrids,dt])
    
    d_tmp = pd.concat([dall,dall_hybrids])
    cmap = ListedColormap(custom_colors)
    
    
    # Set consistent figure size and DPI
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    # Set consistent axis limits based on data
    x_min, x_max = d_tmp['o1'].min(), d_tmp['o1'].max()
    y_min, y_max = d_tmp['o2'].min(), d_tmp['o2'].max()
    z_min, z_max = d_tmp['o3'].min(), d_tmp['o3'].max()
    
    # Add small padding to limits
    padding = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)
    ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
    ax.set_zlim(z_min - z_range * padding, z_max + z_range * padding)
    
    # Create scatter plots for each algorithm separately to get proper legend entries
    for i, algorithm in enumerate(ALGORITHMS):
        mask = dall['algorithm'] == algorithm
        ax.scatter(dall.loc[mask, 'o1'], 
                  dall.loc[mask, 'o2'], 
                  dall.loc[mask, 'o3'],
                  c=[cmap(i)],  # Use single color for each algorithm
                  label=algorithm,
                  marker=markers[i],
                  alpha=0.7)  # Slight transparency for better visibility
    if with_hybrids:
        cmap_hybrids = ListedColormap(custom_colors_hybrids)
        
        ax.scatter(dall_hybrids.loc[:, 'o1'], 
                dall_hybrids.loc[:, 'o2'], 
                dall_hybrids.loc[:, 'o3'],
                c=[cmap_hybrids(0)],  # Use single color for each algorithm
                label="Hybrid",
                marker=markers_hybrids[0],
                s=50,
                alpha=0.7)  # Slight transparency for better visibility

    ax.set_xlabel('Distance', labelpad=10)
    ax.set_ylabel('Occupation Variance', labelpad=10)
    ax.set_zlabel('Power Consumption', labelpad=10)
    
    # Set consistent legend position and size
    legend = ax.legend(bbox_to_anchor=(1.15, 0.5),  # Position legend outside the plot
                      loc='center left',
                      markerscale=1.5,  # Make the legend markers slightly larger
                      fontsize=10)  # Consistent font size
    
    # Set consistent tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set consistent view parameters
    ax.view_init(elev=20., azim=45)  # Initial view
    
    # Adjust layout with consistent parameters
    plt.tight_layout(pad=2.0)  # Consistent padding
    
    if do_gif:
        # Generate all frames with consistent parameters
        for ii in range(0,360,1):
            ax.view_init(elev=20., azim=ii)  # Keep elevation constant
            # Ensure the plot maintains its size and format
            fig.set_size_inches(12, 8)
            plt.tight_layout(pad=2.0)
            save_plot(fig, f"pf_movie{ii}")
    else:
        save_plot(fig, 'PF_plot')
        plt.show()
    
    plt.close(fig)  # Close the figure to free memory



# def get_mask_pareto_front(df):
#     """
#     Get a boolean mask indicating which solutions in a specific generation are part of the Pareto front.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing the optimization results
#         generation (int): The generation number to get the Pareto front from
        
#     Returns:
#         pd.Series: Boolean mask where True indicates a solution is part of the Pareto front
#     """
#     # Filter for the specific generation
   
#     gen_data = df
    
#     # Get the objective columns
#     obj_cols = ['o1', 'o2', 'o3']
    
#     # Initialize boolean mask for Pareto front solutions
#     is_pareto = pd.Series(True, index=gen_data.index)
    
#     # For each solution in the generation
#     for idx, row in gen_data.iterrows():
#         # Compare with all other solutions
#         for other_idx, other_row in gen_data.iterrows():
#             if idx == other_idx:  # Skip self-comparison
#                 continue
                
#             # Check if other solution dominates current solution
#             # A solution dominates another if it's better or equal in all objectives
#             # and strictly better in at least one objective
#             dominates = True
#             strictly_better = False
            
#             for obj in obj_cols:
#                 if other_row[obj] > row[obj]:  # Assuming minimization for all objectives
#                     dominates = False
#                     break
#                 elif other_row[obj] < row[obj]:
#                     strictly_better = True
            
#             if dominates and strictly_better:
#                 is_pareto[idx] = False
#                 break
    
#     # Create a full-length boolean series with False for all non-generation rows
#     full_mask = pd.Series(False, index=df.index)
#     full_mask[gen_data.index] = is_pareto
    
#     return full_mask


# def prepare_pareto_front(path_exp, file, ALGORITHMS, replicas, output_folder):
#     """
#     Prepare Pareto front data for all algorithms and replicas.
    
#     Args:
#         path_exp (str): Path to input experiment data
#         file (str): File pattern for the data
#         ALGORITHMS (list): List of algorithm names
#         replicas (range): Range of replica numbers
#         output_folder (str): Path to save the processed files
#     """
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     for replica in range(1,2): #TODO: Change to replicas
#         for algorithm in ALGORITHMS:
#             print(f"Preparing Pareto front for algorithm {algorithm} and replica {replica}")
            
#             # Load the data
#             df = load_data(path_exp, file, replica, algorithm)
            
#             # Create a copy of the DataFrame to avoid SettingWithCopyWarning
#             df = df.copy()
            
#             # Initialize pareto_front column
#             df['pareto_front'] = False
            
#             # Process each generation
#             for generation in range(1, 600):
#                 print(f"Updating Pareto front for algorithm {algorithm} and replica {replica} generation {generation}")
#                 # Get mask for current generation
#                 gen_mask = df['generation'] == generation
#                 gen_data = df.loc[gen_mask].copy()
#                 pf_mask = get_mask_pareto_front(gen_data)
#                 df.loc[gen_mask, 'pareto_front'] = pf_mask.values
            
#             # Save the updated DataFrame
#             output_path = os.path.join(output_folder, file.format(algorithm=algorithm, replica=replica))
#             df.to_csv(output_path, index=False, sep=" ", header=False)
#             print(f"Completed {algorithm} replica {replica}")

def plot_metrics(path_exp, file_metrics, ALGORITHMS):
    """
    Plot metrics for different algorithms.
    
    Args:
        path_exp (str): Path to the experiment directory
        file_metrics (str): Name of the metrics file
        ALGORITHMS (list): List of algorithm names to plot
    """
    # Read the metrics file
    df = pd.read_csv(path_exp + file_metrics)
    
    # Create a figure with subplots for each metric
    metrics = ['GD', 'IGD', 'HV', 'S', 'STE']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
    # fig.suptitle('Metrics Comparison Across Algorithms', fontsize=16)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Create box plot for each algorithm
        data = [df[df['Algorithm'] == alg][metric] for alg in ALGORITHMS]
        ax.boxplot(data, labels=ALGORITHMS)
        
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start from 0 for GD, IGD, and S
        if metric in ['GD', 'IGD', 'S']:
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the suptitle
    save_plot(fig, 'metrics_boxplot')
    plt.show()

def plot_metrics_time_series(path_exp, file_metrics,file_metrics_hybrids, algorithms):
    """
    Plot time series of metrics (GD, IGD, HV, S, STE) across generations for all algorithms.
    Each metric has its own subplot, with different algorithms shown as lines.
    
    Args:
        path_exp (str): Path to the experiment directory
        file_metrics (str): Name of the metrics file
        algorithms (list): List of algorithm names to plot
    """
    # Read the metrics file
    df = pd.read_csv(path_exp + file_metrics)
    df_hybrids = pd.read_csv(path_exp + file_metrics_hybrids)

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
        for j, alg in enumerate(algorithms):
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
        
        grouped_hybrids = df_hybrids.groupby('Generation')[metric].agg(['mean', 'std'])
        print(grouped_hybrids)
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
        
        # Add metric name as text in the top-left corner
        ax.text(0.02, 0.95, metric, 
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold',
                verticalalignment='top')
    
    # Add a single legend at the top of the figure
    fig.legend(handles, labels,
              loc='upper center',
              bbox_to_anchor=(0.5, 0.98),
              ncol=len(algorithms),
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


path_exp = "data_individualexp/"
path_exp_hybrids = "data_individualexp/hybrids/"
file = "{algorithm}_{replica}_400-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
file_hybrids = "{algorithm}_{replica}_100-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
replicas = range(1,11)
ALGORITHMS = ['NSGA2','NSGA3','UNSGA3','SMSEMOA']
# Colores personalizados (pueden ser RGB o hex)
custom_colors = ['#FF595E', '#8AC926', '#1982C4', '#6A4C93']
# custom_colors = ['#ffadad', '#caffbf', '#a0c4ff', '#ffc6ff']
cmap = ListedColormap(custom_colors)
markers = ['o', 'o', 'o', 'o'] 
# based_hybrid = ['NSGA2','NSGA3','UNSGA3','SMSEMOA']
based_hybrid = ['NSGA2',"NSGA3","UNSGA3","SMSEMOA"]
# based_hybrid = ['Hybrids']
# custom_colors_hybrids = ['#d00000','#70e000','#0496ff','#7209b7']
# markers_hybrids = ['v','v','v','v']
markers_hybrids = ['v']
custom_colors_hybrids = ['#ff8500']


if __name__ == "__main__":
    # plot_3d_scatter("NSGA2",1,file_out="animation",do_gif=False,)
    # create_animation_gif("NSGA2_animation.gif",start_name="movie",duration=0.05)  # 50ms per frame

    # ONly one specific generation
    do_gif = False
    plot_PF_plots(replica=1,last_generation=600,
                  with_hybrids=True,
                  do_gif=do_gif)
    if do_gif:
        create_animation_gif("pf_animation_all.gif",start_name="pf_movie",duration=0.05)  # 50ms per frame
    
    # Clean up the individual movie frame files
    # cleanup_movie_frames()

    # # Perfecta 
    # file_metrics = "table_400:600.csv"
    # file_metrics_hybrids = "hybrids/table_merge.csv"
    # plot_metrics_time_series(path_exp, file_metrics,file_metrics_hybrids, ALGORITHMS)