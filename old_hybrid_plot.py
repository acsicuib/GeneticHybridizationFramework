import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import Counter

from resopt.plot.plot_utils import GenerationPicker, get_recommended_ticks, \
        getXYZLabel, linear_growth, asymptotic_growth
from resopt.plot.color_mixing import mix_colors
from resopt.files.file_utils import parse_file, get_solution_array

from resopt.analysis.txt_to_csv import txt_to_csv, stackplot, normalplot

def plot(configs):
    # TODO: MAKE IT WORK FOR VARIABLE NUMBER OF OBJECTIVES AND ALGORITHMS
    alg_list = configs.hybrid_legend
    df = txt_to_csv(configs.input)[0]
    print(df.shape)
    print(df.columns)
    # DATAFRAME PLOT
    fig = plt.figure(configs.title)
    fig.suptitle(configs.title)

    ax_list = []
    for j in range(2):
        ax_list.append(fig.add_subplot(2, 1, j+1))
        ax_list[j].grid(which='both', axis='x', linestyle='--')
        ax_list[j].tick_params(axis='x', rotation=45)

    normalplot(df, ax_list[0], alg_name="", from_last=False)
    normalplot(df, ax_list[1], alg_name="", from_last=True)
    #ax_list[0].set_xticks(gen_steps)
    #ax_list[1].set_xticks(gen_steps)

    plt.tight_layout()
    plt.show()

def plot_stack(configs):
    # TODO: MAKE IT WORK FOR VARIABLE NUMBER OF OBJECTIVES AND ALGORITHMS
    alg_list = configs.hybrid_legend
    df = txt_to_csv(configs.input)[0]

    # DATAFRAME PLOT
    fig = plt.figure(configs.title)
    fig.suptitle(configs.title)

    ax_list = []
    for j in range(2):
        ax_list.append(fig.add_subplot(2, 1, j+1))
        ax_list[j].grid(which='both', axis='x', linestyle='--')
        ax_list[j].tick_params(axis='x', rotation=45)

    stackplot(df, ax_list[0], alg_name="", from_last=False)
    stackplot(df, ax_list[1], alg_name="", from_last=True)
    #ax_list[0].set_xticks(gen_steps)
    #ax_list[1].set_xticks(gen_steps)

    plt.tight_layout()
    plt.show()

def plot_variance(configs):
    # TODO: MAKE IT WORK FOR VARIABLE NUMBER OF OBJECTIVES AND ALGORITHMS
    alg_list = configs.hybrid_legend
    df_list = txt_to_csv(configs.input)

    # DATAFRAME PLOT
    fig = plt.figure(configs.title)
    fig.suptitle(configs.title)

    ax_list = []
    for j in range(2):
        ax_list.append(fig.add_subplot(2, 1, j+1))

    varplot(df_list, ax_list[0], alg_name="", from_last=False)
    varplot(df_list, ax_list[1], alg_name="", from_last=True)

    plt.tight_layout()
    plt.show()

def plot_convergence_hybrid(configs):
    n_obj = configs.n_objectives
    n_algs = len(configs.hybrid_legend)

    parsed = parse_file(
            configs.input[0],
            n_obj=n_obj,
            n_algs=n_algs,
            timestamps=True,
            population=True,
            return_dict=True
        )

    generation = parsed.get('generation', [])
    o = parsed.get('objective', [])
    is_sol = parsed.get('is_solution', [])
    gl_orig = parsed.get('genetic_load_origin', [[]] * len(o))
    gl_last = parsed.get('genetic_load_last', [[]] * len(o))

    is_sol = np.array(is_sol, dtype=np.bool_)
    o = np.array(o)[:, is_sol]
    generation = np.array(generation)[is_sol]
    gl_orig = np.array(gl_orig)[:, is_sol]
    gl_last = np.array(gl_last)[:, is_sol]

    if configs.max_gen == 0 or generation[-1] < configs.max_gen:
        max_gen = generation[-1]
    else:
        max_gen = configs.max_gen

    # Get max idx considering max_gen option
    if max_gen < generation[-1]:
        max_idx = generation.index(max_gen + 1)
    else:
        max_idx = len(generation)

    generation = generation[:max_idx]
    o1 = o[0][:max_idx]
    o2 = o[1][:max_idx]
    if configs.n_objectives > 2:
        o3 = o[2][:max_idx]

    color_array = np.array([
            mcolors.to_rgb(c)
            for c in mcolors.TABLEAU_COLORS.values()
        ][:n_algs])
    color_select = np.array([True, False, False, True])
    color = []
    for idx in range(generation.size):
        color.append(mix_colors(
                gl_last[:,idx][color_select][:, np.newaxis] * \
                color_array[color_select]
            ))

    sizes = np.array([30 for _ in generation])

    fig = plt.figure(configs.title)
    if configs.n_objectives == 2:
        ax = fig.add_subplot()
    elif configs.n_objectives > 2:
        ax = fig.add_subplot(projection='3d')

    if configs.n_objectives == 2:
        points = ax.scatter(
                o1, o2,
                s=30, c=color, vmin=1, vmax=max_gen)
    else:
        points = ax.scatter(
                o1, o2, o3, 
                s=30, depthshade=False, c=color, vmin=1, vmax=max_gen)

    # X axis ticks
    o1_min, o1_max = min(o1), max(o1)
    o1_ticks = get_recommended_ticks(o1_min, o1_max)
    ax.set_xticks(o1_ticks)
    half_x = (o1_ticks[1] - o1_ticks[0]) / 2
    ax.set_xlim(o1_ticks[0] - half_x, o1_ticks[-1] + half_x)

    # Y axis ticks
    o2_min, o2_max = min(o2), max(o2)
    o2_ticks = get_recommended_ticks(o2_min, o2_max)
    ax.set_yticks(o2_ticks)
    half_y = (o2_ticks[1] - o2_ticks[0]) / 2
    ax.set_ylim(o2_ticks[0] - half_y, o2_ticks[-1] + half_y)

    if configs.n_objectives > 2:
        # Z axis ticks
        o3_min, o3_max = min(o3), max(o3)
        o3_ticks = get_recommended_ticks(o3_min, o3_max)
        ax.set_zticks(o3_ticks)
        half_z = (o3_ticks[1] - o3_ticks[0]) / 2
        ax.set_zlim(o3_ticks[0] - half_z, o3_ticks[-1] + half_z)

    # Title and labels
    if configs.x_label is None:
        ax.set_xlabel(getXYZLabel(0, configs.objectives))
    else:
        ax.set_xlabel(configs.x_label)

    if configs.y_label is None:
        ax.set_ylabel(getXYZLabel(1, configs.objectives))
    else:
        ax.set_ylabel(configs.y_label)

    if configs.n_objectives == 3:
        if configs.z_label is None:
            ax.set_zlabel(getXYZLabel(2, configs.objectives))
        else:
            ax.set_zlabel(configs.z_label)

    # Fill missing names in legend so we plot all solutions even if there's any
    # without a label
    legend = configs.hybrid_legend if configs.hybrid_legend else []
    patches = [mpatches.Patch(color=c, label=l) for c,l in zip(color_array, legend)]
    leg = ax.legend(handles=patches)

    plt.title(configs.title)

    img = plt.imshow(np.array([[generation[0],generation[-1]]]), cmap='Greys')
    img.set_visible(False)
    cbar = fig.colorbar(img)

    cbar.ax.hover_event = True
    cbar.ax.set_picker(1)
    gen_picker = GenerationPicker(generation.tolist(), points) 

    if configs.output:
        plt.savefig(configs.output)
    else:
        plt.tight_layout()
        plt.show()

def plot_scatter_gen_load(configs):
    n_obj = configs.n_objectives
    n_algs = len(configs.hybrid_legend)

    parsed = parse_file(
            configs.input[0],
            n_obj=n_obj,
            n_algs=n_algs,
            timestamps=True,
            population=True,
            return_dict=True
        )

    generation = parsed.get('generation', [])
    #is_sol = parsed.get('is_solution', [])
    gl_orig = parsed.get('genetic_load_origin', [[]] * n_algs)
    gl_last = parsed.get('genetic_load_last', [[]] * n_algs)

    #is_sol = np.array(is_sol, dtype=np.bool_)
    generation = np.array(generation)#[is_sol]
    gl_orig = np.array(gl_orig)#[:, is_sol]
    gl_last = np.array(gl_last)#[:, is_sol]

    if configs.max_gen == 0 or generation[-1] < configs.max_gen:
        max_gen = generation[-1]
    else:
        max_gen = configs.max_gen

    # Get max idx considering max_gen option
    if max_gen < generation[-1]:
        max_idx = generation.index(max_gen + 1)
    else:
        max_idx = len(generation)

    generation = generation[:max_idx]

    color_array = np.array([
            mcolors.to_rgb(c)
            for c in mcolors.TABLEAU_COLORS.values()
        ][:n_algs])
    color_select = np.array([True, False, False, True])
    color = []
    for idx in range(generation.size):
        color.append(mix_colors(
                gl_last[:,idx][color_select][:, np.newaxis] * \
                color_array[color_select]
            ))


    fig = plt.figure(configs.title)
    max_count = generation[generation == 1].size # only for population
    for i in range(n_algs):
        curr_alg = configs.hybrid_legend[i]
        ax = fig.add_subplot(2,2,i+1)

        # Duplicates increase point size
        sizes = [
                linear_growth(c, max_count, vmin=5., vmax=50.)
                for g in range(1, generation[-1] + 1)
                for c in Counter(gl_last[i][generation == g]).values()
                for _ in range(c)
            ]

        # Plot
        points = ax.scatter(
                generation, gl_last[i],
                s=sizes, c=color, vmin=1, vmax=max_gen, marker='d')

        # Title and labels
        ax.title.set_text("{}'s genetic load".format(curr_alg))
        ax.set_xlabel('Generation')
        ax.set_ylabel('Genetic load')

    # Legend
    #legend = configs.hybrid_legend if configs.hybrid_legend else []
    #patches = [mpatches.Patch(color=c, label=l) for c,l in zip(color_array, legend)]
    #leg = ax.legend(handles=patches)

    plt.suptitle(configs.title)

    if configs.output:
        plt.savefig(configs.output)
    else:
        plt.tight_layout()
        plt.show()
    pass

if __name__ == "__main__":
    from resopt.param.parameters import configs

    if configs.stack:
        plot(configs) ## funciona
        # plot_stack(configs)
    else:
        plot_scatter_gen_load(configs)
    plot_convergence_hybrid(configs)


