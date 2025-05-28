import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from resopt.plot.plot_utils import GenerationPicker, LegendPicker, \
        get_recommended_ticks, getXYZLabel
from resopt.files.file_utils import parse_file, get_solution_array

def plot_scatter_legend(configs):
    solutions = []

    for f in configs.input:
        o = get_solution_array(f, n_obj=configs.n_objectives)[-1]
        solutions.append(o)

    fig = plt.figure()
    if configs.n_objectives == 2:
        ax = fig.add_subplot()
    elif configs.n_objectives == 3:
        ax = fig.add_subplot(projection='3d')

    o1_all = sum([o[:,0].tolist() for o in solutions if np.any(o)], [])
    o2_all = sum([o[:,1].tolist() for o in solutions if np.any(o)], [])
    if configs.n_objectives == 3:
        o3_all = sum([o[:,2].tolist() for o in solutions], [])

    if configs.ref_points is not None:
        ref_points = np.array(configs.ref_points)
        o1_all += ref_points[:,0].tolist()
        o2_all += ref_points[:,1].tolist()
        if configs.n_objectives == 3:
            o3_all += ref_points[:,2].tolist()

    # X axis ticks
    o1_min, o1_max = min(o1_all), max(o1_all)
    o1_ticks = get_recommended_ticks(o1_min, o1_max)
    ax.set_xticks(o1_ticks)
    if o1_ticks.size == 1:
        half_x = 0.5
    else:
        half_x = (o1_ticks[1] - o1_ticks[0]) / 2
    ax.set_xlim(o1_ticks[0] - half_x, o1_ticks[-1] + half_x)

    # Y axis ticks
    o2_min, o2_max = min(o2_all), max(o2_all)
    o2_ticks = get_recommended_ticks(o2_min, o2_max)
    ax.set_yticks(o2_ticks)
    if o2_ticks.size == 1:
        half_y = 0.5
    else:
        half_y = (o2_ticks[1] - o2_ticks[0]) / 2
    ax.set_ylim(o2_ticks[0] - half_y, o2_ticks[-1] + half_y)

    if configs.n_objectives == 3:
        # Z axis ticks
        o3_min, o3_max = min(o3_all), max(o3_all)
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

    plt.title(configs.title)

    # Fill missing names in legend so we plot all solutions even if there's any without a label
    legend = configs.legend if configs.legend else []
    names = legend + [''] * (len(solutions) - len(legend))

    dots_sol = dict()
    if configs.ref_points is not None:
        color = mcolors.TABLEAU_COLORS['tab:cyan']
        if configs.n_objectives == 2:
            dots_sol[configs.ref_points_legend] = ax.scatter(
                    ref_points[:,0],
                    ref_points[:,1],
                    s=50,
                    facecolors="None",
                    edgecolors=color,
                    label=configs.ref_points_legend,
                    marker='D'
                )
        elif configs.n_objectives == 3:
            dots_sol[configs.ref_points_legend] = ax.scatter(
                    ref_points[:,0],
                    ref_points[:,1],
                    ref_points[:,2],
                    s=50,
                    facecolors="None",
                    edgecolors=color,
                    label=configs.ref_points_legend,
                    marker='D'
                )

    for sol, color, name in zip(solutions, mcolors.TABLEAU_COLORS, names):
        if not np.any(sol): continue
        if configs.n_objectives == 2:
            dots_sol[name] = ax.scatter(
                    sol[:,0], 
                    sol[:,1], 
                    s=30, 
                    facecolors=color,
                    edgecolors=color,
                    label=name
                )
        elif configs.n_objectives == 3:
            dots_sol[name] = ax.scatter(
                    sol[:,0], 
                    sol[:,1],
                    sol[:,2],
                    s=30,
                    facecolors=color,
                    edgecolors=color,
                    label=name
                )

    leg = ax.legend()
    for legtext in leg.get_texts():
        legtext.set_picker(5)

    legend_picker = LegendPicker(dots_sol)

    if configs.output:
        plt.savefig(configs.output)
    else:
        plt.show()

                
def plot_convergence(configs):
    generation, o = parse_file(configs.input[0], n_obj=configs.n_objectives)

    if configs.max_gen == 0 or generation[-1] < configs.max_gen:
        max_gen = generation[-1]
    else:
        max_gen = configs.max_gen

    if configs.trim_gen:
        # Exclude generations after convergence
        idx = generation.index(generation[-1])
        step = len(generation) - idx

        # Get convergence moving idx backwards from the end
        o_cmp = [set(o[i][idx:]) for i in range(configs.n_objectives)]
        o_set = [set(o[i][idx-step:idx]) for i in range(configs.n_objectives)]
        convergence = np.any([o_set[i] == o_cmp[i] for i in range(configs.n_objectives)])
        while convergence:
            idx -= step
            o_set = [set(o[i][idx-step:idx]) for i in range(configs.n_objectives)]
            convergence = np.any([o_set[i] == o_cmp[i] for i in range(configs.n_objectives)])

        # Get max idx considering max_gen option
        if max_gen < generation[idx]:
            max_idx = generation.index(max_gen + 1)
        else:
            max_idx = idx + step

        generation = generation[:max_idx]
        o1 = o[0][:max_idx]
        o2 = o[1][:max_idx]
        if configs.n_objectives > 2:
            o3 = o[2][:max_idx]
    else:
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

    cmap = mpl.colormaps['viridis']
    color = [item for item in generation]
    sizes = np.array([30 for _ in generation])

    fig = plt.figure()
    if configs.n_objectives == 2:
        ax = fig.add_subplot()
    elif configs.n_objectives > 2:
        ax = fig.add_subplot(projection='3d')

    if configs.n_objectives == 2:
        points = ax.scatter(
                o1, o2,
                s=30, c=color, cmap=cmap, vmin=1, vmax=max_gen)
    else:
        points = ax.scatter(
                o1, o2, o3, 
                s=30, depthshade=False, c=color, cmap=cmap, vmin=1, vmax=max_gen)

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

    plt.title(configs.title)
    cbar = fig.colorbar(points)

    cbar.ax.hover_event = True
    cbar.ax.set_picker(1)
    gen_picker = GenerationPicker(generation, points) 

    if configs.output:
        plt.savefig(configs.output)
    else:
        plt.show()

if __name__ == '__main__':
    pass


