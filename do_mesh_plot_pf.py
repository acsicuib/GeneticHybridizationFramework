import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
import pyvista as pv
import os

path_reference_points = "results_hybrid_400_500/reference_points.txt"

# Load all datasets
df_pareto = pd.read_csv(path_reference_points,sep="\t")
columns = [f"o{i+1}" for i,_ in enumerate(range(3))] + ["algorithm"] + ["replica"]
df_pareto.columns = columns

custom_colors = ['#ff8500', '#FF595E', '#1982C4', '#6A4C93', "#8AC926"]
algorithms = df_pareto["algorithm"].unique()


# === Crear escena en PyVista ===
plotter = pv.Plotter()

# Add Pareto front points for each algorithm
for ix,algorithm in enumerate(algorithms):
    if algorithm.startswith("h_"):
        algorithm_label = "Hybrids"
        ix_color = len(custom_colors)-1
    else:
        algorithm_label = algorithm
        ix_color = ix
    algorithm_points = df_pareto[df_pareto['algorithm'] == algorithm][['o1', 'o2', 'o3']].values
    
    # Create spheres for each point
    for i, point in enumerate(algorithm_points):
        sphere = pv.Sphere(radius=0.002, center=point)
        plotter.add_mesh(sphere, label=algorithm_label, color = custom_colors[ix_color])

# Add legend with supported parameters
plotter.add_legend(
    bcolor='white',  # Background color
    face='r',        # Position (r = right)
    size=[0.2, 0.2]  # Size of the legend
)

# Ajustar escena
plotter.add_axes()
plotter.show_grid()
plotter.show_bounds(xtitle="Distance", ytitle="Occupation Var.", ztitle="Power")

# Export to HTML with interactive features
plotter.export_html(
    'visualization.html',
 
)

# Show the plot
plotter.show()
