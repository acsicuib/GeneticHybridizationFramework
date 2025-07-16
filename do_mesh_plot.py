import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
import pyvista as pv
from scipy.spatial.distance import pdist, squareform
import os
from do_algorithms_plot import custom_colors,custom_colors_hybrids
import matplotlib.pyplot as plt
from utils import load_data_normalized, load_data_merged_hybrids

path_exp = "results_hybrid_400_500/"
file = "{algorithm}_{replica}_400-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.normalized.txt"
ALGORITHMS = ['NSGA2',"NSGA3","UNSGA3","SMSEMOA"]
replica = 1

path_exp_hybrids = "results_hybrid_400_500/hybrids/"
file_hybrids =path_exp_hybrids+ "{algorithm}_{replica}_100-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
based_hybrid = ['NSGA2',"NSGA3","UNSGA3","SMSEMOA"]


# Load both datasets
df_original = load_data_normalized(path_exp, file, replica)
df_hybrids = load_data_merged_hybrids(file_hybrids)

# Filter for generation 600
df_original = df_original.loc[df_original["generation"] == 600].copy()
df_hybrids = df_hybrids.loc[df_hybrids["generation"] == 600].copy()

# Get points for both datasets

points_hybrids = df_hybrids[['o1', 'o2', 'o3']].values

# === Crear escena en PyVista ===
plotter = pv.Plotter()

def create_surface(points, color, opacity=0.5):
    # Calcular matriz de distancias
    distances = squareform(pdist(points))
    n_points = len(points)
    
    # Crear conexiones basadas en proximidad
    k = 5  # número de vecinos más cercanos a considerar
    connections = []
    for i in range(n_points):
        nearest_indices = np.argsort(distances[i])[1:k+1]
        for j in nearest_indices:
            if i < j:  # Evitar duplicados
                connections.append([i, j])
    
    # Crear malla de PyVista con las conexiones
    point_cloud = pv.PolyData(points)
    lines = np.array([[2, i, j] for i, j in connections])
    point_cloud.lines = lines
    
    # Crear superficie a partir de las conexiones
    surface = point_cloud.delaunay_2d()
    return surface

# Create and add surfaces
surface_original = []
for i,algorithm in enumerate(ALGORITHMS):
    points_original = df_original[df_original['algorithm'] == algorithm][['o1', 'o2', 'o3']].values
    surface_original.append(create_surface(points_original, custom_colors[i]))

surface_hybrids = create_surface(points_hybrids, custom_colors_hybrids[0],opacity=0.5)

# Add surfaces to plotter with more descriptive labels
for i,algorithm in enumerate(ALGORITHMS):
    plotter.add_mesh(surface_original[i], color=custom_colors[i], opacity=0.5, show_edges=True, 
                    label=f"{algorithm}")

plotter.add_mesh(surface_hybrids, color=custom_colors_hybrids[0], opacity=0.5, show_edges=True, 
                label=f"Hybrid")

# Add points with descriptive labels (only for first point of each type)
# for i, point in enumerate(points_original):
#     sphere = pv.Sphere(radius=0.0001, center=point)
#     label = f"{algorithm} Points" if i == 0 else None
#     plotter.add_mesh(sphere, color="blue", label=label)

# for i, point in enumerate(points_hybrids):
#     sphere = pv.Sphere(radius=0.0001, center=point)
#     label = f"{algorithm_hybrids} Hybrid Points" if i == 0 else None
#     plotter.add_mesh(sphere, color="purple", label=label)

# Add legend with supported parameters
plotter.add_legend(
    bcolor='white',  # Background color
    face='r',        # Position (r = right)
    size=[0.2, 0.2]  # Size of the legend
)

# Ajustar escena
plotter.add_axes()
plotter.show_grid()
plotter.show_bounds(xlabel="Distance", ylabel="Occupation Var.", zlabel="Power")

# Export to HTML with interactive features
plotter.export_html(
    'visualization.html',
 
)

# Show the plot
plotter.show()
