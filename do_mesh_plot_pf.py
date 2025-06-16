import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
import pyvista as pv
from scipy.spatial.distance import pdist, squareform
import os
from do_algorithms_plot import custom_colors,custom_colors_hybrids


path_exp_hybrids = "data_individualexp/hybrids/"
file_hybrids = "{algorithm}_{replica}_100-600_SV0-CV2-MV1_MM0.2-MC0.1-MB0.1.txt"
based_hybrid = ['NSGA2',"NSGA3","UNSGA3","SMSEMOA"]


def load_data_hybrids(path_exp,file_hybrids,replica=1):
    pairs = [f"tt{i}" for i in range(97)]
    columns = ["date", "time", "pf","generation"] 
    columns += [f"o{i+1}" for i,_ in enumerate(range(3))]
    columns += pairs    
 
    df_hybrids = pd.DataFrame()
    for algorithm in based_hybrid:
        df = pd.read_csv(path_exp + file_hybrids.format(algorithm=algorithm,replica=replica),sep=" ",header=None)
        df_hybrids = pd.concat([df_hybrids,df])
    # print(len(df.columns))
    # df.drop(columns=[len(df.columns)-1],inplace=True) #WITH NORMALIZED FILES 
    df_hybrids.columns = columns
    return df_hybrids   


def load_pareto_front_data(file_path="pareto_front_single_algorithms.txt"):
    """Load Pareto front data from the specified file"""
    df_pareto = pd.read_csv(file_path, sep='\t')
    return df_pareto


# Load all datasets
df_hybrids = load_data_hybrids(path_exp_hybrids, file_hybrids)
df_pareto = load_pareto_front_data()

# Filter for generation 600
df_hybrids = df_hybrids.loc[df_hybrids["generation"] == 600].copy()

# Get points for all datasets
points_hybrids = df_hybrids[['o1', 'o2', 'o3']].values
points_pareto = df_pareto[['o1', 'o2', 'o3']].values

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


surface_hybrids = create_surface(points_hybrids, custom_colors_hybrids[0],opacity=0.5)


plotter.add_mesh(surface_hybrids, color=custom_colors_hybrids[0], opacity=0.5, show_edges=True, 
                label=f"Hybrid")

# Add Pareto front points for each algorithm
for algorithm in df_pareto['algorithm'].unique():
    algorithm_points = df_pareto[df_pareto['algorithm'] == algorithm][['o1', 'o2', 'o3']].values
    
    # Create spheres for each point
    for i, point in enumerate(algorithm_points):
        sphere = pv.Sphere(radius=0.002, center=point)
        plotter.add_mesh(sphere)

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
