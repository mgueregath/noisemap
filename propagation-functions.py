import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter

# Cargar archivos shapefile
poligonos1 = gpd.read_file('Colectoras-line.shp')
poligonos2 = gpd.read_file('Servicios-line.shp')
poligonos3 = gpd.read_file('Locales-line.shp')

# Asignar niveles de ruido
poligonos1['nivel_ruido'] = 68.8
poligonos2['nivel_ruido'] = 66.7
poligonos3['nivel_ruido'] = 63.9

# Combinar geometrías
poligonos = pd.concat([poligonos1, poligonos2, poligonos3], ignore_index=True)

# Generar puntos a lo largo de las líneas
average_length = poligonos.length.mean()
optimal_distance = average_length / 50
geometries, nivel_ruido = [], []

for idx, row in poligonos.iterrows():
    line = row.geometry
    num_points = int(line.length // optimal_distance)
    points = [line.interpolate(i * optimal_distance) for i in range(num_points + 1)]
    geometries.extend(points)
    nivel_ruido.extend([row['nivel_ruido']] * len(points))

# Crear GeoDataFrame
puntos_df = gpd.GeoDataFrame({'nivel_ruido': nivel_ruido}, geometry=geometries, crs=poligonos.crs)

# Extraer coordenadas para interpolación
X = np.array([(point.x, point.y) for point in puntos_df.geometry])
y = puntos_df['nivel_ruido'].values

# Crear grilla para predicción
lat_min, lat_max = X[:, 1].min(), X[:, 1].max()
lon_min, lon_max = X[:, 0].min(), X[:, 0].max()
lat_grid, lon_grid = np.meshgrid(np.linspace(lat_min, lat_max, 250), np.linspace(lon_min, lon_max, 250))

# Funciones de propagación de ruido
def rapid_exponential_decay(x_coords, y_coords, values, grid_x, grid_y, power=2, decay_rate=1.0, threshold=0.0005):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.exp(-decay_rate * dist) / (dist**power + 1e-10)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights) if (dist < threshold).any() else 0
    return grid_z

def exponential_decay(x_coords, y_coords, values, grid_x, grid_y, power=2, decay_rate=0.1, threshold=0.002):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.exp(-decay_rate * dist) / (dist**power + 1e-10)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights) if (dist < threshold).any() else 0
    return grid_z

def gaussian_propagation(x_coords, y_coords, values, grid_x, grid_y, sigma=3):
    grid_z = exponential_decay(x_coords, y_coords, values, grid_x, grid_y, decay_rate=0.1)
    return gaussian_filter(grid_z, sigma=sigma)

def inverse_distance_weighting(x_coords, y_coords, values, grid_x, grid_y, power=1.2):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = 1 / (dist**power + 1e-10)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

def quadratic_decay(x_coords, y_coords, values, grid_x, grid_y, decay_factor=0.5):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = 1 / ((dist + decay_factor)**2)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

def combined_decay(x_coords, y_coords, values, grid_x, grid_y, short_decay=0.5, long_decay=0.05, threshold=0.0015):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.where(dist < threshold, np.exp(-short_decay * dist), np.exp(-long_decay * dist))
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

def inverse_distance_smooth(x_coords, y_coords, values, grid_x, grid_y, power=1, smooth=0.1):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2) + smooth
            weights = 1 / (dist**power)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

def linear_decay(x_coords, y_coords, values, grid_x, grid_y):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.maximum(0, 1 - dist)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return grid_z

# Modelos para comparar
modelos = [
    ("Rapid Exponential Decay", rapid_exponential_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Exponential Decay", exponential_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Gaussian Propagation", gaussian_propagation(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Inverse Distance Weighting", inverse_distance_weighting(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Quadratic Decay", quadratic_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Combined Decay", combined_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Inverse Distance Smooth", inverse_distance_smooth(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Linear Decay", linear_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("IDW with Custom Smooth", inverse_distance_weighting(X[:, 0], X[:, 1], y, lon_grid, lat_grid, power=3))
]

# Definir los límites y colores proporcionados dinámicamente
all_values = np.concatenate([model[1].flatten() for model in modelos])
vmin, vmax = all_values.min(), all_values.max()
bounds = np.linspace(vmin, vmax, 12)
cmap = ListedColormap([
    (0, 255/255, 255/255, 0), (121/255, 173/255, 18/255, 1), (0, 97/255, 76/255, 1),
    (243/255, 240/255, 68/255, 1), (165/255, 115/255, 0, 1), (254/255, 136/255, 0, 1),
    (223/255, 23/255, 23/255, 1), (156/255, 12/255, 12/255, 1), (145/255, 70/255, 153/255, 1),
    (70/255, 146/255, 221/255, 1), (1/255, 90/255, 156/255, 1)
])
norm = BoundaryNorm(bounds, cmap.N)

# Graficar cada modelo con la nueva escala de colores
fig, axs = plt.subplots(3, 3, figsize=(20, 18))
axs = axs.ravel()

for i, (nombre, prediccion) in enumerate(modelos):
    ax = axs[i]
    contour = ax.contourf(lon_grid, lat_grid, prediccion, levels=bounds, cmap=cmap, norm=norm)
    ax.set_title(nombre)
    ax.axis('off')

# Añadir barra de color para representar la escala
fig.colorbar(contour, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1, ticks=bounds)

plt.tight_layout()
plt.show()
