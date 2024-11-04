import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, median_filter

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
optimal_distance = average_length / 100
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

# Define rapid exponential decay IDW function
def rapid_decay_exponential_idw(x_coords, y_coords, values, grid_x, grid_y, power=2, max_distance=0.0005, decay_rate=20.0):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            valid = dist < max_distance
            if valid.any():
                weights = np.exp(-decay_rate * dist[valid]) / (dist[valid]**power + 1e-10)
                grid_z[i, j] = np.sum(weights * values[valid]) / np.sum(weights)
            else:
                grid_z[i, j] = 0
    return grid_z

# Funciones de propagación de ruido avanzadas con mayor suavizado y menor alcance
def exponential_decay_gaussian(x_coords, y_coords, values, grid_x, grid_y, decay_rate=10.0, sigma=0.5):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.exp(-decay_rate * dist)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return gaussian_filter(grid_z, sigma=sigma)

def polynomial_decay(x_coords, y_coords, values, grid_x, grid_y, order=6):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = 1 / (dist**order + 1e-10)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return median_filter(grid_z, size=3)

def combined_gaussian_decay(x_coords, y_coords, values, grid_x, grid_y, short_decay=10.0, long_decay=1.5, sigma=0.5, threshold=0.0003):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.where(dist < threshold, np.exp(-short_decay * dist), np.exp(-long_decay * dist))
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return gaussian_filter(grid_z, sigma=sigma)

def radial_basis_smooth(x_coords, y_coords, values, grid_x, grid_y, epsilon=1.0):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = np.exp(-epsilon * dist**2)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return gaussian_filter(grid_z, sigma=1)

def spline_based_decay(x_coords, y_coords, values, grid_x, grid_y, power=4, smooth=0.1):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2) + smooth
            weights = 1 / (dist**power)
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return gaussian_filter(grid_z, sigma=2)

def logistic_decay(x_coords, y_coords, values, grid_x, grid_y, growth_rate=20, mid_point=0.0002):
    grid_z = np.zeros_like(grid_x, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dist = np.sqrt((x_coords - grid_x[i, j])**2 + (y_coords - grid_y[i, j])**2)
            weights = 1 / (1 + np.exp(growth_rate * (dist - mid_point)))
            grid_z[i, j] = np.sum(weights * values) / np.sum(weights)
    return gaussian_filter(grid_z, sigma=1)

# Modelos para comparar
modelos = [
    ("Rapid Exponential Decay", rapid_decay_exponential_idw(X[:, 0], X[:, 1], y, lon_grid, lat_grid, power=2, max_distance=0.0003, decay_rate=1.0)),
    ("Exponential Decay with Gaussian", exponential_decay_gaussian(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Polynomial Decay", polynomial_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Combined Gaussian Decay", combined_gaussian_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Radial Basis Smooth", radial_basis_smooth(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Spline Based Decay", spline_based_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid)),
    ("Logistic Decay", logistic_decay(X[:, 0], X[:, 1], y, lon_grid, lat_grid))
]

# Definir los límites y colores proporcionados
bounds_provided = [-0.01, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 999999]
colors_provided = [
    (0, 255/255, 255/255, 0),
    (121/255, 173/255, 18/255, 1),
    (0, 97/255, 76/255, 1),
    (243/255, 240/255, 68/255, 1),
    (165/255, 115/255, 0, 1),
    (254/255, 136/255, 0, 1),
    (223/255, 23/255, 23/255, 1),
    (156/255, 12/255, 12/255, 1),
    (145/255, 70/255, 153/255, 1),
    (70/255, 146/255, 221/255, 1),
    (1/255, 90/255, 156/255, 1)
]

# Crear el mapa de colores y la norma de límites
cmap = ListedColormap(colors_provided)
norm = BoundaryNorm(bounds_provided, cmap.N)

# Graficar cada modelo con la nueva escala de colores
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.ravel()

for i, (nombre, prediccion) in enumerate(modelos):
    ax = axs[i]
    contour = ax.contourf(lon_grid, lat_grid, prediccion, levels=bounds_provided, cmap=cmap, norm=norm)
    ax.set_title(nombre)
    ax.axis('off')

# Añadir barra de color para representar la escala
fig.colorbar(contour, ax=axs, orientation='horizontal', fraction=0.05, pad=0.1, ticks=bounds_provided)

plt.tight_layout()
plt.show()
