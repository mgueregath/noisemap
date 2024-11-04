import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter

# Load shapefiles
poligonos1 = gpd.read_file('Colectoras-line.shp')
poligonos2 = gpd.read_file('Servicios-line.shp')
poligonos3 = gpd.read_file('Locales-line.shp')

# Assign noise levels
poligonos1['nivel_ruido'] = 68.8
poligonos2['nivel_ruido'] = 66.7
poligonos3['nivel_ruido'] = 63.9

# Combine geometries
poligonos = pd.concat([poligonos1, poligonos2, poligonos3], ignore_index=True)

# Generate points along lines
average_length = poligonos.length.mean()
optimal_distance = average_length / 100
geometries = []
nivel_ruido = []

for idx, row in poligonos.iterrows():
    line = row.geometry
    num_points = int(line.length // optimal_distance)
    points = [line.interpolate(i * optimal_distance) for i in range(num_points + 1)]
    for punto in points:
        geometries.append(Point(punto.x, punto.y))
        nivel_ruido.append(row['nivel_ruido'])

# Create GeoDataFrame
puntos_df = gpd.GeoDataFrame({'nivel_ruido': nivel_ruido}, geometry=geometries, crs=poligonos.crs)

# Extract coordinates for interpolation
X = pd.DataFrame({'Longitude': puntos_df.geometry.x, 'Latitude': puntos_df.geometry.y}).values
y = puntos_df['nivel_ruido'].values

# Define grid for prediction
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

# Perform the rapid exponential decay IDW interpolation
y_pred_rapid_decay = rapid_decay_exponential_idw(X[:, 0], X[:, 1], y, lon_grid, lat_grid, power=2, max_distance=0.0003, decay_rate=1.0)

# Apply Gaussian smoothing
y_pred_smoothed_rapid_decay = gaussian_filter(y_pred_rapid_decay, sigma=1)

# Define color map and bounds
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
cmap_provided = ListedColormap(colors_provided)
norm_provided = BoundaryNorm(bounds_provided, ncolors=len(colors_provided))

# Calculate dynamic road_noise_colors based on bounds and colors
road_noise_colors = {}
for noise_level in [68.8, 66.7, 63.9]:
    for i, bound in enumerate(bounds_provided[:-1]):
        if bound <= noise_level < bounds_provided[i + 1]:
            road_noise_colors[noise_level] = colors_provided[i]
            break

# Plot the noise map
fig, ax = plt.subplots(figsize=(15, 15))
noise_map = ax.contourf(lon_grid, lat_grid, y_pred_smoothed_rapid_decay, levels=bounds_provided, cmap=cmap_provided, norm=norm_provided)

# Add colorbar and labels
#plt.colorbar(noise_map, ax=ax, label='Nivel de Ruido (Leq)')

ax.axis('off')  # Remove axes

# Plot streets with reduced line thickness and rapid decay model
for noise_level, color in road_noise_colors.items():
    subset = poligonos[poligonos['nivel_ruido'] == noise_level]
    
    # Minimal blending and reduced thickness for street representation
    for i in range(2, 0, -1):
        alpha = 0.2 * i
        linewidth = 0.8 * i
        subset.plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha, zorder=2)
    
    # Main line on top with reduced thickness
    subset.plot(ax=ax, color=color, linewidth=0.8, zorder=3)

plt.show()

output_path = './mapa_ruido_clean.png'  # Define the output file path
fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

# Close the figure to free up memory
plt.close(fig)