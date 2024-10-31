import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, mapping, LineString
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
import json
from skimage import measure

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

# Generate points along lines for interpolation
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

# Create GeoDataFrame for points
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
y_pred_rapid_decay = rapid_decay_exponential_idw(X[:, 0], X[:, 1], y, lon_grid, lat_grid, power=2, max_distance=0.00035, decay_rate=100.0)

# Apply Gaussian smoothing
y_pred_smoothed_rapid_decay = gaussian_filter(y_pred_rapid_decay, sigma=1)

# Define color map and bounds
bounds_provided = [-0.01, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 999999]

# Initialize GeoJSON dictionaries for contours and streets
geojson_contours = {
    "type": "FeatureCollection",
    "features": []
}

geojson_streets = {
    "type": "FeatureCollection",
    "features": []
}

# Generate and add contours for interpolated noise levels
for i in range(len(bounds_provided) - 2, -1, -1):  # Process from high to low bounds
    level = (bounds_provided[i] + bounds_provided[i + 1]) / 2  # Midpoint for contour level
    contours = measure.find_contours(y_pred_smoothed_rapid_decay, level=level)

    for contour in contours:
        lat_idx, lon_idx = contour[:, 0], contour[:, 1]
        lat_coords = lat_grid[0, 0] + (lat_grid[-1, 0] - lat_grid[0, 0]) * (lat_idx / lat_grid.shape[0])
        lon_coords = lon_grid[0, 0] + (lon_grid[0, -1] - lon_grid[0, 0]) * (lon_idx / lon_grid.shape[1])
        
        # Form a polygon from the contour coordinates
        polygon = Polygon(zip(lon_coords, lat_coords))
        if polygon.is_valid:
            feature = {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "value": f"{bounds_provided[i]}"  # Assign noise level range
                }
            }
            geojson_contours["features"].append(feature)

# Convert street LineStrings to buffered Polygons with "tipo": "calle" property
buffer_distance = 0.00005  # Buffer distance for streets
for idx, row in poligonos.iterrows():
    if isinstance(row.geometry, LineString):
        buffered_polygon = row.geometry.buffer(buffer_distance)  # Convert to Polygon
        feature = {
            "type": "Feature",
            "geometry": mapping(buffered_polygon),
            "properties": {
                "value": row['nivel_ruido'],
                "tipo": "calle"  # Special property for street polygons
            }
        }
        geojson_streets["features"].append(feature)

# Combine sorted contours and street polygons into a single GeoJSON structure
combined_geojson = {
    "type": "FeatureCollection",
    "features": geojson_contours["features"] + geojson_streets["features"]
}

# Save the final combined GeoJSON file
with open('valdivia2024_combined_contours_streets.geojson', 'w') as geojson_file:
    json.dump(combined_geojson, geojson_file)
