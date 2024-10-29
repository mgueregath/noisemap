import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import utm
from shapely.geometry import Polygon

from pyogrio import set_gdal_config_options
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
from shapely import vectorized

# Función para generar una rejilla de puntos dentro de un polígono
def generar_puntos_rejilla(line, distance):
    num_points = int(line.length // distance)
    points = [line.interpolate(i * distance) for i in range(num_points + 1)]
    return points



set_gdal_config_options({
    'SHAPE_RESTORE_SHX': 'YES',
})


# 1. Cargar los tres archivos SHP
poligonos1 = gpd.read_file('./Colectoras-line.shp')
poligonos2 = gpd.read_file('./Servicios-line.shp')
poligonos3 = gpd.read_file('./Locales-line.shp')


crs_original = "EPSG:4326"  # Reemplaza con el EPSG correcto de tus datos originales
if poligonos1.crs is None:
    poligonos1.crs = crs_original
if poligonos2.crs is None:
    poligonos2.crs = crs_original
if poligonos3.crs is None:
    poligonos3.crs = crs_original

# 2. Reproyectar al mismo CRS si es necesario
crs_objetivo = {'init': 'epsg:4326'}

poligonos1 = poligonos1.to_crs(crs_objetivo)
poligonos1['nivel_ruido'] = 50
poligonos2 = poligonos2.to_crs(crs_objetivo)
poligonos2['nivel_ruido'] = 62
poligonos3 = poligonos3.to_crs(crs_objetivo)
poligonos3['nivel_ruido'] = 75

geometries = []
nivel_ruido = []

# 3. Combinar los GeoDataFrames
poligonos = pd.concat([poligonos1, poligonos2, poligonos3], ignore_index=True)


# Calcular una distancia más corta para aumentar la cantidad de puntos
average_length = poligonos.length.mean()
optimal_distance = average_length / 20 

for idx, row in poligonos.iterrows():
    puntos_rejilla = generar_puntos_rejilla(row.geometry, optimal_distance)  # Ajusta el spacing según lo necesario
    if not puntos_rejilla:  # Verificar si no se generan puntos
        print(f"No se generaron puntos para el polígono {idx}")
    for punto in puntos_rejilla:
        geometries.append(punto)
        nivel_ruido.append(row['nivel_ruido'])

if not geometries:
    raise ValueError("No se generaron puntos en ninguno de los polígonos. Ajusta el valor de 'spacing' o revisa los datos.")


# Crear el GeoDataFrame usando las listas separadas
puntos = gpd.GeoDataFrame({'nivel_ruido': nivel_ruido}, geometry=geometries, crs=poligonos.crs)

#datos = pd.read_csv('datos2024.csv')

def utm_to_latlon(este, norte, huso):
    lat, lon = utm.to_latlon(este, norte, huso, northern=False)
    return lat, lon

# Aplicar la función a los datos
#datos[['Latitud', 'Longitud']] = datos.apply(lambda row: utm_to_latlon(row['Este'], row['Norte'], row['Huso']), axis=1, result_type="expand")

# Verificar los resultados


# Extraer coordenadas y valores de Leq
# Extraer coordenadas y valores de Leq
X = pd.DataFrame({
    'Latitude': puntos.geometry.x,
    'Longitude': puntos.geometry.y
}).values
y = pd.DataFrame({
    'Leq': nivel_ruido
}).values

# Crear un kernel para Gaussian Process (Kriging)
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# Ajustar el modelo con los datos
gp.fit(X, y)

# Generar una cuadrícula de latitudes y longitudes
lat_max = -39.8
lat_min = -39.865
lon_max = -73.3
lon_min = -73.18
lat_grid, lon_grid = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))

# Aplanar la cuadrícula para hacer predicciones
grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

# Predecir los valores en la cuadrícula
y_pred, sigma = gp.predict(grid_points, return_std=True)

# Reshape los resultados para que coincidan con la cuadrícula
y_pred = y_pred.reshape(lat_grid.shape)

# Definir los límites para cada rango de valores
bounds = [-0.01, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 999999]
colors_mpl = [
    (0, 255/255, 255/255, 0),      # Igual a 0
    (121/255, 173/255, 18/255, 1), # >0 y <=35
    (0, 97/255, 76/255, 1),        # >40 y <=45
    (243/255, 240/255, 68/255, 1), # >45 y <=50
    (165/255, 115/255, 0, 1),      # >50 y <=55
    (254/255, 136/255, 0, 1),      # >55 y <=60
    (223/255, 23/255, 23/255, 1),  # >60 y <=65
    (156/255, 12/255, 12/255, 1),  # >65 y <=70
    (145/255, 70/255, 153/255, 1), # >70 y <=75
    (70/255, 146/255, 221/255, 1), # >75 y <=80
    (1/255, 90/255, 156/255, 1)    # >80
]

# Crear un colormap y un normalizador
cmap = ListedColormap(colors_mpl)
norm = BoundaryNorm(boundaries=bounds, ncolors=len(colors_mpl))

# Generar los contornos de la predicción
contours = plt.contourf(lon_grid, lat_grid, y_pred, levels=bounds, cmap=cmap, norm=norm)
plt.colorbar(label='Predicción Leq')
plt.title('Capa Interpolada de Leq')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()

polygons = []
value_ranges = []

min_distance_threshold = 0.01

# Iterar a través de los niveles de contorno y generar polígonos
for i in range(len(bounds) - 1):
    # Usar contour para obtener las coordenadas de los contornos
    contour = plt.contour(lon_grid, lat_grid, y_pred, levels=[bounds[i], bounds[i+1]], linewidths=0, colors='none')
    
    for collection in contour.collections:
        for path in collection.get_paths():
            # Crear un polígono a partir de las coordenadas del contorno
            coords = path.vertices
            if len(coords) > 0:  # Verificar que haya coordenadas
                coords = np.array(coords)  # Convertir a array para operaciones posteriores
                polygon_parts = []

                # Agrupar puntos en polígonos basados en la distancia
                current_polygon = []
                for j in range(len(coords)):
                    if j == 0 or np.linalg.norm(coords[j] - coords[j - 1]) < min_distance_threshold:
                        current_polygon.append(coords[j])
                    else:
                        if len(current_polygon) > 2:  # Debe haber al menos 3 puntos para formar un polígono
                            polygon_parts.append(Polygon(current_polygon))
                        current_polygon = [coords[j]]

                # Agregar el último grupo si hay suficientes puntos
                if len(current_polygon) > 2:
                    polygon_parts.append(Polygon(current_polygon))

                # Agregar los polígonos generados
                for polygon in polygon_parts:
                    polygons.append(polygon)
                    value_ranges.append((bounds[i], bounds[i + 1]))

# Crear un GeoDataFrame a partir de los polígonos
geo_df = gpd.GeoDataFrame({'geometry': polygons, 'value': [f"{low}" for low, high in value_ranges]})

# Especificar el sistema de referencia de coordenadas (CRS) - por ejemplo, WGS84
geo_df.set_crs(epsg=4326, inplace=True)

# Guardar el GeoDataFrame como un archivo GeoJSON
geo_df.to_file('valdivia2024.geojson', driver='GeoJSON')