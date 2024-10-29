import pandas as pd
import geopandas as gpd
from pyogrio import set_gdal_config_options
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point
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
poligonos1['nivel_ruido'] = 62
poligonos3 = poligonos3.to_crs(crs_objetivo)
poligonos1['nivel_ruido'] = 75

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

# 5. Extraer coordenadas y valores de ruido
puntos['X'] = puntos.geometry.x
puntos['Y'] = puntos.geometry.y
X = puntos['X'].values
Y = puntos['Y'].values
Z = puntos['nivel_ruido'].values  # Asegúrate del nombre correcto de la columna

# 6. Crear malla de puntos para la interpolación
xmin, ymin, xmax, ymax = puntos.total_bounds
grid_resolution = 100
grid_x, grid_y = np.mgrid[xmin:xmax:grid_resolution*1j, ymin:ymax:grid_resolution*1j]

# 7. Realizar la interpolación
grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')

# 8. Visualizar el mapa de ruido
plt.figure(figsize=(10, 8))
plt.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='jet')
plt.colorbar(label='Nivel de Ruido (dB)')
plt.scatter(X, Y, c='black', s=10, label='Centroides')
plt.legend()
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Mapa de Ruido Interpolado')
plt.show()

# 9. Guardar el resultado como un raster
pixel_size_x = (xmax - xmin) / grid_resolution
pixel_size_y = (ymax - ymin) / grid_resolution
transform = from_origin(xmin, ymax, pixel_size_x, pixel_size_y)

with rasterio.open(
    'mapa_ruido.tif',
    'w',
    driver='GTiff',
    height=grid_z.shape[1],
    width=grid_z.shape[0],
    count=1,
    dtype=grid_z.dtype,
    crs=poligonos.crs,
    transform=transform,
) as new_dataset:
    new_dataset.write(grid_z.T, 1)
