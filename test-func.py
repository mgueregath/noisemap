import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Cargar el shapefile
shapefile_path = './Colectoras-line.shp'
gdf = gpd.read_file(shapefile_path)

# Calcular una distancia más corta para aumentar la cantidad de puntos
average_length = gdf.length.mean()
optimal_distance = average_length / 20  # Reducir el divisor aumenta la cantidad de puntos

# Función para generar puntos a intervalos regulares sobre una LineString
def generate_points_on_line(line, distance):
    num_points = int(line.length // distance)
    points = [line.interpolate(i * distance) for i in range(num_points + 1)]
    return points

# Generar puntos para cada LineString en el GeoDataFrame
points_data = []
for _, row in gdf.iterrows():
    line = row.geometry
    points = generate_points_on_line(line, optimal_distance)
    points_data.extend(points)

# Crear un nuevo GeoDataFrame con los puntos generados
points_gdf = gpd.GeoDataFrame(geometry=points_data, crs=gdf.crs)

# Visualizar las líneas originales y los puntos generados
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='blue', linewidth=1, label='Líneas originales')
points_gdf.plot(ax=ax, color='red', markersize=5, label='Puntos generados')

# Configuración de la visualización
plt.legend()
plt.title('Líneas y Puntos Generados con Mayor Densidad')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.show()
