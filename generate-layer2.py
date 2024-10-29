import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from geokrige.methods import SimpleKriging
import matplotlib.pyplot as plt
import utm

# Cargar los datos
datos = pd.read_csv('datos.csv')

# Convertir las coordenadas UTM a latitud y longitud
def utm_to_latlon(este, norte, huso):
    lat, lon = utm.to_latlon(este, norte, huso, northern=False)
    return lat, lon

# Aplicar la conversión
datos[['Latitud', 'Longitud']] = datos.apply(lambda row: utm_to_latlon(row['Este'], row['Norte'], row['Huso']), axis=1, result_type="expand")

data = {
    "geometry": [Point(lon, lat) for lon, lat in zip(datos['Longitud'], datos['Latitud'])],
    "Leq": datos['Leq']
}
df = gpd.GeoDataFrame(data)
print(df.geometry.head())

kgn = SimpleKriging()
kgn.load(df, y='Leq')

kgn.variogram()
kgn.fit(model='exp')
