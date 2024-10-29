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

datos = pd.read_csv('datos.csv')

def utm_to_latlon(este, norte, huso):
    lat, lon = utm.to_latlon(este, norte, huso, northern=False)
    return lat, lon

# Aplicar la función a los datos
datos[['Latitud', 'Longitud']] = datos.apply(lambda row: utm_to_latlon(row['Este'], row['Norte'], row['Huso']), axis=1, result_type="expand")

datos.to_csv('datos_2.csv')