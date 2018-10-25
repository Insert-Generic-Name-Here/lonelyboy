import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from random import choice
import contextily as ctx
from shapely.geometry import Point, LineString, shape
import numpy as np


def distance_to_nearest_port(point, ports):
				return ports.geom.distance(point).min()


def get_outliers(series, alpha = 3):
				q25, q75 = series.quantile((0.25, 0.75))
				iqr = q75 - q25
				q_high = q75 + alpha*iqr
				q_low = q25 - alpha*iqr
				return series.loc[(series >q_high) | (series<q_low)].index


def resample_geospatial(sample_ves, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False):
				'''
				Resample and interpolate linearly a sample vessel.

				'''
				sample_ves['datetime'] = pd.to_datetime(sample_ves['ts'],unit='s')
				upsampled = sample_ves.resample(rule,on='datetime', loffset=True, kind='timestamp').first()
				interpolated = upsampled.interpolate(method=method)
				interpolated['geom'] = interpolated[['lon', 'lat']].apply(lambda x: Point(x[0], x[1]), axis=1)
				interpolated['datetime'] = interpolated.index
				interpolated.reset_index(drop=True, inplace=True)
				if drop_lon_lat:
								interpolated.drop(['lat', 'lon'], axis=1)
				return gpd.GeoDataFrame(interpolated, crs = crs, geometry='geom')
