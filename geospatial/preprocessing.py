import pandas as pd
from haversine import haversine
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


def calculate_velocity(gdf):
				'''
				Return given dataframe with an extra velocity column that is calculated using the distance covered in a given amount of time
				'''
				gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
				gdf['next_loc'] = gdf.geom.shift(-1)
				gdf = gdf[:-1]
				gdf['next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
				gdf['velocity'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).multiply(3600/gdf.ts.diff(-1).abs())
				gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
				return gdf
				


def calculate_distance_traveled(gdf):
				'''
				Return given dataframe with an extra velocity column that is calculated using the distance covered in a given amount of time
				'''
				gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
				gdf['next_loc'] = gdf.geom.shift(-1)
				gdf = gdf[:-1]
				gdf['next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
				gdf['distance'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).cumsum()
				gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
				return gdf
