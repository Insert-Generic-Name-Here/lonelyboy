import pandas as pd
from haversine import haversine
import geopandas as gpd
import matplotlib.pyplot as plt
from random import choice
import contextily as ctx
from shapely.geometry import Point, LineString, shape
import numpy as np


def distance_to_nearest_port(point, ports):
				'''
				Calculates the minimum distance between the point and the lists of ports. Can be used to determine if the ship is sailing or not
				'''
				return ports.geom.distance(point).min()


def get_outliers(series, alpha = 3):
				'''
				Returns a series of indexes of row that are to be concidered outliers, using the quantilies of the data. 
				'''
				q25, q75 = series.quantile((0.25, 0.75))
				iqr = q75 - q25
				q_high = q75 + alpha*iqr
				q_low = q25 - alpha*iqr
				# return the indexes of rows that are over/under the threshold above
				return series.loc[(series >q_high) | (series<q_low)].index


def resample_geospatial(sample_ves, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False):
				'''
				Resample and interpolate linearly a sample vessel.

				'''
				#convert unix to datetime
				sample_ves['datetime'] = pd.to_datetime(sample_ves['ts'],unit='s')
				#resample and interpolate using the method given. Linear is suggested
				upsampled = sample_ves.resample(rule,on='datetime', loffset=True, kind='timestamp').first()
				interpolated = upsampled.interpolate(method=method)
				#interpolate the geom column with the correct point objects using lat and lon
				interpolated['geom'] = interpolated[['lon', 'lat']].apply(lambda x: Point(x[0], x[1]), axis=1)
				# reset the index to normal and use the old index as new timestamp
				interpolated['datetime'] = interpolated.index
				interpolated.reset_index(drop=True, inplace=True)
				#drop lat and lon if u like
				if drop_lon_lat:
								interpolated.drop(['lat', 'lon'], axis=1)
				return gpd.GeoDataFrame(interpolated, crs = crs, geometry='geom')


def calculate_velocity(gdf, smoothing=False, window=15, center=False):
				'''
				Return given dataframe with an extra velocity column that is calculated using the distance covered in a given amount of time
				TODO - use the get distance method to save some space
				'''
				#create columns for current and next location. Drop the last columns that contains the nan value
				gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
				gdf['next_loc'] = gdf.geom.shift(-1)
				gdf = gdf[:-1]
				gdf['next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
				# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots) 
				gdf['velocity'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).multiply(3600/gdf.ts.diff(-1).abs())
				if smoothing:
								gdf['velocity'] = gdf['velocity'].rolling(window, center=center).mean()
				gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
				return gdf
				


def calculate_distance_traveled(gdf):
				'''
				Returns given dataframe with an added column ('distance') that contains the accumulated distance travel up to a specific point
				'''
				gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
				gdf['next_loc'] = gdf.geom.shift(-1)
				gdf = gdf[:-1]
				gdf['next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
				gdf['distance'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).cumsum()
				gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
				return gdf


def pick_random_group(gdf, column):
				return gdf.loc[gdf[column] == choice(gdf[column].unique())]					

















