import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from random import choice
import contextily as ctx
from shapely.geometry import Point, LineString, shape
import numpy as np
from lonelyboy.geospatial.preprocessing import get_outliers


def map_plot(df1, df2 = None, color=['r', 'g'], figsize = (15,15)):
				'''
				Plot one or two dataframes on top of eachother.

				TODO - Add support for N Dataframes and more parameters, other that figsize.
				'''
				df1.crs = {'init': 'epsg:4326'}
				ax = df1.to_crs(epsg=3857).plot(figsize=figsize,c=color[0])
				if df2 is not None:
								df2.crs = {'init': 'epsg:4326'}
								df2.to_crs(epsg=3857).plot(figsize=figsize,c=color[1], ax=ax)
				ctx.add_basemap(ax)


def plot_segments(gdf, feature='velocity', alpha=1.5, color='r'):
				plt.axvline(x=min(get_outliers(gdf[feature], alpha=alpha)), c=color)
				plt.axvline(x=max(get_outliers(gdf[feature], alpha=alpha)), c=color)
