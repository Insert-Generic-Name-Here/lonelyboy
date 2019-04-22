import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from random import choice
import contextily as ctx
from shapely.geometry import Point, LineString, shape
import numpy as np
from lonelyboy.geospatial.preprocessing import get_outliers


def map_plot(df1, df2 = None, title=None, fontsize=25, color=['r', 'g'], figsize = (15,15), attribution="", **kwargs):
	'''
	Plot one or two dataframes on top of eachother.

	TODO - Add support for N Dataframes and more parameters, other that figsize.
	'''
	df1.crs = {'init': 'epsg:4326'}
	ax = df1.to_crs(epsg=3857).plot(figsize=figsize,color=color[0], **kwargs)
	if title is not None:
		ax.set_title(title, fontsize=fontsize)
	if df2 is not None:
		df2.crs = {'init': 'epsg:4326'}
		df2.to_crs(epsg=3857).plot(figsize=figsize,color=color[1], ax=ax, **kwargs)
	ctx.add_basemap(ax, attribution=attribution)


def plot_segments(gdf, feature='velocity', alpha=1.5, color='r'):
	plt.axvline(x=min(get_outliers(gdf[feature], alpha=alpha)), c=color)
	plt.axvline(x=max(get_outliers(gdf[feature], alpha=alpha)), c=color)


def plot_clusters(gdf, clusters):
	label_color = pd.DataFrame([], index=gdf.index, columns=['color'])

	cluster_indices = [gdf.loc[gdf.mmsi.isin(cluster)].index for cluster in clusters]
	for color_idx, cluster in enumerate(cluster_indices):
		label_color.loc[label_color.index.isin(cluster), 'color'] = color_idx

	ax = gdf.to_crs(epsg=3857).plot(figsize=(10, 10), c=label_color.color.values)
	ctx.add_basemap(ax, zoom=11)
	plt.show()
