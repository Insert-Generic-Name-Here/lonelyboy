import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix


def index_of_cluster(item, cluster_list):
	position = [ind for ind, subl in enumerate(cluster_list) if item in subl]
	if len(position)>1:
		raise ValueError
	if not position:
		return [-1]
	else:
		return position


def connected_edges(data):
	G = nx.Graph()
	G.add_edges_from(data)
	return [list(cluster) for cluster in nx.connected_components(G)]


def pairs_in_radius(df, radius):
	df.reset_index(inplace=True)
	distances = np.triu(distance_matrix(df[['lat', 'lon']].values, df[['lat', 'lon']].values),0)
	distances[distances == 0] = np.inf
	return np.vstack(np.where(distances<=radius)).T


def get_flock_labels(timeframe,radius):
	timeframe.reset_index(drop=True , inplace=True)
	data = pairs_in_radius(timeframe, radius)
	clusters = connected_edges(data)
	timeframe['flock_label'] = timeframe.apply( lambda x: index_of_cluster(x.name, clusters)[0], axis=1)
	return timeframe


def flocks(df,radius):
	df['flock_label'] = np.nan
	df =  df.groupby('ts', as_index=False).apply(get_flock_labels, radius)
	return df