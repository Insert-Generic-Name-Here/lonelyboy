import pandas as pd
import numpy as np
from haversine import haversine
import networkx as nx
from tqdm import tqdm as tqdm
import geopandas as gpd
import psycopg2
import contextily as ctx
from shapely.geometry import Point, LineString, shape
import datetime
import time, datetime

import os, sys
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.distributed import io


def pairs_in_radius(df, diam=1000):
	'''
	Get all pairs with distance < diam
	'''
	res = []
	for ind_i, ind_j, val_i, val_j in nparray_combinations(df):
		dist = haversine(val_i, val_j)*1000
		if (dist<diam):
			res.append((ind_i,ind_j))
	return res


def connected_edges(data, circular=True):
	'''
	Get circular (all points inside circle of diameter=diam) or density based (each pair with distance<diam)
	'''
	G = nx.Graph()
	G.add_edges_from(data)
	if circular:
		return [sorted(list(cluster)) for cluster in nx.find_cliques(G)]
	else:
		return [sorted(list(cluster)) for cluster in nx.connected_components(G)]


def nparray_combinations(arr):
	'''
	Get all combinations of points
	'''
	for i in range(arr.shape[0]):
		for j in range(i+1, arr.shape[0]):
			yield i, j, arr[i,:], arr[j,:]


def translate(sets, sdf):
	'''
	Get mmsis from clustered indexes
	'''
	return [sorted(tuple([sdf.iloc[point].mmsi for point in points])) for points in sets]


def get_clusters(timeframe, diam, circular=True):
	pairs = pairs_in_radius(timeframe[['lon', 'lat']].values, diam)
	return connected_edges(pairs, circular=circular)


def find_existing_flocks(x, present, past, last_ts):
	'''
	Find all clusters (present) that existed in the past (cluster subset of flock)
	'''
	# find the indices of past Dataframe where current cluster is subset of flock
	indcs = [set(x.clusters) < set(val) for val in past.loc[past.et==last_ts].clusters.values]
	# get the indices of the past dataframe where that occurs
	past.loc[(indcs)].index.tolist()


def replace_with_existing_flocks(x, present, to_keep, past):
	'''
	Replace current cluster with his existing flock
	'''
	if to_keep.iloc[x.name]:
		if len(past.iloc[to_keep.iloc[x.name]])>1:
			raise Exception('len > 1, something is wrong')

		x.dur = past.iloc[to_keep.iloc[x.name]].dur.values[0] +1
		x.st = past.iloc[to_keep.iloc[x.name]].st.values[0]
	return x


def get_current_clusters(sdf, ts, diam=1000, circular=True):
	'''
	Get clusters and init them as a single flock
	'''
	present = pd.DataFrame([[tuple(val)] for (val) in translate(get_clusters(sdf, diam, circular=circular), sdf )], columns=['clusters'])
	present['st'] = present['et'] = ts
	present['dur'] = 1
	return present


def present_new_or_subset_of_past(present, past, last_ts):
	'''
	Find and treat current clusters that exist in the past as a subset of a flock (used when flocks break apart to many smaller ones).
	'''
	to_keep = present.apply(find_existing_flocks, args=(present,past,last_ts,) , axis=1)

	present = present.apply(replace_with_existing_flocks, args=(present,to_keep,past,), axis=1)

	new = present.merge(past,on='clusters', how='left',suffixes=['','tmp'], indicator=True)
	new = new[new['_merge']=='left_only'].drop(['_merge'],axis=1).dropna(axis=1)

	return new


def past_is_subset_or_set_of_present(present, past, ts, last_ts):
	'''
	Find and propagate a flock if it's subset of a current cluster.
	'''
	# get if tuple of tmp1 is subset or equal of a row of tmp2
	to_keep = past.apply(lambda x: (True in [set(x.clusters) <= set(val) for val in present.clusters.values]) and (x.et == last_ts), axis=1)
	past.loc[to_keep,'et'] = ts
	past.loc[to_keep,'dur']= past.loc[to_keep].dur.apply(lambda x : x+1)
	return past


def merge_pattern(new_clusters, clusters_to_keep):
	'''
	Concatenate all possible flocks to get the full flock dataframe
	'''
	return pd.concat([new_clusters,clusters_to_keep]).reset_index(drop=True)


def check_for_checkpoint(df_checksum, params):
	try:
		ckpnt = io.load_pickle('gp_checkpoint.pckl')
		if ckpnt['checksum'] == df_checksum and ckpnt['params'] == params:
			return (ckpnt['current_ts'], ckpnt['last_ts'], ckpnt['patterns'], ckpnt['ind'])
	except:
		return False

def group_patterns(df, mode, min_diameter=3704, min_cardinality=10, time_threshold=30, checkpoints=True, checkpoints_freq=0.001, save_result=True):
	# circular (flag for convoys/flocks)	-> mode	(string; flocks(f)|convoys(c)|spherical swarms(fs)|dense swarms(cs))
	save_name = f'{mode}_min_diameter{min_diameter}_time_threshold{time_threshold}_min_cardinality{min_cardinality}.csv'
	start = 0
	total = df.datetime.nunique()
	if checkpoints:
		print('[+] Looking for checkpoint...')
		checkpoint_interval = round(checkpoints_freq*df.datetime.nunique())
		print(checkpoint_interval)
		df_checksum = io.get_checksum_of_dataframe(df)
		params = {'mode': mode, 'min_diameter':min_diameter, 'min_cardinality':min_cardinality, 'time_threshold':time_threshold}

		ckpnt = check_for_checkpoint(df_checksum, params)
		if ckpnt:
			print('[+] Loading from checkpoint...')
			ts, last_ts, mined_patterns, start = ckpnt
			df = df.loc[df.datetime >= ts]
		else:
			print('[-] No checkoint available')

	print('CHECK ', df.datetime.min())
	for ind, (ts, sdf) in tqdm(enumerate(df.groupby('datetime'), start=start), total=total, initial=start):


		if checkpoints and start != ind and ind % checkpoint_interval == 0 :
			ckpnt = {'checksum': df_checksum, 'params': params, 'current_ts': ts, 'last_ts': last_ts, 'patterns': mined_patterns, 'ind': ind}
			io.save_pickle(ckpnt,'gp_checkpoint.pckl')
			print('[+] Saving checkpoint...')

		if mode == 'flocks' or mode == 'f':
			present = get_current_clusters(sdf, ts, min_diameter, circular=True)
		elif mode == 'convoys' or mode == 'c':
			present = get_current_clusters(sdf, ts, min_diameter, circular=False)
		elif mode == 'swarms' or mode == 's':
			raise NotImplementedError('Current mode is not Implemented atm.')

		# Init the first present as mined_patterns
		if ind == 0:
			mined_patterns	= present
			last_ts			= ts
			continue

		new_subsets 		= present_new_or_subset_of_past(present, mined_patterns, last_ts)
		old_subsets_or_sets = past_is_subset_or_set_of_present(present, mined_patterns, ts, last_ts)

		if len(new_subsets)==0:
			if len(old_subsets_or_sets)==0:
				print('Shieeeet')
				break
			else:
				mined_patterns = old_subsets_or_sets
		else:
			if len(old_subsets_or_sets)==0:
				mined_patterns = new_subsets
			else:
				mined_patterns = merge_pattern(new_subsets, old_subsets_or_sets)

		# Only keep the entries that are either:
		# 1. Currently active -> (mined_patterns.et==ts)
		# or,
		# 2. Been active for more that time_threshold time steps -> (mined_patterns.dur>time_threshold).
		# and
		# 3. Num of vessels in flock >= min_cardinality -> ([len(clst)>=min_cardinality for clst in mined_patterns.clusters])
		mined_patterns = mined_patterns.loc[((mined_patterns.et==ts) | (mined_patterns.dur>time_threshold)) & ([len(clst)>=min_cardinality for clst in mined_patterns.clusters])]
		last_ts = ts


	# keep this df and use it again as the db for the real time implementation
	# print('Calculating mean velocity per flock...')
	# mined_patterns['mean_vel'] = np.nan
	# mined_patterns['mean_vel'] = mined_patterns.apply(lambda x: df.loc[(df.mmsi.isin(eval(x.clusters))) & (df.ts >= x.st) & (df.ts <= x.et)].velocity.mean(), axis=1)
	if save_result:
		print('Saving Result...')
		mined_patterns.to_csv(save_name, index=False)
