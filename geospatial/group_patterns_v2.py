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
from operator import xor


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
	indcs = past.apply(lambda val: (val.et == last_ts) and set(x.clusters) <= set(val.clusters), axis=1)
		#indcs = [set(x.clusters) < set(val.clusters.values) and (val.et==last_ts) for val in past]
	# get the indices of the past dataframe where that occurs
	if indcs.values.any():
		#print(type(past[indcs].to_frame.st))
		x.st = past[indcs].st.min()

	return x

def find_shrunked(x, present, past, current_ts):
	'''
	Find all clusters (present) that existed in the past (cluster subset of flock)
	'''
	# find the indices of past Dataframe where current cluster is subset of flock
	indcs = present.apply(lambda val: (val.st == current_ts) and set(x.clusters) < set(val.clusters), axis=1)
	#indcs = [set(x.clusters) < set(val.clusters.values) and (val.et==last_ts) for val in past]
	# get the indices of the past dataframe where that occurs
	if indcs.values.any():
		#print(type(past[indcs].to_frame.st))
		x.et = present[indcs].et.max()
	
	return x

def get_current_clusters(sdf, ts, diam=1000, circular=True):
	'''
	Get clusters and init them as a single flock
	'''
	present = pd.DataFrame([[tuple(val)] for (val) in translate(get_clusters(sdf, diam, circular=circular), sdf )], columns=['clusters'])
	present['st'] = present['et'] = ts
	return present


def present_new_or_subset_of_past(present, past, last_ts):
	'''
	Find and treat current clusters that exist in the past as a subset of a flock (used when flocks break apart to many smaller ones).
	'''
	#to_keep = present.apply(find_existing_flocks, args=(present,past,last_ts,) , axis=1)

	#if len(to_keep) != 0:
	#	to_keep.to_csv('wtf.csv', header=None)
	#	print(len(to_keep))
	#	print('this should not be empty:', to_keep)
	present = present.apply(find_existing_flocks, args=(present,past,last_ts,), axis=1)

	return present

def past_is_subset_or_set_of_present(present, past, ts):

	past = past.apply(find_shrunked, args=(present,past,ts,), axis=1)

	return past[~past.clusters.isin(present.clusters)]


def merge_pattern(new_clusters, clusters_to_keep):
	'''
	Concatenate all possible flocks to get the full flock dataframe
	'''
	return pd.concat([new_clusters,clusters_to_keep]).reset_index(drop=True)


def _merge_partitions(dfA, dfB, time_threshold):
	present = dfB.copy()
	mined_patterns = dfA.copy()

	present.et = present.et.apply(pd.to_datetime)
	present.st = present.st.apply(pd.to_datetime)
	mined_patterns.st = mined_patterns.st.apply(pd.to_datetime)
	mined_patterns.et = mined_patterns.et.apply(pd.to_datetime)

	last_et = mined_patterns.et.max()
	current_st = present.st.min()
	ts = present.et.max()

	closed_patterns_A = mined_patterns.loc[mined_patterns.et != last_et]
	mined_patterns = mined_patterns.loc[mined_patterns.et == last_et]

	closed_patterns_B = present.loc[present.st != current_st]
	present = present.loc[present.st == current_st]

	new_subsets = present_new_or_subset_of_past(present, mined_patterns, last_et)
	old_subsets_or_sets = past_is_subset_or_set_of_present(present, mined_patterns, current_st)
    
	# Only keep the entries that are either:
	# 1. Currently active -> (mined_patterns.et==ts)
	# or,
	# 2. Been active for more that time_threshold time steps -> (mined_patterns.dur>time_threshold).
	# and
	# 3. Num of vessels in group pattern >= min_cardinality -> ([len(clst)>=min_cardinality for clst in mined_patterns.clusters])
	final_patterns = merge_pattern(new_subsets, old_subsets_or_sets) 
	return pd.concat([closed_patterns_A, final_patterns.loc[(final_patterns.et - final_patterns.st >= datetime.timedelta(minutes=time_threshold))],
					closed_patterns_B])


def reduce_partitions(dfs, time_threshold, res_rate):	
	complete = dfs[pd.to_datetime([df.et.max() for df in dfs]).argmin()]
	for i in range(len(dfs)-1):

		#THIS WHOLE THING IS NEEDED TO FIND THE MINIMUM POSITIVE NUMBER OF DIFFS !!!!!!
		diffs = pd.Series([pd.to_datetime(df.st.min()) - pd.to_datetime(complete.et.max()) for df in dfs]).values.astype(float)
		nxt = np.where(diffs<0, np.inf, diffs).argmin()

		if dfs[nxt].st.min() - complete.et.max() == pd.Timedelta(res_rate):
			complete = _merge_partitions(complete, dfs[nxt], time_threshold)
		else:
			complete = complete.append(dfs[nxt], ignore_index=True)
	return complete



# def reduce_partitions(dfs):
# 	complete = pd.DataFrame()
# 	for df in tqdm(dfs):
# 		if complete.empty:
# 			complete = complete.append(df)
# 		else:
# 			complete = _merge_partitions(complete, df)
# 	return complete


def check_for_checkpoint(df_checksum, params):
	try:
		ckpnt = io.load_pickle('gp_checkpoint.pckl')
		if ckpnt['checksum'] == df_checksum and ckpnt['params'] == params:
			return (ckpnt['current_ts'], ckpnt['last_ts'], ckpnt['patterns'], ckpnt['ind'])
		else:
			return False
	except:
		return False

def group_patterns(df, mode, min_diameter=3704, min_cardinality=10, time_threshold=30, checkpoints=True, checkpoints_freq=0.1, save_result=True):
	# circular (flag for convoys/flocks)	-> mode	(string; flocks(f)|convoys(c)|spherical swarms(fs)|dense swarms(cs))
	save_name = f'{mode}_min_diameter{min_diameter}_time_threshold{time_threshold}_min_cardinality{min_cardinality}.csv'

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


	mined_patterns,_,_ = mine_patterns(df, mode, min_diameter=min_diameter, min_cardinality=min_cardinality, time_threshold=time_threshold, checkpoints=checkpoints, checkpoints_freq=checkpoints_freq)


	# keep this df and use it again as the db for the real time implementation
	# print('Calculating mean velocity per flock...')
	# mined_patterns['mean_vel'] = np.nan
	# mined_patterns['mean_vel'] = mined_patterns.apply(lambda x: df.loc[(df.mmsi.isin(eval(x.clusters))) & (df.ts >= x.st) & (df.ts <= x.et)].velocity.mean(), axis=1)
	if save_result:
		print('Saving Result...')
		mined_patterns.to_csv(save_name, index=False)


def mine_patterns(df, mode, min_diameter=3704, min_cardinality=10, time_threshold=30, checkpoints=False, checkpoints_freq=0.1, total=None, start=0, last_ts=None, mined_patterns=None, disable_progress_bar=True):

	closed_patterns = pd.DataFrame()

	if not total:
		total = df.datetime.nunique()

	if checkpoints:
		checkpoint_interval = round(checkpoints_freq*df.datetime.nunique())


	if start != 0:
		if (not last_ts) and (not mined_patterns):
			raise

	#if mined patterns are not empty, split them to active (mined_patterns) and inactive (closed_patterns)
	if mined_patterns is not None:
		closed_patterns = closed_patterns.append(mined_patterns.loc[mined_patterns.et!=last_ts])
		mined_patterns = mined_patterns.loc[mined_patterns.et==last_ts]


	for ind, (ts, sdf) in tqdm(enumerate(df.groupby('datetime'), start=start), total=total, initial=start, disable=disable_progress_bar):

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


		present = present.loc[present.clusters.apply(len)>=min_cardinality]
		# Init the first present as mined_patterns
		if ind == 0:
			mined_patterns	= present
			last_ts			= ts
			continue


		new_subsets 		= present_new_or_subset_of_past(present, mined_patterns, last_ts)
		old_subsets_or_sets = past_is_subset_or_set_of_present(present, mined_patterns, ts)

		mined_patterns = merge_pattern(new_subsets, old_subsets_or_sets)

		# Only keep the entries that are either:
		# 1. Currently active -> (mined_patterns.et==ts)
		# or,
		# 2. Been active for more that time_threshold time steps -> (mined_patterns.dur>time_threshold).
		# and
		# 3. Num of vessels in flock >= min_cardinality -> ([len(clst)>=min_cardinality for clst in mined_patterns.clusters])
		mined_patterns = mined_patterns.loc[((mined_patterns.et==ts) | (mined_patterns.et - mined_patterns.st >= datetime.timedelta(minutes=time_threshold))) & ([len(clst)>=min_cardinality for clst in mined_patterns.clusters])]
		#Add all the inactive patterns to closed_patterns df
		closed_patterns = closed_patterns.append(mined_patterns.loc[mined_patterns.et!=ts])
		# Keep only the active dfs
		mined_patterns = mined_patterns.loc[mined_patterns.et==ts]
		last_ts = ts
#		if ind % 100 == 0:
#			print(f'Mined size -> {len(mined_patterns)}, Hist size -> {len(closed_patterns)}')


	return pd.concat([mined_patterns,closed_patterns]), ind+1, last_ts
