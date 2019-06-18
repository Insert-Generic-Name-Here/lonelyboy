import pandas as pd
import numpy as np
from haversine import haversine
import networkx as nx
from tqdm import tqdm as tqdm
import geopandas as gpd
import time, datetime


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


def get_current_clusters(sdf, ts, diam=1000, circular=True):
	'''
	Get clusters and init them as a single flock
	'''
	present = pd.DataFrame([[tuple(val)] for (val) in translate(get_clusters(sdf, diam, circular=circular), sdf )], columns=['clusters'])
	present['st'] = present['et'] = ts
	return present


def find_links(x, past, min_cardinality, inters_lst):
    tmp = past.apply(check_interesection, args=(x, min_cardinality, inters_lst, ), axis=1)
#     print(tmp)
    if set(x.clusters) not in tmp.tolist():
        x.clusters = set(x.clusters)
        inters_lst.append(x.tolist())
		
    
def check_interesection(A, B, min_cardinality, inters_lst):
    tmp = pd.DataFrame()
    inters = set(A.clusters).intersection(set(B.clusters))
    if (len(inters)>=min_cardinality): 
        inters_lst.append([inters, A.st, B.et])
        return inters
    else:
        return 0
	

def find_gps(present, past, min_cardinality):
	active = []

	present.apply(find_links, args=(past, min_cardinality, active,), axis=1)

	active = pd.DataFrame(active, columns=present.columns)
	
# 	print(active[active.clusters.apply(lambda x: set(x) == set([205204000, 228064900, 234597000, 228797000, 259019000]))])
	
	inactive = past[past.clusters.apply(lambda x: set(x) not in active.clusters.tolist())]
	
	active.clusters = active.clusters.apply(sorted).apply(tuple)

# 	active['clst'] = active.clusters.apply(sorted).apply(tuple)
	
# 	inactive.clusters = inactive.clusters.apply(sorted).apply(tuple)

	filtered_dups = active[active.duplicated('clusters', keep=False)].groupby('clusters', group_keys=False, as_index=False).apply(lambda x: pd.Series([x.clusters.unique()[0], x.st.min(), x.et.max()]))
	
# 	active.drop(['clst'], axis=1, inplace=True)
	
	if filtered_dups.duplicated(keep=False).sum() != 0:
		print(active.duplicated(keep=False).sum(), 'dups before')
		print(filtered_dups.duplicated(keep=False).sum(), 'dups after')
	try:
		filtered_dups.columns = present.columns
	except:
		pass

	return pd.concat([active[~active.duplicated('clusters', keep=False)], filtered_dups]).reset_index(drop=True), inactive.reset_index(drop=True)
				  

def _merge_partitions(dfA, dfB, min_cardinality):
	present = dfB.copy()
	mined_patterns = dfA.copy()
	#mined_patterns.to_pickle(f'mined_{mined_patterns.et}.pckl')
	#present.to_pickle(f'present_{present.st}.pckl')

	present.et = present.et.apply(pd.to_datetime)
	present.st = present.st.apply(pd.to_datetime)
	mined_patterns.st = mined_patterns.st.apply(pd.to_datetime)
	mined_patterns.et = mined_patterns.et.apply(pd.to_datetime)

	closed_patterns_A = mined_patterns.loc[(mined_patterns.et != mined_patterns.et.max())]
	mined_patterns = mined_patterns.loc[mined_patterns.et == mined_patterns.et.max()]

	closed_patterns_B = present.loc[present.st != present.st.min()]
	present = present.loc[present.st == present.st.min()]

# 	mined_patterns.to_csv(f'logs/dfA_{mined_patterns.et.unique()[0]}.csv', index=False)
# 	present.to_csv(f'logs/dfB_{present.st.unique()[0]}.csv', index=False)
	right, left = find_gps(present, mined_patterns, min_cardinality)
	
# 	pd.concat([right,left]).to_csv(f'logs/res_{present.st.unique()[0]}.csv', index=False)
	return pd.concat([closed_patterns_A, left, right, closed_patterns_B])


def reduce_partitions(dfs, min_cardinality, res_rate, time_threshold, max_ts=None):	
		
	order = pd.Series([df.st.min()] for df in dfs).sort_values().index.tolist()
	print(order)
	complete = dfs[order[0]]
	for nxt in order[1:]:		
		if pd.to_datetime(dfs[nxt].st.min()) - pd.to_datetime(complete.et.max()) == pd.Timedelta(res_rate):
			print(f'chained -> {dfs[nxt].st.min()} with {complete.et.max()}')
			complete = _merge_partitions(complete, dfs[nxt], min_cardinality)
		else:
			print('nope')
			complete = complete.append(dfs[nxt], ignore_index=True)

	complete.et = complete.et.apply(pd.to_datetime)
	complete.st = complete.st.apply(pd.to_datetime)

	if max_ts is not None:
		return complete[(pd.to_datetime(complete.et) == pd.to_datetime(max_ts)) | (complete.st == complete.st.min()) | (pd.to_datetime(complete.et) - pd.to_datetime(complete.st) >= pd.Timedelta(minutes=time_threshold))]
	else:
		return complete[(complete.et == complete.et.max()) | (complete.st == complete.st.min()) | (pd.to_datetime(complete.et) - pd.to_datetime(complete.st) >= pd.Timedelta(minutes=time_threshold))]


def mine_patterns(df, mode, min_diameter=3704, min_cardinality=10, time_threshold=30, active=pd.DataFrame(), closed_patterns=pd.DataFrame(), total=None, start=0, disable_progress_bar=True):
    
	if not total:
		total = df.datetime.nunique()


	for ind, (ts, sdf) in tqdm(enumerate(df.groupby('datetime'), start=start), total=total, initial=start, disable=disable_progress_bar):

		if mode == 'flocks' or mode == 'f':
			present = get_current_clusters(sdf, ts, min_diameter, circular=True)
		elif mode == 'convoys' or mode == 'c':
			present = get_current_clusters(sdf, ts, min_diameter, circular=False)
		elif mode == 'swarms' or mode == 's':
			raise NotImplementedError('Current mode is not Implemented atm.')


		present = present.loc[present.clusters.apply(len)>=min_cardinality]


		if active.empty:
			active = present
			continue


		if present.empty:
			closed_patterns = closed_patterns.append(active.loc[(active.et - active.st >= pd.Timedelta(minutes=time_threshold))])
			active = pd.DataFrame()
			continue


		active, inactive = find_gps(present, active, min_cardinality)

		closed_patterns = closed_patterns.append(inactive.loc[(inactive.et - inactive.st >= pd.Timedelta(minutes=time_threshold))])

# 		if not ind % 100:
# 			print('## UPDATE ###')

# 			print(f'Size of closed: {len(closed_patterns)}')
# 			print(f'Size of currently active: {len(active)}')
# 			print('#############')


	return active, closed_patterns, ind+1


def mine_patterns_dist(df, mode, min_diameter=3704, min_cardinality=10, time_threshold=30, active=pd.DataFrame(), closed_patterns=pd.DataFrame(), total=None, start=0, disable_progress_bar=True):
    
	if not total:
		total = df.datetime.nunique()


	for ind, (ts, sdf) in tqdm(enumerate(df.groupby('datetime'), start=start), total=total, initial=start, disable=disable_progress_bar):

		if mode == 'flocks' or mode == 'f':
			present = get_current_clusters(sdf, ts, min_diameter, circular=True)
		elif mode == 'convoys' or mode == 'c':
			present = get_current_clusters(sdf, ts, min_diameter, circular=False)
		elif mode == 'swarms' or mode == 's':
			raise NotImplementedError('Current mode is not Implemented atm.')


		present = present.loc[present.clusters.apply(len)>=min_cardinality]


		if active.empty:
			active = present
			continue


		if present.empty:
			closed_patterns = closed_patterns.append(active.loc[(active.et - active.st >= pd.Timedelta(minutes=time_threshold)) | (active.st == df.datetime.min())])
			active = pd.DataFrame()
			continue


		active, inactive = find_gps(present, active, min_cardinality)

		closed_patterns = closed_patterns.append(inactive.loc[(inactive.et - inactive.st >= pd.Timedelta(minutes=time_threshold)) | (inactive.st == df.datetime.min())])

# 		if not ind % 100:
# 			print('## UPDATE ###')

# 			print(f'Size of closed: {len(closed_patterns)}')
# 			print(f'Size of currently active: {len(active)}')
# 			print('#############')

	
	return pd.concat([active, closed_patterns])


