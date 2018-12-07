import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import networkx as nx
from sklearn import metrics
from functools import partial
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Point, LineString, shape
from haversine import haversine
import itertools

# from lonelyboy.geospatial.metrics import haversine_distance as haversine

def MinMax_Scaler(X, feature_range=(0, 1), copy=True):
    scaler = MinMaxScaler(feature_range, copy)
    return scaler.fit_transform(X)


def cmc_convoy_verification(cluster_left, cluster_right, min_samples):
    ''' CMC = Coherent Moving Cluster -- Convoy Verification Process '''
    return len(cluster_left.intersection(cluster_right)) >= min_samples


def cmc_flock_verification(cluster_left, cluster_right, min_samples):
    ''' CMC = Coherent Moving Cluster -- Flock Verification Process '''
    # return len(cluster_left.intersection(cluster_right)) >= min_samples
    return (cluster_left == cluster_right)


def KMeans_Clustering(X, init='k-means++', n_init=10, n_jobs=-1, precompute_distances=True, random_state=0, verbose=0):
    ''' Determine the Optimal Number of Clusters using the Silhouette Score on Multiple Runs of KMeans '''
    clusters = []   
    silhouettes = []
    
    for k in range(1, len(X)+1):
        kmeanTest = KMeans(n_clusters=k, init=init, n_init=n_init, n_jobs=n_jobs, precompute_distances=precompute_distances, random_state=random_state, verbose=verbose).fit(X)
        label = kmeanTest.labels_
        clusters.append(label)
        try:
            sil_coeff = silhouette_score(X, label, metric='euclidean')
            silhouettes.append(sil_coeff)
        except ValueError:
            silhouettes.append(0)
            
    n_clusters_opt = np.argmax(np.array(silhouettes))
    return clusters[n_clusters_opt]


def DBSCAN_Clustering(X, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs).fit(X) 
    return clustering.labels_


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
    return [sorted(list(cluster)) for cluster in nx.connected_components(G)]


def pairs_in_radius(df, radius):
    distances = np.triu(distance_matrix(df[['lat', 'lon']].values, df[['lat', 'lon']].values), 0)
#     distances = sample_timeFrame[['lon', 'lat']].T.apply(lambda A: sample_timeFrame[['lon', 'lat']].T.apply(lambda B: haversine((A[0], A[1]), (B[0], B[1]))))
    distances = np.triu(distances, 0)
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
    df =  df.groupby('datetime', as_index=False).apply(get_flock_labels, radius)
    return df


def point_from_lat_lon(df_w_lat_lon):
    df_w_lat_lon['geom'] = np.nan
    df_w_lat_lon['geom'] = df_w_lat_lon[['lon', 'lat']].apply(lambda x: Point(x), axis=1)
    return gpd.GeoDataFrame(df_w_lat_lon, geometry='geom')


def get_correct_label(present, future):
    lst = future.loc[future.mmsi.isin(present.mmsi)].flock_label.value_counts().index
    if lst[0] != -1 or len(lst)==1:
        return lst[0]
    else:
        return lst[1]


def swap(x, pair):
    return pair[pair.index(x)-1] if x in pair else x


def window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win

    for e in i:
        win = win[1:] + [e]
        yield win


def label_tracing(df):
    grouped = df.groupby('datetime')
    for  ind, (ts, group) in enumerate(list(grouped)[:-1]):
        print (ind, end='\r')
        for label, present in group.groupby('flock_label'):
            future = list(grouped)[ind+1][1]
            new_label = get_correct_label(present, future)
            df.loc[df.datetime == future.iloc[0].datetime, 'flock_label'] = future['flock_label'].apply(swap, args=((new_label, label),))
    return df


def hasNext(x):
    try:
        return next(x)
    except StopIteration:
        return []


def group_patterns_mining(cluster_history, mode, time_threshold=5, min_samples=3):
    endOfTime = cluster_history.datetime.max()
    mined_patterns = pd.DataFrame([], columns=[mode, 'start', 'end'])
    cluster_history_window = cluster_history.groupby('mmsi').agg({'cluster_label': lambda x: window(list(x), time_threshold), 'datetime': lambda x: pd.Timestamp(min(x))})

    while (len(cluster_history_window) != 0):
        # Set the Start of Time
        startTime = cluster_history_window.datetime.min()
        endTime = startTime + pd.offsets.Minute(time_threshold)
        print (f'Datetime of Interest: {startTime}', end='\r')

        # Get the History Window according to the above Timestamps
        timeFrameClusters = cluster_history_window.loc[cluster_history_window.datetime == startTime]['cluster_label'].apply(lambda x: hasNext(x)).apply(tuple)
        # Group by the History Window
        for label_hist_window, mmsis in timeFrameClusters.groupby(timeFrameClusters):
            if ((len(mmsis.index) >= min_samples) and (-1 not in label_hist_window)):
                foi = mined_patterns.loc[mined_patterns[mode].apply(tuple) == tuple(mmsis.index)]
    #             if (-1 in label_hist_window):
    #                 continue
                    # TODO - Refine Here the End Timestamp for Existing Patterns (Minor)
    #             else:
                if (len(foi) != 0):
                    mined_patterns.at[foi.index[0], 'end'] = endTime
                else:
                    newPattern = pd.DataFrame([{mode: tuple(mmsis.index),  'start': startTime, 'end': endTime}], columns=[mode, 'start', 'end'])                     
                    mined_patterns = mined_patterns.append(newPattern, ignore_index=True)

        # Prepare for Next Iteration
        #     * Clean the Redundant Records
        ioi = timeFrameClusters.loc[timeFrameClusters.apply(len) == 0].index
        cluster_history_window.drop(list(ioi), inplace=True)
        #     * Refresh the Start of Time Timestamp
        cluster_history_window['datetime'] += pd.offsets.Minute(1)

        if (startTime > pd.Timestamp(endOfTime)):
            break
        
    return mined_patterns

#### flocks v2 ####

def in_radius(df, diam=500):
    '''
    Return pairs with distance < diam
    '''
    res = []
    for ind_i, ind_j, val_i, val_j in nparray_combinations(df):
        dist = haversine(val_i, val_j)*1000
        if (dist<diam):
            res.append((ind_i,ind_j))   
    return res
            

def nparray_combinations(arr):
    for i in range(arr.shape[0]):
        for j in range(i+1, arr.shape[0]):
            yield i, j, arr[i,:], arr[j,:]


def eval_candidate(candidate, pairs):
    '''
    Evaluate a candidate based on if all included points exist in pairs
    '''
    for pair in itertools.combinations(candidate,2):
        if pair not in pairs:
            return False
    return True


def circle_cluster(timeframe, min_cardinality = 2, diam_in_meters = 500):
    '''
    For each connected graph, get all r-length combinations (min cardinality<r<len(sets)) and check if each of them is valid.
    '''
    pairs = in_radius(timeframe[['lon', 'lat']].values, diam_in_meters)
    full_sets = connected_edges(pairs)
    clusters = []
    for individual_sets in full_sets:
        # TODO MAYBE FROM BIG TO SMALL INSTEAD OF SMALL TO BIG
        for j in range(min_cardinality,len(individual_sets)+1):
            for candidate in itertools.combinations(individual_sets,j):
                 if eval_candidate(candidate, pairs):
                        clusters.append(candidate)
    return clusters