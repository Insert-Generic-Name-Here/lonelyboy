import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from functools import partial
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from lonelyboy.geospatial.metrics import haversine_distance as haversine


def MinMax_Scaler(X, feature_range=(0, 1), copy=True):
    scaler = MinMaxScaler(feature_range, copy)
    return scaler.fit_transform(X)


# CMC = Coherent Moving Cluster -- Convoy Verification Process
def cmc_convoy_verification(cluster_left, cluster_right, min_samples):
    return len(cluster_left.intersection(cluster_right)) >= min_samples


# CMC = Coherent Moving Cluster -- Flock Verification Process
def cmc_flock_verification(cluster_left, cluster_right, min_samples):
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


def join_geospatial(df_left, df_right, condition, mode):
    if (len(df_left) == 0):
        return df_right
    
    df_result = pd.DataFrame([], columns=[mode, 'start_time', 'end_time'])
    indices = []
    
    for idx_left, cluster_left in enumerate(df_left[mode]):
        for idx_right, cluster_right in enumerate(df_right[mode]):
            if condition(cluster_left, cluster_right):                
                res = pd.DataFrame([{mode:cluster_left.intersection(cluster_right), 'start_time':df_left.iloc[idx_left].start_time, 'end_time':df_right.iloc[idx_right].start_time}], columns=[mode, 'start_time', 'end_time'])  
                df_result = df_result.append(res, ignore_index=True)
                indices.append(idx_right)
                break
        
    indices = np.delete(df_right.index.values, indices)
    df_result = df_result.append(df_right.iloc[indices], ignore_index=True)
    return df_result


def group_patterns_mining(gdf, normalizing_algorithm, clustering_algorithm, verification_process, mode, time_threshold=5, min_samples=2, resampling_rate=60): 
    '''
    Search for Flocks/Convoys, given a GeoDataFrame.
    '''
    
    gdf[['lon', 'lat']] = gdf['geom'].apply(lambda x: pd.Series({'lon':x.x, 'lat':x.y})) 
    gdf.drop('geom', axis=1)
    
    cmc_verification_partial = partial(verification_process, min_samples=min_samples)
    gp_history = pd.DataFrame([], columns=[mode, 'start_time', 'end_time'])
    
    for doi, timeFrame in gdf.groupby(['datetime'], as_index=False):   
        print (f'Datetime of Interest: {doi}\r', end='')    
        X = normalizing_algorithm(timeFrame[['lon', 'lat']].values)
        cluster_n = clustering_algorithm(X)
        # Create the DataFrame (Structure: <INDEX_OF_CLUSTER>, <LIST_OF_TIMEFRAME_INDICES>)
        tmp = pd.DataFrame(np.array([gdf.loc[timeFrame.index]['mmsi'], cluster_n]).T, columns=[mode, 'cluster_idx'])
        tmp = tmp.loc[tmp.cluster_idx != -1].groupby('cluster_idx')[mode].apply(list).apply(set)
        tmp = pd.DataFrame({mode:tmp, 'start_time':np.array([doi]*len(tmp))}, columns=[mode, 'start_time', 'end_time'])
        # Append to Convoy History
        gp_history = join_geospatial(gp_history, tmp, cmc_verification_partial, mode=mode)
        
    gp_history = gp_history.fillna(pd.Timestamp(gdf.datetime.unique()[-1]))
    gp_history.end_time = pd.to_datetime(gp_history.end_time, unit='ns')
    return gp_history.loc[(gp_history.end_time - gp_history.start_time >= np.timedelta64(time_threshold*resampling_rate, 's')) & (gp_history[mode].apply(len) >= min_samples)].reset_index(drop=True)