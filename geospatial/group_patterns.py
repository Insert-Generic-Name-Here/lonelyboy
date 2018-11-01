import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from lonelyboy.geospatial.metrics import haversine_distance as haversine


def KMeans_Silhouette(X, n_sims, init='k-means++', n_init=10, n_jobs=-1, precompute_distances=True, random_state=0, verbose=0):
    ''' Determine the Optimal Number of Clusters using the Silhouette Score on Multiple Runs of KMeans '''
    clusters = []   
    silhouettes = []
    
    for k in tqdm(range(1,n_sims+1), leave=False):
        kmeanTest = KMeans(n_clusters=k, init=init, n_init=n_init, n_jobs=n_jobs, precompute_distances=precompute_distances, random_state=random_state, verbose=verbose).fit(X);
        label = kmeanTest.labels_
        clusters.append(label)
        try:
            sil_coeff = silhouette_score(X, label, metric='euclidean')
            silhouettes.append(sil_coeff)
        except ValueError:
            silhouettes.append(0)
            
    n_clusters_opt = np.argmax(np.array(silhouettes))
    return n_clusters_opt+1, clusters[n_clusters_opt]


# TODO #1: Add a Time Threshold for tracking the end_time of the Flock.
# TODO #2: Check if the Application of Haversine Formula Enhances the Cluster Quality (Hint: Check the Data Projection).
def flock_mining(gdf, doi=None, init='k-means++', n_init=10, n_jobs=-1, precompute_distances=True, random_state=0, verbose=0):
    # Get the Useful Features
    gdf[['lon', 'lat']] = gdf['geom'].apply(lambda x: pd.Series({'lon':x.x, 'lat':x.y})) 
    gdf = gdf.drop('geom', axis=1) 
    
    if doi is None:
        datetimes = gdf['datetime'].unique()
    else:
        datetimes = doi
    
    flocks = pd.DataFrame([], columns=['flocks', 'start_time', 'end_time'])

    for datetime_of_interest in tqdm(datetimes):
        # timeFrame = gdf[['lon','lat', 'course']].loc[gdf['datetime'] == datetime_of_interest]
        timeFrame = gdf[['lon','lat']].loc[gdf['datetime'] == datetime_of_interest]
        # Normalize
        scaler = MinMaxScaler()
        X = scaler.fit_transform(timeFrame.values)
        # Cluster
        if (len(timeFrame) == 1):
            continue
        n_flocks, labels = KMeans_Silhouette(X, len(timeFrame), init, n_init, n_jobs, precompute_distances, random_state, verbose)
        # Create the DataFrame (Structure: <INDEX_OF_CLUSTER>, <LIST_OF_TIMEFRAME_INDICES>)
        tmp = pd.DataFrame(np.array([gdf.loc[timeFrame.index]['mmsi'], labels]).T, columns=['flocks', 'flock_idx'])
        tmp = tmp.loc[tmp.flock_idx != -1].groupby('flock_idx')['flocks'].apply(list)
        # Append to Flock History
        flocks = flocks.append(pd.DataFrame(tmp, columns=['flocks', 'start_time', 'end_time']))
        flocks.start_time = datetime_of_interest

    return flocks


# TODO #1: Add a Time Threshold for tracking the end_time of the Convoy.
def convoy_mining(gdf, time_threshold=5, min_samples=3, eps=2.5, metric=haversine, metric_params=None, algorithm='auto', leaf_size=50, p=None, n_jobs=-1): 
    gdf[['lon', 'lat']] = gdf['geom'].apply(lambda x: pd.Series({'lon':x.x, 'lat':x.y})) 
    gdf = gdf.drop('geom', axis=1)

    convoys = pd.DataFrame([], columns=['convoys', 'start_time', 'end_time'])

    for datetime_of_interest in tqdm(gdf['datetime'].unique()):      
        # Get the Useful Features
        timeFrame = gdf[['lon', 'lat']].loc[gdf['datetime'] == datetime_of_interest]
        # Normalize
        scaler = MinMaxScaler()
        X = scaler.fit_transform(timeFrame.values)
        # Cluster
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,\
                            algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs).fit(X)
        cluster_n = clustering.labels_
        # Create the DataFrame (Structure: <INDEX_OF_CLUSTER>, <LIST_OF_TIMEFRAME_INDICES>)
        tmp = pd.DataFrame(np.array([gdf.loc[timeFrame.index]['mmsi'], cluster_n]).T, columns=['convoys', 'cnv_idx'])
        tmp = tmp.loc[tmp.cnv_idx != -1].groupby('cnv_idx')['convoys'].apply(list)
        # Append to Convoy History
        convoys = convoys.append(pd.DataFrame(tmp, columns=['convoys', 'start_time', 'end_time']))
        convoys.start_time = datetime_of_interest
        
    return convoys