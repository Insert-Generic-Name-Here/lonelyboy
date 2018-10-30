import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def KMeans_Silhouette(X, n_sims, n_init=10, n_jobs=-1, precompute_distances=True, random_state=0, verbose=0):
    ''' Determine the Optimal Number of Clusters using the Silhouette Score on Multiple Runs of KMeans '''
    clusters = []   
    silhouettes = []
    
    for k in tqdm(range(1,n_sims+1), leave=True):
        kmeanTest = KMeans(n_clusters=k, n_init=n_init, n_jobs=n_jobs, precompute_distances=precompute_distances, random_state=random_state, verbose=verbose).fit(X);
        label = kmeanTest.labels_
        clusters.append(label)
        try:
            sil_coeff = silhouette_score(X, label, metric='euclidean')
            silhouettes.append(sil_coeff)
        except ValueError:
            silhouettes.append(0)
            
    n_clusters_opt = np.argmax(np.array(silhouettes))
    return n_clusters_opt+1, clusters[n_clusters_opt]


def flock_mining(gdf, doi=None, n_init=10, n_jobs=-1, precompute_distances=True, random_state=0, verbose=0):
    gdf[['lon', 'lat']] = gdf['geom'].apply(lambda x: pd.Series({'lon':x.x, 'lat':x.y})) 
    gdf = gdf.drop('geom', axis=1) 
    
    if doi is None:
        datetimes = gdf['datetime'].unique()
    else:
        datetimes = doi
    
    flocks = {}
    for datetime_of_interest in tqdm(datetimes):
        # timeFrame = gdf[['lon','lat', 'course']].loc[gdf['datetime'] == datetime_of_interest]
        timeFrame = gdf[['lon','lat']].loc[gdf['datetime'] == datetime_of_interest]
        
        scaler = MinMaxScaler()
        X_std = scaler.fit_transform(timeFrame.values)

        if (len(timeFrame) == 1):
            continue
        n_flocks, labels = KMeans_Silhouette(X_std, len(timeFrame), n_init, n_jobs, precompute_distances, random_state, verbose)
        flocks[str(datetime_of_interest)] = (n_flocks, timeFrame.index, labels)
    
    return flocks