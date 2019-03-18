import pandas as pd
from haversine import haversine
import geopandas as gpd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import contextily as ctx
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm
import numpy as np
from collections import Counter


def read_csv_generator(file_path, chunksize=50000, sep=',', **kwargs):
    pd_iter = pd.read_csv(file_path, chunksize=chunksize, sep=sep, **kwargs)
    return pd_iter


def gdf_from_df(df, crs=None):
	# {'init':'epsg:4326'}
	df['geom'] = np.nan
	df.geom = df[['lon', 'lat']].apply(lambda x: Point(x[0],x[1]), axis=1)
	return gpd.GeoDataFrame(df, geometry='geom', crs=crs)


def distance_to_nearest_port(point, ports):
	'''
	Calculates the minimum distance between the point and the lists of ports. Can be used to determine if the ship is sailing or not
	'''
	return ports.geom.distance(point).min()


def ts_from_str_datetime(df):
	print (f'Droping {df.datetime.isna().sum()} rows..')
	df.dropna(subset=['datetime'], inplace=True)
	df['ts'] = pd.to_datetime(df.datetime).values.astype(np.int64) // 10 ** 9


def get_outliers(series, alpha = 3):
	'''
	Returns a series of indexes of row that are to be concidered outliers, using the quantilies of the data.
	'''
	q25, q75 = series.quantile((0.25, 0.75))
	iqr = q75 - q25
	q_high = q75 + alpha*iqr
	q_low = q25 - alpha*iqr
	# return the indexes of rows that are over/under the threshold above
	return series.loc[(series >q_high) | (series<q_low)].index , (q_low, q_high)


def resample_geospatial(vessel, rule = '60S', method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False):
	'''
	Resample and interpolate linearly a sample vessel.
	'''
	#convert unix to datetime
	vessel['datetime'] = pd.to_datetime(vessel['ts'], unit='s')
	#resample and interpolate using the method given. Linear is suggested
	upsampled = vessel.resample(rule,on='datetime', loffset=True, kind='timestamp').first()
	interpolated = upsampled.interpolate(method=method)
	interpolated['real_point'] = interpolated.datetime.apply(lambda x: 1 if type(x)==pd._libs.tslibs.timestamps.Timestamp else 0)
	#interpolate the geom column with the correct point objects using lat and lon
	interpolated['geom'] = interpolated[['lon', 'lat']].apply(lambda x: Point(x[0], x[1]), axis=1)
	# reset the index to normal and use the old index as new timestamp
	interpolated['datetime'] = interpolated.index
	interpolated.reset_index(drop=True, inplace=True)
	#drop lat and lon if u like
	if drop_lon_lat:
		interpolated = interpolated.drop(['lat', 'lon'], axis=1)
	return gpd.GeoDataFrame(interpolated, crs = crs, geometry='geom')


def calculate_velocity(gdf, smoothing=False, window=15, center=False):
	'''
	Return given dataframe with an extra velocity column that is calculated using the distance covered in a given amount of time
	TODO - use the get distance method to save some space
	'''
	# if there is only one point in the trajectory its velocity will be the one measured from the speedometer
	if len(gdf) == 1:
		gdf['velocity'] = gdf.speed
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf['next_loc'] = gdf.geom.shift(-1)
	gdf = gdf[:-1]
	gdf.loc[:,'next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
	# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots)
	gdf.loc[:,'velocity'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).multiply(3600/gdf.ts.diff(-1).abs())

	if smoothing:
		gdf.loc[:,'velocity'] = gdf['velocity'].rolling(window, center=center).mean().bfill().ffill()

	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	gdf = gdf.loc[gdf['mmsi'] != 0]
	gdf.dropna(subset=['mmsi', 'geom'], inplace=True)

	return gdf.fillna(0)


def PotentialAreaOfActivity(gdf, velocity_threshold):
	'''
	Detect Invalid GPS points by calculating the Potential Area of Activity (PAA) and removing them based on a velocity threshold.
	'''
	indices = [item for sublist in [x for x in gdf.groupby(['mmsi'])['velocity'].apply(lambda x: x.loc[x.values >= velocity_threshold].index)] for item in sublist]
	gdf = gdf.drop(indices)
	return gdf


def calculate_distance_traveled(gdf):
	'''
	Returns given dataframe with an added column ('distance') that contains the accumulated distance travel up to a specific point
	'''
	gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf['next_loc'] = gdf.geom.shift(-1)
	gdf = gdf[:-1]
	gdf['next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
	gdf['distance'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).cumsum()
	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	return gdf


def pick_random_group(gdf, column, group_size=1):
	return gdf.loc[gdf[column].isin(np.random.choice(gdf[column].unique(), group_size))]


def detect_POIs(df, feature='velocity', alpha=20, window=100):
	'''
	Detect Points Of Interest based on the 1st order difference series of the selected feature.

	Parameters
	----------
	df : the pandas dataframe to be used

	feature: the column of the dataframe that will be used to mine the needed info, default = 'velocity' or velocity based trajectory segmentation

	alpha : the multiplier that will be used to detect the outling values of the 1st order difference series of the selected feture

	window : the span of the sliding window that will be used for smoothing

	Returns
	-------

	pois : list of the indicies where a change is detected (ex. a change in speed). These indicies can be used for trajectory segmentation.
	'''
	#calculate the 1st order difference series of the feature, while applying smoothing in both series, the original one and the difference series.
	diff_series = df[feature].rolling(window).mean().diff().rolling(window).mean()
	# diff_series = df[feature].rolling(window).mean().diff().rolling(window).mean()

	#detect the outliers of the above series.
	outlier_groups, (qlow, qhigh) = get_outliers(diff_series.dropna(), alpha=alpha)

	try:
		#create the pois list and add the first point of the outlier_groups list that is a guaranteed POI
		pois = [0,outlier_groups[0]]
		#if a point is not a part of a concecutive list add it to the poi list
		#that way only the first point of every group of outliers will be concidered a POI
		for ind, point in enumerate(outlier_groups[1:-1],1):
			# if (point != outlier_groups[ind-1]+1) or (point != outlier_groups[ind+1]-1):
			if (point != outlier_groups[ind-1]+1):
				pois.append(point)
		pois.append(len(df)-1)
	except : # No Outliers?! Maybe you Need to Tune the Function's Parameters
		pois=[0, len(df)-1]
	return pois, (qlow, qhigh)


def segment_trajectories(gdf,pois_alpha=80, pois_window=100, n_jobs=1, np_split=True, feature='mmsi'):
	cores = _get_n_jobs(n_jobs)
	if cores==1:
		print(pois_alpha, pois_window)
		gdf = _segment_trajectories_grouped(gdf, ports, pois_alpha=pois_alpha, pois_window=pois_window)
	else:
		#TODO
		gdf = parallelize_dataframe(gdf, _segment_trajectories_grouped)

	return gdf


def detect_POIs_approx(vessel, window):
	lst = []
	alpha = 1
	while True:
		lst.append(tuple(detect_POIs(vessel, alpha=alpha, window=window)[0]))
		pois, freq = Counter(lst).most_common(1)[0]
		if len(lst[-1]) == 2 or alpha>=100:
			return pois
		alpha += 1


def _segment_vessel(vessel, ports,pois_alpha, pois_window, semantic=False):
	vessel.reset_index(drop=True, inplace=True)
	if len(vessel) == 1 :
		vessel.traj_id  = 0.0
		return vessel

	if pois_alpha != -1:
		pois, _ = detect_POIs(vessel, alpha=pois_alpha, window=pois_window)
	else:
		pois = detect_POIs_approx(vessel, window=pois_window)
	# OLD STUPID STUPID CODE
	# vessel['traj_id'] = vessel.apply(get_trajetory_segment , args=(pois,), axis=1)
	# NEW SMARTER ONE, I GUESS
	# added simple semantic enrichment
	# semantic_id = 0 -> Stationary (near port)
	# semantic_id = 1 -> Stationary (not near port)
	# semantic_id = 2 -> Accelerating
	# semantic_id = 3 -> Decelerating
	for i in range(len(pois)-1):
		if semantic:
			slice = vessel.iloc[pois[i]:pois[i+1]]
			if slice.velocity.mean()<1:
				if slice['geom'].apply(lambda x: distance_to_nearest_port(x, ports)).mean()<=0.1:
					vessel.loc[pois[i]:pois[i+1], 'semantic_id'] = 0
				else:
					vessel.loc[pois[i]:pois[i+1], 'semantic_id'] = 1
			else:
				if slice.velocity.diff().mean()<0:
					vessel.loc[pois[i]:pois[i+1], 'semantic_id'] = 3
				else:
					vessel.loc[pois[i]:pois[i+1], 'semantic_id'] = 2
		vessel.loc[pois[i]:pois[i+1], 'traj_id'] = i
	vessel['pois'] = [pois]*len(vessel)
	return vessel


def _segment_trajectories_grouped(gdf, pois_alpha=20, pois_window=100):
	gdf['traj_id'] = np.nan
	gdf['pois'] = np.nan
	gdf['semantic_id'] = np.nan
	ports = pd.read_pickle('ports.pckl')
# tqdm.pandas()
	gdf = gdf.groupby(['mmsi'], group_keys=False).apply(_segment_vessel, ports, pois_alpha, pois_window).reset_index(drop=True)
	# ts_from_str_datetime(gdf)
	return gdf

def _get_n_jobs(n_jobs):
	if n_jobs>cpu_count():
		raise ValueError("n_jobs can't be more than available cpu cores")
	if n_jobs==1:
		return 1
	else:
		if n_jobs == -1:
			return cpu_count()
		else:
			return n_jobs

def pd2gdf(df):
	df.geom = df.apply(lambda x: Point(x.lon, x.lat), axis=1)
	traj = gpd.GeoDataFrame(df, geometry='geom')
	return traj


def resample_and_calculate_velocity(gdf, velocity_window=3, velocity_drop_alpha=3, smoothing=True, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, n_jobs=1):
	if type(gdf) == pd.core.frame.DataFrame:
		gdf = pd2gdf(gdf)

	gdf['velocity'] = np.nan
	cores = _get_n_jobs(n_jobs)
	if cores==1:
		gdf = _resample_and_calculate_velocity_grouped(gdf, velocity_window=velocity_window, velocity_drop_alpha=velocity_drop_alpha, smoothing=smoothing, res_rule=res_rule, res_method=res_method, crs=crs, drop_lon_lat=drop_lon_lat, resampling_first=resampling_first, drop_outliers=drop_outliers)
	else:
		#TODO
		gdf = parallelize_dataframe(gdf, _resample_and_calculate_velocity_grouped)
	return gdf


def _resample_and_calculate_velocity_grouped(gdf, velocity_window=3, velocity_drop_alpha=3, smoothing=True, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False):
	gdf = gdf.groupby(['mmsi'], group_keys=False).apply(_resample_and_calculate_velocity_vessel,velocity_window, velocity_drop_alpha, smoothing, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers).reset_index(drop=True)
	gdf.reset_index(inplace=True, drop=True)
	return gdf


def _resample_and_calculate_velocity_vessel(vessel, velocity_window, velocity_drop_alpha, smoothing, res_rule, res_method, crs , drop_lon_lat, resampling_first, drop_outliers):
	if len(vessel) == 1 :
		vessel.velocity = vessel.speed
		return vessel
	if resampling_first:
		vessel = resample_geospatial(vessel, rule=res_rule, method=res_method, crs = crs, drop_lon_lat = drop_lon_lat)
		vessel = calculate_velocity(vessel, smoothing=smoothing, window=velocity_window)
	else:
		vessel = calculate_velocity(vessel, smoothing=smoothing, window=velocity_window)
		vessel = resample_geospatial(vessel, rule=res_rule, method=res_method, crs = crs, drop_lon_lat = drop_lon_lat)
	if drop_outliers:
		vessel = vessel.drop(get_outliers(vessel.velocity, alpha=velocity_drop_alpha)[0], axis=0)
	# vessel.reset_index(inplace=True, drop=True)
	return vessel


def clean_gdf(gdf):
	gdf.drop_duplicates(['ts', 'mmsi'], inplace=True)
	gdf.drop([item for sublist in [x for x in gdf.groupby(['mmsi'], group_keys=False)['ts'].apply(lambda x: get_outliers(x)[0]) if x != []] for item in sublist], axis=0, inplace=True)
	gdf.reset_index(inplace=True, drop=True)
	gdf.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')
	return gdf


def partition_geospatial(gdf, feature='mmsi', num_partitions=1):
	partitions = []
	tmp_X = gpd.GeoDataFrame([], columns=gdf.columns)
	for _, x in gdf.groupby([feature]):
		tmp_X = tmp_X.append(x)
		if len(tmp_X) >= len(gdf)//num_partitions:
			partitions.append(tmp_X)
			tmp_X = tmp_X.iloc[0:0] # Drop all Rows
	# partitions.append(tmp_X.reset_index(drop=True))
	partitions.append(tmp_X)
	return partitions


def parallelize_dataframe(df, func, np_split=False, feature='mmsi', num_partitions=8):
	 print('starting..')
	 if np_split:
	 	partitions = np.array_split(df, cpu_count())
	 else:
	 	partitions = [grp[1] for grp in df.groupby(['mmsi'])]
	 print (len(partitions))
	 pool = Pool(num_partitions)
	 df = pd.concat(pool.map(func, partitions))
	 pool.close()
	 pool.join()
	 return df


def _pipeline_apply(vessel, ports, velocity_window=3, velocity_drop_alpha=3, smoothing=True, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, pois_alpha=-1, pois_window=100, semantic=False ):
	'''
	Full automated pipeline. To be used on a single mmsi, either manually, or using .groupby
	'''
	vessel.drop_duplicates(['ts'], inplace=True)
	vessel.sort_values('ts', inplace=True)
	vessel.reset_index(inplace=True, drop=True)
	vessel.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')
	vessel['geom'] = vessel[['lon', 'lat']].apply(lambda x: Point(x[0],x[1]), axis=1)
	vessel=  gpd.GeoDataFrame(vessel, geometry='geom')
	vessel = _resample_and_calculate_velocity_vessel(vessel, velocity_window, velocity_drop_alpha, smoothing, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers)
	vessel = _segment_vessel(vessel, ports, pois_alpha, pois_window, semantic)
	return vessel


def resample_and_segment(vessel, ports, pre_segment_threshold=12, velocity_window=3, velocity_drop_alpha=3, smoothing=True, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, pois_alpha=-1, pois_window=100, semantic=False):
	'''
	Function that calls _pipeline_apply.
	pre_segment_threshold is the number of hours that a transmitter needs to be off in order to brake up a trajectory into usefull ones. If 0 then do not pre segment
	'''
	if pre_segment_threshold != 0:
		# split vessel into segments that correspond to a specific fishing trip
		brake_points = vessel.ts.diff(-1).abs().index[vessel.ts.diff()>60*60*pre_segment_threshold]
		dfs = np.split(vessel, brake_points)
		# apply pipeline to each segment and concatenate
		dfs_prepd = [_pipeline_apply(df, ports, velocity_window, velocity_drop_alpha, smoothing, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers, pois_alpha, pois_window, semantic) for df in dfs if len(df)>1]
        ####### exp #######
		for i in range(1,len(dfs_prepd)):
			dfs_prepd[i].loc[:,'traj_id'] = dfs_prepd[i].traj_id.apply(lambda x: x+dfs_prepd[i-1].traj_id.max()+1)
		df_fn = pd.concat(dfs_prepd)
		df_fn.sort_values('ts', inplace=True)
		df_fn.reset_index(inplace=True, drop=True)
	else:
		df_fn = _pipeline_apply(vessel, ports, velocity_window, velocity_drop_alpha, smoothing, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers, pois_alpha, pois_window, semantic)
	return df_fn, brake_points
