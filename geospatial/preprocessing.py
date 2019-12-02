import pandas as pd
from haversine import haversine
import geopandas as gpd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import contextily as ctx
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from collections import Counter
import math



def read_csv_generator(file_path, chunksize=50000, sep=',', **kwargs):
	pd_iter = pd.read_csv(file_path, chunksize=chunksize, sep=sep, **kwargs)
	return pd_iter

def merc(Coordinates):
	'''
	Transforms Coordinates (lat,lon) to mercator. Usefull for bokeh plotting
	''' 
	lat = Coordinates[0]
	lon = Coordinates[1]

	r_major = 6378137.000
	x = r_major * math.radians(lon)
	scale = x/lon
	y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale    
	return (x, y)


def gdf_from_df(df, crs=None):
	# {'init':'epsg:4326'}
	df.loc[:, 'geom'] = np.nan
	df.geom = df[['lon', 'lat']].apply(lambda x: Point(x[0],x[1]), axis=1)
	df.loc[:, 'merc_x'] = np.nan
	df.loc[:, 'merc_y'] = np.nan
	df.loc[:, 'merc_x'] = df[['lat','lon']].apply(lambda x: merc(x)[0],axis=1)
	df.loc[:, 'merc_y'] = df[['lat','lon']].apply(lambda x: merc(x)[1],axis=1)
	return gpd.GeoDataFrame(df, geometry='geom', crs=crs)


def gdf_from_df_v2(df, coordinate_columns=['lon', 'lat'], crs={'init':'epsg:4326'}):
	'''
		Create a GeoDataFrame from a DataFrame in a much more generalized form.
	'''
	
    df.loc[:, 'geom'] = np.nan
    df.geom = df[coordinate_columns].apply(lambda x: Point(*x), axis=1)
    
    return gpd.GeoDataFrame(df, geometry='geom', crs=crs)


def distance_to_nearest_port(point, ports):
	'''
	Calculates the minimum distance between the point and the lists of ports. Can be used to determine if the ship is sailing or not.
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


# def resample_geospatial(df, features=['lat', 'lon'], rule='60S', method='linear', crs={'init': 'epsg:4326'}, drop_lon_lat=False):
def resample_geospatial(df, features=['lat', 'lon'], rate=1, method='linear', crs={'init': 'epsg:4326'}, drop_lon_lat=False):
    df['datetime'] = pd.to_datetime(df['ts'], unit='s')
    x = df['datetime'].values.astype(np.int64)
    y = df[features].values
    
    # scipy interpolate needs at least 2 records 
    if (len(df) <= 1):
        return df.iloc[0:0]
    
    dt_start = df['datetime'].min().replace(second=0)
    dt_end = df['datetime'].max().replace(second=0)
    
    f = interp1d(x, y, kind=method, axis=0)
	# xnew_V2 = pd.date_range(start=df['datetime'].min().replace(second=0), end=df['datetime'].max().replace(second=0), freq=rule, closed='right')
    xnew_V3 = pd.date_range(start=dt_start.replace(minute=rate*(dt_start.minute//rate)), end=dt_end, freq=f'{rate*60}S', closed='right') 
   
    
    df_RESAMPLED = pd.DataFrame(f(xnew_V3), columns=features)      
    df_RESAMPLED.loc[:, 'datetime'] = xnew_V3
    
    if (len(df_RESAMPLED) == 0):
        df_RESAMPLED.insert(len(df_RESAMPLED.columns), 'geom', '')
    else:
        df_RESAMPLED.loc[:, 'geom'] = df_RESAMPLED[['lon', 'lat']].apply(lambda x: Point(x[0], x[1]), axis=1)

    #drop lat and lon if u like
    if drop_lon_lat:
        df_RESAMPLED = df_RESAMPLED.drop(['lat', 'lon'], axis=1)
        
    return gpd.GeoDataFrame(df_RESAMPLED, crs=crs, geometry='geom')


def calculate_angle(point1, point2):
	'''
		Calculating initial bearing between two points
	'''
	lon1, lat1 = point1[0], point1[1]
	lon2, lat2 = point2[0], point2[1]

	dlat = (lat2 - lat1)
	dlon = (lon2 - lon1)
	numerator = np.sin(dlon) * np.cos(lat2)
	denominator = (
		np.cos(lat1) * np.sin(lat2) -
		(np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
	)

	theta = np.arctan2(numerator, denominator)
	theta_deg = (np.degrees(theta) + 360) % 360
	return theta_deg


def calculate_bearing(gdf):
	'''
	Return given dataframe with an extra bearing column that
	is calculated using the course over ground (in degrees in range [0, 360))
	'''
	# if there is only one point in the trajectory its bearing will be the one measured from the accelerometer
	if len(gdf) == 1:
		gdf['bearing'] = gdf.course
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf['current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf['next_loc'] = gdf.geom.shift(-1)
	gdf = gdf[:-1]
	gdf.loc[:,'next_loc'] = gdf.next_loc.apply(lambda x : (x.x,x.y))
	# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots)
	gdf.loc[:,'bearing'] = gdf[['current_loc', 'next_loc']].apply(lambda x: calculate_angle(x[0], x[1]), axis=1).bfill().ffill()

	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	gdf = gdf.loc[gdf['mmsi'] != 0]
	gdf.dropna(subset=['mmsi', 'geom'], inplace=True)

	return gdf


def calculate_acceleration(gdf):
	'''
	Return given dataframe with an extra acceleration column that
	is calculated using the rate of change of velocity over time.
	'''
	# if there is only one point in the trajectory its acceleration will be zero (i.e. constant speed)
	if len(gdf) == 1:
		gdf['acceleration'] = 0
		return gdf

	gdf['acceleration'] = gdf.velocity.diff().divide(gdf.ts.diff())
	# gdf['acceleration'] = gdf.velocity.diff(-1)

	gdf = gdf.loc[gdf['mmsi'] != 0]
	gdf.dropna(subset=['mmsi', 'geom'], inplace=True)

	return gdf.fillna(0)


def calculate_velocity(gdf):
	'''
	Return given dataframe with an extra velocity column that 
	is calculated using the distance covered in a given amount of time.
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
		gdf = _segment_trajectories_grouped(gdf, pois_alpha=pois_alpha, pois_window=pois_window)
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


def _segment_vessel(vessel, ports, pois_alpha, pois_window, semantic=False):
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
	# gdf['semantic_id'] = np.nan
	# ports = pd.read_pickle('ports.pckl')
	ports = None
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


def resample_and_calculate_velocity(gdf, velocity_drop_alpha=3, rate = 1, res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, n_jobs=1):
	if type(gdf) == pd.core.frame.DataFrame:
		gdf = pd2gdf(gdf)

	gdf['velocity'] = np.nan
	cores = _get_n_jobs(n_jobs)
	if cores==1:
		gdf = _resample_and_calculate_velocity_grouped(gdf, velocity_drop_alpha=velocity_drop_alpha, rate=rate, res_method=res_method, crs=crs, drop_lon_lat=drop_lon_lat, resampling_first=resampling_first, drop_outliers=drop_outliers)
	else:
		#TODO
		gdf = parallelize_dataframe(gdf, _resample_and_calculate_velocity_grouped)
	return gdf


def _resample_and_calculate_velocity_grouped(gdf, velocity_drop_alpha=3, rate = 1, res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False):
	gdf = gdf.groupby(['mmsi'], group_keys=False).apply(_resample_and_calculate_velocity_vessel, velocity_drop_alpha, rate, res_method, crs, drop_lon_lat, resampling_first, drop_outliers).reset_index(drop=True)
	gdf.reset_index(inplace=True, drop=True)
	return gdf


def _resample_and_calculate_velocity_vessel(vessel, velocity_drop_alpha, rate, res_method, crs , drop_lon_lat, resampling_first, drop_outliers):
	if len(vessel) == 1 :
		vessel.velocity = vessel.speed
		return vessel
	if resampling_first:
		vessel = resample_geospatial(vessel, rate=rate, method=res_method, crs = crs, drop_lon_lat = drop_lon_lat)
		vessel = calculate_velocity(vessel)
	else:
		vessel = calculate_velocity(vessel)
		vessel = resample_geospatial(vessel, rate=rate, method=res_method, crs = crs, drop_lon_lat = drop_lon_lat)
	if drop_outliers:
		vessel = vessel.drop(get_outliers(vessel.velocity, alpha=velocity_drop_alpha)[0], axis=0)
	# vessel.reset_index(inplace=True, drop=True)
	return vessel


def clean_gdf(gdf):
	gdf.drop_duplicates(['ts', 'mmsi'], inplace=True)
	gdf.drop([item for sublist in [x for x in gdf.groupby(['mmsi'], group_keys=False)['ts'].apply(lambda x: get_outliers(x)[0]) if x != []] for item in sublist], axis=0, inplace=True)
	gdf.reset_index(inplace=True, drop=True)
	# gdf.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')
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


def _pipeline_apply(vessel, ports, velocity_drop_alpha=3, rate = 1, res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, pois_alpha=-1, pois_window=100, semantic=False ):
	'''
	Full automated pipeline. To be used on a single mmsi, either manually, or using .groupby
	Rate: Resampling rate in minutes
	'''
	vessel.drop_duplicates(['ts'], inplace=True)
	vessel.sort_values('ts', inplace=True)
	vessel.reset_index(inplace=True, drop=True)
	vessel.drop(['id', 'status'], axis=1, inplace=True, errors='ignore')
	vessel['geom'] = vessel[['lon', 'lat']].apply(lambda x: Point(x[0],x[1]), axis=1)
	vessel=  gpd.GeoDataFrame(vessel, geometry='geom')
	vessel = _resample_and_calculate_velocity_vessel(vessel, velocity_drop_alpha, rate, res_method, crs, drop_lon_lat, resampling_first, drop_outliers)
	vessel = _segment_vessel(vessel, ports, pois_alpha, pois_window, semantic)
	return vessel


def resample_and_segment(vessel, ports, pre_segment_threshold=12, velocity_drop_alpha=3, res_rule = '60S', res_method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False, resampling_first=True, drop_outliers=False, pois_alpha=-1, pois_window=100, semantic=False):
	'''
	Function that calls _pipeline_apply.
	pre_segment_threshold is the number of hours that a transmitter needs to be off in order to brake up a trajectory into usefull ones. If 0 then do not pre segment
	'''
	if pre_segment_threshold != 0:
		# split vessel into segments that correspond to a specific fishing trip
		brake_points = vessel.ts.diff(-1).abs().index[vessel.ts.diff()>60*60*pre_segment_threshold]
		try:
			dfs = np.split(vessel, brake_points)
		except ValueError:
			return vessel, None
		# apply pipeline to each segment and concatenate
		dfs_prepd = [_pipeline_apply(df, ports, velocity_drop_alpha, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers, pois_alpha, pois_window, semantic) for df in dfs if len(df)>1]
		####### exp #######
		for i in range(1,len(dfs_prepd)):
			dfs_prepd[i].loc[:,'traj_id'] = dfs_prepd[i].traj_id.apply(lambda x: x+dfs_prepd[i-1].traj_id.max()+1)
		df_fn = pd.concat(dfs_prepd)
		df_fn.sort_values('ts', inplace=True)
		df_fn.reset_index(inplace=True, drop=True)
	else:
		df_fn = _pipeline_apply(vessel, ports, velocity_drop_alpha, res_rule, res_method, crs, drop_lon_lat, resampling_first, drop_outliers, pois_alpha, pois_window, semantic)
	return df_fn, brake_points


def create_port_bounds(ports, epsg=2154, port_radius=2000):
	'''
	Given some Datapoints, create a circular bound of _port_radius_ kilometers.
	'''
	ports2 = ports.copy()
	init_crs = ports2.crs
	# We convert to a CRS where the distance between two points is returned in meters (e.g. EPSG-2154 (France), EPSG-3310 (North America)),
	# so the buffer function creates a circle with radius _port_radius_ meters from the center point (i.e the port's location point)
	ports2.loc[:, 'geom'] = ports2.geom.to_crs(epsg=epsg).buffer(port_radius).to_crs(init_crs)
	# After we create the ports bounding circle we convert back to its previous CRS.
	return ports2


# def segment_trajectories_v2(vessel, ports, port_radius=2000, port_epsg=2154, cardinality_threshold=2):
def segment_trajectories_v2(vessel, ports, port_radius=2000, port_epsg=2154):
	'''
	Segment trajectories based on port entrance/exit
	'''
	sindex = vessel.sindex # create the spatial index (r-tree) of the vessel's data points

	if (ports.geom.type == 'Point').all():
		ports = create_port_bounds(ports, port_radius=port_radius, epsg=port_epsg)

	# find the points that intersect with each subpolygon and add them to _points_within_geometry_ DataFrame
	points_within_geometry = pd.DataFrame()
	for poly in ports.geom:
		# find approximate matches with r-tree, then precise matches from those approximate ones
		possible_matches_index = list(sindex.intersection(poly.bounds))
		possible_matches = vessel.iloc[possible_matches_index]
		precise_matches = possible_matches[possible_matches.intersects(poly)]
		points_within_geometry = points_within_geometry.append(precise_matches)
		
	points_within_geometry = points_within_geometry.drop_duplicates(subset=['mmsi', 'ts'])
	points_outside_geometry = vessel[~vessel.isin(points_within_geometry)].dropna(how='all')

	vessel.loc[:,'traj_id'] = np.nan
	# When we create the _traj_id_ column, we label each record with 0, 
	# if it's outside the port's radius and -1 if it's inside the port's radius. 
	vessel.loc[vessel.index.isin(points_within_geometry.index), 'traj_id'] = -1
	vessel.loc[vessel.index.isin(points_outside_geometry.index), 'traj_id'] = 0
	vessel.loc[:,'label'] = vessel['traj_id'].values
	
	# we drop the consecutive -1 rows, except the first and last one, and segment the trajectory by the remaining -1 points
	vessel = vessel.loc[vessel.traj_id[vessel.traj_id.replace(-1,np.nan).ffill(limit=1).bfill(limit=1).notnull()].index]
	vessel.reset_index(inplace=True, drop=True)
	dfs = np.split(vessel, vessel.loc[vessel.traj_id == -1].index)
	# dfs = [df for df in dfs if len(df) > 0]    # remove the fragments that are empty
	# print (f'@Port-Segmentation BEFORE FILTERING: {[len(tmp_df) for tmp_df in dfs]}')
	dfs = [df for df in dfs if len(df) != 0]    # remove the fragments that have at most 1 point
	# print (f'@Port-Segmentation AFTER FILTERING: {[len(tmp_df) for tmp_df in dfs]}')
	
	if (len(dfs) == 0):
		# return gpd.GeoDataFrame([], columns=['mmsi', 'speed', 'lon', 'lat', 'ts', 'geom', 'traj_id', 'traj_id_12h_gap'], geometry='geom', crs={'init':'epsg:4326'}) 
		return vessel.iloc[0:0]

	dfs[0].loc[:,'traj_id'] = 0    # ensure that the points in the first segments have the starting ID (0)
	# then for each sub-trajectory, we assign an incrementing number (id) to each trajectory segment, starting from 0 
	# for i in range(1,len(dfs)):
	# 	if (len(dfs[i]) == 1):
	# 		dfs[i].loc[:,'traj_id'] = dfs[i].traj_id.apply(lambda x: dfs[i-1].traj_id.max())
	# 	else:
	# 		dfs[i].loc[:,'traj_id'] = dfs[i].traj_id.apply(lambda x: x+dfs[i-1].traj_id.max()+1)				

	for i in range(1,len(dfs)):        
			if (len(dfs[i]) == 1):
					dfs[i].loc[:,'traj_id'] = dfs[i-1].traj_id.max()
			else:
					dfs[i].loc[:,'traj_id'] = dfs[i-1].traj_id.max()+1

	df_fn = pd.concat(dfs)
	df_fn.sort_values('ts', inplace=True)
	df_fn.reset_index(inplace=True, drop=True)

	return df_fn


def __temporal_segment(vessel, temporal_threshold=12, cardinality_threshold=2):
	if len(vessel) == 0:
		# return [gpd.GeoDataFrame([], columns=['mmsi', 'speed', 'lon', 'lat', 'ts', 'geom', 'traj_id', 'traj_id_12h_gap'], geometry='geom', crs={'init':'epsg:4326'})]
		vessel['traj_id_12h_gap'] = None
		return [vessel.iloc[0:0]]

	print(f"Vessel mmsi:{vessel.mmsi.unique()[0]}")
	print(f"Segments Before: {len(vessel.traj_id.unique())}")
	vessel['traj_id_12h_gap'] = 0
	vessel.sort_values(['ts'], inplace=True)
	temporal_threshold = 12 # in hrs

	dfs_temporal = []

	for traj_id, sdf in vessel.groupby('traj_id'):
			df = sdf.reset_index()
			break_points = df.ts.diff(-1).abs().index[df.ts.diff()>60*60*temporal_threshold]
			
			dfs = np.split(df, break_points)
			dfs_temporal.extend(dfs)
			#NOTE #1: Check np.split if break_points=[], returns traj
	
	# print (f'@Temporal-Segmentation BEFORE FILTERING: {[len(tmp_df) for tmp_df in dfs_temporal]}')
	dfs_temporal = [tmp_df for tmp_df in dfs_temporal if len(tmp_df) >= cardinality_threshold]
	# print (f'@Temporal-Segmentation AFTER FILTERING: {[len(tmp_df) for tmp_df in dfs_temporal]}')
	print(f"Segments After: {len(dfs_temporal)}")
	
	if (len(dfs_temporal) == 0):
		# return [gpd.GeoDataFrame([], columns=['mmsi', 'speed', 'lon', 'lat', 'ts', 'geom', 'traj_id', 'traj_id_12h_gap'], geometry='geom', crs={'init':'epsg:4326'})]
		return [vessel.iloc[0:0]]
		
	dfs_temporal[0].loc[:,'traj_id_12h_gap'] = 0
	for idx in range(1, len(dfs_temporal)):
		dfs_temporal[idx].loc[:,'traj_id_12h_gap'] = dfs_temporal[idx].traj_id_12h_gap.apply(lambda x: x+dfs_temporal[idx-1].traj_id_12h_gap.max()+1)

	return dfs_temporal


def segment_resample_v2(vessel, ports, port_epsg=2154, port_radius=2000, temporal_threshold=12, cardinality_threshold=2, resample_trips=False, rate = 1, method='linear', crs = {'init': 'epsg:4326'}, drop_lon_lat = False):                                               
	'''
	After the Segmentation Stage, for each sub-trajectory:
	  * we resample each trajectory
	  * calculate the velocity (per-point)
	  * we use our implementation on trajectory segmentation
		in order to add tags regarding the vessel's activity
	'''
	port_bounds = create_port_bounds(ports, epsg=port_epsg, port_radius=port_radius)
	# port_segmented_trajectories = segment_trajectories_v2(vessel, port_bounds, cardinality_threshold=cardinality_threshold)
	port_segmented_trajectories = segment_trajectories_v2(vessel, port_bounds)
	temporal_segmented_trajectories = __temporal_segment(port_segmented_trajectories, temporal_threshold=temporal_threshold, cardinality_threshold=cardinality_threshold)

	if (resample_trips):
		for idx in range(0, len(temporal_segmented_trajectories)):
			if len(temporal_segmented_trajectories[idx]) == 0:
				continue
			# print (f'@Temporal-Segmentation BEFORE RESAMPLING: {len(temporal_segmented_trajectories[idx])}')
			# print (temporal_segmented_trajectories[idx].ts.diff().values)
			temporal_segmented_trajectories[idx] = resample_geospatial(temporal_segmented_trajectories[idx], rate=rate, method=method, crs=crs, drop_lon_lat=drop_lon_lat)
			# print (f'@Temporal-Segmentation AFTER RESAMPLING: {len(temporal_segmented_trajectories[idx])}')
			# temporal_segmented_trajectories[idx] = calculate_velocity(temporal_segmented_trajectories[idx])

	# tmp = []
	# for traj_id in vessel_fn.traj_id.unique():
	# 	sub_traj = vessel_fn.loc[vessel_fn.traj_id == traj_id]
	# 	# TODO: Adjust _segment_vessel function to make use of the status codes (as presented in thesis-tasks doc) as well as the sub_traj_id column
	# 	# sub_traj_tagged = _segment_vessel(sub_traj.copy(), None, pois_alpha=pois_alpha, pois_window=pois_window, semantic=semantic)
	# 	# sub_traj['sub_traj_id'] = sub_traj_tagged.traj_id.values
	# 	tmp.append(sub_traj)

	vessel_fn = pd.concat(temporal_segmented_trajectories, ignore_index=True)
	vessel_fn.sort_values('ts', inplace=True)
	vessel_fn.reset_index(inplace=True, drop=True)
	#vessel_fn.drop(['index'], axis=1, inplace=True)
	return vessel_fn
