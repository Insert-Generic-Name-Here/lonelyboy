import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from shapely.geometry import Point, LineString, shape
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
from collections import Counter
import math


def haversine(p_1, p_2):
	'''
		Return the Haversine Distance of two points in KM
	'''
	lon1, lat1, lon2, lat2 = map(np.deg2rad, [p_1[0], p_1[1], p_2[0], p_2[1]])   
	
	dlon = lon2 - lon1
	dlat = lat2 - lat1    
	a = np.power(np.sin(dlat * 0.5), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon * 0.5), 2)    
	
	return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def getGeoDataFrame_v2(df, coordinate_columns=['lon', 'lat'], crs={'init':'epsg:4326'}):
	'''
		Create a GeoDataFrame from a DataFrame in a much more generalized form.
	'''
	
	df.loc[:, 'geom'] = np.nan
	df.geom = df[coordinate_columns].apply(lambda x: Point(*x), axis=1)
	
	return gpd.GeoDataFrame(df, geometry='geom', crs=crs)



def calculate_acceleration(gdf, spd_column='velocity', ts_column='ts'):
	'''
	Return given dataframe with an extra acceleration column that
	is calculated using the rate of change of velocity over time.
	'''
	# if there is only one point in the trajectory its acceleration will be zero (i.e. constant speed)
	if len(gdf) == 1:
		gdf.loc[:, 'acceleration'] = 0
		return gdf
	
	gdf.loc[:, 'acceleration'] = gdf[spd_column].diff(-1).divide(gdf[ts_column].diff(-1).abs())
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf


def calculate_velocity(gdf, spd_column='velocity', ts_column='ts'):
	'''
	Return given dataframe with an extra velocity column that 
	is calculated using the distance covered in a given amount of time.
	TODO - use the get distance method to save some space
	'''
	# if there is only one point in the trajectory its velocity will be the one measured from the speedometer
	if len(gdf) == 1:
		gdf.loc[:, 'velocity'] = gdf[spd_column]
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf.loc[:, 'current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf.loc[:, 'next_loc'] = gdf.geom.shift(-1)
	gdf.loc[:, 'dt'] = gdf[ts_column].diff(-1).abs()
	
	gdf = gdf.iloc[:-1]
	gdf.next_loc = gdf.next_loc.apply(lambda x : (x.x,x.y)) 
		
	# get the distance traveled in n-miles and multiply by the rate given (3600/secs for knots)
	gdf.loc[:,'velocity'] = gdf[['current_loc', 'next_loc']].apply(lambda x : haversine(x[0], x[1])*0.539956803 , axis=1).multiply(3600/gdf.dt)

	gdf.drop(['current_loc', 'next_loc', 'dt'], axis=1, inplace=True)
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf


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
		gdf.loc[:, 'bearing'] = gdf.course
		return gdf

	# create columns for current and next location. Drop the last columns that contains the nan value
	gdf.loc[:, 'current_loc'] = gdf.geom.apply(lambda x: (x.x,x.y))
	gdf.loc[:, 'next_loc'] = gdf.geom.shift(-1)
	gdf = gdf.iloc[:-1]
	
	gdf.next_loc = gdf.next_loc.apply(lambda x : (x.x,x.y))
	
	gdf.loc[:,'bearing'] = gdf[['current_loc', 'next_loc']].apply(lambda x: calculate_angle(x[0], x[1]), axis=1)

	gdf.drop(['current_loc', 'next_loc'], axis=1, inplace=True)
	gdf.dropna(subset=['geom'], inplace=True)
	
	return gdf  



def create_area_bounds(spatial_areas, epsg=2154, area_radius=2000):
	'''
	Given some Datapoints, create a circular bound of _area_radius_ kilometers.
	'''
	spatial_areas2 = spatial_areas.copy()
	init_crs = spatial_areas2.crs
	# We convert to a CRS where the distance between two points is returned in meters (e.g. EPSG-2154 (France), EPSG-3310 (North America)),
	# so the buffer function creates a circle with radius _area_radius_ meters from the center point (i.e the port's location point)
	spatial_areas2.loc[:, 'geom'] = spatial_areas2.geom.to_crs(epsg=epsg).buffer(area_radius).to_crs(init_crs)
	# After we create the spatial_areas bounding circle we convert back to its previous CRS.
	return spatial_areas2


def __fix_traj_ids__(traj_sgdf):
    traj_sgdf.reset_index(inplace=True, drop=True)
    dfs = np.split(traj_sgdf, traj_sgdf.loc[traj_sgdf.traj_id == -1].index)
    # print (f'@Port-Segmentation BEFORE FILTERING: {[len(tmp_df) for tmp_df in dfs]}')
    dfs = [df for df in dfs if len(df) != 0]    # remove the fragments that have at most 1 point
    # print (f'@Port-Segmentation AFTER FILTERING: {[len(tmp_df) for tmp_df in dfs]}')

    if (len(dfs) == 0):
        return traj_sgdf.iloc[0:0]

    dfs[0].loc[:,'traj_id'] = 0    # ensure that the points in the first segments have the starting ID (0)
    # then for each sub-trajectory, we assign an incrementing number (id) to each trajectory segment, starting from 0 
    for i in range(1,len(dfs)):        
            if (len(dfs[i]) == 1):
                    dfs[i].loc[:,'traj_id'] = dfs[i-1].loc[:, 'traj_id'].max()
            else:
                    dfs[i].loc[:,'traj_id'] = dfs[i-1].loc[:, 'traj_id'].max()+1

    return pd.concat(dfs)



def classify_area_proximity(trajectories, spatial_areas, o_id_column='id', ts_column='t_msec', area_radius=2000, area_epsg=2154):
    # create the spatial index (r-tree) of the trajectories's data points
    sindex = trajectories.sindex 
  
    # find the points that intersect with each subpolygon and add them to _points_within_geometry_ DataFrame
    points_within_geometry = pd.DataFrame()
    
    if (spatial_areas.geom.type == 'Point').all():
        spatial_areas = create_area_bounds(spatial_areas, area_radius=area_radius, epsg=area_epsg)
    
    for poly in tqdm(spatial_areas.itertuples(), desc='Classifying Spatial Proximity'):
    # for poly in spatial_areas.itertuples():
        # find approximate matches with r-tree, then precise matches from those approximate ones
        possible_matches_index = list(sindex.intersection(poly.geom.bounds))
        possible_matches = trajectories.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly.geom)]
        
        if (precise_matches.empty):
            continue
            
        trajectories.loc[precise_matches.index, 'area_id'] = poly.Index
        points_within_geometry = points_within_geometry.append(precise_matches)
        
    points_within_geometry = points_within_geometry.drop_duplicates(subset=[o_id_column, ts_column])
    points_outside_geometry = trajectories[~trajectories.isin(points_within_geometry)].dropna(how='all')

    # When we create the _traj_id_ column, we label each record with 0, 
    # if it's outside the port's radius and -1 if it's inside the port's radius. 
    trajectories.loc[trajectories.index.isin(points_within_geometry.index), 'traj_id'] = -1
    trajectories.loc[trajectories.index.isin(points_outside_geometry.index), 'traj_id'] = 0
    trajectories.loc[:,'label'] = trajectories.loc[:, 'traj_id'].values
    
    return trajectories



def spatial_segmentation(trajectories, spatial_areas, o_id_column='id', ts_column='t_msec', classify_points=True, area_radius=2000, area_epsg=2154):
    '''
    Segment trajectories based on area proximity
    '''
    
    if classify_points:
        classify_area_proximity(trajectories, spatial_areas, o_id_column=o_id_column, ts_column=ts_column, area_radius=area_radius, area_epsg=area_epsg)
    
    tqdm.pandas()
    # We drop the consecutive -1 rows, except the first and last one, and segment the trajectory by the remaining -1 points
    #     trajectories = trajectories.loc[trajectories.traj_id[trajectories.traj_id.replace(-1,np.nan).ffill(limit=1).bfill(limit=1).notnull()].index]
    trajectories = trajectories.groupby(o_id_column, group_keys=False).progress_apply(lambda gdf: gdf.loc[gdf.traj_id[gdf.traj_id.replace(-1,np.nan).ffill(limit=1).bfill(limit=1).notnull()].index])

    df_fn = trajectories.groupby(o_id_column).progress_apply(__fix_traj_ids__)
    df_fn.sort_values(ts_column, inplace=True)
    df_fn.reset_index(inplace=True, drop=True)

    return df_fn