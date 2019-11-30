import geopandas as gpd
import pandas as pd
import numpy as np
import datetime

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents/Insert-Generic-Name-Here/'))
# sys.path
from scipy import interpolate

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns_v2 as gsgp

from collections import deque
from tqdm import tqdm



def get_line_from_vector(vct_start, heading):
	# h = np.deg2rad(heading)
	
	sin_h = np.around( np.sin(np.deg2rad(heading)), decimals=7)
	cos_h = np.around( np.cos(np.deg2rad(heading)), decimals=7)
	
	if cos_h != 0:
		line_slope = sin_h/(cos_h)
		line_interception = (vct_start[1]*cos_h - vct_start[0]*sin_h)/(cos_h)
	else: 
		line_slope = np.inf
		line_interception = np.inf

	return np.around(line_slope, decimals=2), np.around(line_interception, decimals=2)


def get_vectors_intersecting_point(vct1, heading1, vct2, heading2):
	m_1, b_1 = get_line_from_vector(vct1, heading1)
	m_2, b_2 = get_line_from_vector(vct2, heading2)
	
	#     print (f'Slope: {m_1}\t Interception: {b_1}')
	#     print (f'Slope: {m_2}\t Interception: {b_2}')
	
	if (m_1 == m_2):
		return np.mean(np.array([vct1, vct2]), axis=0)
	elif (m_1 == np.inf):
		X = vct1[0]
		Y = m_2*X + b_2
		return np.around(np.array([X, Y]), decimals=2)
	elif (m_2 == np.inf):
		X = vct2[0]
		Y = m_1*X + b_1
		return np.around(np.array([X, Y]), decimals=2)
	else:
		X = (b_2 - b_1)/(m_1 - m_2)
		Y = m_1*X + b_1
		return np.around(np.array([X, Y]), decimals=2)


def quadraticBezier(initialPoint, controlPoint, finalPoint, t):
	'''
		Quadratic (n=2) Bezier Interpolation 
		t : The timestamp that we want to interpolate to expressed in curve percentage - i.e. the timedelta ratio of the two points of a trajectory (t - t_1)/(t_2 - t_1)
	'''
	interp_point = []
	
	tmp = np.vstack((initialPoint, controlPoint, finalPoint))
	for dim in tmp.T:
		res = (np.power(1-t, 2) * dim[0]) + ((1-t) * 2 * t * dim[1]) + (np.power(t, 2) * dim[2])
		interp_point.append(res)
	
	return np.array(interp_point)


def mround(x, base=5):
	return int(base * np.around(np.float(x)//base))


def extrap1d_ts(vessel_window, interp_ts):
	tmp = np.array(vessel_window)
	try:
		f = interpolate.interp1d(tmp[:,0], tmp[:,1:], kind='linear', fill_value='extrapolate', axis=0)
		return f(interp_ts)
	except ValueError:
		return None


def interp1d_ts(vessel_window, interp_ts):
	tmp = np.array(vessel_window)
	try:
		f = interpolate.interp1d(tmp[:,0], tmp[:,1:], kind='linear', axis=0)
		return f(interp_ts)
	except ValueError:
		return None


def interp1d_quadratic_bezier(window, timestamp, interpolate=True):
	# Step 0: Getting (via the timedelta ratio) the point of the curve to interpolate
	timedelta_ratio = (timestamp - window[0]['ts'])/(window[1]['ts'] - window[0]['ts'])
	
	# Prevent Extrapolation if the mode is set to interpolate
	if interpolate and timedelta_ratio > 1:
		return None    
		
	### Step 1: Getting Vector Info for Point A
	vct1 = np.array(window[0][['lon', 'lat', 'heading']])
	# m_1, b_1 = get_line_from_vector(vct1[:-1], vct1[-1])
	# print (f'Slope: {m_1}\t Interception: {b_1}')

	### Step 1.5: Getting Vector Info for Point A
	vct2 = np.array(window[1][['lon', 'lat', 'heading']])
	# m_2, b_2 = get_line_from_vector(vct2[:-1], vct2[-1])
	# print (f'Slope: {m_2}\t Interception: {b_2}')

	### Step 2: Getting the Control Point (the vectors' intersection)
	vctC = get_vectors_intersecting_point(vct1[:-1], vct1[-1], vct2[:-1], vct2[-1])
	# print (f'Intersecting Point: {(vctC[0], vctC[1])}')
	
	# vctC_enriched = np.concatenate(([window[0]], [window[1]]), axis=0).mean(axis=0)
	vctC_enriched = pd.DataFrame([np.concatenate(([window[0]], [window[1]]), axis=0).mean(axis=0)], columns=window[0].index.tolist())
	vctC_enriched.loc[:, 'lon'] = vctC[0]
	vctC_enriched.loc[:, 'lat'] = vctC[1]
	vctC_enriched = vctC_enriched.values.flatten()
	# print (f'Intersecting Point (enriched): {(vctC[0], vctC[1])}')
	
	### Step 3: Interpolate (via Quadratic Bezier Curves) and return the result
	interp1d_res = pd.DataFrame([quadraticBezier(window[0].values, vctC_enriched, window[1].values, timedelta_ratio)], columns=window[0].index.tolist())
	interp1d_res.loc[:,'mmsi'] = window[0].mmsi
	interp1d_res.loc[:,'ts'] = timestamp
	interp1d_res.loc[:,'traj_id'] = window[0].traj_id
	interp1d_res.loc[:,'trip_id'] = window[0].trip_id
	
	return interp1d_res.values.flatten()   


def align_points(data_points, timestamp, mode, extrapolation_counter, extrapolation_limit=3):
	# aligned_pnt = None
	if mode=='extrap1d' and extrapolation_counter < extrapolation_limit:
		return extrap1d_ts(data_points, timestamp)

	elif mode=='interp1d':
		return interp1d_ts(data_points, timestamp)

	elif mode=='bezierExtrap1d' and extrapolation_counter < extrapolation_limit:
		return interp1d_quadratic_bezier(data_points, timestamp, interpolate=False)

	elif mode=='bezierIinterp1d':
		return interp1d_quadratic_bezier(data_points, timestamp, interpolate=True)
	# return aligned_pnt




def delayed_online_interpolation(df, rate=5, mode='interp1d', extrapolation_limit=3):
	'''
		mode = {interp1d, bezierIinterp1d}
	'''
	dt_const = rate*60                                        # Variable 1 -- Output Interval (in UNIX timestamp unit -- Seconds)

	dt_start = df['ts'].min()                                 # Variable 2 -- Starting Timestamp - For Emulation Purposes
	pending_timestamp = mround(df['ts'].min(), base=dt_const) # Variable 3 -- Pending Timestamp - Timestamp to Interpolate

	vessel_windows = {}                                       # Variable 4 -- Object Sliding Windows
	df_aligned_points_interp1d = np.empty((0, len(df.columns)))
	# n = 1 # FOR DEBUG -- COUNT THE CANDIDATE TIMESTAMPS FOR INTERPOLATION


	for i in tqdm(np.arange(dt_start, df['ts'].max(), 1)):    
		for mmsi, data_point in df.loc[df.ts == i].groupby(by='mmsi', group_keys=False):    # Iterating for each object
			if (mmsi not in vessel_windows): vessel_windows[mmsi] = deque(maxlen=2)         # Create a sliding window (of length 2) for each object, if it doesn't exist.
			vessel_windows[mmsi].append(np.concatenate([[i], data_point.values[0]]))        # Append the timestamp and the signal value to the object's window, if it isn't NaN

			if len(vessel_windows[mmsi]) >= 2:
				interp1d_val = align_points(vessel_windows[mmsi], pending_timestamp, mode, np.inf, np.inf)
				if interp1d_val is not None:
					df_aligned_points_interp1d = np.append(df_aligned_points_interp1d, [interp1d_val], axis=0)
				else:
					continue
		
		if (i % dt_const == 0 and i > pending_timestamp):  # Output Time
			# print (df_aligned_points_interp1d[df_aligned_points_interp1d[:,-1] == pending_timestamp])           # Let's say we output the data points
			pending_timestamp = i
			# n+=1

	return pd.DataFrame(df_aligned_points_interp1d, columns=df.columns).drop_duplicates(subset=['mmsi', 'ts'], keep='last')



def online_alignment_v3(df, rate=5, mode='extrap1d', extrapolation_limit=3):
	'''
		mode = {extrap1d, interp1d}
	'''
	dt_const = rate*60                                        # Variable 1 -- Output Interval (in UNIX timestamp unit -- Seconds)
	dt_start = df['ts'].min()                                 # Variable 2 -- Starting Timestamp - For Emulation Purposes
	pending_timestamp = mround(dt_start, base=dt_const)       # Variable 3 -- Pending Timestamp - Timestamp to Interpolate

	vessel_windows = {}                                       # Variable 4 -- Object Sliding Windows
	extrap1d_counter = {}                                     # Variable 5 -- Object Extrapolator Counter (Limits the number of extrapolations/Resets when the window changes)
	
	df_aligned_points_interp1d = np.empty((0, len(df.columns)))


	for i in tqdm(np.arange(dt_start, df['ts'].max(), 1)):    
		for mmsi, data_point in df.loc[df.ts == i].groupby(by='mmsi', group_keys=False):    # Iterating for each object
			if (mmsi not in vessel_windows): vessel_windows[mmsi] = deque(maxlen=2)         # Create a sliding window (of length 2) for each object, if it doesn't exist.
		
			if mode == 'extrap1d':
				vessel_windows[mmsi].append(np.concatenate([[i], data_point.values[0]]))    # Append the timestamp and the signal value to the object's window, if it isn't NaN
				extrap1d_counter[mmsi]=0
			elif mode == 'interp1d':
				try:
					if not (pending_timestamp <= vessel_windows[mmsi][-1][0]):
						vessel_windows[mmsi].append(np.concatenate([[i], data_point.values[0]]))        # Append the timestamp and the signal value to the object's window, if it isn't NaN
				except IndexError:
						vessel_windows[mmsi].append(np.concatenate([[i], data_point.values[0]]))

		if (i % dt_const == 0): 
			for mmsi, window in vessel_windows.items():
				if mode == 'extrap1d':
					interp1d_val = align_points(window, i, mode, extrap1d_counter[mmsi], extrapolation_limit)
				elif mode == 'interp1d' and i > pending_timestamp:
					interp1d_val = align_points(window, pending_timestamp, mode, np.inf, np.inf)
				else:
					break

				if interp1d_val is not None:
					df_aligned_points_interp1d = np.append(df_aligned_points_interp1d, [interp1d_val], axis=0)	
					if (mode == 'extrap1d'): extrap1d_counter[mmsi] += 1					

			pending_timestamp = i

	return pd.DataFrame(df_aligned_points_interp1d, columns=df.columns).drop_duplicates(subset=['mmsi', 'ts'], keep='last')


# if __name__ == "__main__":
# 	df = pd.read_csv('dynamic_ships_segmented_12h_min_trip_card_3.csv')
# 	# df = toy_df.loc[toy_df.ts.apply(datetime.datetime.fromtimestamp).dt.date == datetime.date(2015, 10, 6)]

# 	print (df.mmsi.nunique())

# 	df_extrap1d = online_alignment_v3(df, mode='extrap1d')
# 	print (f'EXTRAPOLATED {len(df_extrap1d)} DATAPOINTS')

# 	df_extrap1d = online_alignment_v3(df, mode='interp1d')
# 	print (f'INTERPOLATED {len(df_extrap1d)} DATAPOINTS')

# 	tmp = set(df_extrap1d.ts.unique())

# 	df_interp1d = delayed_online_interpolation(df)
# 	print (f'INTERPOLATED {len(df_interp1d)} DATAPOINTS')

# 	tmp2 = set(df_interp1d.ts.unique())
# 	print (tmp2 - tmp)