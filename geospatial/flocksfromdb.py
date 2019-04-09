import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Documents/Insert-Generic-Name-Here/'))
# sys.path

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns_v2 as gsgp

import psycopg2
import numpy as np
import configparser
import pandas as pd
import geopandas as gpd
import contextily as ctx
from random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from shapely.geometry import Point, LineString, shape
from haversine import haversine
from datetime import datetime, timedelta


from multiprocessing import cpu_count, Pool
from functools import partial
import datetime


num_partitions=5

properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']
print('loading data')

# traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min '


# gsgp.group_patterns(df = traj, mode = 'convoys', min_diameter=1852, min_cardinality=10, time_threshold=30, save_result=True)

# gsgp.group_patterns(df = traj, mode = 'flocks', min_diameter=3000, min_cardinality=2, time_threshold=30, save_result=True)




if num_partitions!=1:

	save_name = f'tmptmp.csv'
	dt_sql = 'SELECT datetime FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE ts>1456802710 AND ts<1457575510'

	con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

	# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
	datet = pd.read_sql_query(dt_sql,con=con)
	total = datet.datetime.nunique()

	# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
	# ports.geom = ports.geom.apply(lambda x: x[0])

	con.close()


	parts = pd.cut(datet.datetime,num_partitions+1, retbins=True)[1]

	print(f'No. of partitions -> {len(parts)-1}')
	datet = None
	for i in range(num_partitions):
		if i ==0:
			mined_patterns = None
			start = 0
			last_ts = None
		traj_sql = f"SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE datetime>'{str(parts[i])}' AND datetime<='{str(parts[i+1])}'"

		print(f'Loading Partition #{i+1}')
		con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

		# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
		traj = pd.read_sql_query(traj_sql,con=con)


		# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
		# ports.geom = ports.geom.apply(lambda x: x[0])

		con.close()

		print(f'Starting Partition #{i+1} ---- {traj.datetime.max()}, {traj.datetime.min()}')

		mined_patterns, start, last_ts = gsgp.mine_patterns(df = traj, mode = 'flocks', min_diameter=3000, min_cardinality=2, time_threshold=30, checkpoints=False, checkpoints_freq=0.1, total=total, start=start, last_ts=last_ts, mined_patterns=mined_patterns)

	print('Saving Result...')
	mined_patterns.to_csv(save_name, index=False)


else:
	traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE ts>1456802710 AND ts<1457575510'

	con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

	# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
	traj = pd.read_sql_query(traj_sql,con=con)

	# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
	# ports.geom = ports.geom.apply(lambda x: x[0])

	con.close()

	print('done, len -> ', len(traj))

	gsgp.group_patterns(df = traj, mode = 'flocks', min_diameter=3000, min_cardinality=2, time_threshold=30, save_result=True)
