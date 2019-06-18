import os, sys
sys.path.append(os.path.join(os.path.expanduser('~')))
# sys.path

import plots as gsplt
import preprocessing as gspp
import group_patterns_v3 as lbgp

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

#### PARAMS ####
gp_type = 'flocks'
cardinality = 5
dt = 20
distance = 926
num_partitions=3
print(f'Discovering {gp_type} with card={cardinality}, dt={dt} and distance={distance}')
################

properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = 'localhost'
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

# traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min '


# gsgp.group_patterns(df = traj, mode = 'convoys', min_diameter=1852, min_cardinality=10, time_threshold=30, save_result=True)

# gsgp.group_patterns(df = traj, mode = 'flocks', min_diameter=3000, min_cardinality=2, time_threshold=30, save_result=True)


if num_partitions!=1:

	save_name = f'{gp_type}_card_{cardinality}_dt_{dt}_dist_{distance}.csv'
	dt_sql = 'SELECT datetime FROM ais_data.dynamic_ships_min_trip_card_3_segmented_12h_resampled_1min_v2'

	con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)
	print('loading data')

	# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
	datet = pd.read_sql_query(dt_sql,con=con)
	total = datet.datetime.nunique()

	# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
	# ports.geom = ports.geom.apply(lambda x: x[0])

	con.close()


	parts = pd.cut(datet.datetime,num_partitions, retbins=True)[1]

	print(f'No. of partitions -> {len(parts)-1}')
	datet = None
	for i in range(num_partitions):
		if i ==0:
			active = pd.DataFrame()
			closed_patterns = pd.DataFrame()
			start = 0
		traj_sql = f"SELECT * FROM ais_data.dynamic_ships_min_trip_card_3_segmented_12h_resampled_1min_v2 WHERE datetime>'{str(parts[i])}' AND datetime<='{str(parts[i+1])}'"

		print(f'Loading Partition #{i+1}')
		con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

		# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
		traj = pd.read_sql_query(traj_sql,con=con)


		# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
		# ports.geom = ports.geom.apply(lambda x: x[0])

		con.close()

		print(f'Starting Partition #{i+1} ---- {traj.datetime.max()}, {traj.datetime.min()}')
		
		active, closed_patterns, start = lbgp.mine_patterns(traj, gp_type, min_diameter=distance, min_cardinality=cardinality, time_threshold=dt, active=active, closed_patterns=closed_patterns, total=total, start=start, disable_progress_bar=False)

	print('Saving Result...')
	pd.concat([closed_patterns, active]).to_csv(save_name, index=False)


else:
	traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE ts>1444044420 AND ts<1444444020'

	con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

	# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
	traj = pd.read_sql_query(traj_sql,con=con)

	# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
	# ports.geom = ports.geom.apply(lambda x: x[0])

	con.close()

	print('done, len -> ', len(traj))

	mined_patterns, start, last_ts = gsgp.mine_patterns(df = traj, mode = gp_type, min_diameter=distance, min_cardinality=cardinality, time_threshold=dt, checkpoints=False, checkpoints_freq=0.1, disable_progress_bar=False)
    
	print('Saving Result...')
	mined_patterns.to_csv('tmp2c.csv', index=False)
