import os, sys
import json
sys.path.append(os.path.join(os.path.expanduser('~'), 'dist'))
import paramiko
# sys.path

import plots as gsplt
import preprocessing as gspp
import group_patterns_v3 as gsgp

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
import pickle


from multiprocessing import cpu_count, Pool
from functools import partial
import datetime



def wpr(params, partitions):
	return gsgp.mine_patterns_dist(partitions, *params)


with open('info.json', 'r') as f:
	json_file = json.load(f)


#### PARAMS ####
slave_no = json_file['slave_no']
gp_type = json_file['gp_type']
cardinality = json_file['cardinality']
dt = json_file['dt']
distance = json_file['distance']
res_rate = json_file['res_rate']
print(f'Slave{slave_no} -> Discovering {gp_type} with card={cardinality}, dt={dt} and distance={distance}')
################

properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = '192.168.1.1'
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

save_name = f'Slave{slave_no}_{gp_type}_card_{cardinality}_dt_{dt}_dist_{distance}.csv'



traj_sql = json_file['sql']

params = [gp_type, distance, cardinality, dt]

con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)

# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
traj = pd.read_sql_query(traj_sql,con=con)

# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
# ports.geom = ports.geom.apply(lambda x: x[0])

con.close()

print(f'Slave {slave_no}: Starting..')

# 	partitions = [grp[1] for grp in traj.groupby(['datetime'])]
partitions = [grp[1] for grp in traj.groupby(pd.cut(traj.datetime,cpu_count()))] # no of cores
print (len(partitions), 'partitions')
pool = Pool(len(partitions))
dfs = pool.map(partial(wpr, params), partitions)
pool.close()
pool.join()

print(f'Slave {slave_no}: Reducing...')
mined_patterns = gsgp.reduce_partitions(dfs ,cardinality, res_rate, dt)
print(f'Slave {slave_no}: Saving Result...')
mined_patterns.to_csv(save_name, index=False)

#with open(f'pckl{slave_no}.pkl', 'wb') as f:
#	pickle.dump(dfs, f)

host = '192.168.1.1'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username='user')
print (f'[+] Success for Slave {slave_no} with len = {len(mined_patterns)}')
ftp_client=ssh.open_sftp()
ftp_client.put(save_name,f'/home/user/distdata/{save_name}')
ftp_client.close()
