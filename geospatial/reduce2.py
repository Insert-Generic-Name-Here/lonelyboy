import os, sys, json, configparser
import psycopg2
import pandas as pd

sys.path.append(os.path.join(os.path.expanduser('~')))
from lonelyboy.geospatial import group_patterns_v3 as lbgp

properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = 'localhost'
# host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

mx_sql = 'SELECT MAX(datetime) FROM ais_data.dynamic_ships_segmented_12h_resampled_1min'

con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)
print('loading data')

# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
traj = pd.read_sql_query(mx_sql,con=con)

with open('info_master.json', 'r') as f:
	json_file = json.load(f)

print('Merging on master node...')
csvs = []
for csvname in os.listdir('/home/user/distdata'):
    csvs.append(pd.read_csv(os.path.join('/home/user/distdata', csvname), engine='python'))

for csvdf in csvs:
	csvdf.clusters = csvdf.clusters.apply(eval)
	
fndf = lbgp.reduce_partitions(csvs,json_file['cardinality'], json_file['res_rate'], json_file['dt'], traj.iloc[0].max())
print('Saving final Dataframe')
fndf[(fndf.et - fndf.st >= pd.Timedelta(minutes=json_file['dt'])) | (fndf.et == traj.iloc[0].max())].to_csv(os.path.join('/home/user/data_dist' ,csvname.split('_',1)[1]), index=False)
