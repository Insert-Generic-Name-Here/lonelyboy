import os, sys, pickle
sys.path.append(os.path.join(os.path.expanduser('~')))
# sys.path

from lonelyboy.geospatial import plots as gsplt
from lonelyboy.geospatial import preprocessing as gspp
from lonelyboy.timeseries import lbtimeseries as tspp
from lonelyboy.geospatial import group_patterns_v3 as lbgp
from tqdm import tqdm as tqdm

import pandas as pd
import configparser
import psycopg2
import numpy as np
pd.set_option('display.max_colwidth', -1)  # or 199


properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = '192.168.1.1'
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

traj_sql = 'SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE ts>1444044420 AND ts<1444444020'

con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)
print('Fetching from db')
# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
traj = pd.read_sql_query(traj_sql,con=con)

# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
# ports.geom = ports.geom.apply(lambda x: x[0])

con.close()

print('done, len -> ', len(traj))

# mined_patterns, start, last_ts = gsgp.mine_patterns(df = traj, mode = gp_type, min_diameter=distance, min_cardinality=cardinality, time_threshold=dt, checkpoints=False, checkpoints_freq=0.1, disable_progress_bar=False)

# print('Saving Result...')
# mined_patterns.to_csv('tmp2c.csv', index=False)


# In[4]:


traj.datetime.nunique()


# In[38]:

lbgp.mine_patterns(traj, 'flocks', min_diameter=500, min_cardinality=3, time_threshold=5, active=pd.DataFrame(), closed_patterns=pd.DataFrame(), total=None, start=0, disable_progress_bar=False)
