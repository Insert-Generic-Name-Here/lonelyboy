import os, sys, json
# sys.path.append(os.path.join(os.path.expanduser('~')))
# sys.path
import pandas as pd
import psycopg2
import configparser


#### PARAMS ####
params = {
	'gp_type' : 'flocks',
	'cardinality' : 5,
	'dt' : 10,
	'distance' : 2778
}
num_of_slaves = 5
# num_of_slaves = int(sys.argv[1])

print(f"Discovering {params['gp_type']} with card={params['cardinality']}, dt={params['dt']} and distance={params['distance']}")
################

properties = configparser.ConfigParser()
properties.read(os.path.join('.','sql_server.ini'))
properties = properties['SERVER']

host    = 'localhost'
# host    = properties['host']
db_name = properties['db_name']
uname   = properties['uname']
pw      = properties['pw']
port    = properties['port']

dt_sql = 'SELECT datetime FROM ais_data.dynamic_ships_segmented_12h_resampled_1min '

con = psycopg2.connect(database=db_name, user=uname, password=pw, host=host, port = port)
print('loading data')

# traj = gpd.GeoDataFrame.from_postgis(traj_sql, con)
datet = pd.read_sql_query(dt_sql,con=con)
total = datet.datetime.nunique()

# ports = gpd.GeoDataFrame.from_postgis(ports_sql, con, geom_col='geom' )
# ports.geom = ports.geom.apply(lambda x: x[0])

con.close()
json_list = [params for _ in range(num_of_slaves)]
parts = pd.cut(datet.datetime, num_of_slaves, retbins=True)[1]
# print(parts)

for i in range(len(parts)-1):
	json_list[i]['sql'] = (f"SELECT * FROM ais_data.dynamic_ships_segmented_12h_resampled_1min WHERE datetime>'{str(parts[i])}' AND datetime<='{str(parts[i+1])}'")
	json_list[i]['slave_no'] = i+1
	with open(f'info{i}.json', 'w') as f:
		json.dump(json_list[i], f)
