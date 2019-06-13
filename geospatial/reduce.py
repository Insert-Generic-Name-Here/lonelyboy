import os, sys, json
import pandas as pd

sys.path.append(os.path.join(os.path.expanduser('~')))
from lonelyboy.geospatial import group_patterns_v2 as lbgp

with open('info_master.json', 'r') as f:
	json_file = json.load(f)

print('Merging on master node...')
csvs = []
for csvname in os.listdir('/home/user/distdata'):
    csvs.append(pd.read_csv(os.path.join('/home/user/distdata', csvname), engine='python'))


fndf = lbgp.reduce_partitions(csvs, json_file['dt'], json_file['res_rate'])
print('Saving final Dataframe')
fndf.to_csv(os.path.join('/home/user/data_dist' ,csvname.split('_',1)[1]), index=False)
