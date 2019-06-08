import os, sys
import pandas as pd

sys.path.append(os.path.join(os.path.expanduser('~')))
from lonelyboy.geospatial import group_patterns_v2 as lbgp

csvs = []
for csvname in os.listdir('/home/user/distdata'):
    csvs.append(pd.read_csv(os.path.join('/home/user/distdata', csvname)))


fndf = lbgp.reduce_partitions(csvs)
fndf.to_csv(os.path.join('/home/user/data' ,csvname.split('_',1)[1]), header=False, index=False)
