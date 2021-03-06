#!/usr/bin/env bash

#send current lb version to all workers

mkdir /home/user/distdata
for ipadr in 192.168.1.{2..6}
do
  scp -r  ../../lonelyboy $ipadr:/home/user/dist/
done

#create jsons
python mapper.py #ARGS GO HERE 0/1-FLOCKS/CONVOYS, CARD, DT, DISTANCE

#send jsons and run
i=0
for ipadr in 192.168.1.{2..6}
do
  echo $ipadr
  scp "info$i.json" $ipadr:/home/user/dist/lonelyboy/geospatial/info.json ;rm "info$i.json"
  ((i++))
done

python distribute.py

python reduce2.py

#rm -rf /home/user/distdata

rm 'info_master.json'
