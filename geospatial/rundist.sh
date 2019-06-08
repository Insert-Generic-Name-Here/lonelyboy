#!/usr/bin/env bash

#send current lb version to all workers

for ipadr in 192.168.1.{2..6}
do
  scp -r ../../lonelyboy $ipadr:/home/user/dist/
done

#create jsons
python mapper.py

#send jsons and run
i=0
for ipadr in 192.168.1.{2..6}
do
  echo $ipadr
  scp "info$i.json" $ipadr:/home/user/dist/lonelyboy/geospatial/info.json ;rm "info$i.json"
  ((i++))
done

python distribute.py

python reduce.py
