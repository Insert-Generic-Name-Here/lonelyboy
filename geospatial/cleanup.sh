#!/usr/bin/env bash


rm -rf /home/user/distdata/*

ssh slave1 rm -rf /home/user/dist/lonelyboy
ssh slave2 rm -rf /home/user/dist/lonelyboy
ssh slave3 rm -rf /home/user/dist/lonelyboy
ssh slave4 rm -rf /home/user/dist/lonelyboy
ssh slave5 rm -rf /home/user/dist/lonelyboy

