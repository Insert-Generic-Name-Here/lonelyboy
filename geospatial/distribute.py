import os, sys
import subprocess
import paramiko
import threading
import configparser
from functools import partial
from multiprocessing import cpu_count, Pool


def connect_exec(command, host):
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	try:
		ssh.connect(host, username='user')
		print (f'[+] Success for host: {host}')
		stdin, stdout, stderr = ssh.exec_command(command)

		for line in iter(stdout.readline, ""):
		    print(line, end="")
		stdout.channel.recv_exit_status()
		# print(stderr.read())
		# return stderr.read()
	except:
		print(f"[-] Host {host} is Unavailable.")



#command='date'
command = 'cd /home/user/dist/lonelyboy/geospatial; /home/user/anaconda3/envs/gs_beta/bin/python worker.py'
slaves = ['192.168.1.2', '192.168.1.3', '192.168.1.4', '192.168.1.5', '192.168.1.6']

pool = Pool(len(slaves))
logs = pool.map(partial(connect_exec, command), slaves)
pool.close()
pool.join()
