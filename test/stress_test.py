import sys
import httplib
import urllib
import json


import time
import os

HOSTNAME = "ec2-52-25-58-241.us-west-2.compute.amazonaws.com:8000"

import numpy
import json
import requests
from multiprocessing import Pool

def run_all():
	n = int(10)

	# print simulate_one_client( (n,0) )

	num_clients = 200

	pool = Pool(processes=num_clients)   

	client_args = []
	for cl in range(num_clients):
		client_args.append( (n,cl) )
	results = pool.map(simulate_one_client, client_args )

	for result in results:
		print result



def simulate_one_client( input_args ):
	n,participant_uid = input_args
	avg_response_time = 1.

	rtt = []
	try:
		for i in range(n):
			data = json.dumps({'search': 'abcd', 'text': 1,'sleep_time':0.1})
			url =  'http://'+HOSTNAME+'/testconn'
			ts = time.time()
			response = requests.post(url,data,headers={'content-type':'application/json'})
			te = time.time()
			data = response.text
			data_dict = eval(data)
			rtt.append(te-ts)
			print "worker_uid=participant_uid=%s   request_num=%d  response=%s   trip_duration=%.4f" % (participant_uid,i, str(data_dict), te-ts)
			time.sleep(  avg_response_time*numpy.log(1/numpy.random.rand())  )
	except:
		pass

	rtt.sort()
	return_str = '%s \t rtt (%d/%d) \t : %f (5),    %f (50),    %f (95)' % (participant_uid,len(rtt),n,rtt[int(.05*len(rtt))],rtt[int(.50*len(rtt))],rtt[int(.95*len(rtt))])
	return return_str

if __name__ == '__main__':
  print HOSTNAME
  run_all()