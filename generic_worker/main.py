"""
Load some data from disk, do some shit and save to S3

Assumes following are set as environment variables:
AWS_ACCESS_ID
AWS_SECRET_ACCESS_KEY
"""

import json
import numpy
from scipy.linalg import norm
from datetime import datetime
import time


import boto_conn

import os
ACTIVE_MASTER = os.environ.get('ACTIVE_MASTER', 'localhost')

#### TEST BOTO CONN ####
filename = ACTIVE_MASTER+"_test.txt"
test_string = 'This is some fake text generated at '+str(datetime.now())+'\n'
with open(filename, "a") as myfile:
    myfile.write(test_string)
if not boto_conn.write_to_s3(local_filename_path=filename,s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename):
	raise




#### LOAD DATA FROM DISK ####
try:
	filename = 'data/data.json'
	fid = open(filename)
	raw_data = fid.read()
	print raw_data
except:
	print 'Your file \'' + str(filename) + '\' failed to load'
	raise



#### DO THINGS ####




#### SAVE TO S3 ####
filename = ACTIVE_MASTER+"_some_fake_output.txt"
test_string = 'This is some fake output generated at '+str(datetime.now())
with open(filename, "a") as myfile:
    myfile.write(test_string)
boto_conn.write_to_s3(local_filename_path=filename,s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename)

