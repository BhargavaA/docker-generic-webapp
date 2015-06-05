"""
Load some data from disk, do some shit and save to S3
"""



import json
import numpy
from scipy.linalg import norm
from datetime import datetime
import time



#### LOAD SHIT ####
try:
	filename = 'data/data.json'
	fid = open(filename)
	raw_data = fid.read()
	data = eval(raw_data)
except:
	print 'Your file \'' + str(filename) + '\' failed to load'
	raise



#### DO SHIT TO S3 ####




#### SAVE SHIT ####
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
import os
AWS_ACCESS_ID = os.environ.get('AWS_ACCESS_ID', '')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
conn = S3Connection(AWS_ACCESS_ID,AWS_SECRET_ACCESS_KEY)
b = conn.get_bucket('fake-bucket-12ej1dj2912d21d12')

print AWS_ACCESS_ID
print AWS_SECRET_ACCESS_KEY

full_filename = datetime.now() + '_' + str(filename)

while True:
	try:
		print "trying to save " + str(full_filename)
		k = Key(b)
		k.key = str(full_filename)
		bytes_saved = k.set_contents_from_filename( str(full_filename) )
		break
		# bytes_saved = k.set_contents_from_string(pickle_string)
	except:
		print "FAILED!"
		pass

print "[ %s ] done with backup of file %s to S3...  %d bytes saved" % (str(datetime.now()),full_filename,bytes_saved)



