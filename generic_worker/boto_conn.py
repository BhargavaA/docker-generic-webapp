"""
Load some data from disk, do some shit and save to S3
"""

import json
from datetime import datetime
import time


from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
import os



def write_to_s3(local_filename_path,s3_path,verbose=False):
	"""
	local_filename_path: test.txt
	s3_path: <bucket>/<directory1>/test.txt
	"""

	AWS_ACCESS_ID = os.environ.get('AWS_ACCESS_ID', '')
	AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')

	split_path = s3_path.split('/')
	AWS_BUCKET = split_path.pop(0)
	AWS_FILENAME = '/'.join(split_path)

	if verbose:
		print AWS_ACCESS_ID
		print AWS_SECRET_ACCESS_KEY
		print AWS_BUCKET
		print AWS_FILENAME

	conn = S3Connection(AWS_ACCESS_ID,AWS_SECRET_ACCESS_KEY)
	b = conn.get_bucket(AWS_BUCKET)

	ell = 0
	while True:
		try:
			if verbose: print "trying to save " + local_filename_path + "  to " + AWS_FILENAME
			k = Key(b)
			k.key = AWS_FILENAME
			bytes_saved = k.set_contents_from_filename( local_filename_path )
			break
		except:
			ell+=1
			if ell < 10:
				print "Failed to save to s3... %d/10 times" % ell
				pass
			else:
				return False

	if verbose: print "[ %s ] done with backup of file %s to S3...  %d bytes saved" % (str(datetime.now()),local_filename_path,bytes_saved)
	return True


def download_from_s3(s3_path,local_path='.',verbose=False):
	"""
	s3_path: '<bucket>/<directory1>'
	local_path: 'new_dir'

	Downloads all files in directory of s3_path, including the directory, and puts them at local_path, creates directories if necessary
	"""

	AWS_ACCESS_ID = os.environ.get('AWS_ACCESS_ID', '')
	AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')

	split_path = s3_path.split('/')
	AWS_BUCKET = split_path.pop(0)
	AWS_DIRECTORY = '/'.join(split_path)
	if AWS_DIRECTORY =='':
		AWS_DIRECTORY = None


	conn = S3Connection(AWS_ACCESS_ID,AWS_SECRET_ACCESS_KEY)
	b = conn.get_bucket(AWS_BUCKET)
	for key in b.list(AWS_DIRECTORY):
	    try:
	    	filename = '/'.join([local_path,key.name])
	    	directory = os.path.dirname(filename)
	    	if not os.path.exists(directory):
				os.makedirs(directory)
	        res = key.get_contents_to_filename(filename)
	    except:
	        print (key.name+":"+"FAILED")

