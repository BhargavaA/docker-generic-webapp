"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_iter. This value may be different for other datasets
"""
import sys
sys.path.append("../")
import mnist_nn.nn_logistic as problem

from datetime import datetime
import numpy
import copy
import time
# rng = numpy.random.RandomState(12345)
rng = numpy.random.RandomState() # generates from cache or something always random



########## EC2 PREAMBLE ##########
import boto_conn

import os
ACTIVE_MASTER = os.environ.get('ACTIVE_MASTER', 'localhost')

# TEST BOTO CONN 
filename = ACTIVE_MASTER+"_test.txt"
test_string = 'This is some fake text generated at '+str(datetime.now())+'\n'
with open(filename, "a") as myfile:
    myfile.write(test_string)
if not boto_conn.write_to_s3(local_filename_path=filename,s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename):
    raise
########################################



min_iter = 4000
max_iter = problem.get_max_train_size()
param_info = problem.get_param_ranges()


def run_trials(num_arms,num_pulls,verbose=False):
    n = num_arms
    train_size = num_pulls
    min_err = float('inf')

    for i in range(n):
        # generate hyperparameters
        params = [ 10**rng.uniform( p['range'][0] , p['range'][1] )  for p in param_info  ]

        ts = time.time()
        this_error = problem.run(params,train_size,verbose)
        this_dt = time.time()-ts

        with open("jobs.txt", "a") as myfile:
            this_str = '%d,%.4f,%.4f,%s,%s\n' % (train_size,this_error,this_dt,str(params),str(datetime.now()))
            myfile.write(this_str)

        ########## SAVE TO S3 ##########
        filename = ACTIVE_MASTER+"_jobs.txt"
        boto_conn.write_to_s3(local_filename_path="jobs.txt",s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename)
        ########################################

        if this_error<min_err:
            min_err = this_error
            min_params = params

    return min_err,min_params




with open("results.txt", "a") as myfile:
    this_str = '############\n%s,%f,%f,%s\n' % (str(datetime.now()),1.,0.,str([]))
    myfile.write(this_str)

########## SAVE TO S3 ##########
filename = ACTIVE_MASTER+"_results.txt"
boto_conn.write_to_s3(local_filename_path="results.txt",s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename)
########################################

ts = time.time()
min_err = float('inf')
results = []
for k in range(10):
    B = int((2**k)*max_iter)
    print
    print
    print "Budget B = %d" % B
    print '###################'
    

    num_pulls = max_iter
    num_arms = int(B/num_pulls)
    while num_pulls>=min_iter:

        if num_arms>2:
            print "Starting num_pulls=%d, num_arms=%d" %(num_pulls,num_arms)
            this_ts = time.time()
            pre_error,this_hyperparameters = run_trials(num_arms,num_pulls,verbose=True)
            that_ts = time.time()
            this_error = problem.run(this_hyperparameters,max_iter,verbose=True)
            that_dt = time.time()-that_ts
            this_dt = time.time()-this_ts
            dt = time.time() - ts
            if this_error<min_err:
                min_err = this_error
            res = (this_error,dt,num_pulls,num_arms,this_hyperparameters)
            results.append( res )
            print "Round error: %f" % this_error
            print "Round time elapsed: %f" % this_dt
            print "Round hyperparameters: %s" % str(this_hyperparameters)
            print "Universal minimum error: %f" % min_err
            print "Universal time elapsed error: %f" % dt
            print

            with open("results.txt", "a") as myfile:
                this_str = '%s,%f,%f,%d,%d,%s\n' % (str(datetime.now()),this_error,this_dt,num_pulls,num_arms,str(this_hyperparameters))
                myfile.write(this_str)

            ########## SAVE TO S3 ##########
            filename = ACTIVE_MASTER+"_results.txt"
            boto_conn.write_to_s3(local_filename_path="results.txt",s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename)
            ########################################

            with open("jobs.txt", "a") as myfile:
                this_str = '%d,%.4f,%.4f,%s,%s\n' % (max_iter,this_error,that_dt,str(this_hyperparameters),str(datetime.now()))
                myfile.write(this_str)

            ########## SAVE TO S3 ##########
            filename = ACTIVE_MASTER+"_jobs.txt"
            boto_conn.write_to_s3(local_filename_path="jobs.txt",s3_path='kgjamieson-general-compute/hyperband_nvb/'+filename)
            ########################################

        num_pulls/=2
        num_arms = int(B/num_pulls)




