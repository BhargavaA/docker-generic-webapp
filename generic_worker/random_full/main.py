"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_train_size. This value may be different for other datasets
"""
import sys
sys.path.append("../")
import mnist_nn.nn_logistic as problem

from datetime import datetime
import numpy
import copy
import time
import os
import boto_conn

# rng = numpy.random.RandomState(12345)
rng = numpy.random.RandomState() # generates from cache or something always random

stop_time = 2.5*3600
min_iter = int(2)
max_iter = int(problem.get_max_iter())
min_train_size = int(4000)
max_train_size = int(problem.get_max_train_size())
param_info = problem.get_param_ranges()


def run_trials(num_arms,train_size,num_iters,UID='',params=None):
    n = num_arms
    min_err = float('inf')

    for i in range(n):
        # generate hyperparameters
        if params==None:
            hyperparameters = [ 10**rng.uniform( p['range'][0] , p['range'][1] )  for p in param_info  ]
        else:
            hyperparameters = params

        ts = time.time()
        validation_loss,test_loss = problem.run(hyperparameters,train_size=train_size,n_epochs=num_iters)
        this_dt = time.time()-ts

        if validation_loss<min_err:
            min_err = validation_loss
            min_params = hyperparameters


        ########## SAVE TO S3 ##########
        with open(UID+"_jobs.txt", "a") as myfile:
            this_str = '%d,%d,%.4f,%.4f,%.4f,%s,%s\n' % (train_size,num_iters,validation_loss,test_loss,this_dt,str(hyperparameters),str(datetime.now()))
            myfile.write(this_str)

        filename = UID+"_jobs.txt"
        boto_conn.write_to_s3(local_filename_path=UID+"_jobs.txt",s3_path='kgjamieson-general-compute/hyperband_data_random/'+filename)
        ########################################

    return min_err,min_params



while True:
    dt = 0.
    UID = os.urandom(16).encode('hex')
    ts = time.time()
    print '\n\n################ STARTING %s ################\n' % UID
    while dt<stop_time:
        this_error,that_hyperparameters = run_trials(num_arms=1,train_size=max_train_size,num_iters=max_iter,UID=UID)
        dt = time.time() - ts


