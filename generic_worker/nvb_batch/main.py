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
min_train_size = int(2000)
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

        validation_loss,test_loss,this_dt = problem.run(hyperparameters,train_size=train_size,n_epochs=num_iters)

        if validation_loss<min_err:
            min_err = validation_loss
            min_params = hyperparameters


        ########## SAVE TO S3 ##########
        with open(UID+"_jobs.txt", "a") as myfile:
            this_str = '%d,%d,%.4f,%.4f,%.4f,%s,%s\n' % (train_size,num_iters,validation_loss,test_loss,this_dt,str(hyperparameters),str(datetime.now()))
            myfile.write(this_str)

        filename = UID+"_jobs.txt"
        boto_conn.write_to_s3(local_filename_path=UID+"_jobs.txt",s3_path='kgjamieson-general-compute/hyperband_data_nvb_batch_round2/'+filename)
        ########################################

    return min_err,min_params


# starts a new simulation once the last one has reached the stopping time
while True:
    dt = 0.
    UID = os.urandom(16).encode('hex')
    ts = time.time()
    min_err = float('inf')
    k=0
    print '\n\n################ STARTING %s ################\n' % UID
    while dt<stop_time:
        B = int((2**k)*max_train_size)
        k+=1
        print "\nBudget B = %d" % B
        print '###################'

        num_pulls = int(max_train_size)
        num_arms = int(B/num_pulls)
        while num_pulls>=min_train_size and dt<stop_time:

            if num_arms>2:
                print "Starting num_pulls=%d, num_arms=%d" %(num_pulls,num_arms)
                pre_error,this_hyperparameters = run_trials(num_arms=num_arms,train_size=num_pulls,num_iters=max_iter,UID=UID)

                # after running a batch, pick the best and train on all the data (unless batch used max_train_size)
                if num_pulls<max_train_size:
                    this_error,that_hyperparameters = run_trials(num_arms=1,train_size=max_train_size,num_iters=max_iter,UID=UID,params=this_hyperparameters)

                dt = time.time() - ts

            num_pulls = int(num_pulls/2)
            num_arms = int(B/num_pulls)




