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
min_iter = int(4)
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

        validation_loss,test_loss,this_dt = problem.run(hyperparameters,train_size=train_size,n_epochs=num_iters)

        if validation_loss<min_err:
            min_err = validation_loss
            min_params = hyperparameters


        ########## SAVE TO S3 ##########
        with open(UID+"_jobs.txt", "a") as myfile:
            this_str = '%d,%d,%.4f,%.4f,%.4f,%s,%s\n' % (train_size,num_iters,validation_loss,test_loss,this_dt,str(hyperparameters),str(datetime.now()))
            myfile.write(this_str)

        filename = UID+"_jobs.txt"
        boto_conn.write_to_s3(local_filename_path=UID+"_jobs.txt",s3_path='kgjamieson-general-compute/hyperband_data_rr_batch_round2/'+filename)
        ########################################

    return min_err,min_params



meta_arms = []
meta_arms.append([2,max_train_size,max_iter])

B = max_train_size
train_size = int(max_train_size)
num_arms = int(B/train_size)
while train_size>=min_train_size:

    if num_arms>2:
        meta_arms.append([num_arms,train_size,max_iter])

    train_size = int(train_size/2)
    num_arms = int(B/train_size)

print meta_arms
meta_arms = sorted(meta_arms,key= lambda student: student[1])
print meta_arms

meta_n = len(meta_arms)



# starts a new simulation once the last one has reached the stopping time
while True:
    dt = 0.
    UID = os.urandom(16).encode('hex')
    meta_max = numpy.zeros(meta_n)
    meta_sum = numpy.zeros(meta_n)
    meta_T = numpy.zeros(meta_n)
    ts = time.time()
    min_err = float('inf')
    k=0
    print '\n\n################ STARTING %s ################\n' % UID
    while dt<stop_time:

        print [ str(meta_sum[i]/max(1.,meta_T[i]))+'['+str(meta_max[i])+']'+'('+str(meta_T[i])+')' for i in range(meta_n)]

        idx = k % meta_n
        k+=1
        
        num_arms,train_size,num_iters = meta_arms[idx]

        print "Starting num_arms=%d,  train_size=%d,  num_iters=%d" %(num_arms,train_size,num_iters)
        pre_error,this_hyperparameters = run_trials(num_arms=num_arms,train_size=train_size,num_iters=num_iters,UID=UID)
        reward = 1.-pre_error

        # after running a batch, pick the best and train on all the data (unless batch used max_train_size)
        if train_size<max_train_size or num_iters<max_iter:
            this_error,that_hyperparameters = run_trials(num_arms=1,train_size=max_train_size,num_iters=max_iter,UID=UID,params=this_hyperparameters)
            reward = max(reward,1.-this_error)
        # reward = 1. - .01*numpy.random.rand() - .02*float(idx)/float(meta_n)

        meta_max[idx] = max(meta_max[idx],reward)
        meta_sum[idx] += reward
        meta_T[idx] += 1.

        

        dt = time.time() - ts




