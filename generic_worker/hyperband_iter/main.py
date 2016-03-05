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


def run_trials(round_hyperparameters,train_size,num_iters,UID='',params=None):
    n = len(round_hyperparameters)
    min_err = float('inf')
    trials_ts = time.time()

    results = []

    for i in range(n):
        hyperparameters = round_hyperparameters[i]

        validation_loss,test_loss,this_dt = problem.run(hyperparameters,train_size=train_size,n_epochs=num_iters)

        results.append( (hyperparameters,validation_loss,test_loss) )

        if validation_loss<min_err:
            min_err = validation_loss
            min_params = hyperparameters


        ########## SAVE TO S3 ##########
        with open(UID+"_jobs.txt", "a") as myfile:
            this_str = '%d,%d,%.4f,%.4f,%.4f,%s,%s\n' % (train_size,num_iters,validation_loss,test_loss,this_dt,str(hyperparameters),str(datetime.now()))
            myfile.write(this_str)

        filename = UID+"_jobs.txt"
        boto_conn.write_to_s3(local_filename_path=UID+"_jobs.txt",s3_path='kgjamieson-general-compute/multilayer_perceptron_hyperband_iter_round1/'+filename)
        ########################################

    trials_dt = time.time() - trials_ts
    return results,trials_dt


# starts a new simulation once the last one has reached the stopping time
max_B = float('inf')
while True:
    dt = 0.
    UID = os.urandom(16).encode('hex')
    ts = time.time()
    min_err = float('inf')
    k=1
    print '\n\n################ STARTING %s ################\n' % UID
    while dt<stop_time:
        B = min(max_B,int((2**k)*max_iter))
        k+=1
        print "\nBudget B = %d" % B
        print '###################'


        eta = 3.
        def logeta(x):
            return numpy.log(x)/numpy.log(eta)

        # s_max defines the number of inner loops per unique value of B
        # it also specifies the maximum number of rounds
        R = float(max_iter)
        r = float(min_iter)
        ell_max = int(min(B/R-1,int(logeta(R/r))))
        ell = ell_max
        while ell >= 0 and dt<stop_time:

            # specify the number of arms and the number of times each arm is pulled per stage within this innerloop
            n = int( B/R*eta**ell/(ell+1.) )

            if n> 0:
                
                round_hyperparameters = []
                for i in range(n):
                    hyperparameters = [ 10**rng.uniform( p['range'][0] , p['range'][1] )  for p in param_info  ]
                    round_hyperparameters.append( hyperparameters )
                s = 0
                while (n+1)*R*(s+1.)*eta**(-s)>B:
                    s+=1
                s-=1

                print 
                print 's=%d, n=%d' %(s,n)
                print 'n_i\tr_k'
                for i in range(s+1):
                    
                    num_pulls = int( R*eta**(i-s) )
                    num_arms = int( n*eta**(-i) )
                    if num_pulls/eta<r:
                        max_B = B
                    print '%d\t%d' %(num_arms,num_pulls)
 
                    results,this_dt = run_trials(round_hyperparameters=round_hyperparameters,train_size=max_train_size,num_iters=num_pulls,UID=UID)

                    # pick the top results
                    n_k1 = int( n*eta**(-i-1) )
                    results = sorted(results,key=lambda x: x[1])
                    if s-i-1>=0:
                        round_hyperparameters = [ x[0] for x in results[0:n_k1] ]
                    else:
                        break


            ell-=1

            dt = time.time() - ts