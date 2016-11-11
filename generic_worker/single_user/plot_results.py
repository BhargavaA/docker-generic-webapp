"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_iter. This value may be different for other datasets
"""
import sys
sys.path.append("../")
sys.path.append(".")
# import mnist_nn.nn_logistic as problem

from datetime import datetime
import numpy
import copy
import time
# rng = numpy.random.RandomState(12345)
rng = numpy.random.RandomState() # generates from cache or something always random

import boto_conn

def get_time_series_on_grid(grid_times,fresh_data=True,result_type='validation'):
    if fresh_data:
        boto_conn.download_from_s3('kgjamieson-general-compute/convolutional_mlp_hyperband_iter_round1','hyperband_iter')
    local_path = 'hyperband_iter/convolutional_mlp_hyperband_iter_round1'
    
    # 1-17, 18-24, 25-29
    # active_set = range(1,18)
    active_set = None
    
    import csv
    from os import listdir
    from os.path import isfile, join

    allfiles = []
    for f in listdir(local_path):
        if isfile(join(local_path, f)):
            split_f = f.split('_')
            if split_f[1]=='jobs.txt':
                allfiles.append(f)

    # allfiles = [f for f in listdir(local_path) if isfile(join(local_path, f))]


    dup_check = {}
    time_series = []
    for f in allfiles:

        full_filename=local_path + '/' + f
        times = [-.0001]
        losses = [0.9]
        duration = 0
        min_err = float('inf')
        with open(full_filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            idx=0
            for row in spamreader:
                idx+=1
                if active_set==None or idx in active_set:
                    train_size = int(row[0])
                    num_iters = int(row[1])
                    validation_error = float(row[2])
                    test_error = float(row[3])
                    dt = float(row[4])
                    params_str = row[0]+row[1]+row[6]+row[7]+row[8]

                    # sometimes same hyperparameter was trained on full dataset multiple times - remove this
                    if params_str not in dup_check:
                        duration += dt
                        if validation_error<min_err:
                            try:
                                losses.append( losses[-1]  )
                                times.append( duration-.0001  )
                            except:
                                pass                            
                            times.append(duration)
                            if result_type=='validation':
                                losses.append(validation_error)
                            else:
                                losses.append(test_error)
                            min_err = validation_error

                        dup_check[params_str] = True
        times.append(duration)
        losses.append(losses[-1])
        if times[-1]>grid_times[-1]:
            time_series.append( (times,losses) )

    N = len(grid_times)
    data = numpy.zeros( (len(time_series),N) )
    for k,t in enumerate(time_series):
        for i in range(N):
            j=len(t[0])-1
            try:
                while t[0][j]>grid_times[i]:
                    j-=1
                data[k][i] = t[1][j]
            except:
                data[k][i] = t[1][0]

    return data