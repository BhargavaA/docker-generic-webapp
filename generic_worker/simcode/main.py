'''
Determine the effect of noise on the relative error of the recovered Gram matrix compared to the original.

We run two trials, on the non-convex optimization and then on the convex optimization.
'''
import sys
sys.path.append("../")

import utils
import time
from numpy import *
from numpy.random import *
from numpy.linalg import *
import matplotlib.pyplot as plt
# import seaborn as sns
import multiprocessing
import cPickle as pickle
import utils
import STEConvexProximal
import STEConvexProjected
import STENonConvex
import STEConvexNucNormProjected
from collections import defaultdict

import boto_conn
import json
import os

CORES = multiprocessing.cpu_count()


def predictionError(n, d, trials, radius, low, high, step, pkl=False, filename=None):
    '''
    Uses a pool of workers to compute Test error on a set of triplets using
    STEConvexProximal, STEConvexProjected, STENonConvex
    Inputs:
    n - number of points
    d - ambient dimension
    trials - number of trials for a given size of training set
    radius - radius of ball to sample points from
    low - low range of size of training set triples
    high - high range of size of training set triples
    step - increment between low and high

    Resulting data structure is a dictionary errors.
    Keys:
    low: low range of training set size
    high: high range of training set size
    step: step size between ranges
    
    There is then a key for each algorithm being run, [alg]
    errors[alg]: it is a dictionary of lists, each key corresponds to a number of samples
    errors[alg][n]: a list of dictionaries with keys emp_error, log_error, rel_error

    Example:

    {'ConvexProximal': {700: [{'emp_error': 0.08142857142857143, 'rel_error': 0.35035935076774405, 'log_error': 0.20367039755304681}, 
                              {'emp_error': 0.13142857142857142, 'rel_error': 0.29053108502509623, 'log_error': 0.27989964439096499},
                              {'emp_error': 0.16285714285714287, 'rel_error': 0.4620931832151261, 'log_error': 0.36401000238199444}],
                        900: [{'emp_error': 0.11777777777777777, 'rel_error': 0.25243215331513741, 'log_error': 0.25353697627493027},
                              {'emp_error': 0.09111111111111111, 'rel_error': 0.26705377085102378, 'log_error': 0.20628384458706706},
                              {'emp_error': 0.13, 'rel_error': 0.19799178956077834, 'log_error': 0.26513291353188262}]},  
     'ConvexProjected': {700: [{'emp_error': 0.06142857142857143, 'rel_error': 0.07172512067331327, 'log_error': 0.14840409308776362},
                               {'emp_error': 0.10714285714285714, 'rel_error': 0.080263921520480327, 'log_error': 0.22675768092567222},
                               {'emp_error': 0.12571428571428572, 'rel_error': 0.15820149320959684, 'log_error': 0.28954553447731962}],
                         900: [{'emp_error': 0.09111111111111111, 'rel_error': 0.068074737211754749, 'log_error': 0.2300172292244832},
                               {'emp_error': 0.07444444444444444, 'rel_error': 0.084949438755809056, 'log_error': 0.16235067724408123},
                               {'emp_error': 0.10888888888888888, 'rel_error': 0.072747007013342233, 'log_error': 0.23407810947531854}]}}
    '''
    errors = defaultdict(dict)
    pool = multiprocessing.Pool(processes=CORES)
    for pulls in arange(low, high, step):
        results = pool.map(predictionRunParallel, [(n, d, pulls, radius) for i in range(trials)])
        for key in results[0].keys():
            errors[key][pulls] = [r[key] for r in results]
    pool.close()
    print "final errors"
    print errors
    errors['low'] = low
    errors['high'] = high
    errors['step'] = step
    if pkl:
        with open(filename,'wb') as f:
            pickle.dump(errors,f)
    return errors

def predictionRunParallel(args):
    '''
    Sets the seed and passes args on to prediction Run
    '''
    seed()
    return predictionRun(*args)

def predictionRun(n, d, pulls, radius):
    '''
    Input:
    n - number of points
    d - ambient dimension
    pulls - number of triples to sample
    radius - radius of ball to sample random points from
    '''
    Xtrue = radius*randn(n,d);
    Xtrue = Xtrue - 1/n*dot(ones((n,n)),Xtrue)
    Mtrue = dot(Xtrue, Xtrue.T)
    Strain, train_error = utils.getTriplets(Xtrue, pulls, True)
    Stest, test_error = utils.getTriplets(Xtrue, 10000, True)
    MhatConvexProximal, emp_loss_train = STEConvexProximal.computeEmbedding(n,d,Strain,
                                                                            l=.001,
                                                                            max_num_passes_SGD=16,
                                                                            num_random_restarts=1,
                                                                            max_iter_GD=100,
                                                                            epsilon=0.0001,
                                                                            verbose=True)
    MhatConvexProjected, emp_loss_train = STEConvexProjected.computeEmbedding(n,d,Strain,
                                                                              max_num_passes_SGD=16,
                                                                              num_random_restarts=1,
                                                                              max_iter_GD=100,
                                                                              epsilon=0.0001)
    
    MhatConvexNucNormProjected, emp_loss_train = STEConvexNucNormProjected.computeEmbedding(n,d,Strain,
                                                                                            max_iter_GD=100,
                                                                                            epsilon=0.0001,
                                                                                            trace_norm=trace(Mtrue))
    
    XhatNonConvex, emp_loss_train = STENonConvex.computeEmbedding(n,d,Strain,
                                                                  max_num_passes_SGD=16,
                                                                  num_random_restarts=1,
                                                                  max_iter_GD=100,
                                                                  epsilon=0.0001)
    
    errors = defaultdict(dict)
    errors['ConvexProximal']['emp_error'], errors['ConvexProximal']['log_error'] = utils.getLossGram(MhatConvexProximal,Stest)
    errors['ConvexProximal']['rel_error'] = relative_error(MhatConvexProximal, Mtrue)

    errors['ConvexProjected']['emp_error'], errors['ConvexProjected']['log_error'] = utils.getLossGram(MhatConvexProjected,Stest)
    errors['ConvexProjected']['rel_error'] = relative_error(MhatConvexProjected, Mtrue)

    errors['ConvexNucNormProjected']['emp_error'], errors['ConvexNucNormProjected']['log_error'] = utils.getLossGram(MhatConvexNucNormProjected,Stest)
    errors['ConvexNucNormProjected']['rel_error'] = relative_error(MhatConvexNucNormProjected, Mtrue)

    # errors['ConvexNucNormProjected'] = (utils.getLossX(MhatConvexNucNormProjected,Stest),
    #                                     relative_error(MhatConvexNucNormProjected, Mtrue))

    XhatNonConvexCentered = XhatNonConvex - 1./n*dot(ones((n,n)), XhatNonConvex)
    errors['NonConvex']['emp_error'], errors['NonConvex']['log_error']  = utils.getLossX(XhatNonConvex,Stest)
    errors['NonConvex']['rel_error'] = relative_error(dot(XhatNonConvexCentered, XhatNonConvexCentered.T), Mtrue)
    
    return errors
    

def relative_error(M, Mtrue):
    return norm(M-Mtrue,'fro')#**2/norm(Mtrue,'fro')**2
    

def plot_errors(filename):
    with open(filename, 'rb') as f:
        errors = pickle.load(f)
    low = errors.pop('low')
    high = errors.pop('high')
    step = errors.pop('step')
    
    plt.figure(1)
    colors = ['r','g','b','c','m']
    
    # Prediction errors
    avg_pred_errors = defaultdict(list)
    avg_pred_stds = defaultdict(list)
    avg_rel_errors = defaultdict(list)
    avg_rel_stds = defaultdict(list)
    for key in errors.keys():
        for samplesize in arange(low,high,step):
            errs = errors[key][samplesize] 
            emp_error_tmp = [r['emp_error'] for r in errs]
            avg_pred_errors[key].append(mean(emp_error_tmp))
            avg_pred_stds[key].append(std(emp_error_tmp))
            rel_error_tmp = [r['rel_error'] for r in errs]
            avg_rel_errors[key].append(mean(rel_error_tmp))
            avg_rel_stds[key].append(std(rel_error_tmp))

    plt.figure()
    count = 0
    for key in errors.keys():
        plt.errorbar(arange(low,high,step),
                 avg_pred_errors[key],
                 avg_pred_stds[key],
                 marker='o',
                 color=colors[count],
                 label=key)
        count +=1
    plt.legend(loc='best')
    plt.xlabel('number of pulls')
    plt.ylabel('prediction error')
    # plt.show()

    plt.figure()
    count = 0
    for key in errors.keys():
        plt.errorbar(arange(low,high,step),
                 avg_rel_errors[key],
                 avg_rel_stds[key],
                 marker='o',
                 color=colors[count],
                 label=key)
        count +=1
    plt.legend(loc='best')
    plt.xlabel('number of pulls')
    plt.ylabel('relativea error')
    plt.show()


    
if __name__== '__main__':

    if not 'AWS_SECRET_ACCESS_KEY' in os.environ.keys() or not 'AWS_ACCESS_KEY_ID' in os.environ.keys():
        print "You must set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY as environment variables"
        sys.exit()

    UID = os.urandom(16).encode('hex')
    data_filename = 'mds_'+UID+'.pkl'
    meta_filename = 'mds_'+UID+'.txt'

    args = {'n':64,
            'd':2,
            'trials':CORES,
            'radius':1.,
            'low':1000,
            'high':100001,
            'step':1000,
            'pkl':True,
            'filename':data_filename}

    with open(meta_filename, 'w') as f:
        json.dump(args,f)
    predictionError(**args)


    # ########## SAVE TO S3 ##########
    S3_PATH = 'triplets-general-compute/16-5-8'
    boto_conn.write_to_s3(local_filename_path=meta_filename,s3_path=S3_PATH+'/'+meta_filename)
    boto_conn.write_to_s3(local_filename_path=data_filename,s3_path=S3_PATH+'/'+data_filename)
    # ########################################

    # plot_errors(data_filename)
