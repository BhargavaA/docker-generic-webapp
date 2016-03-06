"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_iter. This value may be different for other datasets
"""
import sys

from datetime import datetime
import numpy
import time



fresh_data = False
max_time = 1.5*3600
N = 1000
grid_times = numpy.linspace(0,max_time,N)

T = []
# from nvb.plot_results import get_time_series_on_grid
# from random_full.plot_results import get_time_series_on_grid
from random_full.plot_results import get_time_series_on_grid as random_get
random_data = random_get(grid_times,fresh_data)
random_mean_data = numpy.mean(random_data,axis=0)
print 'random timeseries = %d' % len(random_data)


# from nvb_batch.plot_results import get_time_series_on_grid as nvb_get
# nvb_data = nvb_get(grid_times,fresh_data)
# nvb_mean_data = numpy.mean(nvb_data,axis=0)
# print 'nvb_batch timeseries = %d' % len(nvb_data)

# from hyperband_batch.plot_results import get_time_series_on_grid as hyperband_batch_get
# hyperband_batch_data = hyperband_batch_get(grid_times,fresh_data)
# hyperband_batch_mean_data = numpy.mean(hyperband_batch_data,axis=0)
# print 'hyperband_batch timeseries = %d' % len(hyperband_batch_data)


from hyperband_iter.plot_results import get_time_series_on_grid as hyperband_iter_get
hyperband_iter_data = hyperband_iter_get(grid_times,fresh_data)
hyperband_iter_mean_data = numpy.mean(hyperband_iter_data,axis=0)
print 'hyperband_iter timeseries = %d' % len(hyperband_iter_data)


# from nvb_iter.plot_results import get_time_series_on_grid as nvb_iter_get
# nvb_iter_data = nvb_iter_get(grid_times,fresh_data)
# nvb_iter_mean_data = numpy.mean(nvb_iter_data,axis=0)
# print 'nvb_iter timeseries = %d' % len(nvb_iter_data)

# from bandit_batch.plot_results import get_time_series_on_grid as bandit_batch_get
# bandit_batch_data = bandit_batch_get(grid_times,fresh_data)
# bandit_batch_mean_data = numpy.mean(bandit_batch_data,axis=0)
# print 'bandit_batch timeseries = %d' % len(bandit_batch_data)

# from bandit_iter.plot_results import get_time_series_on_grid as bandit_iter_get
# bandit_iter_data = bandit_iter_get(grid_times,fresh_data)
# bandit_iter_mean_data = numpy.mean(bandit_iter_data,axis=0)
# print 'bandit_iter timeseries = %d' % len(bandit_iter_data)


# from rr_batch.plot_results import get_time_series_on_grid as rr_batch_get
# rr_batch_data = rr_batch_get(grid_times,fresh_data)
# rr_batch_mean_data = numpy.mean(rr_batch_data,axis=0)
# print 'rr_batch timeseries = %d' % len(rr_batch_data)

# from rr_iter.plot_results import get_time_series_on_grid as rr_iter_get
# rr_iter_data = rr_iter_get(grid_times,fresh_data)
# rr_iter_mean_data = numpy.mean(rr_iter_data,axis=0)
# print 'rr_iter timeseries = %d' % len(rr_iter_data)

# print grid_times

import mpld3
import matplotlib.pyplot as plt
# plt.plot(grid_times,random_data.T,color='black')
plt.plot(grid_times,random_mean_data,linewidth=2,color='black',label='random' )
# plt.plot(grid_times,nvb_data.T,color='red')
# plt.plot(grid_times,nvb_mean_data,linewidth=2,color='red' ,label='nvb_batch'   )
# plt.plot(grid_times,hyperband_batch_mean_data,linewidth=2,color='blue' ,label='hyperband_batch'   )
plt.plot(grid_times,hyperband_iter_mean_data,linewidth=2,color='blue' ,label='hyperband_iter'   )
# plt.plot(grid_times,hyperband_iter_data.T,color='blue')
# plt.plot(grid_times,hyperband_batch_data.T,color='blue')
# plt.plot(grid_times,nvb_iter_mean_data,linewidth=2,color='blue' ,label='nvb_iter'   )
# plt.plot(grid_times,bandit_batch_mean_data,linewidth=2,color='m' ,label='bandit_batch'   )
# plt.plot(grid_times,bandit_iter_mean_data,linewidth=2,color='g' ,label='bandit_iter'   )
# plt.plot(grid_times,rr_batch_mean_data,linewidth=2,color='c' ,label='rr_batch'   )
# plt.plot(grid_times,rr_iter_mean_data,linewidth=2,color='y' ,label='rr_iter'   )
plt.xlabel('Seconds', fontsize=18)
plt.ylabel('Validation error', fontsize=18)
plt.title('MNIST, 2-layer CVN + 1-layer Hidden, 4 hyperparameters, nepoch=15', fontsize=18)
plt.legend()
# plt.show()
mpld3.show()