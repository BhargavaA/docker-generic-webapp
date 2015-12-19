"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_iter. This value may be different for other datasets
"""
import sys

from datetime import datetime
import numpy
import time




max_time = 2*3600
N = 1000
grid_times = numpy.linspace(0,max_time,N)

T = []
# from nvb.plot_results import get_time_series_on_grid
# from random_full.plot_results import get_time_series_on_grid
from random_full.plot_results import get_time_series_on_grid as random_get
random_data = random_get(grid_times)
random_mean_data = numpy.mean(random_data,axis=0)
print 'random timeseries = %d' % len(random_data)


from nvb_batch.plot_results import get_time_series_on_grid as nvb_get
nvb_data = nvb_get(grid_times)
nvb_mean_data = numpy.mean(nvb_data,axis=0)
print 'nvb_batch timeseries = %d' % len(nvb_data)


from nvb_iter.plot_results import get_time_series_on_grid as nvb_iter_get
nvb_iter_data = nvb_iter_get(grid_times)
nvb_iter_mean_data = numpy.mean(nvb_iter_data,axis=0)
print 'nvb_iter timeseries = %d' % len(nvb_iter_data)

# print grid_times


import matplotlib.pyplot as plt
# plt.plot(grid_times,random_data.T,color='black')
plt.plot(grid_times,random_mean_data,linewidth=2,color='black',label='random' )
# plt.plot(grid_times,nvb_data.T,color='red')
plt.plot(grid_times,nvb_mean_data,linewidth=2,color='red' ,label='nvb_batch'   )
# plt.plot(grid_times,nvb_iter_data.T,color='blue')
plt.plot(grid_times,nvb_iter_mean_data,linewidth=2,color='blue' ,label='nvb_iter'   )
plt.xlabel('Seconds')
plt.ylabel('Validation error')
plt.legend()
plt.show()