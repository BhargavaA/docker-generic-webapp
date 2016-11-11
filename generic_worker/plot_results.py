"""
Runs the 'n versus B' algorithm on 'problem' where 'problem' is mnist that is imported. 
The only thing specific to mnist here is the min_iter. This value may be different for other datasets
"""
import sys

from datetime import datetime
import numpy
import time

fresh_data = False
result_type = 'validation' # {'validation','test'}
max_time = 5*3600
N = 1000
grid_times = numpy.linspace(0,max_time,N)


from random_full.plot_results import get_time_series_on_grid as random_get
random_data = random_get(grid_times,fresh_data=fresh_data,result_type=result_type)
random_mean_data = numpy.mean(random_data,axis=0)
print 'random timeseries = %d' % len(random_data)


from singe_user.plot_results import get_time_series_on_grid as hyperband_iter_get
hyperband_iter_data = hyperband_iter_get(grid_times,fresh_data=fresh_data,result_type=result_type)
hyperband_iter_mean_data = numpy.mean(hyperband_iter_data,axis=0)
print 'hyperband_iter timeseries = %d' % len(hyperband_iter_data)


import mpld3
import matplotlib.pyplot as plt
# plt.plot(grid_times,random_data.T,color='black')
plt.plot(grid_times,random_mean_data,linewidth=2,color='black',label='random' )
# plt.plot(grid_times,hyperband_iter_data.T,color='blue')
plt.plot(grid_times,hyperband_iter_mean_data,linewidth=2,color='blue' ,label='hyperband_iter'   )
plt.xlabel('Seconds', fontsize=18)
plt.ylabel(result_type+' error', fontsize=18)
plt.title('MNIST, 2-layer CVN + 1-layer Hidden, 4 hyperparameters, nepoch=15', fontsize=18)
plt.legend()
# plt.show()
mpld3.show()