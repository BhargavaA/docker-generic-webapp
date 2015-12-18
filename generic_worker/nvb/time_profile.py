
import sys
sys.path.append("../")
import mnist_nn.nn_logistic as problem

import matplotlib.pyplot as plt
from datetime import datetime
import numpy
import copy
import time
rng = numpy.random.RandomState(12345)



min_iter = 2000
max_iter = problem.get_max_train_size()
param_info = problem.get_param_ranges()

num_reps = 10

all_times = []
for rep in range(num_reps):

	rng = numpy.random.RandomState()
	params = [ 10**rng.uniform( p['range'][0] , p['range'][1] )  for p in param_info  ]
	train_size = max_iter
	times = []
	while train_size >= min_iter:
		rng = numpy.random.RandomState(12345)
		ts = time.time()
		this_error = problem.run(params,train_size,verbose=True)
		dt = time.time() - ts
		times.append(dt)
		train_size/=2

	all_times.append(times)

train_sizes = []
train_size = min_iter
while train_size <= max_iter:
	train_sizes.append(train_size)
	train_size*=2

for times in all_times:
	plt.loglog(train_sizes,times)
plt.show()

