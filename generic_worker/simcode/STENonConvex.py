import time
from numpy import *
from numpy.random import *
from numpy.linalg import *
import multiprocessing
import cPickle as pickle
from utils import *

norm = linalg.norm
floor = math.floor
ceil = math.ceil

def run(n, d, plot=False, verbose=False):
    """
    Creates random data and finds an embedding.
    Inputs:
    n: The number of points
    d: The number of dimensions
    plot: Whether to plot the points
    """
    n = 50
    d = 2
    m = 5000#int(ceil(40*n*d*log(n)))  # number of labels
     
    Xtrue = randn(n,d);
    Xtrue = Xtrue -1./n*dot(ones((n,n)), Xtrue)
    Mtrue = dot(Xtrue, Xtrue.T)
    Strain,train_bayes_err = getTriplets(Xtrue, m, noise=True)
    Stest,test_bayes_err = getTriplets(Xtrue, m, noise=True)
    Xhat, emp_loss_train = computeEmbedding(n,
                                            d,
                                            Strain,
                                            max_num_passes_SGD=16,
                                            num_random_restarts=1,
                                            epsilon=0.0001,
                                            verbose=verbose)
    emp_loss_train,log_loss_train = getLossX(Xhat, Strain)
    emp_loss_test,log_loss_test = getLossX(Xhat, Stest)
    Xhat = Xhat - 1/n*dot(ones((n,n)),Xhat)
    Mhat = dot(Xhat, Xhat.T)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},  '
           'Bayes Test loss = {}'
           'Relative Error = {} ').format(emp_loss_train, emp_loss_test, test_bayes_err, norm(Mtrue-Mhat,'fro')**2/norm(Mtrue,'fro')**2)
    if plot:
        _, Xhat = transformGramtoX(Mhat,2)
        _, Xpro, _ = procrustes(Xtrue, Xhat)
        twodplot(Xtrue, Xpro)
        stemplot(Mtrue, Mhat)
        plt.show()


def getPartialGradient(X,q):
    """
    Returns normalized gradient of logistic loss wrt to X and a single query q.
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If loss is logistic_loss then loss(X,q) = log(1+exp(s(X,q)))

    Usage:
        G = getPartialGradient(X,S)
    """
    n,d = X.shape
    # pattern for computing gradient
    H = mat([[2.,0.,-2.],[ 0.,  -2.,  2.],[ -2.,  2.,  0.]])
    # compute gradient 
    G = zeros((n,d))
    score = getTripletScoreX(X,q)
    G[q,:] = 1/(1+exp(-score))*dot(H,X[q,:])
    return G


def getFullGradient(X,S):
    """
    Returns normalized gradient of logistic loss wrt to X and S.
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If loss is logistic_loss then loss(X,q) = log(1+exp(s(X,q)))

    Usage:
        G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq = getFullGradient(X,S)
    """
    n,d = X.shape
    # compute Gradient
    G = zeros((n,d))
    m = len(S)
    for q in S:
        G[q,:] = G[q,:] + getPartialGradient(X,q)[q,:]/m
    # compute statistics about gradient used for stopping conditions
    mu = mean(X,0)
    avg_row_norm_sq = 0.
    avg_grad_row_norm_sq = 0.
    max_grad_row_norm_sq = 0.
    for i in range(n):
        row_norm_sq = 0
        grad_row_norm_sq = 0
        for j in range(d):
            row_norm_sq += (X[i,j]-mu[j])*(X[i,j]-mu[j])
            grad_row_norm_sq += G[i,j]*G[i,j]
        avg_row_norm_sq += row_norm_sq/n
        avg_grad_row_norm_sq += grad_row_norm_sq/n
        max_grad_row_norm_sq = max(max_grad_row_norm_sq,grad_row_norm_sq)
    return G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq


def computeEmbedding(n,d,S,num_random_restarts=0,max_num_passes_SGD=16,max_iter_GD=50,max_norm=1,epsilon=0.01,verbose=False):
    """
    Computes an embedding of n objects in d dimensions usin the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (int) num_random_restarts : number of random restarts (nonconvex
        optimization, may converge to local minima). E.g., 9 random restarts
        means take the best of 10 runs of the optimization routine.
        (int) max_num_passes : maximum number of passes over data SGD makes before proceeding to GD (default equals 16)
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and 
                        b is the avg row norm of the centered embedding matrix X. This is a 
                        means to determine how close the current solution is to the "best" solution.  
    """
    X_old = None
    emp_loss_old = float('inf')
    for restart in range(num_random_restarts):
        if verbose:
            print "Restart:{}/{}".format(restart, num_random_restarts)
        ts = time.time()
        X,acc = computeEmbeddingWithEpochSGD(n,d,S,
                                             max_num_passes=max_num_passes_SGD,
                                             max_norm=max_norm,
                                             epsilon=epsilon,
                                             verbose=verbose)
        te_sgd = time.time()-ts
        ts = time.time()
        X_new,emp_loss_new,log_loss_new,acc_new = computeEmbeddingWithGD(X,S,
                                                                         max_iters=max_iter_GD,
                                                                         max_norm=max_norm,
                                                                         epsilon=epsilon,
                                                                         verbose=verbose)
        te_gd = time.time()-ts        
        if emp_loss_new < emp_loss_old:
            X_old = X_new
            emp_loss_old = emp_loss_new
        if verbose:
            print ("restart %d:   emp_loss = %f,   "
                   "log_loss = %f,   duration=%f+%f") %(restart,
                                                          emp_loss_new,log_loss_new,
                                                          te_sgd,te_gd)
    return X_old,emp_loss_old


def computeEmbeddingWithEpochSGD(n,d,S,max_num_passes=0,max_norm=0,epsilon=0.001,a=1,verbose=False):
    """
    Performs epochSGD where step size is constant across each epoch, epochs are 
    doubling in size, and step sizes are getting cut in half after each epoch.
    This has the effect of having a step size decreasing like 1/T. a0 defines 
    the initial step size on the first epoch. 

    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_num_passes : maximum number of passes over data (default equals 16)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition (default = 0.01)
        (float) a0 : inititial step size (default equals 0.1)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and 
                        b is the avg row norm of the centered embedding matrix X. This is a 
                        means to determine how close the current solution is to the "best" solution.  


    Usage:
        X,gamma = computeEmbeddingWithEpochSGD(n,d,S)
    """
    m = len(S)
    # norm of each object is equal to 1 in expectation
    X = randn(n,d)
    if max_num_passes==0:
        max_iters = 16*m
    else:
        max_iters = max_num_passes*m
    if max_norm == 0:
        max_norm = 10*d
    epoch_length = m
    t = 0
    t_e = 0
    # check losses
    if verbose:
        emp_loss,log_loss = getLossX(X,S)
        print "SGD iter=%d,   emp_loss=%f,   log_loss=%f,   a=%f" % (0,emp_loss,log_loss,a)
    rel_max_grad = None
    while t < max_iters:
        t += 1
        t_e += 1
        # check epoch conditions, udpate step size
        if t_e % epoch_length == 0:
            a = a*0.5
            epoch_length = 2*epoch_length
            t_e = 0
            if epsilon>0 or verbose:
                # get losses
                emp_loss,log_loss = getLossX(X,S)
                # get gradient and check stopping-time statistics
                G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq = getFullGradient(X,S)
                rel_max_grad = sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
                rel_avg_grad = sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )
                if verbose:
                    print ("SGD iter=%d,   emp_loss=%f,   log_loss=%f, "
                           "rel_avg_grad=%f,   rel_max_grad=%f,   a=%f") % (t,
                                                                            emp_loss,
                                                                            log_loss,
                                                                            rel_avg_grad,
                                                                            rel_max_grad,
                                                                            a)
                if rel_max_grad < epsilon:
                    break
        # get random triplet unifomrly at random
        q = S[randint(m)]
        grad_partial = getPartialGradient(X,q)/m
        X[q,:] = X[q,:] - a*grad_partial[q,:]
        # project back onto ball such that norm(X[i])<=max_norm
        for i in q:
            norm_i = norm(X[i])
            if norm_i>max_norm:
                X[i] = X[i] * (max_norm / norm_i)
    return X,rel_max_grad


def computeEmbeddingWithGD(X,S,max_iters=50,max_norm=1.,epsilon=0.01,c1=.00001,rho=0.5,verbose=False):
    """
    Performs gradient descent with geometric amarijo line search (with parameter c1)
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.
    Implements line search algorithm 3.1 of page 37 in Nocedal and Wright (2006) Numerical Optimization

    Inputs:
        (numpy.ndarray) X : input embedding
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, exits if gamma<epsilon (default = 0.01)
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) X : output embedding
        (float) emp_loss : output 0/1 error
        (float) log_loss : output log loss
        (float) gamma : Equal to a/b where a is max row norm of the gradient matrix and 
                        b is the avg row norm of the centered embedding matrix X. 
                        This is a means to determine how close the current solution is to the "best" solution.  
    Usage:
        X,gamma = computeEmbeddingWithGD(X,S)
    """
    n,d = X.shape
    alpha = 1
    t = 0
    emp_loss_0 = float('inf')
    log_loss_0 = float('inf')
    rel_max_grad = float('inf')
    if verbose:
        emp_loss,log_loss = getLossX(X,S)
        print "GD iter=%d,   emp_loss=%f,   log_loss=%f,   a=%f" % (0,emp_loss,log_loss,float('nan'))
    while t < max_iters:
        t+=1
        # get gradient and stopping-time statistics
        G, avg_grad_row_norm_sq, max_grad_row_norm_sq, avg_row_norm_sq = getFullGradient(X,S)
        rel_max_grad = sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
        rel_avg_grad = sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )
        if rel_max_grad < epsilon:
            break
        # perform backtracking line search
        emp_loss_0,log_loss_0 = getLossX(X,S)
        norm_grad_sq_0 = avg_grad_row_norm_sq*n
        emp_loss_k,log_loss_k = getLossX(X-alpha*G, S)
        alpha = 1.1*alpha
        inner_t = 0
        while log_loss_k > log_loss_0 - c1*alpha*norm_grad_sq_0:
            alpha = alpha*rho
            emp_loss_k,log_loss_k = getLossX(X-alpha*G,S)
            inner_t += 1
        X = X-alpha*G                

        #check losses
        if verbose:
            print ("GD iter=%d,   emp_loss=%f,   log_loss=%f,"
                   "rel_avg_grad=%f,   rel_max_grad=%f,   alpha=%f,   i_t=%d") % (t,
                                                                                  emp_loss_k,
                                                                                  log_loss_k,
                                                                                  rel_avg_grad,
                                                                                  rel_max_grad,
                                                                                  alpha,inner_t)
            
    return X,emp_loss_0,log_loss_0,rel_max_grad


if __name__ == "__main__":
    run(100, 2,plot=True, verbose=True)
