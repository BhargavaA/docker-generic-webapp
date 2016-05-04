import time
import numpy as np
from numpy import *
from numpy.random import *
from numpy.linalg import *
import multiprocessing
from utils import *
import matplotlib.pyplot as plt
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
    n = 10
    d = 2
    m = 2000  # number of labels        
    Xtrue = 1*randn(n,d);
    Xtrue = Xtrue -1./n*dot(ones((n,n)), Xtrue)
    Mtrue = dot(Xtrue, Xtrue.transpose())

    trace_norm = trace(Mtrue)
    Strain,train_bayes_err = getTriplets(Xtrue, m, noise=True)
    Stest,test_bayes_err = getTriplets(Xtrue, m, noise=True)
    Mhat, emp_loss_train = computeEmbedding(n,d,Strain,
                                            max_iter_GD=1000,
                                            epsilon=0.0001,
                                            trace_norm=trace_norm,
                                            verbose=verbose)
    emp_loss_train,log_loss_train = getLossGram(Mhat, Strain)
    emp_loss_test,log_loss_test = getLossGram(Mhat, Stest)
    print ('Empirical Training loss = {},   '
           'Empirical Test loss = {},   '
           'Bayes Test loss = {},   '
           'Relative Error = {} ').format(emp_loss_train, emp_loss_test, test_bayes_err, norm(Mtrue-Mhat,'fro')**2/norm(Mtrue,'fro')**2)
    if plot:
        _, Xhat = transformGramtoX(Mhat,2)
        _, Xpro, _ = procrustes(Xtrue, Xhat)
        twodplot(Xtrue, Xpro)
        stemplot(Mtrue, Mhat)
        plt.show()

    return Mtrue, Mhat

def computeEmbedding(n,d,S,num_random_restarts=0,max_iter_GD=1000,trace_norm=1,epsilon=0.0001,verbose=False):
    """
    Computes STE MLE of a Gram matrix M using the triplets of S.
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.

    Inputs:
        (int) n : number of objects in embedding
        (int) d : desired dimension
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n].
        (int) num_random_restarts : number of random restarts (nonconvex
        optimization, may converge to local minima). E.g., 9 random restarts
        means take the best of 10 runs of the optimization routine.
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) trace_norm : the maximum allowed trace norm of the gram matrix
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
    """

    M = randn(n,n)
    M = (M+M.transpose())/2
    M = M - 1./n*dot(ones((n,n)),M)
    ts = time.time()
    M_new,emp_loss_new,log_loss_new,acc_new = computeEmbeddingWithGD(M,S,d,
                                                                     max_iters=max_iter_GD,
                                                                     trace_norm=trace_norm,
                                                                     epsilon=epsilon,
                                                                     verbose=verbose)
    te_gd = time.time()-ts        
    if verbose:
        print ("emp_loss = %f,   "
               "log_loss = %f,   duration=%f") % (emp_loss_new,
                                                 log_loss_new,
                                                 te_gd)
    return M_new,emp_loss_new


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def projected(M, R):
    '''
    Project onto psd nuclear norm ball of radius R
    '''
    if R!=None:
        n, n = shape(M)
        D, V = eigh(M)
        D = euclidean_proj_simplex(D, s=R)
        M = dot(dot(V,diag(D)),V.transpose());

    return M


def computeEmbeddingWithGD(M,S,d,max_iters=1000,trace_norm=None,epsilon=0.0001,c1=.00001,rho=0.5,verbose=False):
    """
    Performs gradient descent with geometric amarijo line search (with parameter c1)
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.
    Implements line search algorithm 3.1 of page 37 in Nocedal and Wright (2006)

    Inputs:
        (numpy.ndarray) M : input Gram matrix
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) trace_norm : the maximum allowed trace norm of gram matrix (default equals inf)
        (float) epsilon : parameter that controls stopping condition, 
                          exits if new iteration does not differ in fro norm by more than epsilon (default = 0.0001)
        (float) c1 : Amarijo stopping condition parameter (default equals 0.0001)
        (float) rho : Backtracking line search parameter (default equals 0.5)
        (boolean) verbose : output iteration progress or not (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
        (float) emp_loss : output 0/1 error
        (float) log_loss : output log loss
    Usage:
        X,gamma = computeEmbeddingWithGD(X,S)
    """
    print 'epsilon = ' + str(epsilon)
    n,n = M.shape
    alpha = 1
    t = 0
    emp_loss_0 = float('inf')
    log_loss_0 = float('inf')
    rel_max_grad = float('inf')
    while t < max_iters:
        t+=1
        # get gradient and stopping-time statistics
        G, avg_grad_row_norm_sq, max_grad_row_norm_sq, avg_row_norm_sq = getFullGradientGram(M,S)
        rel_max_grad = sqrt( max_grad_row_norm_sq / avg_row_norm_sq )
        rel_avg_grad = sqrt( avg_grad_row_norm_sq / avg_row_norm_sq )
        if rel_max_grad < epsilon:
            break
        # perform backtracking line search
        emp_loss_0,log_loss_0 = getLossGram(M,S)
        norm_grad_sq_0 = avg_grad_row_norm_sq*n                             

        M_k = projected(M-alpha*G, trace_norm)
        emp_loss_k,log_loss_k = getLossGram( M_k , S)
        d_k = M_k - M

        Delta = norm(d_k,ord='fro')
        if Delta<epsilon:
            break

        # This linesearch comes from Fukushima and Mine, "A generalized proximal point algorithm for certain non-convex minimization problems"
        beta = rho
        inner_t = 0
        while log_loss_k > log_loss_0 - c1*alpha*norm_grad_sq_0:
            beta = beta*beta
            emp_loss_k,log_loss_k = getLossGram( M+beta*d_k ,S)
            inner_t += 1
            if inner_t > 10:
                break
        if inner_t>0:
            alpha = max(0.1,alpha*rho)
        else:
            alpha = 1.2*alpha

        M = projected(M+beta*d_k, trace_norm)

        if verbose:
            print ("GD iter=%d,   emp_loss=%f,   log_loss=%f,   "
                   "d_k_fro_norm=%f,   alpha=%f,   i_t=%d") % (t,
                                                                                  emp_loss_k,
                                                                                  log_loss_k,
                                                                                  Delta,
                                                                                  alpha,inner_t)
    return M,emp_loss_0,log_loss_0,rel_max_grad


if __name__ == "__main__":
    run(100, 2,plot=True, verbose=True)


