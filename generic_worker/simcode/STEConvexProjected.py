import time
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
    # regularization parameter for nuclear norm
    l = 0.001
    Xtrue = 1*randn(n,d);
    Xtrue = Xtrue -1./n*dot(ones((n,n)), Xtrue)
    Mtrue = dot(Xtrue, Xtrue.transpose())
    Strain,train_bayes_err = getTriplets(Xtrue, m, noise=True)
    Stest,test_bayes_err = getTriplets(Xtrue, m, noise=True)
    Mhat, emp_loss_train = computeEmbedding(n,d,Strain,
                                            max_num_passes_SGD=16,
                                            max_iter_GD=50,
                                            num_random_restarts=1,
                                            epsilon=0.0001,
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

def computeEmbedding(n,d,S,num_random_restarts=0, max_num_passes_SGD=16,max_iter_GD=50,max_norm=1,epsilon=0.01,verbose=False):
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
        (int) max_num_passes_SGD : maximum number of passes over data SGD makes before proceeding to GD (default equals 16)
        (int) max_iter_GD: maximum number of GD iteration (default equals 500)
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, smaller means more accurate (default = 0.01)
        (boolean) verbose : outputs some progress (default equals False)

    Outputs:
        (numpy.ndarray) M : output Gram matrix
    """
    M_old = None
    emp_loss_old = float('inf')
    for restart in range(num_random_restarts):
        if verbose:
            print "Restart:{}".format(restart)

        M = randn(n,n)
        M = (M+M.transpose())/2
        M = M - 1./n*dot(ones((n,n)),M)
        ts = time.time()
        M_new,emp_loss_new,log_loss_new,acc_new = computeEmbeddingWithGD(M,S,d,
                                                                         max_iters=max_iter_GD,
                                                                         max_norm=max_norm,
                                                                         epsilon=epsilon,
                                                                         verbose=verbose)
        te_gd = time.time()-ts        
        if emp_loss_new < emp_loss_old:
            M_old = M_new
            emp_loss_old = emp_loss_new
        if verbose:
            print ("restart %d:   emp_loss = %f,   "
                   "log_loss = %f,   duration=%f") %(restart,
                                                     emp_loss_new,
                                                     log_loss_new,
                                                     te_gd)
    return M_old,emp_loss_old


def projected(M, d):
    '''
    Project onto rank d psd matrices
    '''
    n, n = shape(M)
    D, V = eigh(M)
    perm = D.argsort()
    bound = max(D[perm][-d], 0)
    for i in range(n):
        if D[i] < bound:
            D[i] = 0
    M = dot(dot(V,diag(D)),V.transpose());
    return M


def computeEmbeddingWithGD(M,S,d,max_iters=50,max_norm=1.,epsilon=0.01,c1=.00001,rho=0.5,verbose=False):
    """
    Performs gradient descent with geometric amarijo line search (with parameter c1)
    S is a list of triplets such that for each q in S, q = [i,j,k] means that
    object k should be closer to i than j.
    Implements line search algorithm 3.1 of page 37 in Nocedal and Wright (2006)

    Inputs:
        (numpy.ndarray) M : input Gram matrix
        (list [(int) i, (int) j,(int) k]) S : list of triplets, i,j,k must be in [n]. 
        (int) max_iters : maximum number of iterations of SGD (default equals 40*len(S))
        (float) max_norm : the maximum allowed norm of any one object (default equals 10*d)
        (float) epsilon : parameter that controls stopping condition, 
                          exits if new iteration does not differ in fro norm by more than epsilon (default = 0.001)
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

        alpha = 1.3*alpha
        M_k = projected(M-alpha*G, d)
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

        M = projected(M+beta*d_k, d)

        if verbose:
            print ("GD iter=%d,   emp_loss=%f,   log_loss=%f,   "
                   "d_k_fro_norm=%f,   alpha=%f,   i_t=%d") % (t,
                                                                                  emp_loss_k,
                                                                                  log_loss_k,
                                                                                  Delta,alpha,inner_t)
    return M,emp_loss_0,log_loss_0,rel_max_grad


if __name__ == "__main__":
    run(100, 2,plot=True, verbose=True)
