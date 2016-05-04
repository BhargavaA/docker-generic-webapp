import time
from numpy import *
from numpy.random import *
from numpy.linalg import *
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import multiprocessing
import cPickle as pickle

def getTriplets(X,pulls,noise=False):
    S = []
    n,d = X.shape
    error = 0.
    for i in range(0,pulls):
        # get random triplet
        q = getRandomQuery(n)
        score = getTripletScoreX(X,q)
        # align it so it agrees with Xtrue: "q[2] is more similar to q[0] than q[1]"
        if score > 0:
            q = [q[i] for i in [1,0,2]]
        # add some noise
        if noise:
            if rand() > 1/(1+exp(getTripletScoreX(X,q))):
                q = [ q[i] for i in [1,0,2]]
                error+=1
        S.append(q)   
    error /= float(pulls)
    print error        
    return S,error


def getRandomQuery(n):
    """
    Outputs a triplet [i,j,k] chosen uniformly at random from all possible triplets 
    and score = abs( ||x_i - x_k||^2 - ||x_j - x_k||^2 )
    
    Inputs:
        (numpy.ndarray) X : matrix from which n is extracted from and score is derived
        
    Outputs:
        [(int) i, (int) j, (int) k] q : where k in [n], i in [n]-k, j in [n]-k-j
        (float) score : signed distance to current solution (positive if it agrees, negative otherwise)
        
    Usage:
        q,score = getRandomQuery(X)
    """
    i = randint(n)
    j = randint(n)
    while (j==i):
        j = randint(n)
    k = randint(n)
    while (k==i) | (k==j):
        k = randint(n)
    q = [i, j, k]
    return q


def getPartialGradientGram(M,q):
    """
    Returns normalized gradient of logistic loss wrt to X and a single query q.
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If loss is logistic_loss then loss(X,q) = log(1+exp(s(X,q)))

    Usage:
        G = getPartialGradient(X,S)
    """
    n,n = M.shape
    # pattern for computing gradient
    H = mat([[1.,0.,-1.],[ 0.,  -1.,  1.],[ -1.,  1.,  0.]])
    # compute gradient 
    G = zeros((n,n))
    score = getTripletScoreGram(M,q)
    G[[[x] for x in q],q] = H 
    grad_partial = 1/(1+exp(-score))* G
    return grad_partial


def getFullGradientGram(M,S):
    """
    Returns normalized gradient of logistic loss wrt to X and S.
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If loss is logistic_loss then loss(X,q) = log(1+exp(s(X,q)))

    Usage:
        G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq = getFullGradient(X,S)
    """
    n,n = M.shape
    # compute Gradient
    G = zeros((n,n))
    m = len(S)
    for q in S:
        G = G + getPartialGradientGram(M,q)/m
    # compute statistics about gradient used for stopping conditions
    mu = mean(M,0)
    avg_row_norm_sq = 0.
    avg_grad_row_norm_sq = 0.
    max_grad_row_norm_sq = 0.
    for i in range(n):
        row_norm_sq = 0
        grad_row_norm_sq = 0
        for j in range(n):
            row_norm_sq += (M[i,j]-mu[j])*(M[i,j]-mu[j])
            grad_row_norm_sq += G[i,j]*G[i,j]
        avg_row_norm_sq += row_norm_sq/n
        avg_grad_row_norm_sq += grad_row_norm_sq/n
        max_grad_row_norm_sq = max(max_grad_row_norm_sq,grad_row_norm_sq)
    return G,avg_grad_row_norm_sq,max_grad_row_norm_sq,avg_row_norm_sq


def getTripletScoreGram(M,q):
    """
    Given M,q=[i,j,k] returns score = M_ii - M_jj - 2(M_ik-M_jk)
    If score < 0 then the triplet agrees with the embedding, otherwise it does not 

    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    i,j,k = q

    return M[i,i] -2*M[i,k] + 2*M[j,k] - M[j,j]


def getLossGram(M,S):
    """
    Returns loss on M with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(X,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If loss is hinge_loss then loss(X,q) = max(0,1-s(X,q))
    If loss is emp_loss then loss(X,q) = 1 if s(X,q)<0, and 0 otherwise
    If loss is log_loss then loss(X,q) = -log(1+exp(s(X,q)))
    Usage:
        emp_loss, hinge_loss = getLoss(X,S)
    """
    emp_loss = 0 # 0/1 loss
    log_loss = 0 # logistic loss
    for q in S:
        loss_ijk = getTripletScoreGram(M,q)
        if loss_ijk > 0:
            emp_loss = emp_loss + 1.
        log_loss = log_loss+log(1+exp(loss_ijk))
    emp_loss = emp_loss/len(S)
    log_loss = log_loss/len(S)
    return emp_loss, log_loss 


def getTripletScoreX(X,q):
    """
    Given X,q=[i,j,k] returns score = ||x_j - x_k||^2 - ||x_i - x_k||^2
    If score > 0 then the triplet agrees with the embedding, otherwise it does not 

    Usage:
        score = getTripletScore(X,[3,4,5])
    """
    i,j,k = q

    return dot(X[i],X[i]) -2*dot(X[i],X[k]) + 2*dot(X[j],X[k]) - dot(X[j],X[j])


def getLossX(X,S):
    """
    Returns loss on X with respect to list of triplets S: 1/len(S) \sum_{q in S} loss(X,q).
    Intuitively, q=[i,j,k] "agrees" with X if ||x_j - x_k||^2 > ||x_i - x_k||^2.

    For q=[i,j,k], let s(X,q) = ||x_i - x_k||^2 - ||x_j - x_k||^2
    If loss is hinge_loss then loss(X,q) = max(0,1-s(X,q))
    If loss is emp_loss then loss(X,q) = 1 if s(X,q)<0, and 0 otherwise
    If loss is log_loss then loss(X,q) = -log(1+exp(s(X,q)))
    Usage:
        emp_loss, hinge_loss = getLoss(X,S)
    """
    emp_loss = 0 # 0/1 loss
    log_loss = 0 # logistic loss
    for q in S:
        loss_ijk = getTripletScoreX(X,q)
        if loss_ijk > 0:
            emp_loss = emp_loss + 1.
        log_loss = log_loss+log(1+exp(loss_ijk))
    emp_loss = emp_loss/len(S)
    log_loss = log_loss/len(S)
    return emp_loss, log_loss 


def twodplot(X, Y):
    n,d = X.shape
    plt.figure(1)
    # Plot Xtrue
    plt.subplot(131)
    plt.plot(*zip(*X), marker='o', color='r', ls='')
    # Plot Xhat
    plt.subplot(132)
    plt.plot(*zip(*Y), marker='o', color='b', ls='')

    # Overlap plot
    plt.subplot(133)
    plt.plot(*zip(*X), marker='o', color='r', ls='')
    for i in range(n):
        point = X[i,:].tolist()
        if d==1:
            point = [point[0],0]
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(point[0], point[1]),
                     xytext = (-5, 5),
                     ha = 'right',
                     va = 'bottom',
                     color='red',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))
    plt.plot(*zip(*Y), marker='o', color='b', ls='')
    for i in range(n):
        point = Y[i,:].tolist()
        if d==1:
            point = [point[0],0]
        plt.annotate(str(i),
                     textcoords='offset points',
                     xy=(point[0], point[1]),
                     xytext = (-5, -5),
                     ha = 'right',
                     va = 'bottom',
                     color='blue',
                     arrowprops = dict(arrowstyle = '-',
                                       connectionstyle = 'arc3,rad=0'))


def transformGramtoX(M,d):
    '''
    Get a set of points back from a Gram Matrix
    '''
    n,n = M.shape
    U,s,V = svd(M)
    
    for i in range(d, n):
        s[i] = 0
    s = diag(s)
    Mp = dot(dot(U.real,s),V.real.transpose())
    X = dot(U.real,sqrt(s).real)
    return Mp,X[:,0:2]
        

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    http://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape
    muX = X.mean(0)
    muY = Y.mean(0)
    X0 = X - muX
    Y0 = Y - muY
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    # centred Frobenius norm
    normX = sqrt(ssX)
    normY = sqrt(ssY)
    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = concatenate((Y0, zeros(n, m-my)),0)
    # optimum rotation matrix of Y
    A = dot(X0.T, Y0)
    U,s,Vt = linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = dot(V, U.T)

    if reflection is not 'best':
        # does the current solution use a reflection?
        have_reflection = linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = dot(V, U.T)
    traceTA = s.sum()
    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*dot(Y0, T) + muX
    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*dot(muY, T)
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
    return d, Z, tform


def stemplot(G, M):
    plt.figure(2)
    plt.subplot(121)
    plt.stem(eigh(G)[0], color ='r')
    plt.subplot(122)
    plt.stem(eigh(M)[0], color = 'b')


def runParallel(f, args):
    seed()
    return f(*args)
    
