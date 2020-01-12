"""
This file contains prototype scripts for the approximate simulation
of 1-D and 2-D Gaussian random fields with a specified covariance function
C(x,y)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import time
from scipy import interpolate 
def KL_1DNys(N,M,a,b,Cov,quad = "EOLE"):
    """
    Karhunen-Loeve in 1-Dimension using Nystrom method.
    
    
    Parameters:

    N: Order of the Karhunen-Loeve expansion.

    M: number of quadrature intervals . N <=M

    a,b: domain of simulation, X_t for t in [a,b]

    Cov: The covariance function, a bivariate function

    quad: Quadrature used."EOLE" for the EOLE method. I tried Gauss-Legendre
    before and there was an issue with inaccurate simulation at the end
    points of the simulation domain
    -----
    output
    -----
    X: a 1-D array of the random field

    phi: a 2-D arrray whose columns are the eigenfunctions

    L: an 1-D array of the eigenvalues.

    """
    if N > M:
        raise ValueError('Order of expansion N should be less than quadrature\
points used')
    if quad == "EOLE": # EOLE method
        x = np.linspace(a,b,M+1) # EOLE uniform grid.
        W = (1./M)*(b-a)*np.eye(M+1) #EOLE weight matrix
        x1,x2 = np.meshgrid(x,x)
        C = Cov(x1,x2) # covariance matrix
        B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) #symmetric B matrix.
        L,y = spla.eigsh(B,k=N) #eigenvalues and vectors of B.
        arg_sort = np.argsort(-L) # indices for sorting.
        L,y =L[arg_sort].real, y[:,arg_sort].real #re-order the eigens.
        X = np.zeros(M+1)
        W_inv = np.sqrt((float(M)/(b-a)))*np.eye(M+1) # weights matrix.
        phi = np.dot(W_inv,y) # original eigenvector problem.
        Z = np.random.randn(M+1)
        for i in range(N):
            X += Z[i]*np.sqrt(L[i])*phi[:,i]
        return X, phi, L
    else:
        raise ValueError('We only have EOLE quadrature for now.')

def KL_2DNys(N,M,lims,Cov,quad = "EOLE"):
    """
    Simulate 2D Gaussian random fields with the Karhunen-Loeve approximation
    -----
    input
    -----
    N: The order of the Karhunen-Loeve expansion.
    M: M = [M1,M2] number of grid points along x and y direction.
    lims: lims = [a,b,c,d] simulation domain is [a,b] x [c,d]
    Cov: the covariance function. Should be given as c(x,y), x and y bivariate.
    quad: the quadrature method used. EOLE only implemented for now.
    """
    M1,M2 = M # extract M1 and M2
    n,m  = M1+1,M2+1 # save space. 
    a,b,c,d = lims # extract domain limits
    Om = (b-a)*(d-c) # Omega area of the rectangular domain.
    x, y = np.linspace(a,b,n), np.linspace(a,b,m) 
    W =(Om/(n*m))*np.eye(n*m)
    #create list of coordinates
    xx = np.hstack([np.repeat(x,m).reshape(n*m,1),np.tile(y,n).reshape(n*m,1)])
    xxx = np.hstack([np.repeat(xx,n*m,axis=0),np.tile(xx,[n*m,1])])
    C = Cov(xxx[:,0:2],xxx[:,2:]).reshape(n*m,n*m) #Covariance matrix, check this.
    B = np.dot(np.dot(np.sqrt(W),C),np.sqrt(W)) # symmetric pos def B
    # timing test
    t0 = time.clock()
    #L,y = np.linalg.eigh(B) # eigeevalues and vectors of B.
    L,y = spla.eigsh(B,k=N) #eigenvalues and vectors of B.
    arg_sort = np.argsort(-L)
    L,y =L[arg_sort].real, y[:,arg_sort].real #re-order the eigens.
    #Reverse order of EV and their vectors as eigh returns ascenidng order.
    #L,y = L[::-1], y[:,::-1]
    t1 = time.clock()
    print('Eigenvalue problem solved after: {} units'.format(t1-t0))
    W_inv = np.sqrt(float(n*m)/Om)*np.eye(n*m) # invert W matrix.
    phi = np.dot(W_inv,y)
    X = np.zeros((n,m)) # array to hold solution
    Z = np.random.randn(N) #iid standard normals
    for i in range(N):
        X+= np.sqrt(L[i])*Z[i]*phi[:,i].reshape(n,m)
    return X,phi,L # just return eigensuite for now
if __name__ == "__main__":
    pass
 