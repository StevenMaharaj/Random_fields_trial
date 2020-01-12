import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

def circ_embed1D(g,a,b,Cov):
    """
    The Circulant embedding method in 1-Dimension
    -----
    input
    -----
    g: exponent of the sample size N = 2^g
    a,b: terminals of domain.
    Cov: a stationary covariance function of one argument.
    -----
    output
    -----
    X: a 1-D array of the random field
    """
    N = 2**g
    mesh = (b-a)/N
    #print(mesh)
    x = np.arange(0,N)*mesh # domain grid
    r = np.zeros(2*N) #row defining the symmetric circulant matrix
    r[0:N] = Cov(x[0:N]-x[0])
    r[N+1:2*N] = Cov(x[N-1:0:-1]-x[0])
    L =np.fft.fft(r).real # eigenvalues of circulant matrix.
    neg_Evals = L[L <0] # produce a 'list' of negative eigenvalues
    if len(neg_Evals) == 0:
        pass # if there are no negative eigenvalues, continues
    elif np.absolute(neg_Evals.min()) < 1e-16:
        L[ L <0] = 0 # eigenvalues are zero to machine precision.
    else:
        raise ValueError("Could not find a positive definite circulant matrix")
    V1,V2 = np.random.randn(N), np.random.randn(N) # generate iid normals
    W = np.zeros(2*N, dtype = np.complex_)
    W[0] = np.sqrt(L[0]/(2*N))*V1[0]
    W[1:N] = np.sqrt(L[1:N]/(4*N))*(V1[1:N] +1j*V2[1:N])
    W[N] = np.sqrt(L[N]/(2*N))*V1[0]
    W[N+1:2*N] = np.sqrt(L[N+1:2*N]/(4*N))*(V1[N-1:0:-1] -1j*V2[N-1:0:-1])
    #Take fast Fourier transform of the special vector W
    w = np.fft.fft(W)
    return w[0:N].real # return first half of the vector.
def circ_embed2D(n,m,lims,R):
    """
    To simulate a 2-D stationary Gaussian field with the circulant embedding
    method in two dimensions.
    -----
    input
    -----
    n: number of grid points  in the x-direction.
    m: number of grid points in the y-direction.
    lims: a 4-d vector containing end points of rectangular domain.
    R: Covariance function of the Gaussian process, a bivariate function.
    -----
    output
    -----
    field1: The first field outputed, real part from the embedding method.
    field2: The second field outputed, imaginary part from the emedding method.
    """
    a,b,c,d = lims  #extract interval terminals from lims variable.
    dx,dy  = (b-a)/(n-1),(d-c)/(m-1)
    tx,ty = np.array(range(n),float)*dx, np.array(range(m),float)*dy
    Row, Col = np.zeros((n,m)), np.zeros((n,m))
    Row = R(tx[None,:] - tx[0],ty[:,None]-ty[0]) # Row definining block circulant
    Col = R(-tx[None,:] +tx[0], ty[:,None] - ty[0]) # columns defining block circulant
    #construct the block circulant matrix.
    Blk_R = np.vstack(
        [np.hstack([Row,Col[:,-1:0:-1]]),
         np.hstack([Col[-1:0:-1,:],Row[-1:0:-1,-1:0:-1]])])
    L = np.real(np.fft.fft2(Blk_R))/((2*m-1)*(2*n-1)) # eigenvalues
    neg_vals = L[L < 0]
    if len(neg_vals) == 0:
        pass # If there are no negative values, continue
    elif np.absolute(np.min(neg_vals)) < 10e-15:
        L[ L < 0] = 0 # EV negative due to numerical precision, set to zero.
    else:
        raise ValueError(" Could not find a positive definite embedding")
    L = np.sqrt(L) # component wise square root
    Z = (np.random.randn(2*m -1,2*n-1)
         + 1j*np.random.randn(2*m-1,2*n-1)) # add a standard normal complex
    F = np.fft.fft2(L*Z)
    F = F[:m,:n]
    field1, field2 = np.real(F), np.imag(F)
    return field1, field2
        
    raise ValueError("Could not find a positive definite embedding")
    lam = np.sqrt(lam)
                       
