import argparse
import matplotlib.pyplot as plt
import numpy as np
from GuassRF.KL_code import KL_1DNys,KL_2DNys
from GuassRF.circ_embed_code import circ_embed1D,circ_embed2D


parser = argparse.ArgumentParser(prog="package_test.py",description='Choose method to test')
parser.add_argument('-m', choices=['KL1D', 'KL2D','CE1D', 'CE2D'],type=str)
args = parser.parse_args()


if args.m == "KL1D":
        N = 200 # order of the KL expansion
        M = 200 # M+1 quadrature points
        def Bm(t,s):
            return np.minimum(t,s)
        a, b = 0., 1. # domain of simulation
        X,phi,L = KL_1DNys(N,M,a,b,Bm)
    # plot eigenvalues: pi/L = (k-0.5)**2 for BM
        L_ex = [(k+0.5)**2 for k in range(10)]
        L_app = 1./(L[:10]*np.pi**2)
        plt.plot(L_ex, label = "exact eigenvalues")
        plt.plot(L_app,'x', label = "numerical eigenvalues")
        plt.legend()
        plt.ylabel(r' $\frac{1}{\lambda_k\pi^2}$')
        plt.title(' Eigenvalues')
        plt.savefig("BM_EV_eg.pdf")
        plt.close()
        t= np.linspace(a,b,M+1) # t-grid
        exact = np.sqrt(2)*np.sin(4.5*np.pi*t) # exact fifth eigenfunction
        apprx= np.abs(phi[:,4])*np.sign(exact)# approximate 5th ef. Given same sign as exact.
        plt.plot(t, exact,'x',label= "Exact")
        plt.plot(t, apprx, label = "Numerical")
        plt.title("Eigenfunction, k = {}".format(5))
        plt.legend()
        plt.savefig("BM_EF_eg.pdf")
        plt.close()
        t = np.linspace(a,b,M+1) # time grid
        plt.plot(t,X)
        plt.title(" Brownian motion KL simulation")
        plt.savefig("BM_eg.pdf")
elif args.m == "KL2D":
    N = 100 #KL expansion order
    M  =[50,50] # number of points in x- and y-directions.
    A = np.array([[1,0.8],[0.8,1]]) # anisotropic matrix
    #def Cov(x,y, A = A):
    #    s = x - y
    #    arg= A[0,0]*s[:,0]**2 +(A[1,0]+ A[0,1])*s[:,0]*s[:,1] + A[1,1]*s[:,1]**2
    #    return np.exp(-arg)
    def Cov(x,y,rho =0.1):
        r = np.linalg.norm(x - y,1,axis = 1)
        return np.exp(-r/rho)   
    lims = [0.,1.,0.,1.] # domain corners
    x,y = np.linspace(lims[0],lims[1],M[0]+1),np.linspace(lims[2],lims[3],M[1]+1)
    xx,yy = np.meshgrid(x,y, indexing ='ij')
    X,phi,L = KL_2DNys(N,M,lims,Cov)
    print(L[:3])
    plt.loglog(range(N),L[:N])
    plt.title("The exponential's first {} eigenvalues".format(N))
    plt.savefig("exponential_2D_eigenvalues.pdf")
    plt.close()
    for i in range(6):
        plt.subplot(2,3,i+1).set_title('k = {}'.format(i+1))
        e_func = np.array(phi[:,i]).reshape(M[0]+1,M[1]+1)
        plt.pcolor(xx,yy,e_func)
        plt.colorbar()
    plt.savefig("exponential_eigenfunctions.pdf")
    plt.close()
    #X = np.zeros((200,200)) # array to hold solution
    #Z = np.random.randn(N) #iid standard normals
    #s,t = np.linspace(0.,1.,200), np.linspace(0.,1.,200) # finer grid to evaluate on
    #ss,tt = np.meshgrid(s,t,indexing = 'ij')
    #for i in range(N):
    #    eig_array = np.array(phi[:,i]).reshape(M[0]+1,M[1]+1)
    #    e_func = interpolate.interp2d(x,y,eig_array)
    #    eig_field = e_func(s,t)
    #    X+= np.sqrt(L[i])*Z[i]*eig_field
    plt.pcolor(xx,yy,X)
    plt.colorbar()
    plt.savefig("exponential_RF_test.pdf")
    plt.show()
elif args.m == "CE1D":
    h=0.01 # Hurst parameter for fractional Brownian motion. 
    l = 1.0 # parameter for the exponential covariance
    g= 10
    N = 2**g 
    #t_vals = np.linspace(0,1.,N+1)
    t_vals = np.linspace(-5.,5.,N)
    def Cov(k,H=h):
        return (np.abs(k-1)**(2*H) -2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))/2
    def exp_cov(r, l =l):
        return np.exp(-r/l)
    X = circ_embed1D(g,-5.,5.,exp_cov)
    #X = circ_embed1D(g,0.,float(N),Cov) # simulate a fractional Gaussian noise.
    #X2 = np.insert((1./N)**h*np.cumsum(X),0,0) # fractional Brownian motion + starting point
    plt.plot(t_vals,X)
    plt.xlim([-5,5])
    #plt.title(" H = {:.2f}".format(h))
    #plt.savefig("circ_embed_fBm.pdf")
    plt.title(" Exponential covariance, scale length l = {:.1f}".format(l))
    plt.savefig("circ_embed1D_exp.pdf")
    plt.show()
elif args.m == "CE2D":
    l1 = 15 # scale length in x/y direction
    l2 = 50 # scale length in x/y direction
    def R(x,y, l1 = l1, l2 = l2):
        A  = np.array([[3,1],[1,2]])
        arg = ( (x/l1)**2*A[1,1] + (A[0,1] + A[1,0])*(x/l1)*(y/l2)
                + (y/l2)**2*A[0,0] )
        return np.exp(-np.sqrt(arg))
    lims = [1.,383.,1.,511.] # limits: dx = 1, dy =1
    field1, field2 = circ_embed2D(383,511,lims,R)
    plt.title(r' $l_1$ = {}, $l_2$ = {}'.format(l1,l2))
    plt.imshow(field1)
    plt.savefig('circ_embed2D_aniso.pdf')    
    plt.show()
