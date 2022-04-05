"""
Approximative solution of the charfunction for the HestonOU model
Solving the SDE system (Lemma 3.1) as on page8 in (Teng et al., 2016c)
"""
import numpy as np
from scipy.integrate import solve_ivp
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import scipy.integrate as integrate
# from charfuncStochCorrOU_analytic import charfuncHestonOU

# solving tge complex-valued ODE system; using that B = iu (and thus B^2=-u^2)
def F(t,y,u,kv,mv,sv,kr,mr,sr,rho2,v0,E,r):
    a,c,d = y
    return [(1j*u-0)*r+kv*mv*d+kr*mr*c+0.5*sr**2*c**2+sr*rho2*E*1j*u*c,\
            sv*v0*1j*u*d-kr*c,\
            -0.5*u**2+0.5*sv**2*d**2-0.5*1j*u-kv*d]

def CF_OU(p,x0,r,T,u):
    # numerically integrate the system of ODEs
    # model vars
    kv = p[0]; mv = p[1]; sv = p[2]; v0 = p[3] # variance process param
    kr = p[4]; mr = p[5]; sr = p[6]; rho0 = p[7] # correlation process param
    rho2 = p[8] #correlation between dp and dv (set to be constant)
    m = np.sqrt(mv-sv**2/(8*kv)+0j) # aux var
    # n = np.sqrt(v0) - m
    # integrate, for a given u, the system of ODEs in ]0,T]
    sol = solve_ivp(F,[0,T],[0j,0j,0j],method='BDF', \
    args=(u,kv,mv,sv,kr,mr,sr,rho2,v0,m,r),\
    dense_output=True)
    # get the CF at T
    A = sol.sol(T)[0]; C = sol.sol(T)[1]; D = sol.sol(T)[2]
    
    return np.exp(-r*T+A+1j*u*x0+C*rho0+D*v0)