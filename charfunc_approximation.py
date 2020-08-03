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
    kv = p[0]; mv = p[1]; sv = p[2]; v0 = p[3]; # variance process param
    kr = p[4]; mr = p[5]; sr = p[6]; rho0 = p[7]; # correlation process param
    rho2 = p[8]; #correlation between dp and dv (set to be constant)
    m = np.sqrt(mv-sv**2/(8*kv)+0j); # aux var
    # n = np.sqrt(v0) - m
    # integrate, for a given u, the system of ODEs in ]0,T]
    sol = solve_ivp(F,[0,T],[0j,0j,0j],method='BDF', \
    args=(u,kv,mv,sv,kr,mr,sr,rho2,v0,m,r),\
    dense_output=True)
    # get the CF at T
    A = sol.sol(T)[0]; C = sol.sol(T)[1]; D = sol.sol(T)[2]
    
    return np.exp(-r*T+A+1j*u*x0+C*rho0+D*v0)

# s0 = 100; T = 5; r = 0; K = 100 # underlying
# # model parameters from table 2 in the paper Teng et al. - On the Heston model ...
# kv = 2.1; mv = 0.03; sv = 0.2; v0 = 0.02;
# kr = 3.4; mr = -0.6; sr = 0.1; rho0 = -0.4; rho2 = 0.4;

# s0 = 2461.44; K = 2461.44; T = 1/52; r = 0; q = 0
# v0 = 0.0654; mv = 0.0707; sv = 0.2928; kv = 0.6067; rho0 = -0.7571
# kr = 3.4; mr = -0.6; sr = 0.1; rho0 = -0.4; rho2 = 0.4;
# # x = np.log(s0/K)
# param = [kv,mv,sv,v0,kr,mr,sr,rho0,rho2]
# char_fcn = lambda u: CF_OU(param,np.log(s0),r,T,u)

# u = np.linspace(0,50,401); phi = 1j*np.zeros(len(u))
# for i in range(len(u)):
#     phi[i] = char_fcn(u[i])

# plt.figure()
# plt.plot(u,np.real(phi))
# plt.plot(u,np.imag(phi))
# plt.show()



# uu = np.linspace(-100, 100, 2001)
# k = np.log(K)
# a = 1.5

# phi = char_fcn(uu)
# tmp = np.exp(-1j*u*k) * phi
# call = np.exp(-a*k)/np.pi * integrate.simps(tmp, uu)




# uu = np.linspace(-100, 100, 2001)
# TT = np.linspace(1, 5, 2)
# call = np.zeros(len(TT))
# T, u = np.meshgrid(TT, uu)
# phi = np.zeros((len(u),len(TT)), dtype=complex)
# tmp = np.zeros_like(phi)
# cumSumInt = np.zeros_like(phi)
# k = np.log(K)
# a = 1.5

# for o, m in enumerate(TT):
#     for p, n in enumerate(uu):
#         phi[p, o] = char_fcn(m, n)
    
# tmp[:,o] = np.exp(-1j*u*k) * phi[:,o]
# call[o] = np.exp(-a*k)/np.pi * integrate.simps(tmp[:,o], u[:,0])
# cumSumInt[:,o] = 0.5 * np.pi * np.cumsum(tmp[:,o]) * (uu[1] - uu[0])
    
# plt.plot(u,cumSumInt)
# plt.show()
        
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(T, u, np.real(phi), cmap='viridis', edgecolor='none')

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(T, u, np.imag(phi), cmap='viridis', edgecolor='none')

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(T, u, abs(phi), cmap='viridis', edgecolor='none')

# plt.figure()
# plt.plot(T[0,:], put)
















