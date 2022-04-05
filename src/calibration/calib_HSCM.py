"""
Calibration procedure for extended Heston model

"""
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import pandas as pd
from time import time

# df = pd.read_csv('MarkedData.csv')
df = pd.read_csv('data/Crisostomo1.csv')
# df = pd.read_csv('data/Crisostomo2.csv')
# df = pd.read_csv('data/Shoutens_EUROSTOXX_50_short.csv')
# df = pd.read_csv('data/tesla_short.csv')

tmpK = df['Strike'][0:3]
tmpT = df['Maturity'][0:3]
C_market = df['Mid'][0:3]

# parameter aus paper Teng et al. - on the heston model ...    
s0 = 328.29; t = 0; r = 0 # underlying
alpha = 1.25


def F(t,y,u,kv,mv,sv,kr,mr,sr,rho2,v0,E,r):
    # the (complex) ODE system; we use that B = iu (and thus B^2=-u^2)
    a,c,d = y
    return [(1j*u-0)*r+kv*mv*d+kr*mr*c+0.5*sr**2*c**2+sr*rho2*E*1j*u*c,\
            sv*v0*1j*u*d-kr*c,\
            -0.5*u**2+0.5*sv**2*d**2-0.5*1j*u-kv*d]

def CF_OU(p,x0,r,T,u):
    # numerically integrate the system of ODEs
    # model vars
    kv = p[0]; mv = p[1]; sv = p[2]; v0 = p[3] # variance process param
    kr = p[4]; mr = p[5]; sr = p[6]; rho0 = p[7] # correlation process param
    rho2 = p[8]
    m = np.sqrt(mv-sv**2/(8*kv)+0j) # aux var
    # n = np.sqrt(v0) - m
    # integrate, for a given u, the system of ODEs in ]0,T]
    sol = solve_ivp(F,[0,T],[0j,0j,0j],method='BDF', \
    args=(u,kv,mv,sv,kr,mr,sr,rho2,v0,m,r),\
    dense_output=True)
    # get the CF at T
    A = sol.sol(T)[0]; C = sol.sol(T)[1]; D = sol.sol(T)[2]
    
    return np.exp(-r*T+A+1j*u*x0+C*rho0+D*v0)


def simulate(p):
    
    param = p
    call = np.zeros(len(tmpK))

    for i in range(len(tmpT)):
        T = tmpT[i]
        K = tmpK[i]
        print("T : % 2.5f" %(T))
        print("K : % 2.2f" %(K))
        myFunc = lambda u: np.exp(-1j * u * np.log(K)) * ( (np.exp(-r*T) * ( CF_OU(param,np.log(s0),r,T,u - (alpha + 1)*1j) )) / (alpha**2 + alpha - u**2 + 1j*(2 * alpha + 1)*u ) )
        call[i] = (np.exp(-alpha * np.log(K)) / np.pi) * integrate.quad(myFunc, 0, np.inf )[0]
    
    return call

# define objective
def objective(p):
    # simulate model
    C = simulate(p)
    # calculate objective
    obj = 0.0
    
    for i in range(len(tmpK)):
        obj += ((C[i]-C_market[i])/C_market[i])**2
    
    return obj


t0 = time()

# model parameters from table 2 in the paper
kv = 2.1; mv = 0.03; sv = 0.2; v0 = 0.02
kr = 3.7; mr = -0.5; sr = 0.2; rho0 = -0.3; rho2 = -0.4
p0 = [kv,mv,sv,v0,kr,mr,sr,rho0,rho2]

# show initial objective
print('Initial SSE Objective: ' + str(objective(p0)))

fig1 = plt.figure().gca(projection='3d')
fig1.scatter(tmpK, tmpT, C_market, c='r')
fig1.scatter(tmpK, tmpT, simulate(p0), c='b')
fig1.set_xlabel('Strike')
fig1.set_ylabel('Maturity')
fig1.set_zlabel('C market')



# bounds on variables
bnds = ((0.01, 10.0),    # kv
        (0.01, 10.0),   # mv
        (0.01, 10.0),  # sv
        (0.01, 150),   # v0
        (0.01, 10.0),   # kr
        (-1, 1),   # mr
        (0.01, 10.0),   # sr
        (-1, 1),   # rho0
        (-1, 1))   # rho2

solution = minimize(objective,p0,method='SLSQP',bounds=bnds)
p = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(p)))


fig2 = plt.figure().gca(projection='3d')
fig2.scatter(tmpK, tmpT, C_market, c='r')
fig2.scatter(tmpK, tmpT, simulate(p), c='b')
fig2.set_xlabel('Strike')
fig2.set_ylabel('Maturity')
fig2.set_zlabel('C market')

plt.show()

# # optimized parameter values
# U = p[0]
# alpha1 = p[1]
# alpha2 = p[2]
# print('U: ' + str(U))
# print('alpha1: ' + str(alpha1))
# print('alpha2: ' + str(alpha2))

# # calculate model with updated parameters
# Ti  = simulate(p0)
# Tp  = simulate(p)

# # Plot results
# plt.figure(1)

# plt.subplot(3,1,1)
# plt.plot(t/60.0,Ti[:,0],'y:',label=r'$T_1$ initial')
# plt.plot(t/60.0,T1meas,'b-',label=r'$T_1$ measured')
# plt.plot(t/60.0,Tp[:,0],'r--',label=r'$T_1$ optimized')
# plt.ylabel('Temperature (degC)')
# plt.legend(loc='best')

# plt.subplot(3,1,2)
# plt.plot(t/60.0,Ti[:,1],'y:',label=r'$T_2$ initial')
# plt.plot(t/60.0,T2meas,'b-',label=r'$T_2$ measured')
# plt.plot(t/60.0,Tp[:,1],'r--',label=r'$T_2$ optimized')
# plt.ylabel('Temperature (degC)')
# plt.legend(loc='best')

# plt.subplot(3,1,3)
# plt.plot(t/60.0,Q1,'g-',label=r'$Q_1$')
# plt.plot(t/60.0,Q2,'k--',label=r'$Q_2$')
# plt.ylabel('Heater Output')
# plt.legend(loc='best')

# plt.xlabel('Time (min)')
# plt.show()
