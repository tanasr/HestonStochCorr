"""
Created on Tue Jun 16 18:20:18 2020

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import pandas as pd
from time import time

df = pd.read_csv('data/Crisostomo1.csv')
# df = pd.read_csv('data/Crisostomo2.csv')
# df = pd.read_csv('data/Shoutens_EUROSTOXX_50_short.csv')
# df = pd.read_csv('data/tesla_short.csv')


tmpS0 = df['Spot']
tmpK = df['Strike']
tmpT = df['Maturity']
C_market = df['Mid']
rf = df['Interest rate']
# parameter aus paper Teng et al. - on the heston model ...    
s0 = tmpS0[0]; 
t = 0 #r = 0 # underlying
alpha = 1.25

def HestonVanillaCOS(s0,K,tau,r,q,v0, theta, vol, kappa, rho,N,option="c"):
    
    k = np.arange(0,N)
    x = np.log(s0/K)
    if (0.3 < K/s0 <= 0.5) & (2 > tau < 1/12): 
        L = 30
    elif (K/s0 <= 0.3) & (tau <= 1/12): #was 12 but OTM short-mat were mispriced
        L = 80
    else: L = 14
    # L = 12
    c1 = (r-q)*tau + (1-np.exp(-kappa*tau)) * (theta-v0)/(2*kappa) - 0.5*theta*tau
    c2 = (
        (1/(8*kappa**3)) * 
        (vol*tau*kappa*np.exp(-kappa*tau)*(v0-theta)*(8*kappa*rho-4*vol) 
          + kappa*rho*vol*(1-np.exp(-kappa*tau))*(16*theta-8*v0) 
          + 2*theta*kappa*tau*(-4*kappa*rho*vol + vol**2 + 4*kappa**2) 
          + vol**2*((theta-2*v0)*np.exp(-2*kappa*tau)+theta*(6*np.exp(-kappa*tau)-7)+2*v0) 
          + 8*kappa**2*(v0-theta)*(1-np.exp(-kappa*tau)))
        )
    c4 = 0 #only if L=12, otherwise define c4
    a = c1-L*np.sqrt(c2 + np.sqrt(c4)) #better to use np.abs(c2), for when Feller not satisfied
    b = c1+L*np.sqrt(c2 + np.sqrt(c4))

    def charfuncHeston(u):
        #modified charfunc as in Fang,Oosterlee 2008
        D = np.sqrt((kappa - 1j*rho*vol*u)**2 + (u**2 + 1j*u)*vol**2)
        G = (kappa - 1j*rho*vol*u - D)/(kappa - 1j*rho*vol*u + D)
        return (
                np.exp(1j*u*(x+(r-q)*tau) 
                + v0/vol**2*((1-np.exp(-D*tau))/(1-G*np.exp(-D*tau)))*(kappa-1j*rho*vol*u-D))*
                np.exp((kappa*theta)/vol**2 
                * (tau*(kappa-1j*rho*vol*u-D)-2*np.log((1-G*np.exp(-D*tau))/(1-G))))
                )

    def chiVanilla(c,d):
        # Chi coefficient for plain Vanilla payoff
        return (
                (1/(1+(k*np.pi/(b-a))**2)) 
                *(np.cos(k*np.pi * (d-a)/(b-a)) * np.exp(d)
                - np.cos(k*np.pi * (c-a)/(b-a)) * np.exp(c) 
                + k*np.pi/(b-a) * np.sin(k*np.pi * (d-a)/(b-a)) * np.exp(d) 
                - k*np.pi/(b-a) * np.sin(k*np.pi * (c-a)/(b-a)) * np.exp(c))
                )

    def varphiVanilla(c,d):
        # Varphi coefficient for plain Vanilla payoff
        k2 = np.array(k) #create new array containing a copy of k
        k2[0] = 1 #assign random value, b/c k[0] = 0 otherwise. varphi[0] will be redefined anyways
        varphi = (np.sin(k2*np.pi*(d-a)/(b-a)) - np.sin(k2*np.pi*(c-a)/(b-a))) * (b-a)/(k2*np.pi)
        #if k=0: k is only zero at the first element
        varphi[0] = d-c
        return varphi

    #compute coefficient for the payoff
    if option == "c":
        Vk = (2/(b-a))*(K*(chiVanilla(0,b) - varphiVanilla(0,b)))
    elif option == "p":
        Vk = (2/(b-a))*(K*(-1*chiVanilla(a,0) + varphiVanilla(a,0)))
    else:
        sys.exit("please indicate 'c' for call or 'p' for put")

    #compute coefficient for the process
    Ak = np.real(charfuncHeston(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a)))
    return np.exp(-r*tau)*(np.sum(Ak*Vk) - 0.5*Ak[0]*Vk[0])





def simulate(p):
    
    call = np.zeros(len(tmpK))

    for i in range(len(tmpT)):
        T = tmpT[i]
        K = tmpK[i]
        r = rf[i]
        # print("T : % 2.5f" %(T))
        # print("K : % 2.2f" %(K))
        call[i] =  HestonVanillaCOS(s0,K,T,r,0,p[0],p[1],p[2],p[3],p[4],256,option="c")
    
    return call

# define objective
def objective(p):
    # simulate model
    C = simulate(p)
    # calculate objective
    obj = 0.0
    
    #RMSE
    for i in range(len(tmpK)):
        # obj += ((C[i]-C_market[i])/C_market[i])**2  # SSE
        obj += abs((C[i]-C_market[i]))  # AAE
        
    return obj/len(tmpK)

def reporter(p):
    """Reporter function to capture intermediate states of optimization."""
    global ps
    ps.append(p)


v0=0.2; theta=0.2; vol=0.3; kappa=2; rho=0.5
p0 = [v0, theta, vol, kappa, rho]
ps = [p0]
N=256
k = np.arange(0,N)


# show initial objective
print('Initial SSE Objective: ' + str(objective(p0)))

# bounds on variables
bnds = ((0.001, 1.0),    # v0
        (0.001, 3),      # theta
        (0.001, 1.0),      # vol
        (0.001, 10.0),     # kappa
        (-1, 1))        # rho

t0 = time()
solution = minimize(objective, p0, method='SLSQP', bounds=bnds, callback=reporter, options={'maxiter': 5000, 'ftol': 1e-6, 'disp': True})
p = solution.x
ps = np.array(ps)

elapsed = time() -t0
print('Elapsed time:')
print(elapsed)

# show final objective
print('Final SSE Objective: ' + str(objective(p)))

print("v0   : % 2.3f" %(p[0]))
print("theta: % 2.3f" %(p[1]))
print("vol  : % 2.3f" %(p[2]))
print("kappa: % 2.3f" %(p[3]))
print("rho  : % 2.3f" %(p[4]))


n = 2  # Keeps every 7th label

# camara view
azimut = -73
elevation = 25

initialSol = simulate(p0)
optSol = simulate(p)

# Subplots
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(1,2,1, projection='3d')

# fig1 = plt.figure().gca(projection='3d')
ax1.scatter(tmpK, tmpT, C_market, c='r')
ax1.scatter(tmpK, tmpT, initialSol, marker='x', c='b')
ax1.set_xlabel('Strikes', fontsize='x-large')
ax1.set_ylabel('Maturities', fontsize='x-large')
ax1.set_zlabel('C market', fontsize='x-large', labelpad=5)
plt.title('Pre calibration')
[l.set_visible(False) for (i,l) in enumerate(ax1.xaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax1.yaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax1.zaxis.get_ticklabels()) if i % n != 0]
ax1.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax1.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax1.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax1.view_init(elev=elevation, azim=azimut)


ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.scatter(tmpK, tmpT, C_market, c='r')
ax2.scatter(tmpK, tmpT, optSol, marker='x', c='b')
ax2.set_xlabel('Strikes', fontsize='x-large')
ax2.set_ylabel('Maturities', fontsize='x-large')
ax2.set_zlabel('C market', fontsize='x-large', labelpad=5)
plt.title('Post calibration')
[l.set_visible(False) for (i,l) in enumerate(ax2.xaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax2.yaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax2.zaxis.get_ticklabels()) if i % n != 0]
ax2.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax2.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax2.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax2.view_init(elev=elevation, azim=azimut)
plt.tight_layout()

residual = abs(C_market - optSol)/C_market

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(tmpK, tmpT, residual, c='k')
for i in range(len(tmpK)):
    ax3.plot([tmpK[i], tmpK[i]], [tmpT[i],tmpT[i]], zs=[0, residual[i]], c='k')

ax3.set_xlabel('Strikes', fontsize='x-large')
ax3.set_ylabel('Maturities', fontsize='x-large')
ax3.set_zlabel('Residuals', fontsize='x-large', labelpad=5)
[l.set_visible(False) for (i,l) in enumerate(ax3.xaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax3.yaxis.get_ticklabels()) if i % n != 0]
[l.set_visible(False) for (i,l) in enumerate(ax3.zaxis.get_ticklabels()) if i % n != 0]
ax3.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax3.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax3.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax3.set_zlim(0,0.12)
ax3.view_init(elev=elevation, azim=azimut)
plt.tight_layout()


n_iter = len(ps)
iterRange = np.arange(0,n_iter,1)
fig = plt.figure(figsize=(13,5))
plt.plot(iterRange, ps[:,0], label = 'v0')
plt.plot(iterRange, ps[:,1], label = 'theta')
plt.plot(iterRange, ps[:,2], label = 'vol')
plt.plot(iterRange, ps[:,3], label = 'kappa')
plt.plot(iterRange, ps[:,4], label = 'rho')
plt.legend(loc='best', prop={'size': 12})
plt.xlabel('Iterations', fontsize='x-large')
plt.ylabel('Parameter values', fontsize='x-large')
plt.grid()

plt.show()

# acc. to http://apmonitor.com/pdc/index.php/Main/ArduinoEstimation2
