"""
Heston Model COS Method
As in (Hirsa, 2012) p.58 for Vanilla Call and (Fang, Oosterlee, 2008)
"""

import numpy as np
import sys


def HestonVanillaCOS(s0,K,tau,r,q,v0,theta,vol,kappa,rho,N=128,option="c"):
    
    # define the number of decompositions
    k = np.arange(0,N)
    x = np.log(s0/K)
    if (0.3 < K/s0 <= 0.5) & (2 > tau < 1/12): 
        L = 30
    elif (K/s0 <= 0.3) & (tau <= 1/12): # for deep OTM and short-mat. options
        L = 80
    else: L = 12
    c1 = (r-q)*tau + (1-np.exp(-kappa*tau)) * (theta-v0)/(2*kappa) - 0.5*theta*tau
    c2 = (1/(8*kappa**3)) * \
         (vol*tau*kappa*np.exp(-kappa*tau)*(v0-theta)*(8*kappa*rho-4*vol) 
         + kappa*rho*vol*(1-np.exp(-kappa*tau))*(16*theta-8*v0) 
         + 2*theta*kappa*tau*(-4*kappa*rho*vol + vol**2 + 4*kappa**2) 
         + vol**2*((theta-2*v0)*np.exp(-2*kappa*tau)+theta*(6*np.exp(-kappa*tau)-7)+2*v0) 
         + 8*kappa**2*(v0-theta)*(1-np.exp(-kappa*tau)))
    c4 = 0
    a = c1-L*np.sqrt(abs(c2) + np.sqrt(c4)) # c2<0 if Feller not met 
    b = c1+L*np.sqrt(abs(c2) + np.sqrt(c4))
    
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
        k2 = np.array(k) #new array containing a copy of k
        k2[0] = 1 #assigning random value, b/c k[0] = 0 otherwise. varphi[0] will be re-defined again
        varphi = (np.sin(k2*np.pi*(d-a)/(b-a)) - np.sin(k2*np.pi*(c-a)/(b-a))) * (b-a)/(k2*np.pi)
        #if k=0: k is only zero at the first element
        varphi[0] = d-c
        return varphi
    
    #coefficient for the payoff
    if option == "c":
        Vk = (2/(b-a))*(K*(chiVanilla(0,b) - varphiVanilla(0,b)))
    elif option == "p":
        Vk = (2/(b-a))*(K*(-1*chiVanilla(a,0) + varphiVanilla(a,0)))
    else:
        sys.exit("please indicate 'c' for call or 'p' for put")
    
    #computing coefficient for the model and weighting the first term by 1/2
    Ak = np.real(charfuncHeston(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a)))
    return np.exp(-r*tau)*(np.sum(Ak*Vk) - 0.5*Ak[0]*Vk[0])


# t22 = time.time()
if __name__ == "__main__":
    #parameters from Hirsa p. 61
    # s0=100; r=0; q=0; #underlying
    # v0=0.0175; theta=0.0398; vol=0.5751; kappa=1.5768 #variance
    # rho=-0.5711
    s0 = 2461.44; r = 0.03; q = 0.0
    v0 = 0.0654; kappa = 0.6067; vol = 0.2928; theta = 0.0707; rho = -0.7571
    price = HestonVanillaCOS(s0,s0,3,r,q,v0,theta,vol,kappa,rho,N=128,option='c')
    # param = [v0, theta, vol, kappa, rho]
    # strikes = np.linspace(1080,5500,50)
    # taus = np.linspace(0.024726,3.53692,50)
    # hestonprices = np.zeros([len(strikes),len(taus)])
    # zetta = np.zeros((len(tmpK), len(tmpT)))
    # for i,K in enumerate(strikes):
    #     for j,M in enumerate(taus):
    #         hestonprices[j,i] = HestonVanillaCOS(s0,K,M,r,q,v0,theta,vol,kappa,rho,N=256,option='c') 
    print(price)
# HestonVanillaCOS(s0,5400,5.1,r,q,v0,theta,vol,kappa,rho,N=256,option='c') 
# # elapsed = time.time() - t22
# # print(elapsed)



# fig = plt.figure()      
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(strikes,taus)  
# Z = hestonprices
# # plt.plot(strikes, hestonprices[:,0])
# # ax.plt.plot_surface(X,Y,hestonprices,cmap='viridis',edgecolor='none')
# ax.plot_surface(X,Y,Z)
# ax.set_xlabel('strikes')
# ax.set_ylabel('maturities')
# ax.set_zlabel('call value')
# fig.show()