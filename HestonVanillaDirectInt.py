"""
Heston Model (Semi-Analytic Solution) - Direct Integration with continuous dividends
as in: Heston (1993), Jacquier and Martini (2010), Janek et al. (2011), Mikhailov and Nögel (2003), Yang (2013)
"""
import numpy as np
from scipy.integrate import quad
import sys
import time
# from matplotlib import pyplot as plt

# s0 = Price of underlying at time = 0
# K = Strike
# tau = T-t --> remaining time until maturity
# r = interest rate (risk-free in EMM)
# q = dividend (continuous)
# v0 = Start value for volatility
# theta = long-run variance
# vol = volatility of variance
# kappa = mean-reversion rate
# rho = correlation value
# lambd = price ov volatility risk (0 in equivalent martingale measure EMM)

def HestonVanillaDI(s0,K,tau,r,q,v0,theta,vol,kappa,rho,option="c"):
    def Hestonf(phi,mode):
        if mode == 1:
            u = 0.5
            b = kappa - rho*vol #b = kappa + lambd - rho*vol
        elif mode == 2:
            u = -0.5
            b = kappa #b = kappa + lambd
        else: sys.exit()
        d = np.sqrt((rho*vol*phi*1j-b)**2-(2*u*phi*1j-phi**2)*vol**2)
        g = (b-rho*vol*phi*1j-d)/(b-rho*vol*phi*1j+d) #modified charfunc
        C = ((r-q)*tau*phi*1j + kappa*theta/vol**2 
              * (tau*(b-rho*vol*phi*1j-d)-2*np.log((1-g*np.exp(-d*tau))/(1-g))))
        D = (b-rho*vol*phi*1j-d)/vol**2 * ((1-np.exp(-d*tau))/(1-g*np.exp(-d*tau)))
        return np.exp(C + D*v0 + 1j*phi*np.log(s0))
    
    integralP = lambda phi, mode : np.real((np.exp(-1j*phi*np.log(K))*Hestonf(phi,mode))/(1j*phi))
    P = np.zeros(2)
    P[0] = 0.5 + 1/np.pi * quad(integralP,0,np.inf, args=1)[0] #mode1
    P[1] = 0.5 + 1/np.pi * quad(integralP,0,np.inf, args=2)[0] #mode2
    
    #Value for Vanilla Call and Put (Put-Call Parity)
    Vcall = s0*np.exp(-q*tau)*P[0] - K*np.exp(-(r)*tau)*P[1]
    #as in Rouah, page 17, (1.68)
    #Vput = K*np.exp(-r*tau)*HestonProb(mode=2) - s0*np.exp(-q*tau)*HestonProb(mode=1)
    if option=="c":
        return Vcall
    elif option == "p":
        return Vcall+K*np.exp(-r*tau)-s0*np.exp(-q*tau)

# #only put calculation, no if-loop
# def HestonVanillaPut(s0,K,tau,r,q,v0,theta,vol,kappa,rho):
#     def Hestonf(phi,mode):
#         if mode == 1:
#             u = 0.5
#             b = kappa - rho*vol
#         elif mode == 2:
#             u = -0.5
#             b = kappa
#         else: sys.exit()
#         d = np.sqrt((rho*vol*phi*1j-b)**2-(2*u*phi*1j-phi**2)*vol**2)
#         g = (b-rho*vol*phi*1j-d)/(b-rho*vol*phi*1j+d) #modified charfunc
#         C = ((r-q)*tau*phi*1j + kappa*theta/vol**2 
#               * (tau*(b-rho*vol*phi*1j-d)-2*np.log((1-g*np.exp(-d*tau))/(1-g))))
#         D = (b-rho*vol*phi*1j-d)/vol**2 * ((1-np.exp(-d*tau))/(1-g*np.exp(-d*tau)))
#         return np.exp(C + D*v0 + 1j*phi*np.log(s0))
    
#     integralP = lambda phi, mode : np.real((np.exp(-1j*phi*np.log(K))*Hestonf(phi,mode))/(1j*phi))
#     P = np.zeros(2)
#     P[0] = 0.5 - 1/np.pi * integrate.quad(integralP,0,np.inf, args=1)[0] #mode1
#     P[1] = 0.5 - 1/np.pi * integrate.quad(integralP,0,np.inf, args=2)[0] #mode2
#     return K*np.exp(-r*tau)*P[1] - s0*np.exp(-q*tau)*P[0]

# HestonVanillaPut(100,100,1,0.05,0.0,0.0175,0.0398,0.5751,1.5768,-0.5711)

"something doesn't work when r and q are not 0, the prices between DI and COS are not the same"




# if __name__ == "__main__":
#     t0 = time.time()
#     print(HestonVanillaDI(110,100,1/12,0.01,0.0,0.0175,0.0398,0.5751,1.5768,-0.5711,"c"))
#     #tested with parameters from Hirsa p. 61, and Fang, Oosterlee p. 15
#     #print(HestonVanillaCall(1.0,1.5,5.0,0.0,0.0,0.1,0.1,0.2,2.0,-0.3,0.0))
#     #parameters from Mikhailov and Nögel
#     elapsed = time.time() - t0
#     print(elapsed)

    # t00 = time.time()
    # strikes = np.array(np.linspace(50,150,50))
    # underlying = np.array(np.linspace(50,150,50))
    # prices = np.zeros([len(strikes),len(underlying)]) #3rows, 3columns
    # for count_j, value_j in enumerate(strikes):
    #     for count_i,value_i in enumerate(underlying):
    #         prices[count_j,count_i] = HestonVanillaDI(value_i,value_j,1,0.05,0.0,0.0175,0.0398,0.5751,1.5768,-0.5711,"p")
    
    # elapsed = time.time() - t00
    # print('the elapsed time is:')
    # print(elapsed)
    
    # plt.figure()
    # plt.plot(strikes, underlying, prices)
    # plt.grid()
    # plt.show()




# strikeslist = [80,85,90] #equivalent to strikes
# strikes = np.array([80,85,90])
# prices = np.zeros([3,3]) #3rows, 3columns
# for count, value in enumerate(strikes):
#     prices[count,2] = HestonVanillaDI(100,value,1,0.0,0.0,0.0175,0.0398,0.5751,1.5768,-0.5711,0,"c")

    

# prices[0] is the same as prices[0,] and prices[0,:] which is the first row and all columns
# prices[0,2] is the first row, but third column

"""
- plot the integral to reveal how quickly it actually converges.
- see kovachev page 11 for plotting the heston probabilites
- see tests in hestontest.py
"""




"""Same code separated in chunks below"""

# s0=140.0; K=100.0; tau=3/12; r=0.15; q=0.25; v0=0.05; kappa=2; theta=0.04; lambd=0.0; vol=0.1; rho=-0.5
# x = np.log(s0)

# def heston_charfunc(phi,mode):
#     if mode == 1:
#         u = 0.5
#         b = kappa + lambd - rho*vol
#     elif mode == 2:
#         u = -0.5
#         b = kappa + lambd
#     else: sys.exit()
#     d = np.sqrt((rho*vol*phi*1j-b)**2-(2*u*phi*1j-phi**2)*vol**2)
#     g = (b-rho*vol*phi*1j-d)/(b-rho*vol*phi*1j+d) #modified charfunc
#     C = ((r-q)*tau*phi*1j + kappa*theta/vol**2 * (tau*(b-rho*vol*phi*1j-d)-2*np.log((1-g*np.exp(-d*tau))/(1-g))))
#     D = (b-rho*vol*phi*1j-d)/vol**2 * ((1-np.exp(-d*tau))/(1-g*np.exp(-d*tau)))
#     return np.exp(C + D*v0 + 1j*phi*x)

# def heston_p(mode):
#     if mode == 1:
#         integralP = lambda phi : np.real((np.exp(-1j*phi*np.log(K))*heston_charfunc(phi,mode=1))/(1j*phi))
#         P = 0.5 + 1/np.pi * integrate.quad(integralP,0,np.inf)[0] #quad returns a tuple, [0]=int.value
#     elif mode ==2:
#         integralP = lambda phi : np.real((np.exp(-1j*phi*np.log(K))*heston_charfunc(phi,mode=2))/(1j*phi))
#         P = 0.5 + 1/np.pi * integrate.quad(integralP,0,np.inf)[0] #quad returns a tuple, [0]=int.value
#     return P

# def get_heston_Call():
#     return s0*np.exp(-q*tau)*heston_p(mode=1) - K*np.exp(-(r)*tau)*heston_p(mode=2)


# t1 = time.time()
# if __name__ == "__main__":
#     print(get_heston_Call())
# # HestonVanillaCall(s0=1.0,K=1.0,tau=5.0,r=0.0,q=0,v0=0.1,kappa=4,theta=0.1,lambd=0.0,vol=0.2,rho=-0.3) #result is zero, which is pretty strange ...
# elapsed = time.time() - t1
# print(elapsed)