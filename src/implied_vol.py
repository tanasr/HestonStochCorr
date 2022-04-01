"""
Computing implied BS-volatilites using Newton's root-finding method'
"""

"try out scipy.optimize.newton"



import numpy as np
from scipy.stats import norm
from scipy import optimize
# import sys
from BlackScholes import BlackScholesCall
from matplotlib import pyplot as plt



def call_implied_vol(Vmarket, s0, K, T, r=0, q=0):  
    tol = 1e-6
    sigma0 = 0.8 #need to give a value so that the while-statement can be tested
    sigma1 = 0.9 #if sigma1 = sigma0, the while-loop would be skipped
    
    while abs(sigma0-sigma1)>tol: # Newton's method
        sigma0 = sigma1		
        d1 = (np.log(s0/K) + (r-q+0.5*sigma0**2)*T) / (sigma0*np.sqrt(T))
        d2 = d1 - sigma0*np.sqrt(T)
        Vmodel = s0 * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        vega = s0 * np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1)
        
        sigma1 = sigma0 - (Vmodel - Vmarket) / vega 
    return sigma1

# print(call_implied_vol(17.5, 586.08, 585, 40/365, 0.0002, 0.0))
# correct implemented according to values in (iv = 0.2192)
# http://www.codeandfinance.com/finding-implied-vol.html


# def put_implied_vol(Vmarket, s0, K, T, r=0, q=0):
#     tol = 1e-6
#     sigma0 = 10 #need to give a value so that the while-statement can be tested
#     sigma1 = 20 #if sigma 1 = sigma0, the while-loop would be skipped
#     while abs(sigma0-sigma1)>tol:
#         sigma0 = sigma1		
#         d1 = (np.log(s0/K) + (r-q+0.5*sigma0**2)*T) / (sigma0*np.sqrt(T))
#         d2 = d1 - sigma0*np.sqrt(T)
#         Vmodel = K*np.exp(-r*T)*norm.cdf(-d2) - s0*np.exp(-q*T)*norm.cdf(-d1)
#         vega = s0 * np.exp(-q*T) * np.sqrt(T) * norm.pdf(d1)
        
#         sigma1 = sigma0 - (Vmodel - Vmarket) / vega
#     return sigma1



# def residual(sigma, s0, K, r, T, q, Vmarket):
#     # d1 = (np.log(s0/K) + (r - q + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
#     # d2 = d1 - sigma*np.sqrt(T)
#     # vega = s0*np.exp(-q*T)*np.sqrt(T)*norm.pdf(d1)
#     res = BlackScholesCall(s0,K,sigma,T,r) - Vmarket
#     return res


# s0 = 100; K = 110; T = 1/12
# r = 0.01; q = 0; Vmarket = 1.9
# sigma = np.linspace(0,1,100)
# iv = optimize.newton(residual,0.2,args=(s0, K, r, T, q, Vmarket))
# print(iv)

# plt.figure(figsize=(6,4.1))
# plt.title('$V^{model} - V^M$')
# plt.plot(sigma,residual(sigma,s0,K,r,T,q,Vmarket), label=r'$f\,(\sigma$)')
# plt.plot([0,1],[0,0],'k',linestyle='dashed')
# plt.ylabel('Residual',fontsize='large')
# plt.xlabel('Volatility',fontsize='large')
# plt.legend(loc='best')
# plt.grid()
# plt.show()
# plt.plot(sigma, call_IV_Newton(sigma,s0,K,T,r,q,Vmarket))


# # no implied vol when
# s0 = 100; K = 95; T = 30/365; t = 0;
# r = 0.01; q = 0; Vmarket = 2.30
# sigma = np.linspace(0,1,100)

# plt.figure()
# plt.title('Finding the root of the residual function')
# plt.plot(sigma,residual(sigma,s0,K,r,T,Vmarket))
# plt.plot([0,1],[0,0],'k',linestyle='dashed')
# plt.ylabel('Vmodel - Vmarket',fontsize='large')
# plt.xlabel('Volatility',fontsize='large')
# plt.grid()
# plt.show()

# lowerbound = s0-np.exp(-r*(T-t))*K




def call_IV(sigma0, s0, K, T, r, q, Vmarket, method='newton'):
    d1 = (np.log(s0/K) + (r - q + 0.5*sigma0**2) * T) / (sigma0*np.sqrt(T))
    vega = lambda sigma0: s0*np.exp(-q*T)*np.sqrt(T)*norm.pdf(d1)
    
    def residual(sigma0):
        res = BlackScholesCall(s0,K,sigma0,T,r) - Vmarket
        return res

    if method == 'newton':
        tol = 1e-6
        iv = optimize.newton(residual,sigma0,vega,tol=tol)
    if method == 'fsolve':
        iv = optimize.fsolve(residual,sigma0,fprime=vega)
    if method == 'brentq':
        iv = optimize.brentq(residual,1/10000,5)
    return iv

# print(call_IV(0.5,s0,K,T,r,0,Vmarket,'brentq'))




def call_IV_Newton(sigma0, s0, K, T, r, q, Vmarket):
    d1 = (np.log(s0/K) + (r - q + 0.5*sigma0**2) * T) / (sigma0*np.sqrt(T))
    vega = lambda sigma0: s0*np.exp(-q*T)*np.sqrt(T)*norm.pdf(d1)
    
    def residual(sigma0):
        res = BlackScholesCall(s0,K,sigma0,T,r) - Vmarket
        return res

    tol = 1e-6
    iv = optimize.newton(residual,sigma0,vega,tol=tol)
    return iv

def call_IV_Brentq(sigma0, s0, K, T, r, q, Vmarket):
    
    def residual(sigma0):
        res = BlackScholesCall(s0,K,sigma0,T,r) - Vmarket
        return res

    iv = optimize.brentq(residual,1/10000,5)
    return iv

#without first derivative vega
def call_IV_Fsolve(sigma0, s0, K, T, r, q, Vmarket):
    
    def residual(sigma0):
        res = BlackScholesCall(s0,K,sigma0,T,r) - Vmarket
        return res

    iv = optimize.fsolve(residual,sigma0)
    return iv




#Vmodel = S_0*np.exp(-q*T)*norm.cdf(d1)-np.exp(-r*T)*K*norm.cdf(d2)-S_0*np.exp(-q*T)+np.exp(-r*T)*K