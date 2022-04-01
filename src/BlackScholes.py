"""Call and Put Options in Black-Scholes Setting - Brownian Motion"""
import numpy as np
from scipy.stats import norm
import sys
#from matplotlib import pyplot as plt

def BlackScholes(s0, K, sigma, tau, r=0, q=0, option='c'):
    #S_0 = Price of underlying
    #K = Strike Value
    #sigma = Variance
    #rf = risk-free rate (default = 0%)
    #tau = T-t in percentage of maturity in years (if T = 2y, t=1.5 (6m left) -> tau=6/24 or 180*2/365)
    #optiontype = "call" or "put" option (default = "call")

    d1 = (np.log(s0/K) + (r - q +0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    if (option=="c"):
        call = s0 * np.exp(-q*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
        return call
    elif (option=="p"): 
        put = K * np.exp(-r*tau) * norm.cdf(-d2) - s0 * np.exp(-q*tau) * norm.cdf(-d1)
    else:
        sys.exit("please indicate 'c' for call or 'p' for put")
        return put

# print(BlackScholes(s0=100, K=120, sigma=0.25, tau=6/73, r=0.1, q=0.0, option="c"))
#with these parameters, according to wolfram alpha black scholes calculator I should get 1.46$
# print(BlackScholes(s0=6248.20, K=5150, sigma=0.347167, tau=93/360, r=0.00934, q=0, option='p'))



def BlackScholesCall(s0, K, sigma, tau, r=0, q=0):
    
    d1 = (np.log(s0/K) + (r - q + 0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    return s0 * np.exp(-q*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    #return call

# print(BlackScholesCall(s0=100, K=80, sigma=0.25, tau=0.1, r=0.1, q=0))


    
def BlackScholesPut(s0, K, sigma, tau, r=0, q=0):
    
    d1 = (np.log(s0/K) + (r - q +0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    
    return K * np.exp(-r*tau) * norm.cdf(-d2) - s0 * np.exp(-q*tau) * norm.cdf(-d1)

# print(BlackScholesPut(s0=100, K=80, sigma=0.25, tau=0.1, r=0.1, q=0))



"""
do now an approximation of black-scholes (FD or Fourier) and test with analytic solution
code the heston model (analytic) and approximative

plot accross strikes and maturities

plot the integrand of gbm and heston as in hirsa page 47-51

calculate implied volas for all strikes in random_dataset and plot as Norbert did

try to calibrate finally
"""