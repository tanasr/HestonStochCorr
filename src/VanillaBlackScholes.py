"""
Call and Put Options in Black-Scholes Setting - Brownian Motion
Comparing Analytic with COS method and MonteCarlo
"""

import numpy as np
from scipy.stats import norm
import sys
import time
from BlackScholesCOS import COSmethodBS
from BlackScholesMC import getMCgbm

# ========== Analytic ==========
def BlackScholesAnalytic(s0, K, sigma, tau, r, q, option):

    d1 = (np.log(s0/K) + (r - q +0.5*sigma**2) * tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    if (option=="call"):
        price = s0 * np.exp(-q*tau) * norm.cdf(d1) - K * np.exp(-r*tau) * norm.cdf(d2)
    elif (option=="put"): 
        price = K * np.exp(-r*tau) * norm.cdf(-d2) - s0 * np.exp(-q*tau) * norm.cdf(-d1)
    else:
        sys.exit("please indicate 'call' or 'put'!")
        
    return price


def main():
    # set parameters
    s0 = 100
    K = 100
    sigma = 0.25
    dt = 1/12
    r = 0.01
    q = 0
    option = "put"
    N = 128 # for COS method
    iterations = 1e6 # for MonteCarlo

    analytic = BlackScholesAnalytic(s0, K, sigma, dt, r, q, option)
    cos_price = COSmethodBS(s0, K, sigma, dt, r, q, N, option)
    mc_price = getMCgbm(s0, K, sigma, dt, r, q, iterations, option)
    # error = np.sqrt((cos_price - analytic)**2)

    print('Analytic price =    {0:.18f}'.format(analytic))
    print('COS price =         {0:.18f}'.format(cos_price)) #K=K
    print('MonteCarlo price =  {0:.18f}'.format(mc_price))
    # print('squared error =     {0:.18f}'.format(error))


if __name__ == "__main__":
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print('elapsed time:       {0:.12f}'.format(elapsed))