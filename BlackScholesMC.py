"""
Monte Carlo Approximation of Black Scholes
Idea: Sampling paths to obtain the expected payoff in a risk-neutral world and then discounting this payoff at the riskfree rate
     1) Sample a random path for S in a risk-neutral world
     S(t) = S(0)*exp([r-0.5*sigma^2]*t + sigma*sqrt(t)*N(0,1))
     2) Calculate the payoff from the derivative
     call: max(S-K,0) or a put: max(K-S,0)
     3) Repeat steps 1 and 2 to get many sample values of the payoff
     4) Calculate the mean of the sample payoffs to get an estimate of the expected payoff (risk-neutral)
     5) Discount this expected payoff at the risk-free rate to get the estimate of the value of the derivative
Note that we have to simulate a stochastic process i.e. the Wiener process dW, and the price of a financial instrument is then determined by a differential equation ds(t) = mu*s(t)dt + sigma*s(t)dW. The Wiener process is a random walk with mean 0 and variance t --> N(0,t) or sqrt(t)*N(0,1), making a risk-neutral assumption, the stock price at time t is given by (see step 1)

try variance reduction aswell, look which discretization scheme is best
"""
import numpy as np
import time
import sys
#from scipy.stat import norm
#from matplotlib import pyplot as plt

#Vectorized MonteCarlo
def getMCgbm(s0, K, sigma, dt, r=0, q=0, option="c"): #is this really geom. brownian motion (log-normal)?
    
    iterations = 1e6
    #create an empty 2d-vector which will store the payoff. first column will be zeros & 2nd column the payoff
    #the firs column with zeros is important b/c we want to know the max of (S-K,0) for example
    prices = np.zeros([int(iterations), 2])
    drift = s0*np.exp(dt*(r-q-0.5*sigma**2))
    #process is a 1d-array with the number of rows equal to iter
    process = np.exp(sigma*np.sqrt(dt)*np.random.normal(0,1,[1, int(iterations)]))
    """creating a vector of random increments for the brownian motion. If only np.random.normal(1,0) you'll get only one singel number. Defining the size of the draw with the third argument, otherwise you will only get a scalar as output."""
    
    if option == "c":
        prices[:,1] = drift*process - K #drift*process = stock price at any time t
    elif option == "p":
        prices[:,1] = K - drift*process
    else:
        sys.exit("please indicate 'c' for call or 'p' for put")
        
    average_sum = np.sum(np.amax(prices, axis=1)) / iterations #np.max would return a scalar being the largest number in that input array, but amax requires specification about the axis, and hence returns the max value for each entry in that axis
    return average_sum * np.exp(-r*dt)

t0 = time.time()
print(getMCgbm(s0=100, K=80, sigma=0.1, dt=1/12, r=0.05, q=0.2))
# for i in range(10):
#     print(getMCgbm(s0=6248.20, K=5150, sigma=0.347167067144529, dt=93/360, r=0.00934, q=0, option='p'))
elapsed = time.time() - t0
print(elapsed)








# def MC_black_scholes_call(s0, K, sigma, T, r=0, q=0):
# 	
#     sim = 1e6
#     #dt = 1
#     payoff = np.zeros(int(sim))
    
#     for i in range(0,int(sim)):
        
#         St = s0 * np.exp((r-q-0.5*sigma**2)*T + sigma*np.random.normal(0,np.sqrt(T)))
#         #"you dont have to iterate over the first term 1mil times, only over randomness actually"
#         #St = s0 * np.exp((r-q-0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.normal(0,1))
#         payoff[i,] = max(St-K,0)
#         #payoff.append(max(St-K,0))
    
#     MCVcall = np.mean(payoff) * np.exp(-r*T)
#     return MCVcall
   
# t = time.time()
# print(MC_black_scholes_call(90,100,0.15,1,0.05,0))
# elapsed = time.time() - t
# print(elapsed)


# def MC_black_scholes_put(s0, K, sigma, T, r=0, q=0):
# 	
#     sim = 1e6
#     #dt = 1
#     payoff = np.zeros(int(sim))
    
#     for i in range(0,int(sim)):
        
#         #St = s0 * np.exp((r-q-0.5*sigma**2)*T + sigma*np.random.normal(0,np.sqrt(T)))
#         St = s0 * np.exp((r-q-0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.normal(0,1))
#         payoff[i,] = max(K-St,0)
#         #payoff.append(max(St-K,0))
    
#     MCVput = np.mean(payoff) * np.exp(-r*T)
#     return MCVput

# t = time.time()
# print(MC_black_scholes_call(857.29, 860, 0.2076, 18/365, 0.0014, 0.0))
# elapsed = time.time() - t
# print(elapsed)