"""
Monte Carlo Approximation of Black Scholes (vectorised)
"""
import numpy as np

def getMCgbm(s0, K, sigma, dt, r, q, iterations, option):
    """
    Simulating a stochastic process i.e. the Wiener process dW, 
    and the price of a financial instrument is then determined by a 
    differential equation ds(t) = mu*s(t)dt + sigma*s(t)dW. 
    The Wiener process is a random walk with mean 0 and var t --> N(0,t) 
    or sqrt(t)*N(0,1), making a risk-neutral assumption, the stock price 
    at time t is given by (see step 1)
    
    Idea: Sampling paths to obtain the expected payoff in a risk-neutral 
    world and then discounting this payoff at the riskfree rate
     1) Sample a random path for S in a risk-neutral world
     S(t) = S(0)*exp([r-0.5*sigma^2]*t + sigma*sqrt(t)*N(0,1))
     2) Calculate the payoff from the derivative
     call: max(S-K,0) or a put: max(K-S,0)
     3) Repeat steps 1 and 2 to get many sample values of the payoff
     4) Calculate the mean of the sample payoffs to get an estimate of the expected payoff (risk-neutral)
     5) Discount this expected payoff at the risk-free rate to get the estimate of the value of the derivative
"""
    # TODO: try variance reduction aswell, look which discretization scheme is best
    #create an empty 2d-vector which will store the payoff
    #first column will be zeros & 2nd column the payoff
    #the firs column with zeros is important b/c we want to know the max of (S-K,0) for example
    prices = np.zeros([int(iterations), 2])
    drift = s0*np.exp(dt*(r-q-0.5*sigma**2))
    
    #process is a 1d-array with the number of rows equal to iter
    # creating a vector of random increments for the brownian motion
    # Define the size of the draw with third argument, otherwise we get a scalar
    process = np.exp(sigma*np.sqrt(dt)*np.random.normal(0,1,[1, int(iterations)]))
    
    if option == "call":
        prices[:,1] = drift*process - K #drift*process = stock price at any time t
    elif option == "put":
        prices[:,1] = K - drift*process
    else:
        sys.exit("please indicate 'call' or 'put'!")
        
    # use np.amax, because np.max would return the max. number, a scalar
    # amax requires specification about axis, hence returns the max value for each entry
    average_sum = np.sum(np.amax(prices, axis=1)) / iterations
    return average_sum * np.exp(-r*dt) #return discounted payoff