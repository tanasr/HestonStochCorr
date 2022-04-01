"""Black-Scholes Modell (geom. brownian motion) with Fourier-COS Method for Vanillas


is it really geom.brownian motion? 

brownian motion = normal distribution
geom. brownian motion = log-normal distribtion """
#name the file BSvanillaCOS and the function can stay the same or idk 

import numpy as np
import time
import sys

# s0=100; K=80; sigma=0.25; dt=0.1; r=0.1; q=0; option="c"; N=64

def BlackScholesCOS(s0, K, sigma, dt, r, q, option, N):
    L = 10; 
    c1 = (r-q)*dt; c2 = sigma**2*dt; c4 = 0 #better to use np.abs(c2), for when Feller not satisfied
    a = c1-L*np.sqrt(c2 + np.sqrt(c4)); b = c1+L*np.sqrt(c2 + np.sqrt(c4))
    x = np.log(s0/K)
    C = np.exp(-r*dt) #np.exp(-r*dt) for path-dependent and non-Europeans
    k = np.arange(0,N) # it's from 0 to N (not N-1), b/c np.arange doesn't include last entry
   
    def charfuncGBM(u):
        return np.exp(1j*(x + (r-q-0.5*sigma**2)*dt)*u - 0.5*dt*sigma**2*u**2)
    
    # u = k*np.pi/(b-a)
    # charfuncGBM = np.exp(1j*(x + (r-q-0.5*sigma**2)*dt)*u - 0.5*dt*sigma**2*u**2)
    
    def chiVanilla(c,d): 
        return (1/(1+(k*np.pi/(b-a))**2)) * (np.cos(k*np.pi * (d-a)/(b-a)) * np.exp(d) -
                                         np.cos(k*np.pi * (c-a)/(b-a)) * np.exp(c) + 
                                         k*np.pi/(b-a) * np.sin(k*np.pi * (d-a)/(b-a)) * np.exp(d) - 
                                         k*np.pi/(b-a) * np.sin(k*np.pi * (c-a)/(b-a)) * np.exp(c))
    
    def varphiVanilla(c,d):
        k2 = np.array(k) #create new array containing the values of k (keep pointing in mind)
        k2[0] = 1
        varphi = (np.sin(k2*np.pi*(d-a)/(b-a)) - np.sin(k2*np.pi*(c-a)/(b-a))) * (b-a)/(k2*np.pi)
        #if k=0: k is only zero at the first element
        varphi[0] = d-c
        return varphi

    #compute coefficient for the process
    Ak = np.real(charfuncGBM(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a)))
    # Ak = np.real(charfuncGBM * np.exp(-1j*a*u))
    
    #compute coefficient for the payoff
    if option == "c":
        Vk = 2*K/(b-a)*(chiVanilla(0,b) - varphiVanilla(0,b))
    elif option == "p":
        Vk = 2*K/(b-a)*(-1*chiVanilla(a,0) + varphiVanilla(a,0))
    else:
        sys.exit("please indicate 'c' for call or 'p' for put")
    return C*(np.sum(Ak * Vk) - 0.5*Ak[0]*Vk[0])



# t0 = time.time()
# if __name__ == "__main__":
#     # for K in np.array([80,100,120]):
#         print(BlackScholesCOS(s0=100, K=120, sigma=0.25, dt=0.1, r=0.1, q=0.0, option='c', N=64)) #K=K
#         # VputCOS = COSblack_scholes(s0=100, K=80, sigma=0.25, dt=0.1, r=0.1, q=0, option='p', N=64)
#         # print(K)
# elapsed = time.time() - t0
# print('the elapsed time is: ' + str(elapsed) + ' seconds')
