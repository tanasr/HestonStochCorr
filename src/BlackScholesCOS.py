"""
Black-Scholes Modell (geom. brownian motion) with Fourier-COS Method for Vanillas
"""
import numpy as np


def COSmethodBS(s0, K, sigma, dt, r, q, N, option):
    L = 10
    c1 = (r-q)*dt
    c2 = sigma**2*dt
    c4 = 0 #better to use np.abs(c2), for when Feller not satisfied
    a = c1-L*np.sqrt(c2 + np.sqrt(c4))
    b = c1+L*np.sqrt(c2 + np.sqrt(c4))
    x = np.log(s0/K)
    C = np.exp(-r*dt) #np.exp(-r*dt) for path-dependent and non-Europeans
    k = np.arange(0,N) # it's from 0 to N (not N-1), b/c np.arange doesn't include last entry
    
    # Characteristic Function of Geom. Brownian Motion
    def charfuncGBM(u):
        return np.exp(1j*(x + (r-q-0.5*sigma**2)*dt)*u - 0.5*dt*sigma**2*u**2)
    # u = k*np.pi/(b-a)
    # charfuncGBM = np.exp(1j*(x + (r-q-0.5*sigma**2)*dt)*u - 0.5*dt*sigma**2*u**2)
    
    def chiVanilla(c,d): 
        part_a = 1/(1+(k*np.pi/(b-a))**2)
        part_b = (np.cos(k*np.pi * (d-a)/(b-a)) * np.exp(d) -
                  np.cos(k*np.pi * (c-a)/(b-a)) * np.exp(c) + 
                  k*np.pi/(b-a) * np.sin(k*np.pi * (d-a)/(b-a)) * np.exp(d) - 
                  k*np.pi/(b-a) * np.sin(k*np.pi * (c-a)/(b-a)) * np.exp(c))
        return part_a * part_b
    
    def varphiVanilla(c,d):
        k2 = np.array(k) #create new array containing the values of k
        k2[0] = 1
        varphi = (np.sin(k2*np.pi*(d-a)/(b-a)) - 
                  np.sin(k2*np.pi*(c-a)/(b-a))) * (b-a)/(k2*np.pi)
        #if k==0: k is only zero at the first element
        varphi[0] = d-c
        return varphi

    # coefficient for the process
    Ak = np.real(charfuncGBM(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a)))
    
    # coefficient for the payoff
    if option == "call":
        Vk = 2*K/(b-a)*(chiVanilla(0,b) - varphiVanilla(0,b))
    elif option == "put":
        Vk = 2*K/(b-a)*(-1*chiVanilla(a,0) + varphiVanilla(a,0))
    else:
        sys.exit("please indicate 'call' or 'put'!")
    return C*(np.sum(Ak * Vk) - 0.5*Ak[0]*Vk[0])