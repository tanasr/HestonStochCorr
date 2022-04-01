"""
Barrier Option up-and-out call under the Heston model with COS method

Method according to Hirsa page 63ff and Fang, Oosterlee (2009)
Parameters from Schoutens et al. - A perfect calibration. Now what? p. 10

Main idea: Calculate coeff. Ak numerically using direct integration of the 
approximated CF for the ext. Heston Model. Then, using the COS expansion for 
the barrier payoff (Vk) and mixing both methods. 
"""

from numpy import exp, pi, log, sqrt, cos, sin
import numpy as np
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import time

s0 = 2461.44; K = 2461.44; H = 1.5*s0; T = 6; M = 6 #6 monts maturity, 6 monitorings
r = 0.03; q = 0; alpha = 1; #alpha=1 for call, =-1 for put (not damping factor)
v0 = 0.0654; mv = 0.0707; sv = 0.2928; kv = 0.6067; 
mr = 0.0707; sr = 0.2928; kr = 0.6067; rho0 = -0.7571; rho2 = -0.5

N = 64 


# Formulas in brackets are from Hirsa page 63ff
def BarrierExtHestonUOC(s0,K,H,T,r,q,kv,mv,sv,v0,kr,mr,sr,rho0,rho2,alpha,M,N):
    # T = observation dates N
    Tvec = np.arange(0,T+1)/(T); dt = Tvec[1]-Tvec[0]; #M = len(Tvec)-1
    x = log(s0/K);
    k = np.arange(0,N,1); h = log(H/K); 

   
    #c1 and c2 calculated using SV parameters from pure Heston
    c1 = (r-q)*dt + (1-np.exp(-kv*dt)) * (mv-v0)/(2*kv) - 0.5*mv*dt
    c2 = (1/(8*kv**3)) * \
         (sv*dt*kv*np.exp(-kv*dt)*(v0-mv)*(8*kv*mr-4*sv) 
         + kv*mr*sv*(1-np.exp(-kv*dt))*(16*mv-8*v0) 
         + 2*mv*kv*dt*(-4*kv*mr*sv + sv**2 + 4*kv**2) 
         + sv**2*((mv-2*v0)*np.exp(-2*kv*dt)+mv*(6*np.exp(-kv*dt)-7)+2*v0) 
         + 8*kv**2*(v0-mv)*(1-np.exp(-kv*dt)))
    c4 = 0 
    L = 14
    a = c1-L*sqrt(abs(c2) + sqrt(c4)) # c2<0 if Feller not met 
    b = c1+L*sqrt(abs(c2) + sqrt(c4))
    
    # chifunction (2.64) in Hirsa
    def chifunc(c,d):
        chi = 1/(1+(k*pi/(b-a))**2) * \
            (cos(k*pi*(d-a)/(b-a)) * exp(d)-cos(k*pi*(c-a)/(b-a)) * \
             exp(c)+k*pi/(b-a) * sin(k*pi*(d-a)/(b-a)) * \
             exp(d)-k*pi/(b-a) * sin(k*pi*(c-a)/(b-a))*exp(c))
        return chi
    
    # phifucntion (2.65) in Hirsa
    def phifunc(c,d):
        phi = (sin(k*pi*(d-a)/(b-a))-sin(k*pi*(c-a)/(b-a)))*(b-a)/(k*pi)
        phi[0] = d-c
        return phi
    
    # Gkfunc (2.63) in Hirsa
    def Gkfunc(x,y,K,alpha):
        return 2/(b-a)*alpha*K*(chifunc(x,y)-phifunc(x,y))
    
    def F(t,y,u,kv,mv,sv,kr,mr,sr,rho2,v0,E,r):
        # the (complex) ODE system; we use that B = iu (and thus B^2=-u^2)
        a,c,d = y
        return [(1j*u-0)*r+kv*mv*d+kr*mr*c+0.5*sr**2*c**2+sr*rho2*E*1j*u*c,\
                sv*v0*1j*u*d-kr*c,\
                -0.5*u**2+0.5*sv**2*d**2-0.5*1j*u-kv*d]
    
    def CF_OU(p,x0,r,T,u):
        # numerically integrate the system of ODEs
        kv = p[0]; mv = p[1]; sv = p[2]; v0 = p[3]; # variance process param
        kr = p[4]; mr = p[5]; sr = p[6]; rho0 = p[7]; # correlation process param
        rho2 = p[8]; #correlation between dp and dv (set to be constant)
        m = np.sqrt(mv-sv**2/(8*kv)+0j); # aux var
        # n = np.sqrt(v0) - m
        # integrate, for a given u, the system of ODEs in ]0,T]
        sol = solve_ivp(F,[0,T],[0j,0j,0j],method='BDF', \
        args=(u,kv,mv,sv,kr,mr,sr,rho2,v0,m,r),\
        dense_output=True)
        A = sol.sol(T)[0]; C = sol.sol(T)[1]; D = sol.sol(T)[2]
        return np.exp(-r*T+A+1j*u*x+C*rho0+D*v0)
    
    # Coefficient Vk(tM) as in (45)-(46) in Fang, Oosterlee
    if h<0:
        if alpha == 1:
            # VkM = Gkfunc(0,b,K,alpha)+0 
            Vkm = np.zeros(N) #because R = 0
        elif alpha == -1:
            Vkm = Gkfunc(a,h,K,alpha)+0
    elif h>=0:
        if alpha == 1:
            Vkm = Gkfunc(0,h,K,alpha)+0
        elif alpha == -1:
            Vkm = Gkfunc(a,0,K,alpha)

    def Mkfunc(x1,x2,j):
    #function (2.60) in Hirsa
    #three possible states, where k=j=0, where k=j and everything else
        # j = np.arange(0,N)
        Mkjc = (exp(1j*(j+k)*(x2-a)*pi/(b-a))-exp(1j*(j+k)*(x1-a)*pi/(b-a)))/(j+k)
        Mkjs = (exp(1j*(j-k)*(x2-a)*pi/(b-a))-exp(1j*(j-k)*(x1-a)*pi/(b-a)))/(j-k)

        if k == 0:
            Mkjc[0] = (x2-x1)*pi*1j/(b-a)
        # if k==j, which is for every k+1, b/c j is the inner loop
        # Mkjs[np.where(np.isnan(Mkjs))] = (x2-x1)*pi*1j/(b-a)
        Mkjs[k] = (x2-x1)*pi*1j/(b-a)
        
        return (-1j/pi)*(Mkjc+Mkjs)
    
    #param for CF_OU approx
    param = [kv,mv,sv,v0,kr,mr,sr,rho0,rho2]
    phi = np.zeros(N, dtype=complex)
    
    #main loop for up-and-out where x1=a, x2=h, c=h and d=b
    for m in range(M,1,-1): #in range M to 2 backwards with steps -1
        Ck = np.zeros(N) #define for each iteration an empty vector for Ck (2.59)
        for k in range(0,N):
            Mkj = Mkfunc(a,h,np.arange(0,N))
            
            # Solve extended heston phi for given u
            # to obtain Ck as in (2.59) in Hirsa, using T = dt
            # Vkm is actually the option value at each monitoring point:
            char_fcn = lambda u: np.real(CF_OU(param,log(s0),r,dt,u))
            for idx, u in enumerate(range(0,N)):
                phi[idx] = char_fcn( u*pi/(b-a) ) # evaluate charfunc at each u
            Ck[k] = exp(-r*dt) * integrate.trapz( np.real(phi * Vkm * Mkj), x=range(0,N))
        Vkm = Ck 
    
    
    char_fcn = lambda u: CF_OU(param,np.log(K),r,dt,u)
    for idx, u in enumerate(range(0,N)):
        phi[idx] = char_fcn( u*pi/(b-a) ) # evaluate charfunc at each u
        
    #integration using integrate.trapz because Vkm is a vector
    #integrate.quad does uses different u values (acc. to roots of polynomial)
    #calculation of call price according to p. 37 Hirsa, second Equation

    #equation (2.68) in Hirsa
    integrand =  phi * exp( -1j * np.arange(0,N) * pi*a/(b-a) ) * Vkm
    v = integrate.trapz( np.real(integrand), x=range(0,N))
    v = exp(-r*dt) * v
    return v 

t0 = time.time()
print(BarrierExtHestonUOC(s0,K,H,T,r,q,kv,mv,sv,v0,kr,mr,sr,rho0,rho2,alpha,M,N))
elapsed = time.time() - t0
print(elapsed)




