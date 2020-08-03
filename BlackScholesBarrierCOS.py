"""
Barrier Option under the GBM with COS method
"""

from numpy import exp, real, pi, log, sqrt, cos, sin
import numpy as np
import time

# Formulas in brackets are from Hirsa page 63ff
def BarrierGbmOUC(s0,K,H,T,r,q,sigma,alpha,N=128):
    # T = observation dates
    # in months
    Tvec = np.arange(0,T+1)/(12); dt = Tvec[1]-Tvec[0]; M = len(Tvec)-1
    #in days
    # Tvec = np.arange(0,T+1)/(250); dt = Tvec[1]-Tvec[0]; #M = len(Tvec)-1
    # dt = T[1]-T[0]; M = len(T)-1
    x = log(s0/K); k = np.arange(0,N,1)
    
    #c1 and c2 according to GBM
    c1 = (r-q-0.5*sigma**2)*dt; c2 = sigma**2*dt; 
    L = 10; 
    a = c1+x-L*sqrt(abs(c2)); b = c1+x+L*sqrt(abs(c2));
    h = log(H/K); 
    
    # chifunction (2.64)
    def chifunc(c,d):
        chi = 1/(1+(k*pi/(b-a))**2) * \
            (cos(k*pi*(d-a)/(b-a)) * exp(d)-cos(k*pi*(c-a)/(b-a)) * \
                                         exp(c)+k*pi/(b-a) * sin(k*pi*(d-a)/(b-a)) * \
                                        exp(d)-k*pi/(b-a) * sin(k*pi*(c-a)/(b-a))*exp(c))
        return chi
    
    # phifucntion (2.65)
    def phifunc(c,d):
        phi = (sin(k*pi*(d-a)/(b-a))-sin(k*pi*(c-a)/(b-a)))*(b-a)/(k*pi)
        phi[0] = d-c
        return phi
    
    # Gkfunc (2.63)
    def Gkfunc(x,y,K,alpha):
        return 2/(b-a)*alpha*K*(chifunc(x,y)-phifunc(x,y))
    
    def charfuncGBM(u):
        # T = 
        return exp(1j*(x + (r-q-0.5*sigma**2)*dt)*u - 0.5*dt*sigma**2*u**2)
    
    # Coefficient Vk(tM) as in (45)-(46) in Fang
    if h<0:
        if alpha == 1:
            # VkM = Gkfunc(0,b,K,alpha)+0 
            Vkm = np.zero(N) #because R = 0
        elif alpha == -1:
            Vkm = Gkfunc(a,h,K,alpha)+0
    elif h>=0:
        if alpha == 1:
            Vkm = Gkfunc(0,h,K,alpha)+0
        elif alpha == -1:
            Vkm = Gkfunc(a,0,K,alpha)

    def Mkfunc(x1,x2,j):
    #function (2.60)
    #three possible states, where k=j=0, where k=j ans everything else
        # j = np.arange(0,N)
        Mkjc = (exp(1j*(j+k)*(x2-a)*pi/(b-a))-exp(1j*(j+k)*(x1-a)*pi/(b-a)))/(j+k)
        Mkjs = (exp(1j*(j-k)*(x2-a)*pi/(b-a))-exp(1j*(j-k)*(x1-a)*pi/(b-a)))/(j-k)

        if k == 0:
            Mkjc[0] = (x2-x1)*pi*1j/(b-a)
        
        # if k==j, which is for every k+1, b/c j is the inner loop
        # Mkjs[np.where(np.isnan(Mkjs))] = (x2-x1)*pi*1j/(b-a)
        Mkjs[k] = (x2-x1)*pi*1j/(b-a)
        
        # if j==k: #create a var. with a boolean TRUE
        #     tmp = j==k
        #     Mkjs[tmp] = (x2-x1)*pi*1j/(b-a)
        # Mkjs[k] = (x2-x1)*pi*1j/(b-a)
        return (-1j/pi)*(Mkjc+Mkjs)
    
    #calculate coeff Ck (2.59) in the outer loop and coeff Vk in the inner loop
    #inner loop for Ckm: loop over j:N-1
    #outer loop for Vkm: loop over m --> from M in -1 steps until 2 (in python until 3)
    
    #main loop for up-and-out where x1=a, x2=h, c=h and d=b
    for m in range(M,1,-1): #in range M to 2 backwards with steps -1
        Ck = np.zeros(N) #define for each iteration an empty vector for Ck (2.59)
        for k in range(0,N):
            Mkj = Mkfunc(a,h,np.arange(0,N))
            Ck[k] = np.sum(real(charfuncGBM(np.arange(0,N)*pi/(b-a))*Vkm*Mkj)) - \
                            0.5*real(charfuncGBM(0*pi/(b-a))*Vkm[0]*Mkj[0])
            Ck[k] = exp(-r*dt)*Ck[k]
        Vkm = Ck
    
    k = np.arange(0,N)
    v = np.sum(real(charfuncGBM(k*pi/(b-a)) * \
                 exp(-1j*k*pi*a/(b-a))) * Vkm) - \
        0.5*real(charfuncGBM(0))*Vkm[0]
    v = exp(-r*dt)*v
    return v

t0 = time.time()
s0 = 100; K = 100; H = 120; T = 10*12; #M = 120 # [M] = 1/year
r = 0.05; q = 0; sigma = 0.2; alpha = 1
# print(BarrierGbmOUC(100, 100, 120, 12, 0.05, 0, 0.2, 1, 128))
print(BarrierGbmOUC(s0,K,H,T,r,q,sigma,alpha,64))
#refvalue according to BSc thesis: 1.8494
elapsed = time.time() - t0
print(elapsed)
