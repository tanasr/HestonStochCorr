"""
Heston Ornstein-Uhlenbeck Model 
Charfunction according to Teng et al. (2016b) - On the Heston Model...
Approach: assuming only the brackets are missing
"""

from numpy import exp, log, log10, sqrt, real, imag, linspace, zeros
from matplotlib import pyplot as plt
import numpy as np


s0 = 100; r = 0.0  # underlying
kv = 2.1; mv = 0.03; sv = 0.2; v0 = 0.02 #variance
kr = 3.4; mr = -0.6; sr = 0.1; rho0 = -0.4; rho2 = 0.4 #rho1=0, #correlation
x0 = log(s0); t = 0; T = 1/12; tau = T-t

def charfuncHestonOU(u, returnFunction):
    # as in lemma 3.1 on page 8-10
    m = sqrt(mv - sv**2/(8*kv)+0j)
    n = sqrt(v0) - m
    d_hat = sqrt((v0*exp(-kv) - sv**2*(1-exp(-kv))/(4*kv)) + \
                 mv*(1-exp(-kv)) + \
                 sv**2*mv*(1-exp(-kv))**2/(8*kv*mv+8*kv*exp(-kv)*(v0-mv)))

    l = -log10(n**-1*(d_hat-m))

    D1 = sqrt(kv**2+sv**2*(u**2+1j*u))
    D2 = (kv-D1)/(kv+D1)

    C1 = 1j*u*(kv-D1)/sv**2

    l1 = -log((exp(-D1)-D2*exp(-D1))/(1-D2*exp(-D1)))

    alpha = kr*mr + m*sr*rho2*u*1j
    beta = n*sr*rho2*u*1j

    C2 = (mv-v0)/(kv+kr-l1) * exp(-kv*T) + ((v0-mv)/(kv+kr)) * exp(-kv*T) \
        - mv/kr + 1/(kr-l1)

    H1 = (1j*u-1)*r*tau + kv*mv/sv**2 \
        * ((kv-D1)*tau - 2*log((1-D2*exp(-D1*tau))/(1-D2)))

    H2c = C1*(v0-mv)*exp(-kv*T)/((kv+kr-l1)*(kv-l1)) \
        - C1*(v0-mv)*exp(-kv*T)/(kv*(kv+kr)) \
        - mv*C1/(kr-l1)*l1 + C1*C2/kr

    H2 = C1*(mv-v0)*exp(kv*(tau-T)-l1*tau)/((kv+kr-l1)*(kv-l1)) \
        + C1*(v0-mv)*exp(kv*(tau-T))/(kv*(kv+kr)) \
        + mv*tau*C1/kr + mv*C1*exp(-l1*tau)/(kr-l1)*l1 \
        - C1*C2*exp(-kr*tau)/kr + H2c

    H3c = C1*(mv-v0)*exp(-T*(kv+l))/((kv+kr-l1)*(kv+l-l1)) \
        + C1*(v0-mv)*exp(-T*(l+kv))/((l+kv)*(kv+kr)) \
        + mv*C1*exp(-l*T)/(kr*l) \
        - mv*C1*exp(-l*T)/((kr-l1)*(l-l1)) \
        + C1*C2*exp(-l*T)/(l-kr)

    H3 = C1*(mv-v0)*exp(tau*(kv+l-l1)-T*(kv+l))/((kv+kr-l1)*(kv+l-l1)) \
        + C1*(v0-mv)*exp((tau-T)*(l+kv))/((l+kv)*(kv+kr)) \
        + mv*C1*exp(l*(tau-T))/(kr*l) \
        - mv*C1*exp(tau*(l-l1)-l*T)/((kr-l1)*(l-l1)) \
        + C1*C2*exp(tau*(l-kr)-l*T)/(l-kr) + H3c

    H4c1 = C1**2*(v0-mv)**2/(2*kv*(kv+kr)**2)
    H4c2 = C1**2*(v0-mv)**2/(2*(2*kv+kr-l1)**2*(kv-l1))
    H4c3 = 2*C1**2*(v0-mv)**2/((kv+kr-l1)*(kv+kr)*(l1-2*kv))
    H4c4 = 2*C1**2*mv**2/(kv*l1*(kr-l1))
    H4c5 = 2*mv*C1**2*C2/(kr**2-l1**2)
    H4c6 = -0.5*C1**2*C2**2/kr
    H4c7 = -2*mv*C1**2*C2/kr**2
    H4c8 = -0.5*mv**2*C1**2/(l1*(kr-l1)**2)
    H4c9 = 2*(v0-mv)*C1**2*C2/((kv+kr)*(kv-kr))
    H4c10 = 2*C1**2*(mv**2-v0*mv)/((kv+kr)*(kv-l1)*(kr-l1))
    H4c11 = 2*(mv-v0)*C1**2*C2/((kv+kr-l1)*(kv-kr-l1))
    H4c12 = 2*C1**2*(mv**2-v0*mv)/(kr*(kv-l1)*(kv+kr-l1))
    H4c13 = 2*C1**2*(v0*mv-mv**2)/((kr-l1)*(kv-2*l)*(kv+kr-l1))
    H4c14 = 2*C1**2*(v0*mv-mv**2)/(kv*kr*(kv+kr)**2)

    H4c = (H4c1+H4c2+H4c3)*exp(-2*kv*T) + H4c4 + H4c5 + H4c6 + H4c7 + \
        H4c8 + (H4c9+H4c10+H4c11+H4c12+H4c13+H4c14)*exp(-kv*T)

    H4 = H4c1*exp(2*kv*(tau-T)) + H4c4*exp(-tau*l1) + H4c5*exp((-l1-kr)*tau) \
        + H4c9*exp(tau*(kv-kr)-kv*T) + H4c2*exp(2*tau*(kv-l1)-2*kv*T) \
        + H4c3*exp(tau*(2*kv-l1)-2*kv*T) + H4c11*exp(tau*(kv-kr-l1)-kv*T) \
        + H4c12*exp(tau*(kv-l1)-kv*T) + H4c13*exp(tau*(kv-2*l1)-kv*T) \
        + H4c6*exp(tau*(-2*kr*tau)) + H4c7*exp(-kr*tau) + H4c8*exp(-2*l1*tau) \
        + H4c10*exp(tau*(kv-l1)-kv*T) + H4c14*exp(kr*(tau-T)) \
        + C1**2*mv**3*tau/kr**2 + H4c


    D = (kv-D1)/sv**2 * (1-exp(-D1*tau))/(1-D2*exp(-D1*tau))
    A = H1 + alpha*H2 + beta*H3 + (sr**2/2)*H4
    C = C1*(mv-v0)/(kv+kr-l1) * exp((kv-l1)*tau-kv*T) \
        + C1*(v0-mv)/(kv+kr)*exp(kv*(tau-T)) \
        + C1*mv/kr \
        - C1*mv/(kr-l1) * exp(-l1) \
        + C1*C2*exp(-kr*tau)
        
    if returnFunction == 'CHAR':
        return exp(-r * T + A + 1j*u * x0 + C * rho0 + D * v0)
    elif returnFunction == 'A':
        return A
    elif returnFunction == 'B':
        return (1j*u + 0)
    elif returnFunction == 'C':
        return C
    elif returnFunction == 'D':
        return D
    else:
        pass


u = linspace(0,8.3,401); phi = 1j*zeros(len(u))
for k in range(len(u)):
    phi[k] = charfuncHestonOU(u[k], 'CHAR')

plt.figure()
plt.plot(u,real(phi))
plt.plot(u,imag(phi))
plt.show()


# u2 = linspace(9.95,10.2,201); phi = 1j*zeros(len(u2))
# for k in range(len(u2)):
#     phi[k] = charfuncHestonOU(u2[k], 'CHAR')

# plt.figure()
# plt.plot(u2,real(phi))
# plt.plot(u2,imag(phi))
# plt.show()