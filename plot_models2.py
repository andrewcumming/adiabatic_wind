import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.animation import FuncAnimation
from scipy import optimize
import re

sys.path.insert(0, '../athena/vis/python')
import athena_read

def read_data(filename, gamma):
    data = athena_read.athdf(filename)
    x = data['x1v']
    rho = data['rho'][0, 0, :]
    vel = data['vel1'][0, 0, :]
    pres = data['press'][0, 0, :]
    cs = np.sqrt(gamma * pres/rho)
    Kvec = pres / (rho**gamma)
    return x, rho, vel, pres, cs, Kvec
    
def analytic_cs(cs, gamma, gm1):
    K = 1/3
    delta = (5-3*gamma)/gm1
    cs2 = cs**2
    f = cs2 - (gamma*K)**(-2/gm1)/16/delta * cs2**delta - (2/delta)*(gamma*K/gm1 - 1)
    #print("in analytic_cs:", cs, delta, f)
    return f

def analytic_Mdot(K):
    # analytic solution for adiabatic wind
    
    gamma = 1.5-K
    gm1 = gamma-1
    K = 1.0/3.0    
    # analytic estimate for sound speed at the sonic point assuming v_star = 0
    csonic_guess = np.sqrt(2*(gamma*K-gm1)/(5-3*gamma))
    # analytic sound speed at the sonic point
    csonic = optimize.fsolve(analytic_cs, x0=csonic_guess, args=(gamma, gm1), factor=0.01)

    # now get Mdot and rs       
    Mdot_guess = np.pi / (gamma * K)**(1/gm1) * csonic**((5-3*gamma)/gm1)
    rs_guess = 0.5 / csonic**2
    alpha = (5-3*gamma)/gm1
    v_guess = csonic**alpha / 4 / (gamma * K)**((3+alpha)/2)     
    return Mdot_guess, rs_guess, v_guess

# Read parameters from the input file
xlims = (0.00003,0.5)

# Initialize the plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(6, 8)) #gridspec_kw={'hspace': 0})
ax1.set_ylabel(r'$\dot{M}$', color='C0')
ax1_right = ax1.twinx()
ax1_right.set_ylabel(r'$r_s/R$', color='C1')

#ax1.set_yscale('log')
ax1.set_xscale('log')
#ax1_right.set_yscale('log')

ax2_right = ax2.twinx()
ax1.set_xlabel(r'$\frac{3}{2} - \gamma$')
ax2.set_xlabel(r'$\frac{3}{2} - \gamma$')
#ax2_right.set_yscale('log')
ax2.set_xscale('log')

ax2.set_ylabel(r'$v, c_s$', color='C0')
ax2_right.set_ylabel(r'$\mathcal{M}_\star$', color='C1')

for ax in [ax1, ax1_right, ax2, ax2_right]:
    ax.tick_params(which='both', direction='in', right=True, top=True)
ax1.tick_params(which='both', direction='in', left=True, right=False, labelleft=True)
ax2.tick_params(which='both', direction='in', left=True, right=False, labelleft=True)

# Plot the analytic curves
#Ka = np.linspace(Kcrit * 1.001, Ksonic, 100)
Ka = np.linspace(-4, -0.5, 100)
Ka = 10**Ka
Mdota = np.zeros_like(Ka)
rsa = np.zeros_like(Ka)
csa = np.zeros_like(Ka)
vsa = np.zeros_like(Ka)
for i, K in enumerate(Ka):
    Mdota[i], rsa[i], vsa[i] = analytic_Mdot(K)    
    #print(i, K, Mdota[i], rsa[i], vsa[i])

csa = np.sqrt((1.5-Ka)/3)

#ax1_right.plot([Kcrit,Kcrit],[0.9,70],'k-.',alpha=0.2)
#ax1_right.plot([Ksonic,Ksonic],[0.9,70],'k-.',alpha=0.2)
#ax2_right.plot([Kcrit,Kcrit],[0.0,2],'k-.',alpha=0.2)
#ax2_right.plot([Ksonic,Ksonic],[0.0,2],'k-.',alpha=0.2)
#ax1_right.plot(xlims,[1.0,1.0],'k-.',alpha=0.2)
ax1.plot(Ka,Mdota,color='C0')
ax1_right.plot(Ka, rsa, ':', color='C1')
ax2.plot(Ka,csa, color='C0')
ax2_right.plot(Ka,vsa/csa, color='C1')
ax2.plot(Ka,vsa, color='C0')

ax1.set_xlim(xlims)
ax2.set_xlim(xlims)

ax2.set_ylim((0,1))
ax2_right.set_ylim((0.6,0.7))
ax1_right.set_ylim((0.9,2.25))
ax1.set_ylim((4.5,6.5))

# And then the data
#these are for gam=1.1:
#Kvals = [10,105,11,12,15,20,25,30,35,40,45]
#these are the gamma=1.4 values:
Kvals = [12000,13000,14000,14500,14900,14990,14995,14998,14999]
Mdotvals = np.zeros(len(Kvals))
rsvals = np.zeros(len(Kvals))
Ks = np.zeros(len(Kvals))
velvals = np.zeros(len(Kvals))
machvals = np.zeros(len(Kvals))
csvals = np.zeros(len(Kvals))

for i,K in enumerate(Kvals):
    filename = 'results_K0333333/adiwind_gam%5d.out1.02000.athdf' % K
    gamma = K*1e-4
    Ks[i] = 1.5 - gamma
    print(filename)
    x, rho, vel, pres, cs, Kvec = read_data(filename, gamma)
    # extract Mdot at the half-way point
    n = len(x)//2
    Mdotvals[i] = 4 * np.pi * x[n]**2 * rho[n] * vel[n]
    mach2 = (vel/cs)**2
    print(mach2)
    rsvals[i] = np.interp(1, mach2, x)
    velvals[i] = vel[0]
    csvals[i] = cs[0]
    machvals[i] = vel[0]/cs[0]
    print(i, Ks[i], Mdotvals[i], rsvals[i], x[n], vel[n], rho[n], n)

print(Mdotvals)
ax1.scatter(Ks, Mdotvals, marker='s', color='C0')
ax1_right.scatter(Ks, rsvals, marker='s', color='C1')
ax2.scatter(Ks, velvals, marker='s', color='C0')
ax2.scatter(Ks, csvals, marker='o', color='C0')
ax2_right.scatter(Ks, machvals, marker='s', color='C1')

plt.tight_layout()
plt.show()
#plt.savefig('plot_K0333333.pdf')
