import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.animation import FuncAnimation
from scipy import optimize
import re

sys.path.insert(0, '../athena/vis/python')
import athena_read

def read_data(i):
    data = athena_read.athdf(f"out/adiwind.out1.{i:05d}.athdf")
    x = data['x1v']
    rho = data['rho'][0, 0, :]
    vel = data['vel1'][0, 0, :]
    pres = data['press'][0, 0, :]
    cs = np.sqrt(gamma * pres/rho)
    Kvec = pres / (rho**gamma)
    return x, rho, vel, pres, cs, Kvec
    
def extract_from_input(field, filename):
    with open(filename, 'r') as file:
        for line in file:
            match = re.search(field + r'\s*=\s*([\d.]+)', line)
            if match and line[0] != '#':
                return float(match.group(1))  # Convert to float if needed
    return None  # Return None if no match is found

# Read parameters from the input file
gamma = extract_from_input('gamma','athinput.adiabatic_wind')
gm1 = gamma - 1.0
K = extract_from_input('K','athinput.adiabatic_wind')
tlim = extract_from_input('tlim','athinput.adiabatic_wind')
dt = extract_from_input('dt','athinput.adiabatic_wind')
rho0 = extract_from_input('rho0','athinput.adiabatic_wind')
Kcrit = gm1/gamma
print('gamma = ', gamma, ' K = ', K, ' Kcrit = ', Kcrit, ' rho0 = ', rho0)
nsteps = int(tlim/dt)
print('tlim, dt, nsteps=', tlim, dt, nsteps)

# Look at the final model to get the sonic radius etc 
x, rho, vel, pres, cs, Kvec = read_data(nsteps)
Mdot = 4*np.pi*x**2 * rho * vel
mach2 = (vel/cs)**2
rs = np.interp(1, mach2, x)
csonic = np.interp(rs, x, cs)
print("Sonic radius = ", rs) 
print("Sound speed at the sonic point = ", csonic) 
print("Final Kvec =", Kvec) 
#K = Kvec[-1]  

print("cstar = ", (gamma*K)**0.5, cs[0])
 
# Initialize the plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(5, 8), gridspec_kw={'hspace': 0})
ax1.set_yscale('log')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_ylabel(r'$\rho$')
ax2.set_ylabel(r'$v, c_s$')
ax3.set_ylabel(r'$\dot{M}$')
ax2.set_xlabel('r')
ax1_right = ax1.twinx()
ax1_right.set_ylabel(r'$P$', color='r')
ax1_right.set_yscale('log')
ax2_right = ax2.twinx()
ax2_right.set_ylabel(r'$\mathcal{M}^2$', color='r')
ax2_right.set_yscale('log')
ax3_right = ax3.twinx()
ax3_right.set_ylabel(r'$E$', color='r')
#ax3_right.set_yscale('log')
for ax in [ax1, ax1_right, ax2, ax2_right, ax3, ax3_right]:
    ax.tick_params(which='both', direction='in', right=True, top=True)
ax1.tick_params(which='both', direction='in', left=True, right=False, labelleft=True)
ax2.tick_params(which='both', direction='in', left=True, right=False, labelleft=True)
ax3.tick_params(which='both', direction='in', left=True, right=False, labelleft=True)

# Plot the analytic result
x, rho, vel, pres, cs, Kvec = read_data(0)

def analytic(u,r):
    #f =  (u**4+3)/u - 1.2*(1 + (2.5*K-1)*r)/K
    #f =  (u**4+3)/u - 1 - (K/Kcrit-1)*r/csonic**2   
    f =  u*u/(gamma*K) - np.log(u*u/gamma/K) - 4*np.log(r/rs) -4*rs/r +3
    return f

def analytic_rho(rho,r,Mdot,E):
    f = 0.5 * (Mdot/4/np.pi/r**2/rho)**2 + gamma*K*(rho**gm1)/gm1 - E - 1/r
    return f

def analytic_cs(cs):
    delta = (5-3*gamma)/gm1
    cs2 = cs**2
    f = cs2 - (gamma*K)**(-2/gm1)/16/delta * cs2**delta - (2/delta)*(gamma*K/gm1 - 1)
    #print("in analytic_cs:", cs, delta, f)
    return f

if K>Kcrit:  # wind

    
    if 0:
       # isothermal wind Mdot
       Mdot_guess = np.pi * np.exp(1.5-1.0/(gamma*K))/(gamma*K)**1.5
       ax3.plot([x[0],x[-1]], [Mdot_guess, Mdot_guess], 'k-.',alpha=0.5)
       u_guess = np.array([])
       for r in x:
           if len(u_guess) == 0:
               x0 = 2*np.exp(-4*rs/r)
           elif r>rs:
               x0 = 2.0
           else:
               x0 = u_guess[-1]
           u = optimize.fsolve(analytic, x0=x0, args=(r,))     
           u_guess = np.append(u_guess, u)
       ax2.plot(x,u_guess,'k-.',alpha=0.5)
       rho_guess = Mdot_guess / (4*np.pi*x**2*u_guess)
       ax1.plot(x,rho_guess,'k-.',alpha=0.5)
    
    else:
       
       # analytic solution for adiabatic wind
       
       # analytic estimate for sound speed at the sonic point assuming v_star = 0
       csonic_guess = np.sqrt(2*(gamma*K-gm1)/(5-3*gamma))
       print("csonic_guess = ", csonic_guess)       

       # analytic sound speed at the sonic point
       csonic_analytic = optimize.fsolve(analytic_cs, x0=csonic_guess)
       print("csonic_analytic = ", csonic_analytic)  
       vstar_analytic = 0.25 * (gamma*K)**(-1/gm1) * csonic_analytic**((5-3*gamma)/gm1)
       print("vstar_analytic = ", vstar_analytic)  
        
       # set "csonic_guess" to whichever version of csonic we want to use for the profiles  
       csonic_guess = csonic_analytic
       
       Mdot_guess = np.pi / (gamma * K)**(1/gm1) * csonic_guess**((5-3*gamma)/gm1)
       ax3.plot([x[0],x[-1]], [Mdot_guess, Mdot_guess], 'k-.',alpha=0.5)
   
       E_guess = 0.5 * csonic_guess**2 * (5-3*gamma)/gm1
       ax3_right.plot([x[0],x[-1]], [E_guess, E_guess], 'r-.',alpha=0.5)
   
       rho_guess = np.array([])
       for r in x:
           if len(rho_guess) == 0:
               rho0 = 0.99
           else:
               rho0 = rho_guess[-1] * 0.98
           u = optimize.fsolve(analytic_rho, x0=rho0, args=(r,Mdot_guess,E_guess))     
           rho_guess = np.append(rho_guess, u)
       ax1.plot(x,rho_guess,'k-.',alpha=0.5)
   
       u_guess =  Mdot_guess / (4*np.pi*x**2*rho_guess)
       ax2.plot(x,u_guess,'k-.',alpha=0.5)
   
       c_guess = np.sqrt(gamma * K * rho_guess**gm1)
       ax2.plot(x,c_guess,'k-.',alpha=0.5)
   
       mach_guess = (u_guess/c_guess)**2
       ax2_right.plot(x,mach_guess,'k-.',alpha=0.5)
   
           
else:   # hydrostatic atmosphere

    rho_guess = np.ones_like(x) * 1e-10
    rmax = 1/(1-gamma*K/gm1)
    print("rmax=",rmax)
    ind = np.where(x < rmax)
    rho_guess[ind] = (1.0 + (gm1/gamma/K)*((1/x[ind])-1))**(1/gm1)
    ax1.plot(x,rho_guess,'k-.',alpha=0.5)


# set up the lines for the data 
line1, = ax1.plot([], [], label='Density')
line2, = ax2.plot([], [], label='Velocity')
line3, = ax2_right.plot([], [], 'r', label=r'Mach squared')
line4, = ax1_right.plot([], [], 'r', label=r'$P$')
line5, = ax2.plot([], [], label='Sound speed')
line6, = ax2.plot([], [], 'g', label='K')
line7, = ax3.plot([], [], label='Mdot')
line8, = ax3_right.plot([], [], 'r', label='E')


# Function to update the plot
def update(i):
    x, rho, vel, pres, cs, Kvec = read_data(i)
    Mdot = 4*np.pi*x**2 * rho * vel
    mach2 = (vel/cs)**2
    E = 0.5*vel*vel + gamma*pres/rho/gm1 - 1/x

    # Update data for each line
    line1.set_data(x, rho)
    line2.set_data(x, vel)
    line3.set_data(x, mach2)
    line4.set_data(x, pres)
    line5.set_data(x, cs)
    line6.set_data(x, Kvec)
    line7.set_data(x, Mdot)
    line8.set_data(x, E)

    if 1:
        # Rescale axes
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax2_right.relim()
        ax2_right.autoscale_view()
        ax1_right.relim()
        ax1_right.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        ax3_right.relim()
        ax3_right.autoscale_view()

    titl.set_text(f"gam = {gamma}, K = {K}, t={i*dt}")
    #titl.set_text(f"Iteration: {i}")
     
    #plt.savefig(f'png/frame_{i:05d}.png')
    
    return line1, line2, line3, line4, line5, line6, line7, line8, titl

titl = ax1.text(0.5, 1.05, f"gam = {gamma}, K = {K}, t=0", transform=ax1.transAxes, 
                      ha="center", fontsize=14)
# Create animation
ani = FuncAnimation(fig, update, frames=range(nsteps+1), interval=1, blit=False, repeat=False)

# Show the animation
plt.tight_layout()
plt.show()
