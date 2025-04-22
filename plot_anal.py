import matplotlib.pyplot as plt
import numpy as np

lam = np.linspace(2,3.5,1000)
#f = 4 * (3.5/lam -1) * 3125 * lam**5 / 537824

b = -1
c = 3.5/lam - 1
a = 3125*lam**5 / 537824

E = (-b - np.sqrt(b**2-4*a*c))/(2*a)
E2 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
plt.plot(lam,E)

plt.plot(lam,c,':')

#for i,ll in enumerate(lam):
#    print(ll, E[i])

lam0 = np.interp(0.5,E[::-1],lam[::-1])
print(lam0,1/lam0,14/5)

#plt.yscale('log')
plt.show()


