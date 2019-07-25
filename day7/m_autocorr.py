import numpy as np
from isingHMC import IsingHMC

J = 1
h = 0

def acf(x, length=100):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

betainv_values = np.linspace(4, 1.4, 51)
for L in [16,32,48,64]:
    for betainv in betainv_values:
        beta = 1/betainv
        filename = f'cfgs/L{L}_binv{betainv:.4f}.npy'
        phi_values = np.load(filename)
        ising = IsingHMC(J=J, L=L, beta=beta, h=h)
        m = ising.magnetization(phi_values)

        times = np.arange(m.size)
        done = False
        nskip = 1
        while not done:
            try:
                # print(m[::nskip].size)
                ac = acf(m[::nskip])

                print('{}, correlation length: {}'.format(
                    filename, times[::nskip][np.where(ac < 1/np.exp(1))[0][0]])
                     )
                done = True

            except IndexError:
                nskip *= 2

