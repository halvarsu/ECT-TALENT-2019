from isingHMC import IsingHMC
import numpy as np

eps = 0.15
J = 1
h = 0
Ncf = 10000

for L in [16,32,48,64]:
    N = L**2
    ising = IsingHMC(J=J, L=L, beta=0, h=h)
    phi = np.random.uniform(0,1,size=N)

    for beta_inv in np.linspace(4, 1.4, 51):
        ising.set_beta(1/beta_inv)

        # initialize with previous phi
        ising.initialize(phi)
        # just save everything
        ensemble = ising.run(Ncf=Ncf, Ncorr=1, Ntherm=1, eps=0.15)
        phi = ensemble[-1]

        fname = 'cfgs/L{}_binv{:.4f}.npy'.format(L, beta_inv)
        print('saving as {}'.format(fname))
        np.save(fname, ensemble)
