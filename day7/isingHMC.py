#!/usr/bin/env python
# coding: utf-8
"""todo: more classes, more generality. Could make HMC class which uses
integrator class on an action class, with ising class (can also be ==
action class) to calculate specific observables.
    """


import numpy as np
import  matplotlib.pyplot as plt


class IsingHMC():
    """This is the second version of the Ising model from the notes of Tom
    Luu, so all phi's are really psi's"""
    def __init__(self, J, L, h=0, beta=0, C=5):
        self.J = J
        
        self.L = L
        self.N = L*L
        if not hasattr(h, 'len'):
            h = h*np.ones(self.N)
        self.h = h
        self.C = C
        self.K = self.generate_K(L)
        self.Ktilde = self.K + C*np.eye(self.N)
        self.Ktildeinv = np.linalg.inv(self.Ktilde)
        
        self.set_beta(beta)
        self.n_accepts = 0
        self.n_updates = 0
        # self.previous_accepts = []

    def set_beta(self, beta):
        self.beta = beta
        self.sqrtbetaJ = np.sqrt(beta*self.J)
        self.beta_h = beta*self.h
        
    
    def generate_K(self, L):
        K = np.zeros((L,L,L,L))
        for i in range(L):
            for j in range(L):
                K[i,j,(i+1)%L,j] += 1
                K[i,j,(i-1)%L,j] += 1
                K[i,j,i,(j+1)%L] += 1
                K[i,j,i,(j-1)%L] += 1
        return K.ravel().reshape((L**2,L**2))

    def initialize(self, phi0=None):
        if phi0 is None:
            self.phi0 = np.random.uniform(size=self.N)
        else:
            self.phi0 = phi0
        self.reset_acceptance()
        return self.phi0

        
    def phidot(self, p, phi):
        return p

    def pdot(self, p, phi):
        Ktildephi = self.Ktilde.dot(phi)
        return (- Ktildephi + self.beta_h/self.sqrtbetaJ                
                + self.sqrtbetaJ*self.Ktilde.dot(np.tanh(self.sqrtbetaJ*Ktildephi)))
    
    def leapfrog(self, phi0, p0, eps, Nmd):
        pi = p0
        phi = phi0
        phi = phi + eps/2*self.phidot(pi, phi)
        
        for i in range(Nmd -1):
            pi = pi + eps*self.pdot(pi, phi)
            phi = phi + eps*self.phidot(pi, phi)
        
        pi = pi + eps*self.pdot(pi, phi)
        phi = phi + eps/2*self.phidot(pi, phi)
        
        return pi, phi
    
    def action(self, p, phi):
        Ktildephi = self.Ktilde.dot(phi)
        return (p.dot(p)/2                
                + 0.5*phi.dot(Ktildephi)                
                - 1/self.sqrtbetaJ * self.beta_h.dot(phi)               
                - np.sum(np.log(2*np.cosh(self.sqrtbetaJ*Ktildephi))))
        
    def sample_p(self):
        return np.random.normal(size=self.N)
        
    def hmc_sample(self, phi0, eps, Nmd):
        p0 = self.sample_p()
        pf, phif = self.leapfrog(phi0, p0, eps, Nmd)
        H0 = self.action(p0, phi0)
        Hf = self.action(pf, phif)
        dS = Hf - H0
        if np.random.uniform(0,1) < min(1, np.exp(-dS)):
            self.n_accepts += 1
        else:
            phif = phi0
            
        self.n_updates += 1
        return phif
    
    def reset_acceptance(self):
        # self.previous_accepts.append([self.n_updates, self.n_accepts])
        self.n_accepts = 0
        self.n_updates = 0
        
    def acceptance(self):
        return self.n_accepts/self.n_updates
               
    def magnetization(self, phi):
        # this is not valid if not h_i == h_j for all i,j
        h = self.h[0]
        return (1/self.sqrtbetaJ*1/self.N*np.sum(phi, axis=-1)              
                - h/self.J*1/self.N*np.sum(self.Ktildeinv))
    
    def betaEps(self, phi):
        # this is not valid if not h_i == h_j for all i,j
        N = self.N
        h = self.h[0]
        Ktilde_phi = np.einsum('ij,...i->...j',self.Ktilde, phi)
        Kscript = 1/N*np.sum(self.Ktildeinv)
        return (
            self.C*self.beta*self.J/2                 
            + (self.beta*h)**2*Kscript/(2*self.beta*self.J)                 
            + self.beta*h/(2*N*self.sqrtbetaJ)*np.sum(phi,axis=-1)                 
            - self.sqrtbetaJ/(2*N)*np.sum(
                Ktilde_phi*np.tanh(self.sqrtbetaJ*Ktilde_phi), axis=-1
                ))
                
    def run(self, T=1, eps=0.1, Ncf=10000, Ncorr=5, Ntherm=100, verbose=True):
        """phi0 must be initialized before running."""
        try:    
            phi = self.phi0
        except AttributeError:
            raise ValueError('Error, must initialize a phi0 before running')

        Nmd = int(T/eps)
        print(Nmd)

        phi_values = []

        thermalized = False
        acc_pre = 0

        for i in range(Ncf):
            phi = self.hmc_sample(phi, eps, Nmd)
            acc = self.acceptance()
            if i%Ncorr == 0:
                if i >= Ntherm:
                    phi_values.append(phi)
                    if not thermalized:
                        # acceptance pre thermalization
                        acc_pre = self.acceptance()
                        self.reset_acceptance()
                        thermalized = True
                if verbose:
                    print(f"L={self.L} conf {i}/{Ncf}, accept: {acc:.2f}. "
                        +f"accept pre therm: {acc_pre:.2f}, eps: {eps}")
        
        # remove initial value so it has to be explicitly reset
        del(self.phi0)
        return np.array(phi_values)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--lattice_spacing', type=int, default=32)
    parser.add_argument('-b', '--betaJ', type = float, default=0.5)

    parser.add_argument('-T', '--T', type = float, default=1)
    parser.add_argument('-eps', '--eps', type = float, default=0.1)
    parser.add_argument('-Ncf', '--Ncf', type = int, default=1000)
    parser.add_argument('-Ncorr', '--Ncorr', type = int, default=5)
    parser.add_argument('-Ntherm', '--Ntherm', type = int, default=100)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    L = args.lattice_spacing
    N = L**2

    h = 0
    J = 1
    beta = args.betaJ/J
    ising = IsingHMC(J=J, L=L, beta=beta, h=h)
    ising.initialize()
    ensemble = ising.run(
        T=args.T, eps=args.eps, Ncf=args.Ncf, 
        Ncorr=args.Ncorr, Ntherm=args.Ntherm
        )
    fname = 'cfgs/L{}_b{:.4f}.npy'.format(L, beta)
    print('saving as {}'.format(fname))
    np.save(fname, ensemble)
