#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

import particles
from particles.state_space_models import StateSpaceModel
from particles import distributions as dists

# documentation:
# https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/_autosummary/particles.state_space_models.html#module-particles.state_space_models

# source code:
# https://github.com/nchopin/particles/blob/master/particles/state_space_models.py


class StochVol(StateSpaceModel):
    '''
    Stochastic Volatility model, i.e. process of the form
        X_t = V_t·Z_t
        log(V_t^2) = ω·(1-α) + α·log(V_t^2) + ξ·U_t
        Z_t, U_t ~iid N(0,1)
        Cov(Z_t, U_t) = ρ

    Parameters:
    -----------
    innov_X: string
        innovation distribution of observations.

    innov_V: string
        innovation distribution of volatilities.

    pf_type: string
        Particle filter type; one of 'bootstrap', 'guided', and 'auxiliary'

    prior: StructDist (from particles.distributions)
        prior distributions on model parameters

    data: (T,)-array
        observations.
    '''

    default_params = {'alpha': 1, 'xi': 0}

    def __init__(self, innov_X, innov_V, rho, pf_type):

        assert innov_X in ['norm', 't'], "Specified innovation distribution not implemented"
        assert innov_V in ['norm', 't'], "Specified innovation distribution not implemented"
        self.innov_X = innov_X
        self.innov_V = innov_V

        assert 0 <= rho <= 1, "rho must be between 0 and 1"
        self.rho = rho

        assert pf_type in ['bootstrap', 'guided'], "invalid Particle Filter type; must be one of 'bootstrap', 'guided', and 'auxiliary'"
        self.pf_type = pf_type

    def PX0(self):
        '''
        distribution of initial latent state V_1
        '''
        if self.innov_X == 'norm':
            px0 = dists.Normal(loc=0, scale=1)
        else:
            px0 = dists.Student(loc=0, scale=1, df=5)

        return px0

    def PX(self, t, xp):
        '''
        transition kernel of latent states, i.e. distribution of V_t given V_{t-1}

        Note: 'xp' is log(V_{t-1}^2)
        '''
        if self.innov_V == 'norm':
            return dists.Normal(loc=self.omega*(1-self.alpha) + self.alpha*xp,
                                scale=self.xi)
        else:
            return dists.Student(loc=self.alpha*xp,
                                 scale=self.xi,
                                 df=5)

    def PY(self, t, xp, x):
        '''
        distribution of X_t given (X_{t-1}, V_t)

        Note: 'x' is log(V_t^2), 'xp' is log(V_{t-1}^2), 'xi' is vol-of-vol
        '''
        if self.innov_X == 'norm':
            return dists.Normal(loc=0, scale=np.exp(x))
        else:
            return dists.Student(loc=0, scale=np.exp(x), df=5)

    def proposal0(self, data):
        '''
        proposal distribution of inital latent state V_1
        '''
        return dists.Normal(scale = self.sigma)

    def proposal(self, t, xp, data):
        '''
        conditional proposal distribution of V_t given V_{t-1} and X_{1:t}
        '''
        if self.pf_type == 'guided':  # Gaussian random walk
            return dists.Normal(loc=self.alpha*xp, scale=self.xi)

