#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.smc_samplers import StaticModel
from particles import distributions as dists


class WhiteNoise(StaticModel):
    '''
    White noise process, i.e. process of the form
        X_t = σ·Z_t
    '''
    def __init__(self, innov, prior, data):
        super().__init__(data, prior)

        assert innov in ['norm', 't'], "Innovation distribution not implemented"
        self.innov = innov

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current observation
        x_t given x_{1:t-1} and model parameters θ^(i)
        '''
        sigma = theta['sigma']
        assert not np.isnan(sigma).any(), "NaNs in volatilities"
        assert all(sigma >= 0), "Volatility is negative for some particles"

        # compute log-likelihood given σ and distribution
        if self.innov == 'norm':
            return -np.log(sigma) - 1/2*np.log(2*np.pi) - 1/(2*sigma**2)*self.data[t]**2
        else:
            return scipy.stats.t.logpdf(self.data[t], loc=0, scale=sigma)

