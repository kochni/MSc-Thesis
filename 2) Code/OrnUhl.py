#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.smc_samplers import StaticModel
from particles import distributions as dists


class OrnUhl(StaticModel):
    '''
    Ornstein-Uhlenbeck process, i.e. process of the form
        dS_t = σ·Z_t
    implying
        X_t = σ/S_{t-1}·Z_t
    '''

    def __init__(self, innov, prior, data):
        super().__init__(self, data, prior)

        assert innov in ['norm', 't'], "Innovation distribution not implemented"
        self.innov = innov

    def logpyt(self, theta, t):
        '''
        compute log-likelihood of current observation
        x_t given observations x_{1:t-1} and model parameters θ^(i)
        '''
        sigma = theta['sigma']
        raise NotImplemented
        s_t = sigma/S[t-1]
        print("Min. σ:", min(s_t))
        assert not np.isnan(sigma).any(), "NaNs in volatilities"
        assert all(sigma >= 0), "Volatility is negative for some particles"

        # compute log-likelihood given σ and distribution
        if self.innov == 'norm':
            loglik = -np.log(sigma) - 1/2*np.log(2*np.pi) - 1/(2*sigma**2)*self.data[t]**2
        else:
            loglik = scipy.stats.t.logpdf(self.data[t], loc=0, scale=sigma)

        return loglik