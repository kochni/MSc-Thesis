#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.smc_samplers import StaticModel
from particles import distributions as dists

import signatory


class SigVol(StaticModel):
    '''
    Signature-based volatility model, i.e. process of the form
        X_t = σ_t·Z_t
        σ_t^2 = l(Sig(X_{1:t}))
    where Sig(X_{1:t}) is the signature of the path X_{1:t} and l() is a linear
    operator

    Parameters:
    -----------

    sig_type: string
        one of 'full' and 'random'; type of the signature use, where 'full'
        corresponds to the original definition and 'random' to the randomized
        signature (Cuchiero et al., 2021)

    q: int
        dimensionality of the signature; if sig_type = 'full' then q is the
        truncation level of the signature, similarly if sig_type = 'random'
        then q is the dimensionality of the randomized signature
    '''

    def __init__(self, sig_type, innov, prior, data):
        super().__init__(data, prior)

        assert innov in ['norm', 't'], "Innovation distribution not implemented"
        self.innov = innov

    def sig(sefl):
        pass

    def randsig(self):
        pass

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current observation
        x_t given observations x_{1:t-1} and model parameters θ^(i)
        '''
        sigma = theta['sigma']
        print("Min. σ:", min(sigma))
        assert not np.isnan(sigma).any(), "NaNs in volatilities"
        assert all(sigma >= 0), "Volatility is negative for some particles"

        # compute log-likelihood given σ and distribution
        if self.innov == 'norm':
            loglik = -np.log(sigma) - 1/2*np.log(2*np.pi) - 1/(2*sigma**2)*self.data[t]**2
        else:
            loglik = scipy.stats.t.logpdf(self.data[t], loc=0, scale=sigma)

        return loglik

    def summary(self):
        return None


