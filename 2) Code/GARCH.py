#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.smc_samplers import StaticModel
from particles import distributions as dists

# documentation:
# https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/_autosummary/particles.smc_samplers.html#module-particles.smc_samplers

# source code:
# https://github.com/nchopin/particles/blob/master/particles/smc_samplers.py


class GARCH(StaticModel):
    '''
    GARCH(1,1) model, i.e. process of the form
        X_t = σ_t·Z_t,
        σ_t = f(X_{t-1}, σ_{t-1}^2)
    where f(.) depends on the GARCH variant

    Parameters:
    -----------
    innov: string
        Innovation distribution; must be one of 'norm' (for Gaussian) and 't'
        (for Student t).

    prior: StructDist (from particles.distributions)
        Prior distributions on the model parameters.

    data: (T,)-array
        Observations.
    '''

    def __init__(self, variant, innov, prior, data, sd=None, activ=None):
        super().__init__(data, prior)

        assert variant in ['standard', 'gjr', 'exp', 'neural'], "GARCH variant not implemented"
        self.variant = variant

        assert innov in ['norm', 't'], "Specified innovation distribution not implemented"
        self.innov = innov

        self.sd = sd
        self.activ = activ

        if variant == 'neural':
            self.W1 = np.random.normal(loc=0, scale=self.sd, size=(100, 2))
            self.W2 = np.random.normal(loc=0, scale=self.sd, size=(3, 100))
            self.b1 = np.random.normal(loc=0, scale=self.sd, size=1)
            self.b2 = np.random.normal(loc=0, scale=self.sd, size=1)

    def vol_std(self, theta, t):
        '''
        Standard GARCH volatility
            σ^2_t = ω + α·X_{t-1}^2 + β·σ_{t-1}^2
        '''
        omega = np.exp(theta['omega'])
        alpha = np.exp(theta['alpha'])
        beta = np.exp(theta['beta'])

        s2_j = np.array([1]*len(theta))

        for j in range(1, t+1):
            s2_prev = s2_j
            s2_j = omega + alpha*self.data[j-1]**2 + beta*s2_prev
            assert not np.isnan(s2_j).any(), "NaNs in volatilities"
            assert all(s2_j >= 0), "Volatility is negative for some particles"

        return s2_j

    def vol_gjr(self, theta, t):
        '''
        GJR-GARCH volatility
            σ^2_t = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2
        '''
        omega = np.exp(theta['omega'])
        alpha = np.exp(theta['alpha'])
        beta = np.exp(theta['beta'])
        gamma = np.exp(theta['gamma'])

        s2_j = np.array([1]*len(theta))

        for j in range(1, t+1):
            s2_prev = s2_j
            s2_j = omega + (alpha + gamma*(self.data[j-1]>0))*self.data[j-1]**2 + beta*s2_prev
            assert not np.isnan(s2_j).any(), "NaNs in volatilities"
            assert all(s2_j >= 0), "Volatility is negative for some particles"

        return s2_j

    def vol_exp(self, theta, t):
        '''
        Exponential-GARCH volatility
            log(σ^2_t) = ω + α·(|Z_{t-1}| - E[|Z|]) + β·log(σ_{t-1}^2)
        '''
        omega, alpha, beta, gamma = theta['omega'], theta['alpha'], theta['beta'], theta['gamma']
        s2_j = np.array([1]*len(theta))

        # expected magnitude of innovations, E[|Z|] (depends on distribution)
        if self.innov == 'norm':
            self.E_absZ = np.sqrt(2/np.pi)
        else:
            from scipy.special import gamma as Gamma
            self.E_absZ = 2/np.sqrt(np.pi)*Gamma((self.df+1)/2)/Gamma(self.df/2)*np.sqrt(self.df-2)/(self.df-1)

        for j in range(1, t+1):
            s2_prev = s2_j
            Z_prev = self.data[j-1]/np.sqrt(s2_prev)
            log_s2_j = omega + alpha*(abs(Z_prev)-self.E_absZ) + gamma*Z_prev + beta*np.log(s2_prev)
            s2_j = np.exp(log_s2_j)
            assert not np.isnan(s2_j).any(), "NaNs in volatilities"
            assert all(s2_j >= 0), "Volatility is negative for some particles"

        return s2_j

    def reservoir(self, t, s2_prev):
        M = np.stack((np.array([self.data[t-1]]*len(s2_prev)), s2_prev), axis=0)
        h1 = np.matmul(self.W1, M) + self.b1  # hidden nodes 1
        h1 = self.activ(h1)

        res = np.matmul(self.W2, h1) + self.b2  # hidden nodes 2
        res = self.activ(res)

        return res

    def vol_res(self, theta, t):
        '''
        Neural volatility
            σ^2_t = W·Res(X_{t-1}, σ^2_{t-1})
        '''
        s2_j = np.array([1]*len(theta))
        for j in range(1, t+1):
            s2_prev = s2_j
            res = self.reservoir(t, s2_prev)
            s2_j = theta['w1']*res[0, :] + theta['w2']*res[1, :] + theta['w3']*res[2, :]
            assert not np.isnan(s2_j).any(), "NaNs in volatilities"
            assert all(s2_j >= 0), "Volatility is negative for some particles"

        return s2_j

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current observation
        x_t given observations x_{1:t-1} and model parameters θ^(i)
        '''
        # compute # σ_t^(1:N)
        if self.variant == 'standard':
            s_t = np.sqrt(self.vol_std(theta, t))
        elif self.variant == 'gjr':
            s_t = np.sqrt(self.vol_gjr(theta, t))
        elif self.variant == 'exp':
            s_t = np.sqrt(self.vol_exp(theta, t))
        else:
            s_t = np.sqrt(self.vol_res(theta, t))

        # compute log-likelihood given σ_t and distribution
        if self.innov == 'norm':
            loglik = -np.log(s_t) - 1/2*np.log(2*np.pi) - 1/(2*s_t**2)*self.data[t]**2
        else:
            loglik = scipy.stats.t.logpdf(self.data[t], loc=0, scale=s_t)

        return loglik

    def summary(self):
        return None


