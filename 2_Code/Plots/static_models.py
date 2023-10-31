#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

# SMC packages
from particles.smc_samplers import StaticModel

# other utilts
from numpy.lib.recfunctions import structured_to_unstructured
from helpers import *

# documentation:
# https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/_autosummary/particles.smc_samplers.html#module-particles.smc_samplers

# source code:
# https://github.com/nchopin/particles/blob/master/particles/smc_samplers.py


class WhiteNoise(StaticModel):
    '''
    White noise process, i.e. process of the form
        X_t = σ·Z_t
    '''

    def __init__(self, innov, prior, data):
        super().__init__(data, prior)

        assert innov in ['N', 't'], "Innovation distribution not implemented"
        self.innov = innov

        self.s2_0 = np.var(data)  # initial volatility

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current
        observation x_t given x_{1:t-1} and model parameters θ^(i)
        '''
        sigma = theta['sigma']
        s2_t = np.maximum(1e-10, sigma**2)

        assert not np.isnan(s2_t).any(), "NaNs in volatilities"

        # compute log-likelihood given σ_t and distribution
        if self.innov == 'N':
            return (-0.5*np.log(s2_t) - 1/2*np.log(2*np.pi) -
                    1/2*self.data[t]**2/s2_t)
        else:
            df = theta['df']
            return stats.t.logpdf(self.data[t], loc=0, scale=np.sqrt(s2_t),
                                  df=df)


class GARCH(StaticModel):
    '''
    GARCH(1,1) model, i.e. process of the form
        X_t = σ_t·Z_t,
        σ_t = f(X_{t-1}, σ_{t-1}^2)
    where f(.) depends on the GARCH variant

    Parameters:
    -----------
    variant: string
        GARCH variant; one of 'basic', 'gjr' (for GJR-GARCH), 'thr' (for
        T-GARCH), 'exp' (for E-GARCH).

    innov: string
        Innovation distribution; must be one of 'N' (for Gaussian) and 't'
        (for Student t).

    prior: StructDist (from particles.distributions)
        Prior distributions on the model parameters.

    data: (T,)-array
        Observations of raw price process.
    '''

    def __init__(self, variant, innov, prior, data, hyper=None):
        super().__init__(data, prior)

        assert variant in ['basic', 'gjr', 'thr', 'exp', 'mix', 'guyon'], (
            "GARCH variant not implemented")
        self.variant = variant

        assert innov in ['N', 't'], ("Specified innovation distribution not "
                                     "implemented")
        self.innov = innov
        self.hyper = hyper

    def vol_std(self, theta, t):
        '''
        Standard GARCH volatility
            σ^2_t = ω + α·X_{t-1}^2 + β·σ_{t-1}^2
                  = ω·Σ_j(β^{j-1}) + α·Σ_j(β^{j-1}·X_{t-j}^2) + β^{t-1}·σ_1^2
        '''
        omega = theta['omega']
        alpha = theta['alpha']
        beta = theta['beta']

        t_grid = np.arange(0, t, 1).reshape(-1, 1)
        X = self.data.reshape(-1, 1)
        s2_t = (omega * np.sum(beta**t_grid, axis=0) +
                alpha*np.sum(beta**t_grid * X[0:t][::-1]**2, axis=0) +
                beta**t * self.s2_0)
        s2_t = np.maximum(s2_t, 1e-10)

        return s2_t

    def vol_gjr(self, theta, t):
        '''
        GJR-GARCH volatility
            σ^2_t = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2
        '''
        omega = theta['omega']
        alpha = theta['alpha']
        beta = theta['beta']
        gamma = theta['gamma']

        s2_j = np.tile(self.s2_0, len(theta))
        for j in range(1, t+1):
            s2_prev = s2_j
            s2_j = (omega + (alpha + gamma*(self.data[j-1]>0))*self.data[j-1]**2
                    + beta*s2_prev)

        s2_j = np.maximum(s2_j, 1e-10)

        return s2_j

    def vol_thr(self, theta, t):

        omega = theta['omega']
        alpha = theta['alpha']
        beta = theta['beta']
        gamma = theta['gamma']

        s_j = np.tile(np.sqrt(self.s2_0), len(theta))  # initial volatility
        for j in range(1, t+1):
            s_prev = s_j
            Z_prev = self.data[j-1] / s_prev

            s_j = omega + alpha*abs(Z_prev) + gamma*Z_prev + beta*s_prev

        s2_j = np.maximum(s_j**2, 1e-10)

        return s2_j

    def vol_exp(self, theta, t):
        '''
        Exponential GARCH volatility
            log(σ^2_t) = ω + α·(|Z_{t-1}| + γ·Z_{t-1}) + β·log(σ^2_{t-1})
        '''
        omega = theta['omega']
        alpha = theta['alpha']
        beta = theta['beta']
        gamma = theta['gamma']

        log_s2_j = np.tile(np.log(self.s2_0), len(theta))
        for j in range(1, t+1):
            log_s2_prev = log_s2_j
            Z_prev = self.data[j-1]/np.exp(log_s2_prev/2)
            log_s2_j = omega + alpha*(abs(Z_prev) + gamma*Z_prev) + beta*log_s2_prev
            log_s2_j = np.minimum(100, np.maximum(-100, log_s2_j))

        s2_j = np.exp(log_s2_j)

        return s2_j

    def vol_mix(self, theta, t):
        '''
        Mixture GARCH volatility,
        '''
        K = self.hyper['K']
        P = np.full([len(theta), K], np.nan)
        for k in range(K-1):
            globals()['p_' + str(k)] = P[:, k] =  theta['p_' + str(k)]
        P[:, -1] = 1 - np.sum(P[:, 0:-1], axis=1)

        omegas = np.full([len(theta), K], np.nan)
        alphas = np.full([len(theta), K], np.nan)
        betas = np.full([len(theta), K], np.nan)
        for k in range(K):
            globals()['omega_' + str(k)] = omegas[:, k] = theta['omega_' + str(k)]
            globals()['alpha_' + str(k)] = alphas[:, k] = theta['alpha_' + str(k)]
            globals()['beta_' + str(k)] = betas[:, k] = theta['beta_' + str(k)]

        E_omega = np.sum(P*omegas, axis=1)
        E_alpha = np.sum(P*alphas, axis=1)
        E_beta = np.sum(P*betas, axis=1)

        t_grid = np.arange(0, t, 1).reshape(-1, 1)
        X = self.data.reshape(-1, 1)

        E_s2_t = (E_omega * np.sum(E_beta**t_grid, axis=0) +
                  E_alpha*np.sum(E_beta**t_grid * X[0:t][::-1]**2, axis=0) +
                  E_beta**t)
        s2_t = np.maximum(E_s2_t, 1e-10)

        return s2_t

    def vol_guyon(self, theta, t):
        '''
        Guyon and Lekeufack's (2023) path-dependent volatility model
        '''
        beta0 = theta['beta0']    # intercept
        beta1 = theta['beta1']    # trend coefficient
        beta2 = theta['beta2']    # volatility coefficient
        alpha1 = theta['alpha1']  # term in trend kernel
        alpha2 = theta['alpha2']  #
        delta1 = theta['delta1']  # term in trend kernel
        delta2 = theta['delta2']  #

        # if constraints of alpha, delta violated, might cause errors, hence
        # set to arbitrary value instead (will not be accepted anyways)
        alpha1 = np.maximum(alpha1, 1 + 1e-10)
        alpha2 = np.maximum(alpha2, 1 + 1e-10)
        delta1 = np.maximum(delta1, 1e-10)
        delta2 = np.maximum(delta2, 1e-10)

        if t == 0:
            s2_t = np.tile(self.s2_0, len(theta))
        else:
            tp_grid = np.arange(0, t, 1)  # grid of all previous time stamps
            # trend kernel weights:
            k1 = tspl(t=t, tp=tp_grid, alpha=alpha1, delta=delta1)
            # volatility kernel weights:
            k2 = tspl(t=t, tp=tp_grid, alpha=alpha2, delta=delta2)
            trend = np.sum(k1 * np.transpose(self.data[tp_grid]), axis=1)
            volat = np.sqrt(np.sum(k2 * np.transpose(self.data[t-tp_grid]**2),
                                   axis=1))

            s_t = beta0 + beta1*trend + beta2*volat
            s2_t = np.maximum(s_t**2, 1e-10)

        return s2_t

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current
        observation x_t given observations x_{1:t-1} and model parameters θ^(i)
        '''
        # get # σ^2_t^(1:N)
        if self.variant == 'basic':
            s2_t = self.vol_std(theta, t)
        elif self.variant == 'gjr':
            s2_t = self.vol_gjr(theta, t)
        elif self.variant == 'thr':
            s2_t = self.vol_thr(theta, t)
        elif self.variant == 'exp':
            s2_t = self.vol_exp(theta, t)
        elif self.variant == 'mix':
            s2_t = self.vol_mix(theta, t)
        elif self.variant == 'guyon':
            s2_t = self.vol_guyon(theta, t)

        assert not np.isnan(s2_t).any(), "NaN volatility for some particles"

        # compute log-likelihood given σ_t and distribution
        if self.innov == 'N':
            return (-0.5*np.log(s2_t) - 0.5*np.log(2*np.pi) -
                    0.5/s2_t*self.data[t]**2)
        else:
            df = theta['df']
            return stats.t.logpdf(self.data[t], loc=0,
                                  scale=np.sqrt(s2_t), df=df)


class RCStatic(StaticModel):
    '''
    Reservoir computers for static volatility models

    Parameters:
    -----------

    rc_type: string
        type of the RC approach use; one of 'elm' (for extreme learning
        machine, ELM), 't_sig' (for truncated signature), and 'r_sig'
        (for randomized signature); ELM is Markovian, while signature-based methods
        model path-dependence.

    q: int
        dimensionality of the reservoir.

    hyper: dict
        dictionary of hyperparameters; must include
            'q' (int): dimensionality of the reservoir.
            'sd' (float): standard deviation of the random initializations of
                the weights resp. random signature components.
            'activ' (func): activation function used in the random neural
                network resp. the generation of the randomized signature.

    innov: string
        innovation distribution; one of 'N' (for Gaussian) and 't' (for
         Student t)

    prior: StructDist
        prior distribution on parameters.

    data: (T,)-array
        data.
    '''

    def __init__(self, rc_type, hyper, innov, prior, data):
        super().__init__(data, prior)

        assert innov in ['N', 't'], "Innovation distribution not implemented"
        self.innov = innov

        self.hyper = hyper

        assert rc_type in ['elm', 't_sig', 'r_sig'], "RC type not implemented"
        self.rc_type = rc_type

        if rc_type == 'elm':  # randomly draw inner parameters of NN
            self.W1 = np.random.normal(loc=0, scale=hyper['sd'], size=(100, 2))
            self.W2 = np.random.normal(loc=0, scale=hyper['sd'],
                                       size=(hyper['q'], 100))
            self.b1 = np.random.normal(loc=0, scale=hyper['sd'], size=1)
            self.b2 = np.random.normal(loc=0, scale=hyper['sd'], size=1)

        elif rc_type == 'r_sig':
            # draw random components of randomized signature
            self.rsig_0 = np.random.normal(loc=0, scale=hyper['sd'],
                                           size=hyper['q'])
            self.A1 = np.random.normal(loc=0, scale=hyper['sd'],
                                       size=(hyper['q'], hyper['q']))
            self.b1 = np.random.normal(loc=0, scale=hyper['sd'],
                                       size=hyper['q'])
            self.A2 = np.random.normal(loc=0, scale=hyper['sd'],
                                       size=(hyper['q'], hyper['q']))
            self.b2 = np.random.normal(loc=0, scale=hyper['sd'],
                                       size=hyper['q'])

    def elm_vol(self, t, theta):
        '''
        compute volatility via a random 2-layer neural network, a.k.a. "extreme
        learning machine (ELM)"
        '''
        theta = structured_to_unstructured(theta)
        theta = np.minimum(np.exp(50), np.maximum(-np.exp(50), theta))

        if self.innov == 'N':
            w = theta[:, 1:]  # weights
            w0 = theta[:, 0]  # bias term
        else:
            w = theta[:, 1:-1]  # last term is df
            w0 = theta[:, 0]

        s2_0 = np.var(self.data)  # initial volatility
        log_s2_j = np.tile(s2_0, len(theta))
        for j in range(1, t+1):
            log_s2_prev = log_s2_j

            # compute reservoir from previous log-return and volatility
            M = np.stack((np.array([self.data[t-1]]*len(theta)), log_s2_prev),
                         axis=0)
            h1 = np.matmul(self.W1, M) + self.b1  # hidden nodes 1
            h1 = self.hyper['activ'](h1)
            h2 = np.matmul(self.W2, h1) + self.b2  # hidden nodes 2
            res = self.hyper['activ'](h2)  # final reservoir
            res = np.transpose(res)

            # get volatility from reservoir & readout
            log_s2_j = np.sum(w*res, axis=1) + w0
            log_s2_j = np.minimum(100, np.maximum(-100, log_s2_j))

        return log_s2_j

    def sig(self, t, theta):
        '''
        compute volatility from the truncated signature of the time-extended
        path of the log-returns, (t, X_t)
        '''
        pass

    def rsig_vol(self, t, theta):
        '''
        compute volatility from the randomized signature of the time-extended
        path of the log-returns, (t, X_t)
        '''
        # rSig_0 is random and identical for all particles
        rsig = np.tile(self.rsig_0.reshape(-1, 1), (1, len(theta)))
        b1 = np.tile(self.b1.reshape(-1, 1), (1, len(theta)))
        b2 = np.tile(self.b2.reshape(-1, 1), (1, len(theta)))
        theta = structured_to_unstructured(theta)
        theta = np.minimum(np.exp(50), np.maximum(-np.exp(50), theta))

        if self.innov == 'N':
            w = theta[:, 0:-1]  # weights
            w0 = theta[:, -1]   # bias term
        else:
            w = theta[:, 0:-2]  # last term is df
            w0 = theta[:, -2]

        s2_0 = np.var(self.data)  # initial volatility
        log_s2_j = np.tile(s2_0, len(theta))
        for j in range(1, t+1):
            log_s2_prev = log_s2_j

            # update rSig
            Z_prev = self.data[j-1] / np.exp(log_s2_prev/2)
            incr_1 = np.matmul(self.A1, rsig) + b1
            incr_1 = self.hyper['activ'](incr_1)
            incr_2 = np.matmul(self.A2, rsig) + b2
            incr_2 = self.hyper['activ'](incr_2) * Z_prev
            rsig += incr_1 + incr_2

            # get volatility from rSig and readout
            log_s2_j = np.sum(w*np.transpose(rsig), axis=1) + w0
            log_s2_j = np.minimum(100, np.maximum(-100, log_s2_j))

        return log_s2_j

    def logpyt(self, theta, t):
        '''
        for each θ-particle θ^(i), compute log-likelihood of current
        observation x_t given observations x_{1:t-1} and model parameters θ^(i)
        '''
        if self.rc_type == 'elm':
            log_s2_t = self.elm_vol(t, theta)
        elif self.rc_type == 'sig':
            log_s2_t = self.sig_vol(t, theta)
        elif self.rc_type == 'r_sig':
            log_s2_t = self.rsig_vol(t, theta)

        assert not np.isnan(log_s2_t).any(), "NaNs in volatilities"
        s_t = np.exp(log_s2_t/2)

        # compute log-likelihood given σ and distribution
        if self.innov == 'N':
            return -np.log(s_t) - 1/2*np.log(2*np.pi) - 1/(2*s_t**2)*self.data[t]**2
        else:
            df = structured_to_unstructured(theta)[:, -1]
            return stats.t.logpdf(self.data[t], loc=0, scale=s_t, df=df)
