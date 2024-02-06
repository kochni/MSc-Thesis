#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

# packages
from particles.smc_samplers import StaticModel
from particles.distributions import F_innov
from particles.hmm import HMM

# models
from white_noise import *
from garch import *
from guyon import *
from rescomp_dv import *


class DetVol(StaticModel):
    '''
    Any model for which either
        - the conditional volatility (and hence likelihood) is known given the
          past observations and the model parameters (e.g. GARCH, ELM)
        - the expected likelihood (over any latent variables) can be computed
          analytically (e.g., models w regime switching or jumps in returns)

    This allows for using the IBIS algorithm.

    '''
    def __init__(self, spec, prior, prices):
        self.S = prices  # raw prices
        X = 100 * np.log(prices[1:]/prices[:-1])  # log-returns
        super().__init__(data=X, prior=prior)

        # define model
        dynamics = spec['dynamics']
        variant = spec.get('variant')
        hyper = spec.get('hyper')
        self.innov_X = spec['innov_X']
        self.theta_FX = {  # parameters of innovation distribution
            'tail': None,
            'shape': None,
            'skew': None
            }
        self.switching = spec.get('switching')
        self.K = K = spec.get('regimes') if self.switching is not None else 1

        err_msg = "Specified multi-regime model but not switching type"
        if self.K > 1: assert self.switching is not None, err_msg

        err_msg = "Can't have jumps in volatility in DetVol, use StochVol"
        assert spec.get('jumps') not in ['V', 'vol', 'volatility'], err_msg
        self.jumps = spec.get('jumps')

        if dynamics == 'constant':
            self.model = WhiteNoise(K)
        elif dynamics == 'garch':
            self.model = GARCH(variant, K)
        elif dynamics == 'elm':
            self.model = ResComp('elm', hyper, K)
        elif dynamics == 'esn':
            self.model = ResComp('esn', hyper, K)
        elif dynamics == 'sig':
            self.model = ResComp(variant, hyper, K)
        elif dynamics == 'guyon':
            self.model = Guyon(K)

    def vols(self, theta, X, t):
        N = len(theta)
        K = self.K

        # Volatilities w/o jump:
        s_t = self.model.vol(theta, X, t)  # (N,K)

        # add volatilities w/ jump:
        if self.jumps is not None:
            lambda_X = np.clip(theta['lambda_X'], 0., 1.)
            phi_X = np.clip(theta['phi_X'], 0., None)
            phi_X = phi_X.reshape(N, 1)  # (N,K=1)
            sd_jump = np.sqrt(s_t**2 + phi_X**2)  # (N,K)
            s_t = np.stack([s_t, sd_jump], axis=2)  # (N,K,2)
        else:
            s_t = s_t.reshape(N, K, 1)  # (N,K,1)
        assert not np.isnan(s_t).any(), "NaNs in volatilities"

        s_t = np.clip(s_t, 1e-20, 1e20)  # (N,K,J)

        return s_t

    def probs(self, theta, X, t):
        '''
        compute probabilities of regimes given filtration,
            P(r_t=j | X_{0:t-1})

        Output: p_t, (N,K,J) array
            where p_t[i, k, j] probability of volatility from i-th particle,
            k-th regime, and conditional on jump (j=1) resp. no jump (j=0)

        '''
        N = len(theta)
        K = self.K
        X = self.data[0:t]  # filtration X_0, ..., X_{t-1}

        # single-regime models
        if K == 1:
            p_t = np.full([N, K], 1.)

        # Mixture Models:
        elif self.switching in ['mix', 'mixing', 'mixture']:
            p_t = np.full([N, K], 0.)
            ps = theta['p_0'] if K == 2 else theta['p']
            ps = ps.reshape(-1, K-1)
            p_t[:, 0:-1] = ps
            p_t[:, -1] = 1 - np.sum(ps, axis=1)

        # Markov-Switching Models:
        else:
            p_t = np.full([N, K], 0.)
            for i in range(N):  # (!) parallelizable ??
                df = theta['df'][i] if 'df' in theta.dtype.names else None
                P = np.full([K, K], 0.)  # transition matrix
                for k in range(K):
                    P[k, 0:-1] = np.clip(theta['P_' + str(k)][i], 0., 1.)
                P[:, -1] = 1 - np.sum(P, axis=1)

                # HMM with given parameters
                hmm = HybridHMM(model=self.model, theta=theta[[i]],
                                X=X, t=t, trans_mat=P, df=df)

                # compute probabilities of regimes given filtration
                if t > 0:
                    bw = BaumWelch(hmm=hmm, data=self.data[0:t])
                    bw.forward()
                    # bw.backward()
                    p_t[i, :] = bw.pred[-1]
                else:
                    p_t[i, :] = 1./K

        # add jumps
        if self.jumps is not None:
            lambda_X = np.clip(theta['lambda_X'], 0., 1.)
            probs_nj = np.einsum('N,NK->NK', 1.0-lambda_X, p_t)
            probs_j = np.einsum('N,NK->NK', lambda_X, p_t)
            p_t = np.stack([probs_nj, probs_j], axis=2)  # (N,K,2)
        else:
            p_t = p_t.reshape(N, K, 1)  # (N,K,1)
        assert not np.isnan(p_t).any(), "NaNs in regime probabilities"

        return p_t

    def logpyt(self, theta, t):
        '''
        (expected) likelihood of parameters

        '''
        N = len(theta)
        K = self.K
        innov_X = self.innov_X
        theta_FX = self.theta_FX
        X = self.data[0:t]  # filtration at time t, X_0, ..., X_{t-1}

        # Volatilities w/o jump:
        self.s_t = s_t = self.vols(theta, X, t)  # (N,K,J)

        # Regime probabilities:
        self.p_t = p_t = self.probs(theta, X, t)  # (N,K,J)

        # get parameters & cap/transform:
        theta_FX = self.theta_FX
        if self.innov_X == 't':
            df_X = theta['df_X']
            df_X = np.clip(df_X, 2.+1e-10, None)
            theta_FX['df'] = df_X.reshape(N, 1, 1)  # (N,1,1)
        elif self.innov_X == 'GH':
            tail_X = theta['tail_X']
            tail_X = np.clip(tail_X, -50., 50.)
            theta_FX['tail'] = tail_X.reshape(N, 1, 1)
            skew_X = theta['skew_X']
            skew_X = np.clip(skew_X, -50., 50.)
            theta_FX['skew'] = skew_X.reshape(N, 1, 1)  # (N,1,1)
            shape_X = theta['shape_X']
            shape_X = np.clip(shape_X, abs(skew_X)+1e-3, 50.+1e-3)
            theta_FX['shape'] = shape_X.reshape(N, 1, 1)  # (N,1,1)

        # Expected likelihood over regimes & jumps:
        liks = F_innov(0., s_t, **theta_FX).pdf(self.data[t])
        E_lik = np.einsum('NKJ,NKJ->N', p_t, liks)
        E_lik = np.clip(E_lik, 1e-100, None)  # avoids error when taking log
        log_E_lik = np.log(E_lik)
        assert not np.isnan(log_E_lik).any(), "NaNs in log-likelihoods"

        return log_E_lik


class HybridHMM(HMM):
    '''
    Student t HMM which automatically becomes Gaussian (if df=None) or Cauchy
    (if df=1) HMM

    Used in Markov-switching models to compute regime probabilities given
    parameters
    '''

    def PY(self, t, xp, x):
        s_t = self.model.vol(theta=self.theta, X=self.X, t=self.t).flatten()
        s_t = np.clip(s_t, 1e-50, 1e50)
        return F_innov(0., s_t[x], **self.theta_FX)
