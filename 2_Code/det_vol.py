#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

# SMC packages
from particles.smc_samplers import StaticModel
from particles.distributions import Mixture, Normal, Student
from particles.hmm import HMM, BaumWelch

# signatures
from esig import stream2sig, stream2logsig

# plotting
import matplotlib.pyplot as plt

# other utilts
from statsmodels.stats.weightstats import DescrStatsW
from numpy.lib.recfunctions import structured_to_unstructured
from operator import itemgetter
# import helpers


class DetVol(StaticModel):
    '''
    Any model for which either
        - the conditional volatility (and hence likelihood) is known given the
          past observations and the model parameters (e.g. GARCH, ELM)
        - the expected likelihood (over any latent variables) can be computed
          analytically (e.g., models w regime switching or jumps in returns)

    This allows for using the IBIS algorithm.

    '''
    def __init__(self, spec, prior, prices, **pred_opts):
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

        # predictions, prediction sets, coverage, & prediction errors
        T = len(X)
        self.pred_opts = pred_opts  # prediction options
        self.predictions = {}  # 1-day ahead predictions
        self.predsets = {}  # 1-day ahead prediction sets

        if dynamics == 'constant':
            self.model = WhiteNoise(K)
        elif dynamics == 'garch':
            if variant == 'elm':
                self.model = ResComp('elm', hyper, K)
            else:
                self.model = GARCH(variant, K)
        elif dynamics == 'sig':
            self.model = ResComp(variant, hyper, K)
        elif dynamics == 'guyon':
            self.model = Guyon(K)

    def probs(self, theta, X, t):
        '''
        compute probabilities of regimes given filtration,
            P(r_t=j | X_{0:t-1})

        Output: p_t, (N,K) array
            where p_t[i, j] probability of j-th regime at time t given
            parameters from i-th particle

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

        return p_t

    def predict(self, s_t, p_t, theta, t):
        '''
        produce 1-day ahead point predictions and prediction sets

        '''
        # frequently used variables:
        N = len(theta)
        K = self.K
        W = np.full(N, 1./N)  # (!) replace by actualy weights!

        # all variables put in shape (N,K,J) such that x[i,k,j] is value
        # associated with i-th particle, k-th regime, and no jump/jump

        # innovations:
        theta_FX = self.theta_FX
        if self.innov_X == 't':
            df_X = theta['df_X']
            df_X = np.clip(df_X, 2.+1e-10, None)
            theta_FX['tail'] = df_X.reshape(N, 1, 1)  # (N,1,1)
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

        # other
        alpha = self.pred_opts['alpha']
        strike = self.S[t] if self.pred_opts['strike'] == 'last' else self.pred_opts['strike']
        probs = p_t * W.reshape(N, 1, 1)  # (N,K,J=1)

        # simulate 1-day ahead returns (used for pred's & predsets)s
        X_sim = F_innov(0., s_t, **theta_FX).rvs([100, N, K, s_t.shape[2]])

        # Point Predictions:
        # (1) Volatility
        var_pred = np.einsum('NKJ,NKJ->N', p_t, s_t**2)
        var_pred = np.clip(var_pred, 0., None)
        vol_pred = np.sqrt(var_pred)
        vol_pred = np.einsum('N,N->...', vol_pred, W)

        # (2) Sq. returns
        X2_pred = np.einsum('NKJ,NKJ,N->...', p_t, s_t**2, W)

        # (3) Price = previous price

        # (4) Option Payouts
        pay_pred = np.mean(np.maximum(X_sim - strike, 0.))

        # Prediction Sets:
        # (1) Volatility
        dsw = DescrStatsW(s_t.flatten(), probs.flatten())
        vol_predset = dsw.quantile([0.5*alpha, 1.-0.5*alpha],
                                   return_pandas=False)

        # GH CDF too intensive to evaluate -> simulate + empirical quantiles
        if self.innov_X != 'GH':
            # (2) Sq. Returns
            X2_predset = inv_cdf(cdf=lambda x: sq_mix_cdf(x, probs, 0., s_t, **theta_FX),
                                 p=[0.5*alpha, 1.-0.5*alpha], lo=0., hi=100.)

            # (3) Price
            X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., s_t, **theta_FX),
                                p=[0.5*alpha, 1.-0.5*alpha], lo=-50., hi=50.)
            S_predset = np.exp(0.01*X_predset) * self.S[t]

            # (4) Option Payout
            X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., s_t, **theta_FX),
                                  p=[1.-alpha], lo=0., hi=50.)
            pay_predset = np.exp(0.01*X_predset) * self.S[t] - strike

        else:
            X_sim = X_sim.flatten()

            # (2) Sq. Returns
            X2_predset = np.quantile(X_sim**2, [0.5*alpha, 1.-0.5*alpha])

            # (3) Price
            X_predset = np.quantile(X_sim, [0.5*alpha, 1.-0.5*alpha])
            S_predset = np.exp(0.01*X_predset) * self.S[t]

            # (4) Option Payouts
            X_predset = np.quantile(X_sim, [1.-alpha])
            pay_predset = np.exp(0.01*X_predset) * self.S[t] - strike

        if t == 0:
            self.predictions['Vol'] = vol_pred
            self.predsets['Vol'] = vol_predset
            self.predictions['X2'] = X2_pred
            self.predsets['X2'] = X2_predset
            self.predsets['S'] = np.array([[np.nan, np.nan], S_predset])
            self.predictions['Pay'] = pay_pred
            self.predsets['Pay'] = pay_predset
        else:
            self.predictions['Vol'] = np.append(self.predictions['Vol'], vol_pred)
            self.predsets['Vol'] = np.vstack([self.predsets['Vol'], vol_predset])
            self.predictions['X2'] = np.append(self.predictions['X2'], X2_pred)
            self.predsets['X2'] = np.vstack([self.predsets['X2'], X2_predset])
            self.predsets['S'] = np.vstack([self.predsets['S'], S_predset])
            self.predictions['Pay'] = np.append(self.predictions['Pay'], pay_pred)
            self.predsets['Pay'] = np.append(self.predsets['Pay'], pay_predset)

    def logpyt(self, theta, t):
        '''
        (expected) likelihood of parameters

        '''
        N = len(theta)
        K = self.K
        theta_FX = self.theta_FX
        X = self.data[0:t]  # filtration at time t, X_0, ..., X_{t-1}

        # Volatilities w/o jump:
        s_t = self.model.vol(theta, X, t)  # (N,K)
        s_t = np.clip(s_t, 1e-20, 1e20)
        # Volatilities w/ jump:
        if self.jumps is not None:
            lambd_X = np.clip(theta['lambda_X'], 0., 1.)
            phi_X = np.clip(theta['phi_X'], 0., None)
            phi_X = phi_X.reshape(N, 1)  # (N,K=1)
            sd_jump = np.sqrt(s_t**2 + phi_X**2)  # (N,K)
            s_t = np.stack([s_t, sd_jump], axis=2)  # (N,K,2)
        else:
            s_t = s_t.reshape(N, K, 1)  # (N,K,1)
        assert not np.isnan(s_t).any(), "NaNs in volatilities"

        # Regime probabilities:
        p_t = self.probs(theta, X, t)  # (N,K)
        p_t = np.clip(p_t, 1e-50, 1.)
        if self.jumps is not None:
            probs_nj = np.einsum('N,NK->NK', 1.0-lambd_X, p_t)
            probs_j = np.einsum('N,NK->NK', lambd_X, p_t)
            p_t = np.stack([probs_nj, probs_j], axis=2)  # (N,K,2)
        else:
            p_t = p_t.reshape(N, K, 1)  # (N,K,1)
        assert not np.isnan(p_t).any(), "NaNs in regime probabilities"

        # get parameters & cap/transform:
        theta_FX = self.theta_FX
        if self.innov_X == 't':
            df_X = theta['df_X']
            df_X = np.clip(df_X, 2.+1e-10, None)
            theta_FX['tail'] = df_X.reshape(N, 1, 1)  # (N,1,1)
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

        # produce point predictions & prediction sets:
        self.predict(s_t, p_t, theta, t)

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
