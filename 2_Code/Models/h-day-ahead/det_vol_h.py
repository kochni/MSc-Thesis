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
        self.switching = spec.get('switching')
        self.K = K = spec.get('regimes') if self.switching is not None else 1

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

        self.jumping = spec.get('jumps')
        self.innov_X = spec.get('innov_X')

        err_msg = "Specified multi-regime model but not switching type"
        if self.K > 1: assert self.switching is not None, err_msg

        # predictions, prediction sets, coverage, & prediction errors
        T = len(X)
        self.pred_opts = pred_opts  # prediction options
        self.predictions = {}  # 1-day ahead predictions
        self.predsets = {}  # 1-day ahead prediction sets

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

        shape = [N, K]

        # single-regime models
        if K == 1:
            p_t = np.full(shape, 1.)

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
                    P[k, 0:-1] = cap(theta['P_' + str(k)][i], 0., 1.)
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
        # innovations:
        if 'df_X' in theta.dtype.names:
            df_X = theta['df_X']  # (N,)
            df_X = cap(df_X, 2.+1e-10)
            df_X = np.tile(df_X[:, np.newaxis], [1, K])  # (N,K)
            df_X = df_X.flatten()
        else:
            df_X = None

        # Volatilities:
        sigmas = s_t  # (N,K); std w/o jump
        # Probabilities:
        probs = p_t * W[:, np.newaxis]  # (N,K); probabilities w/0 jump
        # jumps:
        if 'phi' in theta.dtype.names:
            lambd = theta['lambda']
            phi = theta['phi']
            sigmas_j = np.sqrt(s_t**2 + phi[:, np.newaxis]**2)  # sd w/ jump
            sigmas = np.stack([sigmas, sigmas_j], axis=2)  # (N,K,2)
            probs_nj = probs * (1.-lambd[:, np.newaxis])
            probs_j = probs * lambd[:, np.newaxis]
            probs = np.stack([probs_nj, probs_j], axis=2)  # (N,K,2)
            if 'df_X' in theta.dtype.names:
                df_X = np.tile(df_X[:, :, np.newaxis], [1, 1, 2])
                df_X = df_X.flatten()

        sigmas = sigmas.flatten()
        probs = probs.flatten()

        # other
        alpha = self.pred_opts['alpha']
        strike = self.S[t] if self.pred_opts['strike'] == 'last' else self.pred_opts['strike']

        # Point Predictions & Prediction Sets:
        # (1) Volatility
        var_pred = np.einsum('NK,NK->N', p_t, s_t**2)
        if 'phi' in theta.dtype.names:
            var_pred += np.einsum('N,N->N', lambd, phi**2)
        var_pred = np.clip(var_pred, 0., None)
        vol_pred = np.sqrt(var_pred)
        vol_pred = np.einsum('N,N->...', vol_pred, W)

        dsw = DescrStatsW(sigmas, probs)
        vol_predset = dsw.quantile([0.5*alpha, 1.-0.5*alpha],
                                   return_pandas=False)

        # (2) Squared returns
        X2_pred = np.einsum('NK,NK,N->...', p_t, s_t**2, W)
        if 'phi' in theta.dtype.names:
            X2_pred += np.einsum('N,N,N->...', lambd, phi**2, W)

        X2_predset = inv_cdf(cdf=lambda x: sq_mix_cdf(x, probs, 0., sigmas, df_X),
                             p=[0.5*alpha, 1.-0.5*alpha], lo=0., hi=100.)

        # (3) Price
        X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., sigmas, df_X),
                            p=[0.5*alpha, 1.-0.5*alpha], lo=-50., hi=50.)
        S_predset = np.exp(0.01*X_predset) * self.S[t]

        # (4) option payout
        X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., sigmas, df_X),
                              p=[1.-alpha], lo=0., hi=50.)
        pay_predset = np.exp(0.01*X_predset) * self.S[t] - strike

        if t == 0:
            self.predictions['Vol'] = vol_pred
            self.predsets['Vol'] = vol_predset
            self.predictions['X2'] = X2_pred
            self.predsets['X2'] = X2_predset
            self.predsets['S'] = np.array([[np.nan, np.nan], S_predset])
            self.predsets['Pay'] = np.array([np.nan, pay_predset[0]])
        else:
            self.predictions['Vol'] = np.append(self.predictions['Vol'], vol_pred)
            self.predsets['Vol'] = np.vstack([self.predsets['Vol'], vol_predset])
            self.predictions['X2'] = np.append(self.predictions['X2'], X2_pred)
            self.predsets['X2'] = np.vstack([self.predsets['X2'], X2_predset])
            self.predsets['S'] = np.vstack([self.predsets['S'], S_predset])
            self.predsets['Pay'] = np.append(self.predsets['Pay'], pay_predset)


    def logpyt(self, theta, t):
        '''
        (expected) likelihood of parameters

        '''
        X = self.data[0:t]  # filtration at time t, X_0, ..., X_{t-1}

        # Volatilities:
        s_t = self.model.vol(theta, X, t)  # (N,K) array
        s_t = cap(s_t, 1e-100, 1e100)
        assert not np.isnan(s_t).any(), "NaNs in volatilities"

        # Regime probabilities:
        p_t = self.probs(theta, X, t)  # (N,K) array
        p_t = cap(p_t, 1e-50, 1.)
        assert not np.isnan(p_t).any(), "NaNs in regime probabilities"

        # get parameters & cap/transform:
        if 'df_X' in theta.dtype.names:
            df_X = theta['df_X'].reshape(-1, 1)
            df_X = cap(df_X, 2.+1e-10)
        else:
            df_X = None

        # Make predictions, prediction sets, check coverages:
        self.predict(s_t, p_t, theta, t)

        # Expected likelihood over regimes:
        liks = F_innov(sd=s_t, df=df_X).pdf(self.data[t])
        E_lik = np.einsum('NK,NK->N', p_t, liks)  # Hadamard & row-sum
        E_lik = E_lik.flatten()

        # Expected likelihoods conditional on occurrence of jump:
        if 'lambda_X' in theta.dtype.names:
            lambd_X = cap(theta['lambda_X'], 0., 1.)
            phi_X = cap(theta['phi_X'], 0., None)
            phi_X = np.tile(phi_X.reshape(-1, 1), [1, self.K])

            sd_jump = np.sqrt(s_t**2 + phi_X**2)
            liks_jump = F_innov(sd=sd_jump, df=df_X).pdf(self.data[t])
            E_lik_jump = np.einsum('NK,NK->N', p_t, liks_jump)  # Hadamard & row-sum
            E_lik_jump = E_lik_jump.flatten()

            # overall expected likelihood
            E_lik = (1.-lambd_X)*E_lik + lambd_X*E_lik_jump

        E_lik = cap(E_lik, 1e-100, None)  # avoids error when taking log
        log_E_lik = np.log(E_lik)
        assert not np.isnan(log_E_lik).any(), "NaNs in log-likelihoods"

        # at least step, produce volatility predictions & option prices:
        # if t == T:
        #     S_sim, vol_sim, vol_pred = self.pred_price(theta, self.wgts.W, T,
        #                                               **self.pred_opts)

        return log_E_lik

    def pred_price(self, theta, W, t=None, **pred_opts):
        '''
        predict volatility and price options by simulating future evolutions
        of a final estimated model

        Parameters:
        -----------
        h: int
            prediction horizon, no. of days ahead to simulate
        M: int
            no. of simulated future return paths per θ-particle

        '''
        h = pred_opts['h']
        # M = 1 if h == 1 else pred_opts['M']
        alpha = pred_opts['alpha']
        N = len(theta)
        T = len(self.data) if pred_opts.get('t') is None else pred_opts['t']
        K = self.K  # no of regimes (not strike price!)
        X = self.data

        # get parameters & cap/transform
        if 'df_X' in theta.dtype.names:
            df_X = theta['df_X']  # (N,)
            df_X = cap(df_X, 2.+1e-10, None)
        else:
            df_X = None

        # regime probabilities: fixed if K=1 or mixture; for Markov depend
        # on time
        p_t = self.probs(theta, X, T)
        probs = (p_t*W[:, np.newaxis]).flatten()  # weighted & flattened
        vols = self.model.vol(theta, X, T)  # (N,K) 1-day ahead vol's
        sigmas = vols.flatten()

        # 1-day ahead prediction (set):
        # (1) Volatility
        # prediction:
        self.vol_pred[T] = (sigmas * probs).sum()
        # prediction set;
        dsw = DescrStatsW(sigmas, probs)
        self.vol_predset[:, T] = dsw.quantile([0.5*alpha, 1.-0.5*alpha],
                                               return_pandas=False)

        # (2) Squared returns
        # prediction:
        self.X2_pred[T] = (sigmas**2 * probs).sum()
        # prediction set:
        dsw = DescrStatsW(sigmas**2, probs)
        self.X2_predset[:, T] = dsw.quantile([0.5*alpha, 1.-0.5*alpha],
                                               return_pandas=False)

        # (3)) Squared returns
        # prediction set:
        X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., sigmas, df_X),
                            p=[0.5*alpha, 1.-0.5*alpha])
        self.S_predset[:, T] = np.exp(0.01*X_predset) * self.S[-1]

        # (3) volatility estimates given full posterior
        # vol_pred_post = np.full([T+1], np.nan)
        # for j in range(T+1):
        #     s_t = self.model.vol(theta, X[0:j], j)  # (N,K)
        #     s_t = np.einsum('NK,NK->N', s_t, p_t)  # (N,), pred of particles
        #     vol_pred_post[j] = np.einsum('N,N->...', s_t, W)

        # (4) future path simulation: prediction (sets) for out-sample
        # volatility, returns, prices, & option prices:
        # for h>1, need to simulate future return paths of length h-1:
        if h > 1:
            # N*M different future evolutions; N different option prices
            # --> reshape X to (N,T,M), where (:, 0:T, :) is identical
            X_sim = np.tile(X[np.newaxis, :, np.newaxis], [N, 1, M])  # (N,T,M)
            vols = np.tile(vols[:, :, np.newaxis], [1, 1, M])  # (N,K,M)
            probs = (np.tile(p_t[:, :, np.newaxis], [1, 1, M]) *
                     np.tile(W[:, np.newaxis, np.newaxis], [1, K, M])) / M

            # simulate evolutions & compute vol's:
            for j in range(1, h):
                # pick random regime for particles & each of their copies
                r = np.apply_along_axis(lambda p: np.random.choice(np.arange(K),
                                                                   M, p=p),
                                        1, p_t)  # (N,M)
                row_ind, col_ind = np.ogrid[:N, :M]
                vol = vols[row_ind, r, col_ind]  # (N,M)
                # ...and separate innovation
                Z_next = F_innov(df=df_X).rvs(size=[N, M])  # (N,M)
                X_next = np.einsum('NM,NM->NM', vol, Z_next)  # (N,M)
                X_sim = np.concatenate([X_sim, X_next[:, np.newaxis, :]],
                                       axis=1)  # (N,T+j,M)
                # next day vol prediction & prediction set:
                vols = self.model.vol(theta, X_sim, T+j)  # (N,K,M)
                self.vol_pred[T+j] = np.einsum('NKM,NKM->...', vols, probs)
                dsw = DescrStatsW(vols.flatten(), probs.flatten())
                self.vol_predset[:, T+j] = dsw.quantile([0.5*alpha, 1.-0.5*alpha],
                                                        return_pandas=False)
                # squared return prediction & prediction set:
                self.X2_pred[T+j] = np.einsum('NKM,NKM->...', vols**2, probs)
                X2_predset = inv_cdf(cdf=lambda x: sq_mix_cdf(x, probs.flatten(), 0.,
                                                              vols.flatten()**2,
                                     df_X), p=[0.5*alpha, 1.-0.5*alpha],
                                     lo=0., hi=100.)
                self.X2_predset[:, T+j] = X2_predset


                if self.switching == 'markov':
                    pass

        # convert estimated log-returns to prices
        if h <= 1:
            X_sim = X[np.newaxis, :, np.newaxis]

        X_sim = np.insert(X_sim, 0, 0., 1)
        S_sim = np.exp(0.01 * np.cumsum(X_sim, axis=1)) * self.S[0]
        # (!) unweighted (!) must weight axis 0 by particle weights
        self.S_predset[:, T+1:] = np.quantile(S_sim[:, T+1:, :], axis=(0, 2),
                                              q=[0.5*alpha, 1.-0.5*alpha])

        return S_sim, self.S_predset, self.vol_pred, self.vol_predset, self.X2_pred, self.X2_predset


class HybridHMM(HMM):
    '''
    Student t HMM which automatically becomes Gaussian (if df=None) or Cauchy
    (if df=1) HMM

    Used in Markov-switching models to compute regime probabilities given
    parameters
    '''

    def PY(self, t, xp, x):
        s_t = self.model.vol(theta=self.theta, X=self.X, t=self.t).flatten()
        s_t = cap(s_t, 1e-50, 1e50)
        return F_innov(sd=s_t[x], df=self.df)
