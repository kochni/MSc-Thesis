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

# other utilts
from numpy.lib.recfunctions import structured_to_unstructured
from operator import itemgetter
# from helpers import *


class DetVol(StaticModel):
    '''
    Any model for which either
        - the conditional volatility (and hence likelihood) is known given the
          past observations and the model parameters (e.g. GARCH, ELM)
        - the expected likelihood (over any latent variables) can be computed
          analytically (e.g., models w regime switching or jumps in returns)

    This allows for using the IBIS algorithm.

    '''

    def __init__(self, spec, prior, data):
        super().__init__(data, prior)

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

    def probs(self, theta, X, t):
        '''
        probabilities of volatility regimes

        Output: p_t, (N, K) array
            where p_t[i, j] probability of j-th regime at time t given
            parameters from i-th particle

        '''

        shape = [len(theta), self.K]
        p_t = np.full(shape, np.nan)

        # Single-Regime Models:
        if self.K == 1:
            p_t = np.full(shape, 1.)

        # Mixture Models:
        elif self.switching in ['mix', 'mixing', 'mixture']:
            ps = theta['p_0'] if self.K == 2 else theta['p']
            ps = ps.reshape(-1, self.K-1)
            p_t[:, 0:-1] = ps
            p_t[:, -1] = 1 - np.sum(ps, axis=1)

        # Markov-Switching Models:
        else:
            for i in range(len(theta)):  # (!) vectorize ??
                df = theta['df'][i] if 'df' in theta.dtype.names else None
                P = np.full([self.K, self.K], 0.)  # transition matrix
                for k in range(self.K):
                    P[k, 0:-1] = theta['P_' + str(k)][i]
                P[:, -1] = 1 - np.sum(P, axis=1)

                # HMM with given parameters
                hmm = HybridHMM(model=self.model, theta=theta[[i]],
                                X=self.data, t=t, trans_mat=P, df=df)
                # compute probabilities of regimes given past data
                if t > 0:
                    bw = BaumWelch(hmm=hmm, data=self.data[0:t])
                    bw.forward()
                    bw.backward()
                    # print("p_", t, "(i=", i, "):", bw.pred[-1])
                    p_t[i, :] = bw.pred[-1]
                else:
                    p_t[i, :] = 1. / self.K

        return p_t

    def logpyt(self, theta, t):
        '''
        (expected) likelihood of parameters

        '''
        # volatilities:
        s_t = self.model.vol(theta=theta, X=self.data, t=t)  # (N,K) array
        s_t = cap(s_t, floor=1e-100, ceil=1e100)
        assert not np.isnan(s_t).any(), "NaNs in volatilities"

        # regime probabilities:
        p_t = self.probs(theta, self.data, t)  # (N,K) array
        p_t = cap(p_t, floor=1e-50, ceil=1.)
        assert not np.isnan(p_t).any(), "NaNs in regime probabilities"

        # get parameters & cap/transform:
        if 'df_X' in theta.dtype.names:
            df_X = theta['df_X'].reshape(-1, 1)
            df_X = cap(df_X, floor=2.+1e-10)
        else:
            df_X = None

        # expected likelihood over regimes:
        liks = F_innov(sd=s_t, df=df_X).pdf(self.data[t])
        E_lik = np.einsum('NK,NK->N', p_t, liks)  # Hadamard & row-sum
        E_lik = E_lik.flatten()

        # expected likelihoods conditional on occurrence of jump:
        if 'lambda_X' in theta.dtype.names:
            lambd_X = cap(theta['lambda_X'], floor=0., ceil=1.)
            phi_X = cap(theta['phi_X'], floor=0.)
            phi_X = np.tile(phi_X.reshape(-1, 1), [1, self.K])

            sd_jump = np.sqrt(s_t**2 + phi_X**2)
            liks_jump = F_innov(sd=sd_jump, df=df_X).pdf(self.data[t])
            E_lik_jump = np.einsum('NK,NK->N', p_t, liks_jump)  # Hadamard & row-sum
            E_lik_jump = E_lik_jump.flatten()

            # overall expected likelihood
            E_lik = (1.-lambd_X)*E_lik + lambd_X*E_lik_jump

        E_lik = cap(E_lik, floor=1e-100)  # avoids error when taking log
        log_E_lik = np.log(E_lik)
        assert not np.isnan(log_E_lik).any(), "NaNs in log-likelihoods"

        return log_E_lik

    def predict(self, theta, W, t=None, s=1, M=1):
        '''
        simulate future evolutions of an estimated process

        Parameters:
        -----------
        s: int
            no. of days ahead to simulate
        M: int
            no. of simulated future return paths per θ-particle

        '''
        # for 1-day ahead, simulation of future returns and hence cloning of
        # particles not needed
        M = 1 if s == 1 else M
        N = len(theta)
        t = len(self.data) if t is None else t
        theta = np.repeat(theta, M, 0)  # creates M copies of each θ-particle
        W = np.repeat(W, M, 0) / M
        X = self.data[0:t]

        # regime probabilities: fixed if K=1 or mixture; for Markov depend
        # on time
        if self.K == 1:
            p_t = np.full([M*N, self.K], 1.)
        elif self.switching in ['mix', 'mixing', 'mixture']:
            p_t = np.full([M*N, self.K], np.nan)
            ps = theta['p_0'] if self.K == 2 else theta['p']
            ps = ps.reshape(-1, self.K-1)
            p_t[:, 0:-1] = ps
            p_t[:, -1] = 1 - np.sum(ps, axis=1)

        # (1) in-sample volatility estimates + 1-day ahead prediction
        # (given data until time t)
        reg_vols = np.full([self.K, t+s], np.nan)  # pred's of each regime
        vol_pred = np.full(t+s, np.nan)  # reg_vols avg'd over regime prob's
        for j in range(t+1):
             vols = self.model.vol(theta, X[0:j], j)  # (M*N,K)
             reg_vols[:, j] = np.sum(vols * W.reshape(-1,1), axis=0)
             preds = np.sum(vols * p_t, axis=1)  # (M*N,), pred of particles
             vol_pred[j] = np.sum(preds * W)  # weigh predictions by W_t

        # (2) out-sample volatility predictions:
        # for s>1, need to simulate future return paths of length s-1:
        if s > 1:
            # get parameters & cap/transform
            if 'df_X' in theta.dtype.names:
                df_X = theta['df_X']  # (M*N,)
                df_X = cap(df_X, floor=2.+1e-10)
                df_X = np.tile(df_X.reshape(-1, 1, 1), [1, self.K])  # (M*N, 1, K)
            else:
                df_X = None

            # N*M*K different future evolutions --> reshape X to (M*N, t+s, K)
            shape = [M*N, 1, self.K]
            X = np.tile(self.data[np.newaxis, :, np.newaxis], shape)

            # simulate evolutions & compute vol's:
            for j in range(1, s):
                Z_next = F_innov(df=df_X).rvs(size=shape)
                X_next = vols * Z_next
                X = np.concatenate([X, X_next], axis=1)
                vols = self.model.vol(theta, X, t+j)  # (M*N,K); next vol's

                if self.switching == 'markov':
                    pass

                reg_vols[:, t+j] = np.sum(vols * W.reshape(-1,1), axis=0)
                preds = np.sum(vols * p_t, axis=1)  # (M*N,)
                vol_pred[t+j] = np.sum(preds * W)

        return vol_pred, reg_vols


class WhiteNoise:
    '''
    White noise volatility model:

        X_t = σ·Z_t

    '''

    def __init__(self, K):
        self.K = K

    def vol(self, theta, X=None, t=None):
        if self.K == 1:
            s_t = theta['sigma'].reshape(-1, 1)
        else:
            shape = [len(theta), self.K]
            s_t = np.full(shape, np.nan)
            for k in range(self.K):
                s_t[:, k] = theta['sigma_' + str(k)]

        # transform parameters
        s_t = np.exp(s_t)

        return s_t


class GARCH:
    '''
    GARCH volatility models:

        X_t = σ_t·Z_t
        σ_t^2 = f(X_{t-1}^2, σ_{t-1}^2)

        where f(·,·) depends on the variant.

    '''

    def __init__(self, variant, K, hyper=None):

        msg = "Specified GARCH-variant not implemented"
        assert variant in ['basic', 'gjr', 'thr', 'exp', 'elm'], msg
        self.variant = variant
        self.hyper = hyper
        self.K = K
        self.s2_0 = 1  # initial volatility

    def vol_std(self, theta, X, t=None):
        '''
        compute the standard GARCH volatility for 1 or more regimes at time t
        given the parameters
        '''
        # by default compute volatility for first period outside dataset
        t = len(X) if t is None else t
        N = len(theta)
        shape = [N, self.K]

        if self.K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
        else:
            omegas = np.full(shape, np.nan)
            alphas = np.full(shape, np.nan)
            betas = np.full(shape, np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]

        # transform & cap parameters:
        omegas = cap(omegas, floor=1e-20)
        alphas = cap(alphas, floor=1e-20)
        betas = cap(betas, floor=1e-20, ceil=1.)

        # previous returns in reverse order (first entry = previous return)
        if X.ndim == 1:
            X_rev = X[0:t][::-1]  # (t,)
            X_rev = X_rev.reshape(1, -1, 1)
        else:
            X_rev = X[:, 0:t, :][:, ::-1, :]  # (N,t,K)

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GARCH volatilities:
        # s2_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        beta_pow = betas[:, :, np.newaxis] ** t_grid  # (t,N,K)
        beta_pow = np.swapaxes(beta_pow, 1, 2)  # (N,t,K)
        a = np.einsum('NtK->NK', beta_pow)  # sum over axis 1
        a = np.einsum('NK,NK->NK', omegas, a)  # Hadamard product
        b = np.einsum('NtK,NtK->NtK', beta_pow, X_rev**2)
        b = np.einsum('NtK->NK', b)
        b = np.einsum('NK,NK->NK', alphas, b)
        c = betas**t * self.s2_0
        s2_t = a + b + c
        # --> s2_t[i, j] is squared volatility of i-th particle in j-th regime

        s2_t = cap(s2_t, floor=1e-100)  # avoids error from sqrt
        s_t = np.sqrt(s2_t)

        return s_t

    def vol_gjr(self, theta, X, t):
        '''
        compute the GJR-GARCH volatility for K regimes at time t:

            σ_t^2 = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2

        '''
        N = len(theta)
        shape = [N, self.K]

        if self.K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
            gammas = theta['gamma'].reshape(shape)
        else:
            omegas = np.full(shape, np.nan)
            alphas = np.full(shape, np.nan)
            betas = np.full(shape, np.nan)
            gammas = np.full(shape, np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                gammas[:, k] = theta['gamma_' + str(k)]

        # transform & cap parameters:
        omegas = cap(omegas, floor=1e-20)
        alphas = cap(alphas, floor=1e-20)
        betas = cap(betas, floor=1e-20, ceil=1.)

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
        # else X has shape (N,t,K)

        s2_j = np.tile(self.s2_0, shape)
        for j in range(1, t+1):
            s2_prev = s2_j
            s2_j = (omegas + (alphas + gammas*(X[:, j-1, :]<0))*X[:, j-1, :]**2
                    + betas*s2_prev)

        s2_j = cap(s2_j, floor=1e-100)  # avoids error from sqrt
        s_t = np.sqrt(s2_j)

        return s_t

    def vol_thr(self, theta, X, t):
        '''
        compute the Threshold-GARCH volatility for K regimes at time t:

            σ^2_t = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2

        '''
        shape = [len(theta), self.K]

        if self.K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
            gammas = theta['gamma'].reshape(shape)
        else:
            omegas = np.full(shape, np.nan)
            alphas = np.full(shape, np.nan)
            betas = np.full(shape, np.nan)
            gammas = np.full(shape, np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                gammas[:, k] = theta['gamma_' + str(k)]

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
        # else X has shape (N,t,K)

        s_j = np.tile(np.sqrt(self.s2_0), shape)  # initial volatility
        for j in range(1, t+1):
            s_prev = s_j
            s_j = (omegas + alphas*abs(X[:, j-1, :]) + gammas*X[:, j-1, :] +
                   betas*s_prev)

        return s_j

    def vol_exp(self, theta, X, t):
        '''
        compute the Exponential-GARCH volatility for K regimes at time t:

            log(σ^2_t) = ω + α·(|Z_{t-1}| + γ·Z_{t-1}) + β·log(σ^2_{t-1})

        '''
        shape = [len(theta), self.K]

        if self.K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
            gammas = theta['gamma'].reshape(shape)
        else:
            omegas = np.full(shape, np.nan)
            alphas = np.full(shape, np.nan)
            betas = np.full(shape, np.nan)
            gammas = np.full(shape, np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                gammas[:, k] = theta['gamma_' + str(k)]

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
        # else X has shape (N,t,K)

        log_s2_j = np.tile(np.log(self.s2_0), shape)
        for j in range(1, t+1):
            log_s2_prev = log_s2_j
            log_s2_j = (omegas + alphas*abs(X[:, j-1, :]) + gammas*X[:, j-1, :]
                        + betas*log_s2_prev)
            log_s2_j = cap(log_s2_j, floor=-100, ceil=100)

        s_t = np.exp(log_s2_j/2)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'basic':
            s_t = self.vol_std(theta, X, t)

        elif self.variant == 'gjr':
            s_t = self.vol_gjr(theta, X, t)

        elif self.variant == 'thr':
            s_t = self.vol_thr(theta, X, t)

        elif self.variant == 'exp':
            s_t = self.vol_exp(theta, X, t)

        elif self.variant == 'elm':
            model = ResComp(variant='elm', hyper=self.hyper, K=self.K)
            s_t = model.vol(theta, X, t)

        return s_t


class ResComp:
    '''
    Reservoir computers for static volatility models

    Parameters:
    -----------

    variant: string
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

    def __init__(self, variant, hyper, K):

        self.variant = variant
        self.K = K
        self.q = q = hyper['q']
        sd = hyper.get('sd')

        # draw random components
        if variant == 'elm':  # inner parameters of NN
            H = 20  # hidden layer width
            self.A1 = np.random.normal(scale=sd, size=[H, 2, K])
            self.b1 = np.random.normal(scale=sd, size=[H, 1, K])
            self.A2 = np.random.normal(scale=sd, size=[q, H, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K])

        elif variant == 'r_sig':  # random matrices of Controlled ODE
            self.rsig_0 = np.random.normal(scale=sd, size=[q, 1, K])
            self.A1 = np.random.normal(scale=sd, size=[q, q, K])
            self.b1 = np.random.normal(scale=sd, size=[q, 1, K])
            self.A2 = np.random.normal(scale=sd, size=[q, q, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K])

        else:
            # compute minimum required truncation level to obtain enough
            # components as specified in dimensionality (q)
            self.M = 1
            while 2*(2**self.M - 1) < self.q: self.M += 1

        # extract repeatedly accessed hyperparameters
        self.activ = hyper.get('activ')
        self.s2_0 = 1  # initial volatility

    def elm_vol(self, theta, X, t):
        '''
        compute volatility by passing past volatility & returns through a
        random 2-layer neural network, a.k.a. "extreme learning machine (ELM)"

        '''
        N = len(theta)
        shape0 = [N, self.K]  # shape of w0s (bias term)
        shape = [self.q, N, self.K]  # shape of Ws (linear weights)

        w0s = np.full(shape0, np.nan)
        Ws = np.full(shape, np.nan)

        if self.K == 1:
            w0s[:, 0] = theta['w0']
            for j in range(self.q):
                Ws[j, :, 0] = theta['w' + str(j)]
        else:
            for k in range(self.K):
                w0s[:, k] = theta['w0_' + str(k)]
                for j in range(self.q):
                    Ws[j, :, k] = theta['w' + str(j) + '_' + str(k)]

        w0s = cap(w0s, floor=-1e20, ceil=1e20)
        Ws = cap(Ws, floor=-1e20, ceil=1e20)

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
            X = np.tile(X, [N, 1, self.K]) # (N,t,K)
        # else X has shape (N,t,K) already anyway (but w different entries
        # along axes 0 and 2)

        log_s2_j = np.tile(np.log(self.s2_0), [N, self.K])  # (N,K)
        for j in range(1, t+1):  # replace with functools.reduce() ?
            log_s2_prev = log_s2_j

            # compute reservoir from previous log-return and volatility:
            M = np.stack((X[:, t-1, :], log_s2_prev), axis=0)  # (2,N,K)

            # hidden nodes 1:
            # matrix multiplication for each regime
            h1 = np.einsum('HIK,INK->HNK', self.A1, M) + self.b1
            h1 = self.activ(h1)  # (H,N,K)

            # hidden nodes 2
            h2 = np.einsum('qHK,HNK->qNK', self.A2, h1) + self.b2  # (q,N,K)
            res = self.activ(h2)  # final reservoir of shape (q,N,K)

            # get volatility from reservoir & readout
            log_s2_j = np.einsum('qNK,qNK->qNK', Ws, res) + w0s  # Hadamard
            log_s2_j = np.einsum('qNK->NK', log_s2_j)  # sum over axis 0
            log_s2_j = cap(log_s2_j, floor=-100, ceil=100)

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def sig_vol(self, theta, X, t):
        '''
        compute volatility from the truncated signature of the time-extended
        path of the log-returns, (t, X_t)
        '''
        N = len(theta)

        if t > 0:
            # time-extended path and its signature:
            X_tilde = np.vstack([X[0:t], np.arange(0, t, 1)])  # (2,t)

            if self.variant == 'standard':
                sig = stream2sig(X_tilde.T, self.M)[0:self.q+1]
            else:
                sig = stream2logsig(X_tilde.T, 6)[0:self.q+1]

            # compute volatility from weights & signature components:
            w_names = ['w' + str(j) for j in range(self.q+1)]
            W = itemgetter(*w_names)(theta)  # (q,N)
            W = np.array(W)
            log_s2_t = np.einsum('qN,q->N', W, sig)
            s_t = np.exp(0.5*log_s2_t)
            s_t = s_t.reshape(-1, 1)
        else:
            s_t = np.full([N, self.K], np.sqrt(self.s2_0))

        return s_t


    def rsig_vol(self, theta, X, t):
        '''
        compute volatility from the randomized signature of the time-extended
        path of the log-returns, (t, X_t)
        '''
        N = len(theta)
        shape0 = [N, self.K]
        shape = [self.q, N, self.K]

        # rSig_0 is identical between particles but not regimes
        rsig = np.tile(self.rsig_0, [1, N, 1])  # (q,N,K)

        w0s = np.full(shape0, np.nan)
        Ws = np.full(shape, np.nan)
        if self.K == 1:
            w0s[:, 0] = theta['w0']
            for j in range(self.q):
                Ws[j, :, 0] = theta['w' + str(j)]
        else:
            for k in range(self.K):
                w0s[:, k] = theta['w0_' + str(k)]
                for j in range(self.q):
                    Ws[j, :, k] = theta['w' + str(j) + '_' + str(k)]

        w0s = cap(w0s, floor=-1e20, ceil=1e20)
        Ws = cap(Ws, floor=-1e20, ceil=1e20)

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
            X = np.tile(X, [N, 1, self.K]) # (N,t,K)
        # else X has shape (N,t,K) already anyway

        log_s2_j = np.tile(np.log(self.s2_0), [N, self.K])  # (N,K)
        for j in range(1, t+1):  # (!) replace with functools.reduce()
            log_s2_prev = log_s2_j

            # update rSig
            incr_1 = np.einsum('qpK,pNK->qNK', self.A1, rsig) + self.b1
            incr_1 = self.activ(incr_1)  # (q,N,K)

            Z_prev = X[:, j-1, :] / np.exp(log_s2_prev/2)  # (N,K)
            incr_2 = np.einsum('qpK,pNK->qNK', self.A2, rsig) + self.b2  # matmul
            incr_2 = self.activ(incr_2)  # (q,N,K)
            incr_2 = np.einsum('qNK,NK->qNK', incr_2, Z_prev)  # Hadamard prod
            rsig += incr_1 + incr_2

            # get volatility from reservoir & rSig
            log_s2_j = np.einsum('qNK,qNK->qNK', Ws, rsig) + w0s  # Hadamard
            log_s2_j = np.einsum('qNK->NK', log_s2_j)  # sum over axis 0
            log_s2_j = cap(log_s2_j, floor=-100, ceil=100)

        s_t = np.exp(log_s2_j/2)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'elm':
            s_t = self.elm_vol(theta, X, t)
        elif self.variant == 'r_sig':
            s_t = self.rsig_vol(theta, X, t)
        else:  # variant = t_sig or log_sig
            s_t = self.sig_vol(theta, X, t)

        return s_t


class Guyon:
    '''
    Guyon and Lekeufack's (2023) path-dependent volatility model

    '''

    def __init__(self, K):

        self.K = K
        self.s2_0 = 1  # initial volatility

    def vol(self, theta, X, t):

        N = len(theta)
        shape = [N, self.K]

        if X.ndim == 1:
            X = X.reshape(1, -1, 1)  # (1,t,1)
            X = np.tile(X, [N, 1, self.K]) # (N,t,K)
        # else X has shape (N,t,K) already anyway (but w different entries
        # along axes 0 and 2)

        if self.K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
            a1s = theta['a1'].reshape(shape)
            a2s = theta['a2'].reshape(shape)
            d1s = theta['d1'].reshape(shape)
            d2s = theta['d2'].reshape(shape)
        else:
            omegas = alphas = betas = np.full(shape, np.nan)
            a1s = a2s = d1s = d2s = np.full(shape, np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                a1s[:, k] = theta['a1_' + str(k)]
                a2s[:, k] = theta['a2_' + str(k)]
                d1s[:, k] = theta['d1_' + str(k)]
                d2s[:, k] = theta['d2_' + str(k)]

        # transform/cap parameters
        omegas = cap(omegas, floor=0.)
        alphas = cap(alphas, floor=0.)
        betas = cap(betas, floor=0.)
        a1s = cap(a1s, floor=1.)
        a2s = cap(a2s, floor=1.)
        d1s = cap(d1s, floor=1.)
        d2s = cap(d2s, floor=1.)

        # kernel parameters must be 3D for vectorization
        a1s = a1s[:, :, np.newaxis]
        a2s = a2s[:, :, np.newaxis]
        d1s = d1s[:, :, np.newaxis]
        d2s = d2s[:, :, np.newaxis]

        if t > 0:
            # grid of all previous time stamps
            tp_grid = np.arange(0, t, 1)
            # trend kernel weights (k1[i, j, k] = weight of i-th particle,
            # j-th regime, k-th time lag)
            k1 = tspl(t=t, tp=tp_grid, alpha=a1s, delta=d1s)  # (N,K,t)
            # volatility kernel weights:
            k2 = tspl(t=t, tp=tp_grid, alpha=a2s, delta=d2s)  # (N,K,t)
            k1 = np.swapaxes(k1, 1, 2)  # (N,t,K)
            k2 = np.swapaxes(k2, 1, 2)  # (N,t,K)
            trend = np.sum(k1 * X[:, 0:t, :], axis=1)
            volat = np.sqrt(np.sum(k2 * X[:, 0:t, :]**2, axis=1))

            s_t = omegas + alphas*trend + betas*volat
            s_t = cap(s_t, floor=1e-50, ceil=1e50)

        elif t == 1:
            trend = X[:, 0, :]
            volat = X[:, 0, :]
            s_t = omegas + alphas*trend + betas*volat

        else:  # t == 0:
            s_t = np.sqrt(self.s2_0)
            s_t = np.tile(s_t, shape)

        return s_t


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
        return F_innov(scale=s_t[x], df=self.df)
