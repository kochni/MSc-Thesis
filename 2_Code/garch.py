#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class GARCH:
    '''
    GARCH-type volatility models:

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

        # random initial volatility
        self.s2_0 = np.random.gamma(1., 0.5, size=1)

    def vol_std(self, theta, X, t=None):
        '''
        compute the standard GARCH volatility for 1 or more regimes at time t
        given the parameters
        '''
        # by default compute volatility for first period outside dataset
        t = len(X) if t is None else t
        N = len(theta)
        K = self.K

        if K == 1:
            omegas = theta['omega'].reshape(N, K)
            alphas = theta['alpha'].reshape(N, K)
            betas = theta['beta'].reshape(N, K)
        else:
            omegas = np.full([N, K], np.nan)
            alphas = np.full([N, K], np.nan)
            betas = np.full([N, K], np.nan)
            for k in range(K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]

        # transform & cap parameters:
        omegas = np.clip(omegas, 1e-20, None)
        alphas = np.clip(alphas, 1e-20, 1.)
        betas = np.clip(betas, 1e-20, 1.)

        # previous returns in reverse order (first entry = previous return)
        X_rev = X[0:t][::-1]
        # all time lags
        t_grid = np.arange(0, t, 1)

        # compute GARCH volatilities:
        # s2_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas.reshape(N, K, 1) ** t_grid  # (N,K,t)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NK,NKt->NK', omegas, beta_pow)

        # α·Σ_{j=1}^t β^j X_{t-j}^2
        b = np.einsum('NK,NKt,t->NK', alphas, beta_pow, X_rev**2)

        #  β^t·σ_0^2
        c = betas**t * self.s2_0

        s2_t = a + b + c
        # --> s2_t[i, j] is squared volatility of i-th particle in j-th regime

        s2_t = np.clip(s2_t, 1e-100, None)  # avoids error from sqrt
        s_t = np.sqrt(s2_t)

        return s_t

    def vol_gjr(self, theta, X, t):
        '''
        compute the GJR-GARCH volatility for K regimes at time t:

            σ_t^2 = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2

        '''
        N = len(theta)
        K = self.K

        if K == 1:
            omegas = theta['omega'].reshape(N, K)
            alphas = theta['alpha'].reshape(N, K)
            betas = theta['beta'].reshape(N, K)
            gammas = theta['gamma'].reshape(N, K)
        else:
            omegas = np.full([N, K], np.nan)
            alphas = np.full([N, K], np.nan)
            betas = np.full([N, K], np.nan)
            gammas = np.full([N, K], np.nan)
            for k in range(K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                gammas[:, k] = theta['gamma_' + str(k)]

        # transform & cap parameters:
        omegas = np.clip(omegas, 1e-20, None)
        alphas = np.clip(alphas, 1e-20, 1.)
        betas = np.clip(betas, 1e-20, 1.)

        # previous returns in reverse order (first entry = previous return)
        X_rev = X[0:t][::-1]
        # all time lags
        t_grid = np.arange(0, t, 1)

        # compute GJR-GARCH volatilities:
        # s2_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #        + Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas.reshape(N, K, 1) ** t_grid  # (N,K,t)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NK,NKt->NK', omegas, beta_pow)

        # α·Σ_{j=1}^t β^j X_{t-j}^2
        b = np.einsum('NK,NKt,t->NK', alphas, beta_pow, X_rev**2)

        #  β^t·σ_0^2
        c = betas**t * self.s2_0

        # γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2
        d = np.einsum('NK,NKt,t,t->NK', gammas, beta_pow, 1*(X_rev<0), X_rev**2)

        s2_t = a + b + c + d

        s2_t = np.clip(s2_t, 1e-100, None)  # avoids error from sqrt
        s_t = np.sqrt(s2_t)

        return s_t

    def vol_thr(self, theta, X, t):
        '''
        compute the Threshold-GARCH volatility for K regimes at time t:

            σ^2_t = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2

        '''
        N = len(theta)
        K = self.K

        if self.K == 1:
            omegas = theta['omega'].reshape(N, K)
            alphas = theta['alpha'].reshape(N, K)
            betas = theta['beta'].reshape(N, K)
            gammas = theta['gamma'].reshape(N, K)
        else:
            omegas = np.full([N, K], np.nan)
            alphas = np.full([N, K], np.nan)
            betas = np.full([N, K], np.nan)
            gammas = np.full([N, K], np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                gammas[:, k] = theta['gamma_' + str(k)]

        # previous returns in reverse order (first entry = previous return)
        X_rev = X[0:t][::-1]

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GJR-GARCH volatilities:
        # s_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #        + Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas.reshape(N, K, 1) ** t_grid  # (N,K,t)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NK,NKt->NK', omegas, beta_pow)

        # α·Σ_{j=1}^t β^j |X_{t-j}|
        b = np.einsum('NK,NKt,t->NK', alphas, beta_pow, abs(X_rev))

        # β^t·σ_0
        c = betas**t * self.s2_0

        # γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}
        d = np.einsum('NK,NKt,t,t->NK', gammas, beta_pow, 1*(X_rev<0), X_rev)

        s_t = a + b + c + d

        return s_t

    def vol_exp(self, theta, X, t):
        '''
        compute the Exponential-GARCH volatility for K regimes at time t:

            log(σ^2_t) = ω + α·(|Z_{t-1}| + γ·Z_{t-1}) + β·log(σ^2_{t-1})

        '''
        N = len(theta)
        K = self.K
        shape = [N, K]

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

        # previous returns in reverse order (first entry = previous return
        X_rev = X[0:t][::-1]
        # all time lags
        t_grid = np.arange(0, t, 1)

        # compute GJR-GARCH volatilities:
        # log(s2_t) = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #             + γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas.reshape(N, K, 1) ** t_grid  # (N,K,t)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NK,NKt->NK', omegas, beta_pow)

        # α·Σ_{j=1}^t β^j |X_{t-j}|
        b = np.einsum('NK,NKt,t->NK', alphas, beta_pow, abs(X_rev),
                      optimize='greedy')

        # β^t·σ_0^2
        c = betas**t * self.s2_0

        # γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) |X_{t-j}|
        d = np.einsum('NK,NKt,t,t->NK', gammas, beta_pow, 1*(X_rev<0), X_rev)

        log_s2_t = a + b + c + d
        log_s2_t = np.clip(log_s2_t, -50., 50.)

        s_t = np.exp(0.5*log_s2_t)

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