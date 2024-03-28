#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

    def vol_std(self, theta, X, t=None):
        '''
        compute the standard GARCH volatility for 1 or more regimes at time t
        given the parameters
        '''
        # by default compute volatility for first period outside dataset
        t = len(X) if t is None else t
        N = len(theta)
        K = self.K
        shape = [N, K]

        if K == 1:
            omegas = theta['omega'].reshape(shape)
            alphas = theta['alpha'].reshape(shape)
            betas = theta['beta'].reshape(shape)
        else:
            omegas = np.full(shape, np.nan)
            alphas = np.full(shape, np.nan)
            betas = np.full(shape, np.nan)
            for k in range(K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]

        # transform & cap parameters:
        omegas = cap(omegas, 1e-20, None)
        alphas = cap(alphas, 1e-20, 1.)
        betas = cap(betas, 1e-20, 1.)

        # previous returns in reverse order (first entry = previous return)
        if X.ndim == 1:
            X_rev = X[0:t][::-1]
            X_rev = X_rev[np.newaxis, :, np.newaxis]  # (1,t,1)
        else:
            X_rev = X[:, 0:t, :][:, ::-1, :]  # (N,t,M)

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GARCH volatilities:
        # s2_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        s2_0 = np.random.gamma(1., 0.5, size=[N, K])  # random initial volatility
        beta_pow = betas[:, :, np.newaxis] ** t_grid  # (t,N,K)
        beta_pow = np.swapaxes(beta_pow, 1, 2)  # (N,t,K)
        a = np.einsum('NtK->NK', beta_pow)  # sum over axis 1
        a = np.einsum('NK,NK->NK', omegas, a)  # Hadamard product
        b = np.einsum('NtK,NtM->NtKM', beta_pow, X_rev**2)
        b = np.einsum('NtKM->NKM', b)
        b = np.einsum('NK,NKM->NKM', alphas, b)
        c = betas**t * s2_0  # (N,K)
        s2_t = a[:, :, np.newaxis] + b + c[:, :, np.newaxis]
        # --> s2_t[i, j] is squared volatility of i-th particle in j-th regime

        s2_t = cap(s2_t, 1e-100, None)  # avoids error from sqrt
        s_t = np.sqrt(s2_t)

        # if X is 1-dim, then must return (N,K) array for logpyt
        if X.ndim == 1: s_t = s_t[:, :, 0]

        return s_t

    def vol_gjr(self, theta, X, t):
        '''
        compute the GJR-GARCH volatility for K regimes at time t:

            σ_t^2 = ω + (α + γ·1(X_{t-1}>0))·X_{t-1}^2 + β·σ_{t-1}^2

        '''
        N = len(theta)
        K = self.K
        shape = [N, K]

        if K == 1:
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
        omegas = cap(omegas, 1e-20, None)
        alphas = cap(alphas, 1e-20, 1.)
        betas = cap(betas, 1e-20, 1.)

        # previous returns in reverse order (first entry = previous return)
        if X.ndim == 1:
            X_rev = X[0:t][::-1]
            X_rev = X_rev[np.newaxis, :, np.newaxis]  # (1,t,1)
        else:
            X_rev = X[:, 0:t, :][:, ::-1, :]  # (N,t,M)

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GJR-GARCH volatilities:
        # s2_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #        + Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas[:, :, np.newaxis] ** t_grid  # (t,N,K)
        beta_pow = np.swapaxes(beta_pow, 1, 2)  # (N,t,K)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NtK->NK', beta_pow)  # sum over axis 1
        a = np.einsum('NK,NK->NK', omegas, a)  # Hadamard product
        a = a[:, :, np.newaxis]

        # α·Σ_{j=1}^t β^j X_{t-j}^2
        b = np.einsum('NtK,NtM->NtKM', beta_pow, X_rev**2)
        b = np.einsum('NtKM->NKM', b)
        b = np.einsum('NK,NKM->NKM', alphas, b)

        #  β^t·σ_0^2
        s2_0 = np.random.gamma(1., 0.5, size=[N, K])  # random initial volatility
        c = betas**t * s2_0  # (N,K,M)
        c = c[:, :, np.newaxis]

        # Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2
        d = np.einsum('NtM,NtM->NtM', 1*(X_rev<0), X_rev**2)  # Hadamard
        d = np.einsum('NtK,NtM->NtKM', beta_pow, d)
        d = np.einsum('NtKM->NKM', d)
        d = np.einsum('NK,NKM->NKM', gammas, b)

        s2_t = a + b + c + d

        s2_t = cap(s2_t, 1e-100, None)  # avoids error from sqrt
        s_t = np.sqrt(s2_t)

        if X.ndim == 1: s_t = s_t[:, :, 0]

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

        # previous returns in reverse order (first entry = previous return)
        if X.ndim == 1:
            X_rev = X[0:t][::-1]
            X_rev = X_rev[np.newaxis, :, np.newaxis]  # (1,t,1)
        else:
            X_rev = X[:, 0:t, :][:, ::-1, :]  # (N,t,M)

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GJR-GARCH volatilities:
        # s_t = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #        + Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas[:, :, np.newaxis] ** t_grid  # (t,N,K)
        beta_pow = np.swapaxes(beta_pow, 1, 2)  # (N,t,K)

        # ω·Σ_{j=1}^t β^j
        a = np.einsum('NtK->NK', beta_pow)  # sum over axis 1
        a = np.einsum('NK,NK->NK', omegas, a)  # Hadamard product
        a = a[:, :, np.newaxis]

        # α·Σ_{j=1}^t |β^j X_{t-j}|
        b = np.einsum('NtK,NtM->NtKM', beta_pow, abs(X_rev))
        b = np.einsum('NtKM->NKM', b)
        b = np.einsum('NK,NKM->NKM', alphas, b)

        # β^t·σ_0^2
        c = betas**t * self.s2_0  # (N,K,M)
        c = c[:, :, np.newaxis]

        # Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2
        d = np.einsum('NtK,NtM->NtKM', beta_pow, X_rev)
        d = np.einsum('NtKM->NKM', d)
        d = np.einsum('NK,NKM->NKM', gammas, b)

        s_t = a + b + c + d

        # if X is 1-dim, then must return (N,K) array for logpyt
        if X.ndim == 1: s_t = s_t[:, :, 0]

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

        # previous returns in reverse order (first entry = previous return)
        if X.ndim == 1:
            X_rev = X[0:t][::-1]
            X_rev = X_rev[np.newaxis, :, np.newaxis]  # (1,t,1)
        else:
            X_rev = X[:, 0:t, :][:, ::-1, :]  # (N,t,M)

        t_grid = np.arange(0, t, 1)  # all time lags

        # compute GJR-GARCH volatilities:
        # log(s2_t) = ω·Σ_{j=1}^t β^j + α·Σ_{j=1}^t β^j X_{t-j}^2 + β^t·σ_0^2
        #             + γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) X_{t-j}^2

        # β_{ik}^j, i=1,...,N, k=0,...,K-1, j=0,...,t
        beta_pow = betas[:, :, np.newaxis] ** t_grid  # (t,N,K)
        beta_pow = np.swapaxes(beta_pow, 1, 2)  # (N,t,K)

        # ω·Σ_{j=1}^t β^j; (N,K)
        a = np.einsum('NtK->NK', beta_pow)  # sum over axis 1
        a = np.einsum('NK,NK->NK', omegas, a)  # Hadamard product
        a = a[:, :, np.newaxis]

        # α·Σ_{j=1}^t β^j |X_{t-j}|; (N,K,M)
        b = np.einsum('NtK,NtM->NtKM', beta_pow, abs(X_rev), optimize='greedy')
        b = np.einsum('NtKM->NKM', b, optimize='greedy')
        b = np.einsum('NK,NKM->NKM', alphas, b, optimize='greedy')

        # β^t·σ_0^2; (N,K)
        c = betas**t * np.log(self.s2_0)
        c = c[:, :, np.newaxis]

        # γ·Σ_{j=1}^t β^j 1(X_{t-j}<0) |X_{t-j}|; (N,K,M)
        d = np.einsum('NtM,NtM->NtM', 1*(X_rev<0), X_rev)  # Hadamard
        d = np.einsum('NtK,NtM->NtKM', beta_pow, d, optimize='greedy')
        d = np.einsum('NtKM->NKM', d)
        d = np.einsum('NK,NKM->NKM', gammas, b, optimize='greedy')

        log_s2_t = a + b + c + d
        log_s2_t = cap(log_s2_t, -50., 50.)

        # if X is 1-dim, then must return (N,K) array for logpyt
        if X.ndim == 1: log_s2_t = log_s2_t[:, :, 0]

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