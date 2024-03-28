#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Guyon:
    '''
    Guyon and Lekeufack's (2023) path-dependent volatility model

    '''

    def __init__(self, K):

        self.K = K
        self.s2_0 = 1  # initial volatility

    def vol(self, theta, X, t):

        N = len(theta)
        K = self.K

        shape = [N, K]

        # make X 4-dimensional (N,t,K,M)
        if X.ndim == 1:  # X has shape (t,)
            X_resh = X[np.newaxis, 0:t, np.newaxis, np.newaxis]  # (1,t,1,1)
        else:  # X has shape (N,t,M)
            X_resh = X[:, 0:t, np.newaxis, :]

        if K == 1:
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
        alphas = cap(alphas, 0., None)
        betas = cap(betas, 0., None)
        a1s = cap(a1s, 1., None)
        a2s = cap(a2s, 1., None)
        d1s = cap(d1s, 1., None)
        d2s = cap(d2s, 1., None)

        # make kernel parameters 3D
        a1s = a1s[:, :, np.newaxis]
        a2s = a2s[:, :, np.newaxis]
        d1s = d1s[:, :, np.newaxis]
        d2s = d2s[:, :, np.newaxis]

        if t > 0:
            # grid of all previous time stamps
            tp_grid = np.arange(0, t, 1)

            # trend kernel weights:
            # k1[i,j,k] = weight of i-th particle, j-th regime, k-th time lag
            k1 = tspl(t=t, tp=tp_grid, alpha=a1s, delta=d1s)  # (N,K,t)

            # volatility kernel weights:
            k2 = tspl(t=t, tp=tp_grid, alpha=a2s, delta=d2s)  # (N,K,t)

            k1 = np.swapaxes(k1, 1, 2)  # (N,t,K)
            k2 = np.swapaxes(k2, 1, 2)  # (N,t,K)

            k1 = k1[:, :, :, np.newaxis]  # (N,t,K,1)
            k2 = k2[:, :, :, np.newaxis]  # (N,t,K,1)

            trend = np.sum(k1 * X_resh, axis=1)  # (N,K,M)
            volat = np.sqrt(np.sum(k2 * X_resh**2, axis=1))  # (N,K,M)

            s_t = (omegas[:, :, np.newaxis] +
                   alphas[:, :, np.newaxis] * trend +
                   betas[:, :, np.newaxis] * volat)
            s_t = cap(s_t, 1e-50, 1e50)

            # if X is 1-dim, then must return (N,K) array for logpyt
            if X.ndim == 1: s_t = s_t[:, :, 0]

        elif t == 1:
            trend = X_resh[:, 0, :, :]  # (N,K,M)
            volat = X_resh[:, 0, :, :]  # (N,K,M)
            s_t = (omegas[:, :, np.newaxis] +
                   alphas[:, :, np.newaxis] * trend +
                   betas[:, :, np.newaxis] * volat)

            # if X is 1-dim, then must return (N,K) array for logpyt
            if X.ndim == 1: s_t = s_t[:, :, 0]

        else:  # t == 0:
            s_t = np.sqrt(self.s2_0)
            s_t = np.tile(s_t, shape)

        return s_t