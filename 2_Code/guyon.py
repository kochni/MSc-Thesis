#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# kernel functions

def tspl(t, tp, alpha, delta):
    '''
    "time-shifted power law" kernel
    '''
    alpha = alpha[:, :, np.newaxis]
    delta = delta[:, :, np.newaxis]

    tau = abs(t-tp)  # (t,)
    weights = (tau + delta)**(-alpha)  # (N,K,t)

    # normalize so that axis 2 sums to 1
    weights = weights / weights.sum(axis=2)[:, :, np.newaxis]

    return weights


class Guyon:
    '''
    Guyon and Lekeufack's (2023) handcrafted path-dependent volatility model

    '''

    def __init__(self, K):

        self.K = K

    def vol(self, theta, X, t):

        N = len(theta)
        K = self.K

        if K == 1:
            omegas = theta['omega'].reshape([N, K])
            alphas = theta['alpha'].reshape([N, K])
            betas = theta['beta'].reshape([N, K])
            a1s = theta['a1'].reshape([N, K])
            a2s = theta['a2'].reshape([N, K])
            d1s = theta['d1'].reshape([N, K])
            d2s = theta['d2'].reshape([N, K])
        else:
            omegas = alphas = betas = np.full([N, K], np.nan)
            a1s = a2s = d1s = d2s = np.full([N, K], np.nan)
            for k in range(self.K):
                omegas[:, k] = theta['omega_' + str(k)]
                alphas[:, k] = theta['alpha_' + str(k)]
                betas[:, k] = theta['beta_' + str(k)]
                a1s[:, k] = theta['a1_' + str(k)]
                a2s[:, k] = theta['a2_' + str(k)]
                d1s[:, k] = theta['d1_' + str(k)]
                d2s[:, k] = theta['d2_' + str(k)]

        # transform/cap parameters
        omegas = np.clip(omegas, 0., None)
        alphas = np.clip(alphas, 0., None)
        betas = np.clip(betas, 0., None)
        a1s = np.clip(a1s, 1., None)
        a2s = np.clip(a2s, 1., None)
        d1s = np.clip(d1s, 0., None)
        d2s = np.clip(d2s, 0., None)

        if t > 0:
            # grid of all previous time stamps
            tp_grid = np.arange(0, t, 1)

            # trend kernel weights:
            # k1[i,j,k] = weight of i-th particle, j-th regime, k-th time lag
            k1 = tspl(t, tp_grid, a1s, d1s)  # (N,K,t)

            # volatility kernel weights:
            k2 = tspl(t, tp_grid, a2s, d2s)  # (N,K,t)

            trend = np.einsum('NKt,t->NK', k1, X[0:t])
            volat = np.einsum('NKt,t->NK', k2, X[0:t]**2)

            s_t = (omegas +
                   np.einsum('NK,NK->NK', alphas, trend) +
                   np.einsum('NK,NK->NK', betas, volat))

        else:  # t == 0:
            s2_0 = X[0]**2
            s_t = np.sqrt(s2_0)
            s_t = np.full([N, K], s_t)

        s_t = np.clip(s_t, 1e-50, 1e50)  # (N,K)

        return s_t