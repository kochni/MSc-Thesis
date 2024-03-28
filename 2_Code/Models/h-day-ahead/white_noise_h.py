#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

        # during training. need to return (N,K) array;
        # during simulation, need to return (N,K,M) array
        if X.ndim > 1:  # simulation
            M = X.shape[2]
            s_t = np.tile(s_t[:, :, np.newaxis], [1, 1, M])

        return s_t