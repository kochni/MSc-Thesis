#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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

        return s_t