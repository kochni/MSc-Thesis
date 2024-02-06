#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class NeuralSV(StateSpaceModel):
    '''
    Reservoir computer for stochastic volatility:

        log(V_t^2) = w·r^(1)_t + [v·r^(2)_t]·Z_t

        where r^(1)_t, r^(2)_t are two separate random projections of the past
        return and log-variance

    '''

    # change stuff here to modify model:
    default_params = {
        'vol_drift': 'neural',  # 'basic', 'heston', or 'neural
        'vol_vol': 'heston',    # same; 'basic' = constant, 'heston' =
                                # exponentially decreasing
        'q': 5,  # dim of reservoir; must match shape[0] of A, b below!
        'A_m': np.random.normal(size=[5, 2]),  #
        'A_v': np.random.normal(size=[5, 2]),  # random inner NN weights
        'b_m': np.random.normal(size=[5, 1]),  #
        'b_v': np.random.normal(size=[5, 1]),  #
        'df_X': None, 'df_V': None
        }

    def Res(self, component, logV2_prev, X_prev):
        '''
        Reservoir map

        Parameters
        ----------
        component: 'drift' or 'diff'

        '''
        # input = previous log-variance and return
        X_prev = np.full(len(logV2_prev), X_prev)
        I = np.vstack([logV2_prev, X_prev])  # (2,Nx) or (2,M)

        if component == 'drift':
            A = self.A_m
            b = self.b_m
        elif component == 'diff':
            A = self.A_v
            b = self.b_v

        h = np.matmul(A, I) + b  # (q,Nx)

        return sigmoid(h)

    def EXt(self, theta, logV2_prev=None, k=None, X_prev=None):

        if X_prev is None:
            X_prev = np.array([0.])
        if logV2_prev is None:
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_drift == 'neural':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)  # (q,)
                v = np.array(v)
                Res = self.Res('drift', logV2_prev, X_prev)  # (q,Nx)
                return np.einsum('q,qS->S', v, Res) + theta['v0']

            elif self.vol_drift == 'canon':
                omega = theta['omega']
                alpha = theta['alpha']
                return (1.-alpha) * omega + alpha * logV2_prev

            else:  # Heston
                kappa = theta['kappa']
                nu = theta['nu']
                return (logV2_prev + kappa*(nu * np.exp(-logV2_prev) - 1.)
                        - 0.5*xi**2 * np.exp(-logV2_prev))

        # during prediction, theta is structured array with all N particles
        else:
            M = len(theta)
            if self.vol_drift == 'neural':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)
                v = np.array(v)  # (q,M)
                Res = self.Res('drift', logV2_prev, X_prev)  # (q,M)
                return np.sum(v * Res, axis=0) + theta['v0']

            elif self.vol_drift == 'canon':
                omega = theta['omega']
                alpha = theta['alpha']
                return omega * (1.-alpha) + alpha * logV2_prev

            else:  # Heston
                kappa = theta['kappa']
                nu = theta['nu']
                return (logV2_prev + kappa * (nu * np.exp(-logV2_prev) - 1.)
                        - 0.5*xi**2 * np.exp(-logV2_prev))

    def SDXt(self, theta, logV2_prev=None, k=None, X_prev=None):

        if X_prev is None:
            X_prev = np.array([0.])
        if logV2_prev is None:
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_vol == 'neural':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)  # (q,)
                w = np.array(w)
                Res = self.Res('diff', logV2_prev, X_prev)  # (q,Nx)
                logxi2 = np.einsum('q,qS->S', w, Res) + theta['w0']
                return np.exp(0.5*logxi2)

            elif self.vol_vol == 'canon':
                return theta['xi']

            else:  # Heston
                return theta['xi'] * np.exp(-0.5*logV2_prev)

        # during prediction, theta is structured array with all N particles
        else:
            if self.vol_vol == 'neural':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)
                w = np.array(w)  # (q,M)
                Res = self.Res('diff', x, X_prev)  # (q,M)
                logxi2 = np.sum(w * Res, axis=0) + theta['w0']
                return np.exp(0.5*logxi2)

            elif self.vol_vol == 'canon':
                return theta['xi']

            else:  # Heston
                return theta['xi'] * np.exp(-0.5*logV2_prev)

    def PX0(self, theta, theta_FV):
        return F_innov(self.EXt(theta),
                       self.SDXt(theta),
                       **theta_FV,
                       a=-10., b=10.)

    def PX(self, t, logV2_prev, theta, theta_FV, X_prev):
        return F_innov(self.EXt(theta, logV2_prev, X_prev=X_prev),
                       self.SDXt(theta, logV2_prev, X_prev=X_prev),
                       **theta_FV,
                       a=-10., b=10.)

    def PY(self, t, logV2, theta, theta_FX):
        return F_innov(0., np.exp(0.5*logV2), **theta_FX)