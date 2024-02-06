#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class EchoSV(StateSpaceModel):
    '''
    Echo state network stochastic volatility:

        log(V_t^2) = w·r^(1)_t + v·r^(2)_t + w0
        r^(j)_t = σ(C^(j)·r_{t-1} + A^(j)·z_{t-1} + b^(j))
        z_t = (X_{t-1}, log(V_{t-1}^2))

    '''

    # change stuff here to modify model:
    default_params = {
        'vol_drift': 'esn',  # 'canon', 'heston', or 'esn'
        'vol_vol': 'heston',    # "
        # (random) reservoir parameters:
        'q': 5,  # dim of reservoir; must match shape[0] and shape[1] of C and
                 # and shape[0] of A, b below!
        'C1': np.random.normal(size=[5, 5]),  #
        'C2': np.random.normal(size=[5, 5]),  #
        'A1': np.random.normal(size=[5, 2]),  # random ESN weights
        'A2': np.random.normal(size=[5, 2]),  #
        'b1': np.random.normal(size=[5, 1]),  #
        'b2': np.random.normal(size=[5, 1]),  #
        'df_X': None, 'df_V': None,
        'res1': np.random.normal(size=[5, 1]),  # drift-of-vol reservoir
        'res2': np.random.normal(size=[5, 1]),  # vol-of-vol reservoir
        # model parameters:
        'K': 1,  # no. of regimes; only K=1 implemented
        'w0': None, 'w1': None, 'w2': None, 'w3': None, 'w4': None, 'w5': None,
        'v0': None, 'v1': None, 'v2': None, 'v3': None, 'v4': None, 'v5': None,
        'omega': None, 'alpha': None, 'xi': None,
        'nu': None, 'kappa': None,
        # innovation parameters:
        'df_X': None, 'tail_X': None, 'shape_X': None, 'skew_X': None,
        'df_V': None, 'tail_V': None, 'shape_V': None, 'skew_V': None,
        # jump parameters:
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None
        }

    def get_params(self):
        '''
        extract and cap parameters

        '''
        theta = {}
        theta_FX = {}
        theta_FV = {}

        # (1) Model parameters:
        # if drift-of-vol from ESN:
        if self.w0 is not None:
            theta['w0'] = self.w0
            theta['w1'] = self.w1
            theta['w2'] = self.w2
            theta['w3'] = self.w3
            theta['w4'] = self.w4
            theta['w5'] = self.w5
        # drift-of-vol from Canonical SV:
        if self.omega is not None:
            theta['omega'] = self.omega
            theta['alpha'] = np.clip(self.alpha, 0., 1.-1e-10)
        # if drift-of-vol from Heston:
        elif self.nu is not None:
            theta['nu'] = np.clip(self.nu, 0., None)
            theta['kappa'] = np.clip(self.kappa, 0., 1.-1e-10)

        # if vol-of-vol from ESN:
        if self.v0 is not None:
            theta['v0'] = self.v0
            theta['v1'] = self.v1
            theta['v2'] = self.v2
            theta['v3'] = self.v3
            theta['v4'] = self.v4
            theta['v5'] = self.v5
        # if vol-of-vol from Canonical SV or Heston:
        elif self.xi is not None:
            theta['xi'] = np.clip(self.xi, 1e-20, None)

        # (2) Innovation parameters
        # Return innovations
        # Student t
        if self.df_X is not None:
            self.innov_X = 't'
            df_X = self.df_X
            df_X = np.clip(df_X, 2.+1e-10, None)
            theta_FX['df'] = df_X
        # Gen. Hyperbolic
        elif self.shape_X is not None:
            self.innov_X = 'GH'
            tail_X = self.tail_X
            tail_X = np.clip(tail_X, -50., 50.)
            skew_X = self.skew_X
            skew_X = np.clip(skew_X, -50., 50.)
            shape_X = self.shape_X
            shape_X = np.clip(shape_X, abs(skew_X)+1e-3, 50.+1e-3)
            theta_FX['tail'] = tail_X
            theta_FX['skew'] = skew_X
            theta_FX['shape'] = shape_X
        # Gaussian
        else:
            self.innov_X = 'N'

        # Volatility innovations
        # Student t
        if self.df_V is not None:
            self.innov_V = 't'
            df_V = self.df_V
            df_V = np.clip(df_V, 2.+1e-10, None)
            theta_FV['df'] = df_V
        # Gen. Hyperbolic
        elif self.shape_V is not None:
            self.innov_V = 'GH'
            tail_V = self.tail_V
            tail_V = np.clip(tail_V, -50., 50.)
            skew_V = self.skew_V
            skew_V = np.clip(skew_V, -50., 50.)
            shape_V = self.shape_V
            shape_V = np.clip(shape_V, abs(skew_V)+1e-3, 50.+1e-3)
            theta_FV['tail'] = tail_V
            theta_FV['skew'] = skew_V
            theta_FV['shape'] = shape_V
        # Gaussian
        else:
            self.innov_V = 'N'

        # (3) Jump parameters
        # Jumps in returns
        if self.lambda_X is not None:
            theta['lambda_X'] = np.clip(self.lambda_X, 0., 1.)
            theta['phi_X'] = np.clip(self.phi_X, 0., None)
        # Jumps in volatility
        if self.lambda_V is not None:
            theta['lambda_V'] = np.clip(self.lambda_V, 0., 1.)
            theta['phi_V'] = np.clip(self.phi_V, 0., None)

        return theta, theta_FX, theta_FV

    def update_res(self, logV2_prev=None, X_prev=None):
        '''
        update the reservoir given the new observations of the input variables
        (return and log-variance)

        Parameters
        ----------
        component: string
            'drift' or 'diff'.
        logV2_prev: (Nx,) array
            previous log-variance of each x-particle.
        X_prev: float
            previous return.

        '''
        if self.vol_drift == 'esn':
            # update from previous reservoir
            h1 = np.einsum('qp,qM->qM', self.C1, self.res1)  # (q,Nx) or (q,M)

            # update from previous input variables (log-variance and return)
            X_prev = np.full(len(logV2_prev), X_prev)
            M = np.vstack([logV2_prev, X_prev])  # (2,Nx) or (2,M)
            h1 = h1 + np.einsum('qI,IM->qM', self.A1, M)  # (q,Nx) or (q,M)

            self.res1 = sigmoid(h1 + self.b1)  # (q,Nx) or (q,M)

        if self.vol_vol == 'esn':
            # update from previous reservoir
            h2 = np.matmul(self.C2, self.res2)  # (q,)
            h2 = h2.reshape(self.q, 1)  # (q,1)

            # update from previous input variables (log-variance and return)
            X_prev = np.full(len(logV2_prev), X_prev)
            M = np.vstack([logV2_prev, X_prev])  # (2,Nx) or (2,M)
            h2 = h2 + np.matmul('qI,IM->qM', self.A2, M)  # (q,Nx) or (q,M)

            self.res2 = sigmoid(h1 + h2 + self.b2)  # (q,Nx) or (2,M)

    def EXt(self, theta, logV2_prev=None, X_prev=None, res=None, k=None):

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0
        if X_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_drift == 'esn':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)
                w = np.array(w)  # (q,)
                w = w.reshape(self.q, 1)  # (q,1)
                res = self.res1
                return np.sum(w * res, axis=0) + theta['w0']

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
        # and res is reservoir from different x-particles of different PFs
        else:
            M = len(theta)
            if self.vol_drift == 'esn':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)
                w = np.array(w)  # (q,M)
                return np.sum(w * res, axis=0) + theta['w0']

            elif self.vol_drift == 'canon':
                omega = theta['omega']
                alpha = theta['alpha']
                return omega * (1.-alpha) + alpha * logV2_prev

            else:  # Heston
                kappa = theta['kappa']
                nu = theta['nu']
                return (logV2_prev + kappa * (nu * np.exp(-logV2_prev) - 1.)
                        - 0.5*xi**2 * np.exp(-logV2_prev))

    def SDXt(self, theta, logV2_prev=None, X_prev=None, res=None, k=None):

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0.0
        if X_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_vol == 'neural':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)  # (q,)
                v = np.array(v)
                res = self.res2
                logxi2 = np.sum(v * res, axis=0) + theta['v0']
                return np.exp(0.5*logxi2)

            elif self.vol_vol == 'canon':
                return theta['xi']

            else:  # Heston
                return theta['xi'] * np.exp(-0.5*logV2_prev)

        # during prediction, theta is structured array with all N particles
        # and res is reservoir from different x-particles of different PFs
        else:
            if self.vol_vol == 'neural':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)
                v = np.array(v)  # (q,M)
                logxi2 = np.sum(v * res, axis=0) + theta['v0']
                return np.exp(0.5*logxi2)

            elif self.vol_vol == 'canon':
                return theta['xi']

            else:  # Heston
                return theta['xi'] * np.exp(-0.5*logV2_prev)

    def PX0(self):
        theta, _, theta_FV = self.get_params()

        return F_innov(self.EXt(theta),
                       self.SDXt(theta),
                       **theta_FV,
                       a=-10., b=10.)

    def PX(self, t, logV2_prev, X_prev):
        self.update_res(logV2_prev, X_prev)
        theta, _, theta_FV = self.get_params()

        return F_innov(self.EXt(theta, logV2_prev, X_prev=X_prev),
                       self.SDXt(theta, logV2_prev, X_prev=X_prev),
                       **theta_FV,
                       a=-10., b=10.)

    def PY(self, t, logV2):
        _, theta_FX, _ = self.get_params()
        return F_innov(0., np.exp(0.5*logV2), **theta_FX)