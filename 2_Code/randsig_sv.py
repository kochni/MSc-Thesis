#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class RandSigSV(StateSpaceModel):
    '''
    Randomized Signature stochastic volatility:

        log(V_t^2) = w·r^(1)_t + v·r^(2)_t + w0
        r^(j)_t = σ(C^(j)·r_{t-1} + A^(j)·z_{t-1} + b^(j))
        z_t = (X_{t-1}, log(V_{t-1}^2))

    '''

    # change stuff here to modify model:
    default_params = {
        'vol_drift': 'rsig',  # 'canon', 'heston', or 'randsig'
        'vol_vol': 'heston',     # "
        # (random) reservoir parameters:
        'q': 5,  # dim of randomized signature; must match shape[0] and
                 # shape[1] of C and and shape[0] of A, b below!
        'A1': np.random.normal(size=[5, 2]),  #
        'A2': np.random.normal(size=[5, 2]),  #
        'A3': np.random.normal(size=[5, 2]),  # random projections
        'b1': np.random.normal(size=[5, 1]),  #
        'b2': np.random.normal(size=[5, 1]),  #
        'b3': np.random.normal(size=[5, 1]),  #
        'df_X': None, 'df_V': None,
        'rsig1': np.random.normal(size=[5, 1]),  # drift-of-vol reservoir
        'rsig2': np.random.normal(size=[5, 1]),  # vol-of-vol reservoir
        'logV2_prev': None,  # log(V_t^2) - log(V_{t-1}^2), to update rSig
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
        # if drift-of-vol from rSig:
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

        # if vol-of-vol from rSig:
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

    def update_rsig(self, logV2_incr=None, X_incr=None):
        '''
        update the randomized signature (randomized signature) given the new
        observations of the input variables (increment of return and
        log-variance)

        S_t = S_{t-1} + ϕ(A1·S_{t-1} + b1)
                      + ϕ(A2·S_{t-1} + b2) · (X_t - X_{t-1})
                      + ϕ(A3·S_{t-1} + b3) · (log(V_t^2) - log(V_{t-1}^2))

        Parameters
        ----------
        component: string
            'drift' or 'diff'.
        logV2_incr: (Nx,) array
            log(V_t^2) - log(V_{t-1}^2).
        X_incr: float
            X_t - X_{t-1}.

        '''
        # update based on random projection of previous value and
        # increments of input variables

        if self.vol_drift == 'rsig':
            h1 = np.einsum(self.A1, self.rsig1) + self.b1
            h1 = sigmoid(h1)

            h2 = np.einsum(self.A2, self.rsig1) + self.b2
            h2 = sigmoid(h2) * X_incr

            h3 = np.einsum(self.A3, self.rsig1) + self.b2
            h3 = sigmoid(h3) * logV2_incr

            self.rsig1 = self.rsig1 + h1 + h2 + h3  # (q,Nx) or (q,M)

        if self.vol_vol == 'rsig':
            h1 = np.einsum(self.A1, self.rsig2) + self.b1
            h1 = sigmoid(h1)

            h2 = np.einsum(self.A2, self.rsig2) + self.b2
            h2 = sigmoid(h2) * X_incr

            h3 = np.einsum(self.A3, self.rsig2) + self.b2
            h3 = sigmoid(h3) * logV2_incr

            self.rsig2 = self.rsig2 + h1 + h2 + h3  # (q,Nx) or (q,M)

    def EXt(self, theta, logV2_prev=None, X_prev=None, rsig=None, k=None):

        # at t=0, logV2_prev is None, use logV2_0 = 0.0
        if logV2_prev is None:
            logV2_prev = np.array([0.0])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_drift == 'rsig':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)
                w = np.array(w)  # (q,)
                w = w.reshape(self.q, 1)  # (q,1)
                rsig = self.rsig1
                return np.sum(w * rsig, axis=0) + theta['w0']

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
            if self.vol_drift == 'rsig':
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

    def SDXt(self, theta, logV2_prev=None, X_prev=None, rsig=None, k=None):

        # at t=0, logV2_prev is None, use logV2_0 = 0.0
        if logV2_prev is None:
            logV2_prev = np.array([0.0])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_vol == 'rsig':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)  # (q,)
                v = np.array(v)
                rsig = self.rsig2
                logxi2 = np.sum(v * rsig, axis=0) + theta['v0']
                return np.exp(0.5*logxi2)

            elif self.vol_vol == 'canon':
                return theta['xi']

            else:  # Heston
                return theta['xi'] * np.exp(-0.5*logV2_prev)

        # during prediction, theta is structured array with all N particles
        # and res is reservoir from different x-particles of different PFs
        else:
            if self.vol_vol == 'rsig':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)
                v = np.array(v)  # (q,M)
                logxi2 = np.sum(v * rsig, axis=0) + theta['v0']
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

    def PX(self, t, logV2, data):
        # at t=0, use logV2_incr = logV2
        print("logV2_prev:", self.logV2_prev)
        if t == 1:
            logV2_incr = logV2
        else:
            logV2_incr = logV2 - self.logV2_prev

        X_incr = data[t] - data[t-1]
        self.update_rsig(logV2_incr, X_incr)
        self.logV2_prev = logV2
        print("logV2_prev:", self.logV2_prev)
        theta, _, theta_FV = self.get_params()

        return F_innov(self.EXt(theta),
                       self.SDXt(theta),
                       **theta_FV,
                       a=-10., b=10.)

    def PY(self, t, logV2):
        _, theta_FX, _ = self.get_params()
        return F_innov(0., np.exp(0.5*logV2), **theta_FX)