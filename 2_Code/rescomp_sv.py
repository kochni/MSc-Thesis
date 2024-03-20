#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel
from particles.distributions import *

# signatures
from esig import stream2sig, stream2logsig, sigdim, logsigdim

from operator import itemgetter
from helpers import *


class ResCompSV(StateSpaceModel):
    '''
    Reservoir computers for stochastic volatility models:

        log(V_t^2) = μ_t + ξ_t·U_t
        (μ_t, ξ_t) = W·r_t

    where r_t is the reservoir, determined by the variant

    '''

    # model parameters
    innov_X = 'N'
    innov_V = 'N'
    jumps_X = False
    jumps_V = False
    leverage = False
    K = 1
    switching = None

    # hyperparamaters
    vol_drift = 'rescomp'
    vol_vol = 'canonical'
    q = 5
    sd = 1.0
    sig_len = None

    default_params = {
        # model parameters:
        'w0': None, 'w1': None, 'w2': None, 'w3': None, 'w4': None, 'w5': None,
        'w6': None, 'w7': None, 'w8': None, 'w9': None, 'w10': None,
        'v0': None, 'v1': None, 'v2': None, 'v3': None, 'v4': None, 'v5': None,
        'v6': None, 'v7': None, 'v8': None, 'v9': None, 'v10': None,
        'omega': None, 'alpha': None, 'xi': None,
        'nu': None, 'kappa': None,
        'rho': None,
        # innovation parameters:
        'df_X': None, 'tail_X': None, 'shape_X': None, 'skew_X': None,
        'df_V': None, 'tail_V': None, 'shape_V': None, 'skew_V': None,
        # jump parameters:
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None
        }

    # seed for keeping random matrices constant within run
    seed = np.random.randint(10000)

    @classmethod
    def define_model(cls, vol_drift, vol_vol, q, sd, activ,
                     leverage, innov_X, innov_V, jumps_X, jumps_V):
        ''' called once when defining model '''
        cls.K = 1
        cls.vol_drift = vol_drift
        cls.vol_vol = vol_vol
        cls.q = q
        cls.sd = sd
        # cls.activ = activ
        cls.leverage = leverage
        cls.innov_X = innov_X
        cls.innov_V = innov_V
        cls.jumps_X = jumps_X
        cls.jumps_V = jumps_V

    def get_params(self):
        ''' extract and cap parameters '''
        theta = {}
        theta_FX = {}
        theta_FV = {}

        # (1) Model parameters:
        # if drift-of-vol from ResComp:
        if self.vol_drift == 'rescomp':
            for j in range(self.q+1):
                theta['w' + str(j)] = getattr(self, f"w{j}")
        elif self.vol_drift == 'canonical':
            theta['omega'] = self.omega
            theta['alpha'] = np.clip(self.alpha, 0., 0.99)
        elif self.vol_drift == 'heston':
            theta['nu'] = np.clip(self.nu, 0., 50.)
            theta['kappa'] = np.clip(self.kappa, 0., 50.)

        # if vol-of-vol from ResComp:
        if self.vol_vol == 'rescomp':
            for j in range(self.q+1):
                theta['v' + str(j)] = getattr(self, f"w{j}")
        else:
            theta['xi'] = np.clip(self.xi, 0.01, 20.)

        # (2) Innovation parameters
        # Return innovations
        if self.innov_X == 't':
            df_X = self.df_X
            df_X = np.clip(df_X, 3.0, 500.)
            theta_FX['df'] = df_X
        elif self.innov_X == 'GH':
            tail_X = self.tail_X
            tail_X = np.clip(tail_X, -50., 50.)
            skew_X = self.skew_X
            skew_X = np.clip(skew_X, -50., 50.)
            shape_X = self.shape_X
            shape_X = np.clip(shape_X, abs(skew_X)+0.01, 50.01)
            theta_FX['tail'] = tail_X
            theta_FX['skew'] = skew_X
            theta_FX['shape'] = shape_X

        # Volatility innovations
        if self.innov_V == 't':
            df_V = self.df_V
            df_V = np.clip(df_V, 3.0, 500.)
            theta_FV['df'] = df_V
        elif self.innov_V == 'GH':
            tail_V = self.tail_V
            tail_V = np.clip(tail_V, -50., 50.)
            skew_V = self.skew_V
            skew_V = np.clip(skew_V, -50., 50.)
            shape_V = self.shape_V
            shape_V = np.clip(shape_V, abs(skew_V)+0.01, 50.01)
            theta_FV['tail'] = tail_V
            theta_FV['skew'] = skew_V
            theta_FV['shape'] = shape_V

        # (3) Jump parameters
        # Jumps in returns
        if self.jumps_X is True:
            theta['lambda_X'] = np.clip(self.lambda_X, 0., 1.)
            theta['phi_X'] = np.clip(self.phi_X, 0., 1e20)
        # Jumps in volatility
        if self.jumps_V is True:
            theta['lambda_V'] = np.clip(self.lambda_V, 0., 1.)
            theta['phi_V'] = np.clip(self.phi_V, 0., 50.)

        return theta, theta_FX, theta_FV


    def EXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' E[log(V_t^2) | log(V_{t-1}^2)] '''

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0
        if X_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_drift == 'rescomp':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)  # (q,)
                w = np.array(w)
                w = w.reshape(self.q, 1)  # (q,1)
                EXt = np.sum(w * self.res, axis=0) + theta['w0']

            elif self.vol_drift == 'canonical':
                omega = theta['omega']
                alpha = theta['alpha']
                EXt = (1.-alpha) * omega + alpha * logV2_prev

            elif self.vol_drift == 'heston':
                kappa = theta['kappa']
                nu = theta['nu']
                EXt = (logV2_prev + kappa*(nu * np.exp(-logV2_prev) - 1.)
                        - 0.5*xi**2 * np.exp(-logV2_prev))

        # during prediction, theta is structured array with all N particles
        # and res is reservoir from different x-particles of different PFs
        else:
            M = len(theta)
            if self.vol_drift == 'rescomp':
                w_names = ['w' + str(j) for j in range(1, self.q+1)]
                w = itemgetter(*w_names)(theta)
                w = np.array(w)  # (q,M)
                EXt = np.sum(w * res, axis=0) + theta['w0']

            elif self.vol_drift == 'canonical':
                omega = theta['omega']
                alpha = theta['alpha']
                EXt = omega * (1.-alpha) + alpha * logV2_prev

            elif self.vol_drift == 'heston':
                kappa = theta['kappa']
                nu = theta['nu']
                EXt = (logV2_prev + kappa * (nu * np.exp(-logV2_prev) - 1.)
                        - 0.5*xi**2 * np.exp(-logV2_prev))

        # Leverage effect shifts and scales latent volatility distribution
        if self.leverage is True:
            Z_prev = X_prev * np.exp(-0.5*logV2_prev)
            EXt += theta['rho'] * Z_prev

        EXt = np.clip(EXt, -20., 20.)
        return EXt

    def SDXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' SD(log(V_t^2) | log(V_{t-1}^2)) '''

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0.0
        if X_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.vol_vol == 'rescomp':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)  # (q,)
                v = np.array(v)
                logxi2 = np.sum(v * self.res, axis=0) + theta['v0']
                SDXt = np.exp(0.5*logxi2)

            elif self.vol_vol == 'canonical':
                SDXt = theta['xi']

            elif self.vol_vol == 'heston':
                SDXt = theta['xi'] * np.exp(-0.5*logV2_prev)

        # during prediction, theta is structured array with all N particles
        # and res is reservoir from different x-particles of different PFs
        else:
            if self.vol_vol == 'rescomp':
                v_names = ['v' + str(j) for j in range(1, self.q+1)]
                v = itemgetter(*v_names)(theta)
                v = np.array(v)  # (q,M)
                logxi2 = np.sum(v * res, axis=0) + theta['w0']
                SDXt = np.exp(0.5*logxi2)

            elif self.vol_vol == 'canonical':
                SDXt = theta['xi']

            elif self.vol_vol == 'heston':
                SDXt = theta['xi'] * np.exp(-0.5*logV2_prev)

        # Leverage effect: shifts and scales distribution
        if self.leverage is True:
            Z_prev = X_prev * np.exp(-0.5*logV2_prev)
            SDXt = SDXt * np.sqrt(1-theta['rho']**2)
            if self.innov_X == 't':
                SDXt = SDXt * np.sqrt((theta['df'] + Z_prev**2) / (theta['df'] + 1))

        SDXt = np.clip(SDXt, 0.01, 20.)
        return SDXt

    def PX0(self, X0):
        ''' π(log(V_0^2)) '''

        _, _, theta_FV = self.get_params()

        return F_innov(np.log(X0**2), np.array(1.0), **theta_FV, a=-20., b=20.)

    def PX(self, t, logV2_prev, returns):
        ''' π(log(V_t^2) | log(V_{t-1}^2)) '''

        logV2_prev = np.clip(logV2_prev, -20., 20.)
        theta, _, theta_FV = self.get_params()
        self.update_reservoir(returns, logV2_prev)

        return F_innov(self.EXt(theta, logV2_prev, returns),
                       self.SDXt(theta, logV2_prev, returns),
                       **theta_FV,
                       a=-20., b=20.)

    def PY(self, t, logV2):
        ''' π(X_t | V_t) '''

        logV2 = np.clip(logV2, -20., 20.)
        _, theta_FX, _ = self.get_params()

        return F_innov(0., np.exp(0.5*logV2), **theta_FX)

    # methods for Guided PF:

    def proposal0(self, X0):
        ''' ψ(logV2_0) '''

        _, _, theta_FV = self.get_params()

        return F_innov(np.log(X0**2), np.array(1.0), **theta_FV, a=-20., b=20.)

    def proposal(self, t, logV2_prev, returns):
        ''' ψ(logV2_t | logV2_{t-1}, X_t) '''

        logV2_prev = np.clip(logV2_prev, -20., 20.)
        theta, _, theta_FV = self.get_params()
        EXt = self.EXt(theta, logV2_prev, returns[t-1])
        SDXt = self.SDXt(theta, logV2_prev, returns[t-1])
        mu_star = EXt + 0.25*SDXt**2 * (returns[t]**2 * np.exp(-EXt) - 2.0)
        mu_star = np.clip(mu_star, -20., 20.)

        return F_innov(mu_star, SDXt, **theta_FV, a=-20., b=20.)


class ExtremeSV(ResCompSV):
    '''
    Extreme Learning Machine:
        r_t = ϕ(C·z_t + b)
        z_t = (X_{t-1}, log(V_{t-1}^2))

    '''

    @classmethod
    def generate_matrices(cls):

        # update seed to get different matrices across runs
        cls.seed += 1
        np.random.seed(cls.seed)

        cls.A = np.random.normal(scale=cls.sd, size=[cls.q, 2])
        cls.b = np.random.normal(scale=cls.sd, size=[cls.q, 1])

        # random initial reservoir:
        cls.res = np.random.normal(scale=cls.sd, size=[cls.q, 1])

    def update_reservoir(self, returns, logV2_prev):
        # all reservoirs of shape (q,Nx)

        t = len(returns)
        Nx = len(logV2_prev)

        # input = previous log-variance, previous return
        X_prev = np.full(Nx, returns[t-1])
        z = np.vstack([X_prev, logV2_prev])  # (2,Nx)
        h = np.matmul(self.A, z) + self.b  # (q,Nx)
        self.res = shi(h)


class EchoSV(ResCompSV):
    '''
    Echo State Network:
        r_t = ϕ(A·r_{t-1} + C·z_t + b)
        z_t = (X_{t-1}, log(V_{t-1}^2))

    '''

    @classmethod
    def generate_matrices(cls):

        # update seed to get different matrices across runs
        cls.seed += 1
        np.random.seed(cls.seed)

        cls.A = np.random.normal(scale=cls.sd, size=[cls.q, cls.q])
        cls.C = np.random.normal(scale=cls.sd, size=[cls.q, 2])
        cls.b = np.random.normal(scale=cls.sd, size=[cls.q, 1])

        spec_rad_A = max(abs(np.linalg.eigvals(cls.A)))
        cls.A = cls.A / (spec_rad_A + 0.1)
        cls.C = cls.C / (spec_rad_A + 0.1)
        cls.b = cls.b / (spec_rad_A + 0.1)

        # random initial reservoir:
        cls.res = np.random.normal(scale=cls.sd, size=[cls.q, 1])

    def update_reservoir(self, returns, logV2_prev):
        t = len(returns)
        Nx = len(logV2_prev)

        # recurrent update
        h = np.matmul(self.A, self.res)  # (q,Nx)

        # update from input variables (log-variance and return)
        X_prev = np.full(Nx, returns[t-1])
        z = np.vstack([X_prev, logV2_prev])  # (2,Nx)
        h = h + np.matmul(self.C, z)  # (q,Nx)
        h = h + self.b
        self.res = shi(h)


class BarronSV(ResCompSV):
    '''
    Barron Functional:
        r_t = ϕ(C·u_t + b)
        u_t = ϕ(A·u_{t-1} + C·z_t + b)
        z_t = (X_{t-1}, log(V_{t-1}^2))

    '''
    pass


class SigSV(ResCompSV):
    '''
    Signature:
        r_t = 𝕊(u_{0:t-1}})
        u_t = (t, X_t)

    '''

    @classmethod
    def generate_matrices(cls):
        # random initial reservoir:
        cls.res = np.random.normal(scale=cls.sd, size=[cls.q, 1])

        cls.sig_len = 1
        while logsigdim(2, cls.sig_len) < cls.q:
            cls.sig_len += 1

    def update_reservoir(self, returns, logV2_prev):
        t = len(returns)
        Nx = len(logV2_prev)

        # input path: time-extended path of past returns
        time = np.arange(t)
        u = np.stack([time, returns]).T # (t,2)
        res = stream2logsig(u, self.sig_len)[0:self.q]  # (q,)
        res = res.reshape(self.q, 1)
        self.res = np.tile(res, [1, Nx])  # (q,Nx)


class RandSigSV(ResCompSV):
    '''
    Randomized Signature:

        r_t = 𝕊_{t-1}
        𝕊_t = 𝕊_{t-1} + Σ_j ϕ(A_j·𝕊_t + b_j)·(u_t^{(j)} - u_{t-1}^{(j)})
        u_t = (t, X_t, log(V_t^2))

    '''

    @classmethod
    def generate_matrices(cls):

        # update seed to get different matrices across runs
        cls.seed += 1
        np.random.seed(cls.seed)

        cls.A1 = np.random.normal(scale=cls.sd, size=[cls.q, cls.q])
        cls.b1 = np.random.normal(scale=cls.sd, size=[cls.q, 1])
        cls.A2 = np.random.normal(scale=cls.sd, size=[cls.q, cls.q])
        cls.b2 = np.random.normal(scale=cls.sd, size=[cls.q, 1])
        cls.A3 = np.random.normal(scale=cls.sd, size=[cls.q, cls.q])
        cls.b3 = np.random.normal(scale=cls.sd, size=[cls.q, 1])
        cls.logV2_pp = 0.0

        # random initial reservoir:
        cls.res = np.random.normal(scale=cls.sd, size=[cls.q, 1])

    def update_reservoir(self, returns, logV2_prev):
        t = len(returns)
        Nx = len(logV2_prev)

        # increments of control variables:
        if t == 1:
            X_incr = returns[0]
        else:
            X_incr = returns[t-1] - returns[t-2]
        logV2_incr = logV2_prev - self.logV2_pp
        self.logV2_pp = logV2_prev

        # update of rand-Sig:
        h1 = np.matmul(self.A1, self.res) + self.b1
        h1 = sigmoid(h1)

        h2 = np.matmul(self.A2, self.res) + self.b2
        h2 = sigmoid(h2) * X_incr

        h3 = np.matmul(self.A3, self.res) + self.b3
        h3 = sigmoid(h3) * logV2_incr

        self.res = self.res + h1 + h2 + h3  # (q,Nx)