#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel
from particles.distributions import *


class PWConstSV(StateSpaceModel):
    '''
    Stochastic volatility model with (piecewise) constant volatility:

        V_t = V_{t-1} + J_t

    '''
    K = 1
    leverage = False
    innov_X = 'N'
    innov_V = 'N'
    jumps_X = False
    jumps_V = True

    default_params = {
        # innovation parameters:
        'rho': None,
        'df_X': None, 'tail_X': None, 'shape_X': None, 'skew_X': None,
        'df_V': None, 'tail_V': None, 'shape_V': None, 'skew_V': None,
        # jump parameters:
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None
        }

    @classmethod
    def define_model(cls, leverage, innov_X, innov_V, jumps_X, jumps_V, regimes,
                     switching=None):
        ''' called once when defining model '''

        if leverage is True:
            err_msg = "Can't have GenHyp innovations when using leverage"
            assert innov_X != 'GH', err_msg
            err_msg = "Must have same innovation distributions when using leverage"
            assert innov_X == innov_V, err_msg

        cls.K = regimes
        cls.leverage = leverage
        cls.innov_X = innov_V
        cls.innov_X = innov_V
        cls.jumps_X = jumps_X
        cls.jumps_V = jumps_V
        cls.switching = switching

    def get_params(self):
        ''' extract and cap parameters '''
        theta = {}
        theta_FX = {}
        theta_FV = {}

        # (1) Model parameters:
        if self.jumps_V is True:
            theta['lambda_V'] = np.clip(self.lambda_V, 0., 1.)
            theta['phi_V'] = np.clip(self.phi_V, 0., 50.)
        if self.leverage is True:
            theta['rho'] = np.clip(self.rho, -0.99, 0.99)

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
            theta['phi_X'] = np.clip(self.phi_X, 0., 1e10)

        return theta, theta_FX, theta_FV

    def EXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' E[log(V_t^2) | log(V_{t-1}^2)] '''

        EXt = logV2_prev
        EXt = np.clip(EXt, -20., 20.)
        return EXt

    def SDXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' SD(log(V_t^2) | log(V_{t-1}^2)) '''

        return np.full(logV2_prev.shape, 1e-8)
        # scale=0 would cause error in truncated distributions

    def PX0(self, X0):
        ''' π(log(V_0^2)) '''

        _, _, theta_FV = self.get_params()

        return F_innov(np.log(X0**2), np.array(1.0), **theta_FV, a=-20., b=20.)

    def PX(self, t, logV2_prev, returns):
        ''' π(log(V_t^2) | log(V_{t-1}^2)) '''

        theta, _, theta_FV = self.get_params()
        logV2_prev = np.clip(logV2_prev, -20., 20.)

        lambda_V = theta['lambda_V']
        phi_V = theta['phi_V']

        return Mixture([1.-lambda_V, lambda_V],
                       Dirac(loc=logV2_prev),
                       F_innov(logV2_prev, phi_V, **theta_FV, a=-20., b=20.))

    def PY(self, t, logV2):
        ''' π(X_t | V_t) '''

        theta, theta_FX, _ = self.get_params()
        logV2 = np.clip(logV2, -20., 20.)

        if self.jumps_X is False:
            return F_innov(0., np.exp(0.5*logV2), **theta_FX)

        else:
            sd_jump = np.sqrt(np.exp(logV2) + theta['phi_X']**2)
            return Mixture([1.-theta['lambda_X'], theta['lambda_X']],
                           F_innov(0., np.exp(0.5*logV2), **theta_FX),
                           F_innov(0., sd_jump, **theta_FX)
                           )


# Extension based on Gen. Hyperbolic innovations:

class PWConstGH(PWConstSV):

    innov_V = 'GH'