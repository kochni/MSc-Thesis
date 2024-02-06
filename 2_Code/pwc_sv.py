#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class PWConstSV(StateSpaceModel):
    '''
    Stochastic volatility model with (piecewise) constant volatility:

        V_t = V_{t-1} + J_t

    Note: only interesting if jumps in volatility specified, else use DetVol

    '''

    default_params = {
        'df': None, 'df_X': None, 'df_V': None,
        'tail': None, 'tail_X': None, 'tail_V': None,
        'shape': None, 'shape_X': None, 'shape_V': None,
        'skew': None, 'skew_X': None, 'skew_V': None,
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None,
        }
    predictions = None
    predsets = None

    def PX0(self):
        theta, _, theta_FV = self.get_params()
        return F_innov(0., theta['phi_V'], **theta_FV)

    def PX(self, t, xp):
        theta, _, theta_FV = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        return Mixture([1.-theta['lambda_V'], theta['lambda_V']],
                       Dirac(loc=xp),
                       F_innov(xp, theta['phi_V'], **theta_FV)
                       )

    def PY(self, t, xp, x, data):
        theta, theta_FX, _ = self.get_params()
        x = np.clip(x, -50., 50.)

        # produce point predictions & prediction sets
        self.predict(x)

        if theta['phi_X'] is None:
            return F_innov(0., np.exp(0.5*x), **theta_FX)

        else:
            sd_jump = np.sqrt(np.exp(x) + theta['phi_X']**2)
            return Mixture([1.-theta['lambda_X'], theta['lambda_X']],
                           F_innov(0., np.exp(0.5*x), **theta_FX),
                           F_innov(0., sd_jump, **theta_FX)
                           )

    # methods for Guided PF:

    def _xhat(self, xst, s, yt):
        pass

    def proposal0(self, data):
        return F_innov(df=5)

    def proposal(self, t, xp, data):
        return F_innov(mean=xp, df=5)