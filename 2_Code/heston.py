#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel
from particles.distributions import *


class Heston(StateSpaceModel):
    '''
    Heston model:

    log(V_t^2) = log(V_{t-1}^2) + κ·(ν/V_{t-1}^2 - 1) - ξ^2/V_{t-1}
                 + ξ/V_{t-1}·U_t

    '''

    innov_X = 'N'
    innov_V = 'N'
    jumps_X = False
    jumps_V = False
    leverage = True
    K = 1
    switching = None

    default_params = {
        # model parameters:
        'nu': None, 'kappa': None, 'xi': None,
        'rho': None,
        # innovation parameters:
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
            err_msg = "Can't have Student t innovations when using jumps"
            if jumps_X: assert innov_X in ['N', 'GH'], err_msg
            if jumps_V: assert innov_V in ['N', 'GH'], err_msg

        cls.K = regimes
        cls.leverage = leverage
        cls.innov_X = innov_X
        cls.innov_V = innov_V
        cls.jumps_X = jumps_X
        cls.jumps_V = jumps_V
        cls.switching = switching

    def get_params(self):
        ''' extract and cap parameters '''
        theta = {}
        theta_FX = {}
        theta_FV = {}

        # (1) Model parameters:
        theta['nu'] = np.clip(self.nu, 0.01, 50.)
        theta['kappa'] = np.clip(self.kappa, 0.01, 10.)
        theta['xi'] = np.clip(self.xi, 0.01, 50.)
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
        # Jumps in volatility
        if self.jumps_V is True:
            theta['lambda_V'] = np.clip(self.lambda_V, 0., 1.)
            theta['phi_V'] = np.clip(self.phi_V, 0., 50.)

        return theta, theta_FX, theta_FV

    def EXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' E[log(V_t^2) | log(V_{t-1}^2)] '''

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0.0
        if logV2_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # very negative logV2_prev causes problems here:
        logV2_prev = np.clip(logV2_prev, -5., 20.)

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.K == 1:
                kappa = theta['kappa']
                nu = theta['nu']
                xi = theta['xi']

            else:
                if k == 0:
                    kappa = theta['kappa_0']
                    nu = theta['nu_0']
                    xi = theta['xi_0']
                else:
                    kappa = theta['kappa_1']
                    nu = theta['nu_1']
                    xi = theta['xi_1']

        # during prediction, theta is structured array with all N particles
        else:
            if self.K == 1:
                kappa = theta['kappa']
                nu = theta['nu']
                xi = theta['xi']

            else:
                M = len(theta)
                kappa = np.stack([theta['kappa_0'], theta['kappa_1']],
                                 axis=1)[np.arange(M), k]
                nu = np.stack([theta['nu_0'], theta['nu_1']],
                              axis=1)[np.arange(M), k]
                xi = np.stack([theta['xi_0'], theta['xi_1']],
                              axis=1)[np.arange(M), k]

        EXt = (logV2_prev
               + kappa * (nu*np.exp(-logV2_prev) - 1.0)
               - 0.5*xi**2 * np.exp(-logV2_prev))

        # Leverage effect shifts and scales latent volatility distribution
        if self.leverage is True:
            Z_prev = X_prev * np.exp(-0.5*logV2_prev)
            EXt += theta['rho'] * Z_prev

        EXt = np.clip(EXt, -20., 20.)
        return EXt

    def SDXt(self, theta, logV2_prev=None, X_prev=None, k=None, res=None):
        ''' SD(log(V_t^2) | log(V_{t-1}^2)) '''

        # at t=0 (kwargs are None), use X_prev = logV2_prev = 0
        if logV2_prev is None:
            X_prev = np.array([0.])
            logV2_prev = np.array([0.])

        # very negative logV2_prev causes problems here:
        logV2_prev = np.clip(logV2_prev, -5., 20.)

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if self.K == 1:
                xi = theta['xi']
            else:
                if k == 0:
                    xi = theta['xi_0']
                else:
                    xi = theta['xi_1']

        # during prediction, theta is structured array with all N particles
        else:
            M = len(theta)
            if self.K == 1:
                xi = theta['xi']
            else:
                xi = np.stack([theta['xi_0'], theta['xi_1']],
                              axis=1)[np.arange(M), k]

        SDXt = xi * np.exp(-0.5*logV2_prev)

        # Leverage effect: shifts and scales distribution
        if self.leverage is True:
            Z_prev = X_prev * np.exp(-0.5*logV2_prev)
            SDXt = SDXt * np.sqrt(1-theta['rho']**2)
            if self.innov_X == 't':
                SDXt = SDXt * (theta['df'] + Z_prev**2) / (theta['df'] + 1)

        SDXt = np.clip(SDXt, 0.01, 100.)
        return SDXt

    def PX0(self, X0):
        ''' π(log(V_0^2)) '''

        _, _, theta_FV = self.get_params()

        return F_innov(np.log(X0**2), np.array(1.0), **theta_FV, a=-20., b=20.)

    def PX(self, t, logV2_prev, returns):
        ''' π(log(V_t^2) | log(V_{t-1}^2)) '''

        theta, _, theta_FV = self.get_params()

        EXt = self.EXt(theta, logV2_prev, returns[t-1])
        SDXt = self.SDXt(theta, logV2_prev, returns[t-1])

        # mix over regimes and jumps:
        # single regime:
        if self.K == 1:
            # no jumps
            if self.jumps_V is False:
                return F_innov(EXt, SDXt, **theta_FV, a=-20., b=20.)
            # jumps
            else:
                dist_nojmp = F_innov(EXt, SDXt, **theta_FV, a=-20., b=20.)
                sd_jmp = np.sqrt(SDXt**2 + theta['phi_V']**2)
                dist_jmp = F_innov(EXt, sd_jmp, **theta_FV, a=-20., b=20.)
                lambda_V = theta['lambda_V']
                return Mixture([(1.-lambda_V), lambda_V], dist_nojmp, dist_jmp)

        # 2 regimes
        else:
            p_0 = theta['p_0']
            EXt_0 = self.EXt(theta, logV2_prev, X_prev, 0)
            EXt_1 = self.EXt(theta, logV2_prev, X_prev, 1)
            SDXt_0 = self.SDXt(theta, logV2_prev, X_prev, 0)
            SDXt_1 = self.SDXt(theta, logV2_prev, X_prev, 1)

            # no jumps
            if self.jumps_V is False:
                dist_0 = F_innov(EXt_0, SDXt_0, **theta_FV, a=-20., b=20.)
                dist_1 = F_innov(EXt_1, SDXt_1, **theta_FV, a=-20., b=20.)

                return Mixture([p_0, 1.-p_0], dist_0, dist_1)

            # jumps
            else:
                dist_0_nj = F_innov(EXt_0, EXt_1, **theta_FV, a=-20., b=20.)
                dist_1_nj = F_innov(EXt_1, SDXt_1, **theta_FV, a=-20., b=20.)

                lambda_V = theta['lambda_V']
                SD_jmp_0 = np.sqrt(SDXt_0**2 + theta['phi_V']**2)
                SD_jmp_1 = np.sqrt(SDX1_1**2 + theta['phi_V']**2)
                dist_0_j = F_innov(EX1_0, SD_jmp_0, **theta_FV, a=-20., b=20.)
                dist_1_j = F_innov(EXt_1, SD_jmp_1, **theta_FV, a=-20., b=20.)

                return Mixture([p_0*(1.-lambda_V), p_1*(1.-lambda_V),
                                p_0*lambda_V, p_1*lambda_V],
                               dist_0_nj, dist_1_nj, dist_0_j, dist_1_j)

    def PY(self, t, logV2):
        ''' π(X_t | V_t) '''

        theta, theta_FX, _ = self.get_params()
        logV2 = np.clip(logV2, -20., 20.)

        # regimes fixed at this point, hence only mix over jumps
        # no jumps:
        if self.jumps_X is False:
            return F_innov(0., np.exp(0.5*logV2), **theta_FX)

        # jumps:
        else:
            dist_nojmp = F_innov(0., np.exp(0.5*logV2), **theta_FX)
            sd_jmp = np.sqrt(np.exp(logV2) + theta['phi_X']**2)
            dist_jmp = F_innov(0., sd_jmp, **theta_FX)
            lambda_X = theta['lambda_X']
            return Mixture([(1.-lambda_X), lambda_X], dist_nojmp, dist_jmp)

    # methods for Guided PF:

    def proposal0(self, X0):
        ''' ψ(logV2_0) '''

        _, _, theta_FV = self.get_params()

        return F_innov(np.log(X0**2), np.array(1.0), **theta_FV, a=-20., b=20.)

    def proposal(self, t, logV2_prev, returns):
        ''' ψ(logV2_t | logV2_{t-1}, X_t) '''

        theta, _, theta_FV = self.get_params()
        EXt = self.EXt(theta, logV2_prev, returns[t-1])
        SDXt = self.SDXt(theta, logV2_prev, returns[t-1])
        mu_star = EXt + 0.25*SDXt**2 * (returns[t]**2 * np.exp(-EXt) - 2.0)
        mu_star = np.clip(mu_star, -20., 20.)

        return F_innov(mu_star, SDXt, **theta_FV, a=-20., b=20.)


class Bates(Heston):

    jumps_X = True


class BNS(Heston):

    jumps_V = True


