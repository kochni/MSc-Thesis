#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class CanonSV:
    '''
    Canonical stochastic volatility model:

        log(V_t^2) = ω·(1-α) + α·log(V_{t-1}^2) + ξ·Z_t

    '''

    def EXt(self, theta, logV2_prev, k=None, X_prev=None):
        '''
        expectation of log(V_t^2) conditional on regime, parameters, and
        previous volatility

        E[log(V_t^2)] = ω·(1-α) + α·log(V_{t-1}^2)

        '''
        x = np.clip(x, -50., 50.)

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:

            if 'omega' in theta.keys():
                omega = theta['omega']
                alpha = theta['alpha']

            elif 'omega_0' in theta.keys():
                if k == 0:
                    omega = theta['omega_0']
                    alpha = theta['alpha_0']
                else:
                    omega = theta['omega_1']
                    alpha = theta['alpha_1']

            return omega * (1.0-alpha) + alpha * x

        # during prediction, theta is structured array with all N particles
        else:
            if 'omega' in theta.dtype.names:
                omega = theta['omega']
                alpha = theta['alpha']

            elif 'omega_0' in theta.dtype.names:
                M = len(theta)
                omega = np.stack([theta['omega_0'], theta['omega_1']],
                                 axis=1)[np.arange(M), k]
                alpha = np.stack([theta['alpha_0'], theta['alpha_1']],
                                 axis=1)[np.arange(M), k]

            return omega * (1.0-alpha) + alpha * x

    def SDXt(self, theta, logV2_prev, k=None, X_prev=None):
        '''
        standard deviation of log(V_t^2) conditional on regime, parameter, and
        previous volatility

        Var(log(V_t^2)) = ξ^2

        '''
        x = np.clip(x, -50., 50.)

        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if 'xi' in theta.keys():
                xi = theta['xi']
            elif 'xi_0' in theta.keys():
                if k == 0:
                    xi = theta['xi_0']
                else:
                    xi = theta['xi_1']

            return xi

        # during prediction, theta is structured array with all N particles
        else:
            M = len(theta)
            if 'xi' in theta.dtype.names:
                xi = theta['xi']
            elif 'xi_0' in theta.dtype.names:
                xi = np.stack([theta['xi_0'], theta['xi_1']],
                              axis=1)[np.arange(M), k]

            return xi

    def PX0(self, theta, theta_FV):
        if theta.get('p_0') is None:
            return F_innov(self.EXt(theta, t=0, logV2_prev=0.),
                           self.SDXt(theta, t=0, logV2_prev=0.),
                           **theta_FV,
                           a=-10., b=10.)
        else:
            p_0 = theta['p_0']
            dist_0 = F_innov(self.EXt(theta, t=0, logV2_prev=0., k=0),
                             self.SDXt(theta, t=0, logV2_prev=0., k=0),
                             **theta_FV,
                             a=-10., b=10.)
            dist_1 = F_innov(self.EXt(theta, t=0, logV2_prev=0., k=1),
                             self.SDXt(theta, t=0, logV2_prev=0., k=1),
                             **theta_FV,
                             a=-10., b=10.)
            return Mixture([p_0, 1.-p_0], dist_0, dist_1)

    def PX(self, t, xp, theta, theta_FV, returns=None):
        xp = np.clip(xp, -50., 50.)

        # mix over regimes and jumps:
        if theta.get('p_0') is None:  # single regime
            if theta.get('phi_V') is None:  # no jumps
                return F_innov(self.EXt(theta, xp),
                               self.SDXt(theta, xp),
                               **theta_FV)
            else:  # jumps
                dist_nojmp = F_innov(self.EXt(theta, xp),
                                     self.SDXt(theta, xp),
                                     **theta_FV,
                                     a=-10., b=10.)
                sd_jmp = np.sqrt(theta['xi']**2 + theta['phi_V']**2)
                dist_jmp = F_innov(self.EXt(theta, xp),
                                   sd_jmp,
                                   **theta_FV,
                                   a=-10., b=10.)
                lambd = theta['lambda_V']
                return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

        else:  # 2 regimes
            p_0 = theta['p_0']
            if theta.get('phi_V') is None:  # no jumps
                dist_0 = F_innov(self.EXt(theta, xp, 0),
                                 theta['xi_0'],
                                 **theta_FV,
                                 a=-10., b=10.)
                dist_1 = F_innov(self.EXt(theta, xp, 1),
                                 theta['xi_1'],
                                 **theta_FV,
                                 a=-10., b=10.)
                return Mixture([p_0, 1.-p_0], dist_0, dist_1)
            else:  # jumps
                EXt_0 = self.EXt(theta, xp, 0)
                EXt_1 = self.EXt(theta, xp, 1)
                dist_0_nj = F_innov(EXt_0, theta['xi_0'], **theta_FV,
                                    a=-10., b=10.)
                dist_1_nj = F_innov(EXt_1, theta['xi_1'], **theta_FV,
                                    a=-10., b=10.)
                lambd = theta['lambda_V']
                sd_jmp_0 = np.sqrt(theta['xi_0']**2 + theta['phi_0']**2)
                sd_jmp_1 = np.sqrt(theta['xi_1']**2 + theta['phi_1']**2)
                dist_0_j = F_innov(EXt_0, sd_jmp_0, **theta_FV,
                                   a=-10., b=10.)
                dist_1_j = F_innov(EXt_1, sd_jmp_1, **theta_FV,
                                   a=-10., b=10.)
                return Mixture([p_0*(1.-lambd), (1.-p_0)*(1.-lambd),
                                p_0*lambd, (1.-p_0)*lambd],
                               dist_0_nj, dist_1_nj, dist_0_j, dist_1_j)

    def PY(self, t, x, theta, theta_FX):
        x = np.clip(x, -50., 50.)

        # regimes considered fixed at this point, hence only mix over jumps:
        if theta.get('phi_X') is None:  # no jumps
            return F_innov(0., np.exp(0.5*x), **theta_FX)
        else:  # jumps
            dist_nojmp = F_innov(0., np.exp(0.5*x), **theta_FX)
            sd_jmp = np.sqrt(np.exp(x) + theta['phi_X']**2)
            dist_jmp = F_innov(0., sd_jmp, **theta_FX)
            lambd = theta['lambda_X']
            return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

    # methods for Guided PF:

    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        return F_innov(self._xhat(0, 1, data[0]), sd=1)

    def proposal(self, t, xp, data):
        return F_innov(self._xhat(self.EXt(theta, xp), self.xi, data[t]),
                       self.xi, **theta_FX)
