#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# SMC
from particles.state_space_models import StateSpaceModel


class Heston:
    '''
    Heston model:

    log(V_t^2) = log(V_{t-1}^2) + κ·(ν/V_{t-1}^2 - 1) - 1/2·ξ^2/V_{t-1}^2
                 + ξ/V_{t-1}·Z_t

    '''

    def EXt(self, theta, logV2_prev, k=None, X_prev=None):
        '''
        E[log(V_t^2) | log(V_{t-1}^2)]

        '''
        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if 'kappa' in theta.keys():
                kappa = theta['kappa']
                nu = theta['nu']
                xi = theta['xi']

            elif 'kappa_0' in theta.keys():
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
            if 'kappa' in theta.dtype.names:
                kappa = theta['kappa']
                nu = theta['nu']
                xi = theta['xi']

            elif 'kappa_0' in theta.dtype.names:
                M = len(theta)
                kappa = np.stack([theta['kappa_0'], theta['kappa_1']],
                                 axis=1)[np.arange(M), k]
                nu = np.stack([theta['nu_0'], theta['nu_1']],
                                 axis=1)[np.arange(M), k]
                xi = np.stack([theta['xi_0'], theta['xi_1']],
                                 axis=1)[np.arange(M), k]
        return (logV2_prev + kappa * nu*(np.exp(-logV2_prev) - 1)
                - 0.5*xi**2 * np.exp(-logV2_prev))

    def SDXt(self, theta, logV2_prev, k=None, X_prev=None):
        '''
        SD[log(V_t^2) | log(V_{t-1}^2)]

        '''
        # during Particle Filtering step, theta is dictionary with floats
        if type(theta) == dict:
            if 'xi' in theta.keys():
                xi = theta['xi']
            elif 'xi_0' in theta.keys():
                if k == 0:
                    xi = theta['xi_0']
                else:
                    xi = theta['xi_1']

        # during prediction, theta is structured array with all N particles
        else:
            M = len(theta)
            if 'xi' in theta.dtype.names:
                xi = theta['xi']
            elif 'xi_0' in theta.dtype.names:
                xi = np.stack([theta['xi_0'], theta['xi_1']],
                              axis=1)[np.arange(M), k]

        return xi * np.exp(-logV2_prev)

    def PX0(self, theta, theta_FV):
        '''
        distribution o flog(V_0^2)

        '''
        if theta.get('p_0') is None:
            return F_innov(self.EXt(theta, logV2_prev=0.),
                           self.SDXt(theta, logV2_prev=0.),
                           **theta_FV,
                           a=-10., b=10.)

        else:
            p_0 = theta.get('p_0')
            dist_0 = F_innov(self.EXt(theta, logV2_prev=0., k=0),
                             self.SDXt(theta, logV2_prev=0., k=0),
                             **theta_FV,
                             a=-10., b=10.)
            dist_1 = F_innov(self.EXt(theta, logV2_prev=0., k=1),
                             self.SDXt(theta, logV2_prev=0., k=1),
                             **theta_FV,
                             a=-10., b=10.)
            return Mixture([p_0, 1.-p_0], dist_0, dist_1)

    def PX(self, t, xp, theta, theta_FV, X_prev=None):
        '''
        distribution of log(V_t^2) | log(V_{t-1}^2)

        '''
        # mix over regimes and jumps:
        # single regime:
        if theta.get('p_0') is None:
            # no jumps
            if theta.get('phi_V') is None:
                return F_innov(self.EXt(theta, xp),
                               self.SDXt(theta, xp),
                               **theta_FV,
                               a=-10., b=10.)
            # jumps
            else:
                dist_nojmp = F_innov(self.EXt(theta, xp),
                                     self.SDXt(theta, xp),
                                     **theta_FV,
                                     a=-10., b=10.)
                sd_jmp = np.sqrt(self.SDXt(theta, xp)**2 + theta['phi_V'])
                dist_jmp = F_innov(self.EXt(theta, xp),
                                   sd_jmp,
                                   **theta_FV,
                                   a=-10., b=10.)
                lambd = theta['lambda_V']
                return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

        # 2 regimes
        else:
            p_0 = theta.get('p_0')
            # no jumps
            if theta.get('phi_V') is None:
                dist_0 = F_innov(self.EXt(theta, xp, k=0),
                                 self.SDXt(theta, xp, k=0),
                                 **theta_FV,
                                 a=-10., b=10.)
                dist_1 = F_innov(self.EXt(theta, xp, k=1),
                                 self.SDXt(theta, xp, k=1),
                                 **theta_FV,
                                 a=-10., b=10.)

                return Mixture([p_0, 1.-p_0], dist_0, dist_1)

            # jumps
            else:
                dist_0_nj = F_innov(self.EXt(theta, xp, k=0),
                                    self.SDXt(theta, xp, k=0),
                                    **theta_FV,
                                    a=-10., b=10.)
                dist_1_nj = F_innov(self.EXt(theta, xp, k=1),
                                    self.SDXt(theat, xp, k=1),
                                    **theta_FV,
                                    a=-10., b=10.)

                lambd = theta['lambda_V']
                sd_jmp_0 = np.sqrt(self.SDXt(theta, xp, k=0)**2 + theta['phi_V'])
                sd_jmp_1 = np.sqrt(self.SDXt(theta, xp, k=1)**2 + theta['phi_V'])
                dist_0_j = F_innov(self.EXt(theta, xp, k=0),
                                   sd_jmp_0,
                                   **theta_FV, a=-10., b=10.)
                dist_1_j = F_innov(self.EXt(theta, xp, k=1),
                                   sd_jmp_1,
                                   **theta_FV, a=-10., b=10.)

                return Mixture([p_0*(1.-lambd), p_1*(1.-lambd),
                                p_0*lambd, p_1*lambd],
                               dist_0_nj, dist_1_nj, dist_0_j, dist_1_j)

    def PY(self, t, x, theta, theta_FX):
        '''
        distribution of X_t | V_t

        '''

        # regimes fixed at this point, hence only mix over jumps
        # no jumps:
        if theta.get('phi_X') is None:
            return F_innov(0., np.exp(0.5*x), **theta_FX)

        # jumps:
        else:
            dist_nojmp = F_innov(0., np.exp(0.5*x), **theta_FX)
            sd_jmp = np.sqrt(np.exp(x) + theta['phi_X']**2)
            dist_jmp = F_innov(0., sd_jmp, **theta_FX)
            lambd = theta['lambda_X']
            return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

    # methods for defining Guided PF:

    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        return F_innov(mean=self._xhat(0, 1, data[0]),
                      sd=1)

    def proposal(self, t, xp, data):
        return F_innov(mean=self._xhat(self.EXt(xp), self.xi, data[t]),
                      sd=self.xi)