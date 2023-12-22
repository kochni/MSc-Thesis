#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.state_space_models import StateSpaceModel
from particles.distributions import Normal, Student, Dirac, Mixture

from operator import itemgetter

# from helpers import F_innov, cap


class PiecewiseConst(StateSpaceModel):
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
        'phi_V': None, 'lambda_V': None
        }

    def get_params(self):
        ''' extract all parameters & do capping/transformations '''
        theta = {}
        theta_FX = {}
        theta_FV = {}

        # innovation parameters:
        df_X = self.df_X if self.df_X is not None else self.df
        df_V = self.df_V if self.df_V is not None else self.df
        theta_FX['df_X'] = cap(df_X, floor=2.+1e-10)
        theta_FV['df_V'] = cap(df_V, floor=2.+1e-10)
        # --> if only 'df' specified, df_X = df_V = df;
        # if no df arguments specified, df_X = df_V = None

        # jump parameters:
        theta['lambda_X'] = cap(self.lambda_X, floor=0., ceil=1.)
        theta['lambda_V'] = cap(self.lambda_V, floor=0., ceil=1.)
        theta['phi_X'] = cap(self.phi_X, floor=0.)
        theta['phi_V'] = cap(self.phi_V, floor=0.)

        return theta, theta_FX, theta_FV

    def predict(self):
        pass

    def PX0(self):
        theta = self.get_params()
        return F_innov(0., theta['phi_V'], **theta_FX)

    def PX(self, t, xp):
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        return Mixture([1.-theta['lambda_V'], theta['lambda_V']],
                       Dirac(loc=xp),
                       F_innov(xp, theta['phi_V'], **theta_FX)
                       )

    def PY(self, t, xp, x, data):
        theta = self.get_params()
        x = np.clip(x, -50., 50.)

        if theta['phi_X'] is None:
            return F_innov(0., np.exp(0.5*x), **theta_FX)

        else:
            sd_jump = np.sqrt(np.exp(x) + theta['phi_X']**2)
            return Mixture([1.-theta['lambda_X'], theta['lambda_X']],
                           F_innov(0., np.exp(0.5*x), **theta_FX),
                           F_innov(0., sd_jump, **theta_FX)
                           )

    # methods for defining Guided PF:

    def _xhat(self, xst, s, yt):
        pass

    def proposal0(self, data):
        return F_innov(df=5)

    def proposal(self, t, xp, data):
        return F_innov(mean=xp, df=5)


class StochVol(StateSpaceModel):
    '''
    Canonical stochastic volatility model:

        log(V_t^2) = ω·(1-α) + α·log(V_{t-1}^2) + ξ·Z_t

    '''

    # by default, single Gaussian regime without jumps
    default_params = {
        'p_0': None,
        'omega': None, 'alpha': None, 'xi': None,
        'omega_0': None, 'alpha_0': None, 'xi_0': None,
        'omega_1': None, 'alpha_1': None, 'xi_1': None,
        'df': None,'df_X': None, 'df_V': None,
        'tail': None, 'tail_X': None, 'tail_V': None,
        'shape': None, 'shape_X': None, 'shape_V': None,
        'skew': None, 'skew_X': None, 'skew_V': None,
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None
        }

    def get_params(self):
        ''' extract all parameters & do necessary capping '''
        theta = {}

        theta['p_0'] = cap(self.p_0, floor=0., ceil=1.)
        if theta['p_0'] is None:
            theta['omega'] = self.omega
            theta['alpha'] = cap(self.alpha, floor=0., ceil=1.-1e-10)
            theta['xi'] = cap(self.xi, floor=1e-20)
        else:
            theta['omega_0'] = self.omega_0
            theta['omega_1'] = self.omega_1
            theta['alpha_0'] = cap(self.alpha_0, floor=0., ceil=1.-1e-10)
            theta['alpha_1'] = cap(self.alpha_1, floor=0., ceil=1.-1e-10)
            theta['xi_0'] = cap(self.xi_0, floor=1e-20)
            theta['xi_1'] = cap(self.xi_1, floor=1e-20)

        # innovation parameters:
        df_X = self.df_X if self.df_X is not None else self.df
        df_V = self.df_V if self.df_V is not None else self.df
        theta['df_X'] = cap(df_X, floor=2.+1e-10)
        theta['df_V'] = cap(df_V, floor=2.+1e-10)
        # --> if only 'df' specified, df_X = df_V = df;
        # if no df arguments specified, df_X = df_V = None

        # jump parameters:
        theta['lambda_X'] = cap(self.lambda_X, floor=0., ceil=1.)
        theta['lambda_V'] = cap(self.lambda_V, floor=0., ceil=1.)
        theta['phi_X'] = cap(self.phi_X, floor=0.)
        theta['phi_V'] = cap(self.phi_V, floor=0.)

        return theta

    def PX0(self):
        theta = self.get_params()
        if theta['p_0'] is None:
            return F_innov(theta['omega'],
                           theta['xi']/np.sqrt(1.-theta['alpha']**2),
                           **theta_FV)
        else:
            p_0 = theta['p_0']
            dist_0 = F_innov(theta['omega_0'],
                             theta['xi_0']/np.sqrt(1.-theta['alpha_0']**2),
                             **theta_FV)
            dist_1 = F_innov(theta['omega_1'],
                             theta['xi_1']/np.sqrt(1.-theta['alpha_1']**2),
                             **theta_FV)
            return Mixture([p_0, 1.-p_0], dist_0, dist_1)

    def EXt(self, rt=None, xp=None):
        ''' E[log(V_t^2) | log(V_{t-1}^2)] '''
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        if rt is None:
            omega = theta['omega']
            alpha = theta['alpha']
        else:
            omega = theta['omega_' + str(rt)]
            alpha = theta['alpha_' + str(rt)]

        return (1.-alpha)*omega + alpha*xp

    def PX(self, t, xp):
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        # mix over regimes and jumps:
        if theta['p_0'] is None:  # single regime
            if theta['phi_V'] is None:  # no jumps
                return F_innov(self.EXt(xp=xp), theta['xi'],
                               **theta_FV)
            else:  # jumps
                dist_nojmp = F_innov(self.EXt(xp=xp), theta['xi'],
                                     **theta_FV)
                sd_jmp = np.sqrt(theta['xi']**2 + theta['phi_V']**2)
                dist_jmp = F_innov(self.EXt(xp=xp), sd_jmp,
                                   **theta_FV)
                lambd = theta['lambda_V']
                return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

        else:  # 2 regimes
            p_0 = theta['p_0']
            if theta['phi_V'] is None:  # no jumps
                dist_0 = F_innov(self.EXt(0, xp), theta['xi_0'],
                                 **theta_FV)
                dist_1 = F_innov(self.EXt(1, xp), theta['xi_1'],
                                 **theta_FV)
                return Mixture([p_0, 1.-p_0], dist_0, dist_1)
            else:  # jumps
                dist_0_nj = F_innov(self.EXt(0, xp), theta['xi_0'],
                                    **theta_FV)
                dist_1_nj = F_innov(mean=self.EXt(1, xp), sd=theta['xi_1'],
                                    **theta_FV)
                lambd = theta['lambda_V']
                sd_jmp_0 = np.sqrt(theta['xi_0']**2 + theta['phi_0']**2)
                sd_jmp_1 = np.sqrt(theta['xi_1']**2 + theta['phi_1']**2)
                dist_0_j = F_innov(self.EXt(0, xp), sd_jmp_0,
                                   **theta_FV)
                dist_1_j = F_innov(self.EXt(1, xp), sd_jmp_1,
                                   **theta_FV)
                return Mixture([p_0*(1.-lambd), (1.-p_0)*(1.-lambd),
                                p_0*lambd, (1.-p_0)*lambd],
                               dist_0_nj, dist_1_nj, dist_0_j, dist_1_j)

    def PY(self, t, xp, x, data):
        theta = self.get_params()
        x = cap(x, floor=-50., ceil=50.)

        # regimes considered fixed at this point, hence only mix over jumps:
        if theta['phi_X'] is None:  # no jumps
            return F_innov(0., np.exp(0.5*x), **theta_FX)
        else:  # jumps
            dist_nojmp = F_innov(0., np.exp(0.5*x), **theta_FX)
            sd_jmp = np.sqrt(np.exp(x) + theta['phi_X']**2)
            dist_jmp = F_innov(0., sd_jmp, theta_FX)
            lambd = theta['lambda_X']
            return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

    # methods for Guided PF:

    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        return F_innov(mean=self._xhat(0, 1, data[0]), sd=1)

    def proposal(self, t, xp, data):
        return F_innov(self._xhat(self.EXt(xp), self.xi, data[t]),
                       self.xi, **theta_FX)


class Heston(StateSpaceModel):
    '''
    Heston model:

        log(V_t^2) = log(V_{t-1}^2) + κ·(ν/V_{t-1}^2 - 1) - 1/2·ξ^2/V_{t-1}^2
                     + ξ/V_{t-1}·Z_t

    '''

    # by default, single Gaussian regime without jumps
    default_params = {
        'p_0': None,                                  # regime probabilities
        'nu': None, 'kappa': None, 'xi': None,        #
        'nu_0': None, 'kappa_0': None, 'xi_0': None,  # main model parameters
        'nu_1': None, 'kappa_1': None, 'xi_1': None,  #
        'df': None,'df_X': None, 'df_V': None,        # innovations
        'tail': None, 'tail_X': None, 'tail_V': None,
        'shape': None, 'shape_X': None, 'shape_V': None,
        'skew': None, 'skew_X': None, 'skew_V': None,
        'phi_X': None, 'lambda_X': None,
        'phi_V': None, 'lambda_V': None
        }

    def get_params(self):
        ''' extract all parameters & do necessary capping '''
        theta = {}

        theta['p_0'] = cap(self.p_0, floor=0., ceil=1.)
        if theta['p_0'] is None:
            theta['nu'] = self.nu
            theta['kappa'] = cap(self.kappa, floor=0., ceil=1.-1e-10)
            theta['xi'] = cap(self.xi, floor=1e-20)
        else:
            theta['nu_0'] = self.nu_0
            theta['nu_1'] = self.nu_1
            theta['kappa_0'] = cap(self.kappa_0, floor=0., ceil=1.-1e-10)
            theta['kappa_1'] = cap(self.kappa_1, floor=0., ceil=1.-1e-10)
            theta['xi_0'] = cap(self.xi_0, floor=1e-20)
            theta['xi_1'] = cap(self.xi_1, floor=1e-20)

        # innovation parameters:
        df_X = self.df_X if self.df_X is not None else self.df
        df_V = self.df_V if self.df_V is not None else self.df
        theta['df_X'] = cap(df_X, floor=2.+1e-10)
        theta['df_V'] = cap(df_V, floor=2.+1e-10)
        # --> if only 'df' specified, df_X = df_V = df;
        # if no df arguments specified, df_X = df_V = None

        # jump parameters:
        theta['lambda_X'] = cap(self.lambda_X, floor=0.)
        theta['lambda_V'] = cap(self.lambda_V, floor=0.)
        theta['phi_X'] = cap(self.phi_X, floor=0.)
        theta['phi_V'] = cap(self.phi_V, floor=0.)

        return theta

    def EXt(self, rt=None, xp=None):
        ''' E[log(V_t^2) | log(V_{t-1}^2)] '''
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        if rt is None:
            nu = theta['nu']
            kappa = theta['kappa']
            xi = theta['xi']
        else:
            nu = theta['nu_' + str(rt)]
            kappa = theta['kappa_' + str(rt)]
            xi = theta['xi_' + str(rt)]

        return xp + kappa*(nu*np.exp(-xp) - 1.) - .5*xi**2 * np.exp(-xp)

    def SDXt(self, rt=None, xp=None):
        ''' SD[log(V_t^2) | log(V_{t-1}^2)] '''
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        if rt is None:
            sd = theta['xi'] * np.exp(-0.5*xp)
        else:
            sd = theta['xi_' + str(rt)] * np.exp(-0.5*xp)

        sd = cap(sd, floor=1e-50, ceil=1e50)
        return sd

    def PX0(self):
        theta = self.get_params()
        if theta['p_0'] is None:
            return F_innov(theta['nu'], theta['xi'], theta_FV)
        else:
            p_0 = theta['p_0']
            dist_0 = F_innov(theta['nu_0'], theta['xi_0'], **theta_FV)
            dist_1 = F_innov(theta['nu_1'], theta['xi_1'], **theta_FV)
            return Mixture([p_0, 1.-p_0], dist_0, dist_1)

    def PX(self, t, xp):
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        # mix over regimes and jumps:
        if theta['p_0'] is None:  # single regime
            if theta['phi_V'] is None:  # no jumps
                return F_innov(mean=self.EXt(xp=xp), sd=self.SDXt(xp=xp),
                               df=theta['df_V'])
            else:  # jumps
                dist_nojmp = F_innov(self.EXt(xp=xp), self.SDXt(xp=xp),
                                     **theta_FV)
                sd_jmp = np.sqrt(self.SDXt(xp=xp)**2 + theta['phi_V'])
                dist_jmp = F_innov(self.EXt(xp=xp), sd_jmp, **theta_FV)
                lambd = theta['lambda_V']
                return Mixture([(1.-lambd), lambd], dist_nojmp, dist_jmp)

        else:  # 2 regimes
            p_0 = theta['p_0']
            if theta['phi_V'] is None:  # no jumps
                dist_0 = F_innov(self.EXt(0, xp), self.SDXt(0, xp), **theta_FV)
                dist_1 = F_innov(self.EXt(1, xp), self.SDXt(1, xp), **theta_FV)
                return Mixture([p_0, 1.-p_0], dist_0, dist_1)

            else:  # jumps
                dist_0_nj = F_innov(self.EXt(0, xp), self.SDXt(0, xp),
                                    **theta_FV)
                dist_1_nj = F_innov(self.EXt(1, xp), self.SDXt(1, xp),
                                    **theta_FV)

                lambd = theta['lambda_V']
                sd_jmp_0 = np.sqrt(self.SDXt(0, xp)**2 + theta['phi_0'])
                sd_jmp_1 = np.sqrt(self.SDXt(1, xp)**2 + theta['phi_1'])
                dist_0_j = F_innov(self.EXt(0, xp), sd_jmp_0, **theta_FV)
                dist_1_j = F_innov(self.EXt(1, xp), sd_jmp_1, **theta_FV)
                return Mixture([p_0*(1.-lambd), p_1*(1.-lambd),
                                p_0*lambd, p_1*lambd],
                               dist_0_nj, dist_1_nj, dist_0_j, dist_1_j)

    def PY(self, t, xp, x, data):
        theta = self.get_params()
        x = cap(x, floor=-50., ceil=50.)

        # regimes fixed at this point, hence only mix over jumps:
        if theta['phi_X'] is None:  # no jumps
            return F_innov(0., np.exp(0.5*x), **theta_FX)
        else:  # jumps
            dist_nojmp = F_innov(0., np.exp(0.5*x), **theta_FX)
            sd_jmp = np.sqrt(np.exp(x) + theta['phi_X'])
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


class NeuralSV(StateSpaceModel):
    '''
    Reservoir computer for stochastic volatility:

        log(V_t^2) = ℓ(Res1_t) + ℓ(Res2_t)·Z_t

        where Res1_t, Res2_t are two separate random projections of the past
        returns X_{t-1} and volatilities V_{t-1} and ℓ(·) is a linear map

    '''

    # change stuff here to modify model:
    default_params = {
        'vol_drift': 'heston',  # 'basic', 'heston', or 'neural
        'vol_vol': 'neural',    # same; 'basic' = constant, 'heston' =
                                # exponentially decreasing
        'q': 5,  # dim of reservoir; must match w dim of A2· and b2· below!
        'A11': np.random.normal(size=[20, 2]),  #
        'A12': np.random.normal(size=[20, 2]),  #
        'b11': np.random.normal(size=[20, 1]),  #
        'b12': np.random.normal(size=[20, 1]),  # inner NN weights
        'A21': np.random.normal(size=[5, 20]),  #
        'A22': np.random.normal(size=[5, 20]),  #
        'b21': np.random.normal(size=[5, 1]),   #
        'b22': np.random.normal(size=[5, 1]),    #
        'df': None, 'df_X': None, 'df_V': None
        }

    def get_params(self):
        theta = {}
        if self.vol_drift == 'neural':
            for j in range(self.q+1):
                theta['v' + str(j)] = getattr(self, 'v'+str(j))
        elif self.vol_drift == 'basic':
            theta['omega'] = self.omega
            theta['alpha'] = cap(self.alpha, floor=0., ceil=1.-1e-10)
            theta['xi'] = cap(self.xi, floor=1e-20)
        else:  # Heston
            theta['nu'] = self.nu
            theta['kappa'] = cap(self.kappa, floor=0., ceil=1.-1e-10)
            theta['xi'] = cap(self.xi, floor=1e-20)

        if self.vol_vol == 'neural':
            for j in range(self.q+1):
                theta['w' + str(j)] = getattr(self, 'w'+str(j))
        else:  # Basic SV or Heston
            theta['xi'] = cap(self.xi, floor=1e-20)

        # if only 'df' specified, df_X = df_V = df;
        # if no df arguments specified, df_X = df_V = None
        df_X = self.df_X if self.df_X is not None else self.df
        df_V = self.df_V if self.df_V is not None else self.df
        theta['df_X'] = cap(df_X, floor=2.+1e-10)
        theta['df_V'] = cap(df_V, floor=2.+1e-10)

        return theta

    def Res(self, factor, t, xp=0.):
        if t > 0:
            rp = np.full(len(xp), data[t-1])
        else:
            rp = np.array(0.)
            xp = np.array(1.)

        x = np.vstack([rp, xp])  # (2,Nx)
        if factor == 'drift':
            A1, b1, A2, b2 = self.A11, self.b11, self.A21, self.b21
        else:
            A1, b1, A2, b2 = self.A12, self.b12, self.A22, self.b22

        h1 = np.einsum('HI,IN->HN', A1, x) + b1
        h1 = sigmoid(h1)
        h2 = np.einsum('qH,HN->qN', A2, h1) + b2
        h2 = sigmoid(h2)
        return h2  # (q,Nx)

    def EXt(self, t, xp=0.):
        theta = self.get_params()
        xp = cap(xp, floor=-50., ceil=50.)

        if self.vol_drift == 'neural':
            w_names = ['v' + str(j) for j in range(1, self.q+1)]
            W = itemgetter(*w_names)(theta)  # (q,)
            W = np.array(W)
            Res = self.Res('drift', t, xp)  # (q,Nx)
            EXt = np.einsum('q,qN->N', W, Res) + theta['v0']

        elif self.vol_drift == 'basic':
            EXt = (1.-theta['alpha']) * theta['omega'] + theta['alpha'] * xp

        else:
            EXt = (xp + theta['kappa']*(theta['nu'] * np.exp(-xp) - 1.) -
                    0.5*theta['xi']**2 * np.exp(-xp))

        return EXt

    def SDXt(self, t, xp=0.):  # same as EXt but with w and Res2
        theta = self.get_params()

        if self.vol_vol == 'neural':
            w_names = ['w' + str(j) for j in range(1, self.q+1)]
            W = itemgetter(*w_names)(theta)  # (q,)
            W = np.array(W)
            Res = self.Res('diffus', t, xp)  # (q,Nx)
            SDXt = np.einsum('q,qN->N', W, Res) + theta['w0']

        elif self.vol_vol == 'basic':
            SDXt = theta['xi']

        else:
            SDXt = theta['xi'] * np.exp(-0.5*xp)

        SDXt = cap(SDXt, floor=1e-20, ceil=1e20)
        return SDXt

    def PX0(self):
        theta = self.get_params()
        return F_innov(self.EXt(t=0), self.SDXt(t=0), **theta_FV)

    def PX(self, t, xp):
        theta = self.get_params()
        return F_innov(self.EXt(t, xp), self.SDXt(t, xp), **theta_FV)

    def PY(self, t, xp, x, data):
        theta = self.get_params()
        x = np.clip(x, -50., 50.)
        return F_innov(0., np.exp(0.5*x), **theta_FX)
