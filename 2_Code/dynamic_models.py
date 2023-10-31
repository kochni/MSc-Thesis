#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy

from particles.state_space_models import StateSpaceModel
from particles import distributions as dists

# documentation:
# https://particles-sequential-monte-carlo-in-python.readthedocs.io/en/latest/_autosummary/particles.state_space_models.html#module-particles.state_space_models

# source code:
# https://github.com/nchopin/particles/blob/master/particles


class MSGARCH(StateSpaceModel):
    '''
    Markov-switching GARCH model, i.e. process of the form

    Parameters:
    -----------

    '''

    default_params = {
        'variant': 'basic',
        'innov_X': 'N',
        'innov_V': 'N'
        }

    def PX0(self):
        '''
        distribution of initial latent state, K_0(V_1)
        '''
        p_00 = self.p_00
        p_01 = 1 - p_00
        p_10 = self.p_10
        p_11 = 1 - p_10

        p = p_10/(p_10+p_01)

        return dists.Binomial(n=1, p=p)

    def PX(self, t, xp):
        '''
        transition kernel of latent states, K(V_t | V_{t-1})

        Note: 'xp' is r_{t-1}
        '''
        p = (1-xp)*self.p_00 + xp*self.p_10

        return dists.Binomial(n=1, p=p)

    def PY(self, t, xp, x, data):
        '''
        conditional distribution of observations, P(X_t | X_{t-1}, V_t)

        Note: 'x' is r_t
        '''
        omega = (1-x)*self.omega_0 + x*self.omega_1
        alpha = (1-x)*self.alpha_0 + x*self.alpha_1
        beta = (1-x)*self.beta_0 + x*self.beta_1

        s2_j = np.tile(1, len(omega))  # initial volatility

        for j in range(1, t+1):
            s2_prev = s2_j
            s2_j = omega + alpha*data[j-1]**2 + beta*s2_prev

        s2_t = np.maximum(s2_j, 1e-10)

        return dists.Normal(loc=0, scale=s2_t)


    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        '''
        proposal distribution of inital latent state, nu_0(V_1)
        '''
        return dists.Normal(loc=self._xhat(0, 1, data[0]),
                            scale=1)

    def proposal(self, t, xp, data):
        '''
        conditional proposal distribution of new latent state,
        nu(V_t | V_{t-1}, X_{1:t})
        '''
        xi = self.xi

        return dists.Normal(loc=self._xhat(self.EXt(xp), xi, data[t]),
                            scale=xi)


class StochVol(StateSpaceModel):
    '''
    Stochastic Volatility model, i.e. process of the form
        X_t = V_t·Z_t
        log(V_t^2) = ω·(1-α) + α·log(V_t^2) + ξ·U_t

    Parameters:
    -----------
    variant: string
        specific model; one of 'basic', 'heston', and 'sabr'.

    innov_X: string
        innovation distribution of observations; one of 'N' (for Gaussian)
        and 't' (for Student t).

    innov_V: string
        innovation distribution of volatilities; one of 'N' (for Gaussian)
        and 't' (for Student t).
    '''

    default_params = {
        'variant': 'basic',
        'innov_X': 'N',
        'innov_V': 'N'
        }

    def PX0(self):
        '''
        distribution of initial latent state, K_0(V_1)
        '''
        omega = self.omega
        omega = np.minimum(np.exp(50), np.maximum(-np.exp(50), omega))

        if self.innov_X == 'N':
            return dists.Normal(loc=omega, scale=1)
        else:
            df = self.df
            return dists.Student(loc=omega, scale=1, df=df)

    def EXt(self, xp):
        '''
        conditional expectation of latent state, E[x_t|x_{t-1}]
        '''
        omega = self.omega
        alpha = self.alpha

        return omega*(1 - alpha) + alpha*xp

    def PX(self, t, xp):
        '''
        transition kernel of latent states, K(V_t | V_{t-1})

        Note: 'xp' is log(V_{t-1}^2), 'xi' is vol-of-vol
        '''
        xp = np.minimum(50, np.maximum(-50, xp))
        alpha = self.alpha
        xi = self.xi

        if self.innov_V == 'N':
            return dists.Normal(loc=self.EXt(xp), scale=xi)
        else:
            df = self.df
            return dists.Student(loc=self.EXt(xp), scale=xi, df=df)

    def PY(self, t, xp, x, data):
        '''
        conditional distribution of observations, P(X_t | X_{t-1}, V_t)

        Note: 'x' is log(V_t^2), 'xp' is log(V_{t-1}^2), 'xi' is vol-of-vol
        '''
        x = np.minimum(50, np.maximum(-50, x))

        if self.innov_X == 'N':
            return dists.Normal(loc=0, scale=np.exp(x/2))
        else:
            df = self.df
            return dists.Student(loc=0, scale=np.exp(x/2), df=df)

    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        '''
        proposal distribution of inital latent state, nu_0(V_1)
        '''
        return dists.Normal(loc=self._xhat(0, 1, data[0]),
                            scale=1)

    def proposal(self, t, xp, data):
        '''
        conditional proposal distribution of new latent state,
        nu(V_t | V_{t-1}, X_{1:t})
        '''
        xi = self.xi

        return dists.Normal(loc=self._xhat(self.EXt(xp), xi, data[t]),
                            scale=xi)


class Heston(StateSpaceModel):
    '''
    Heston stochastic volatility model, i.e. process of the form
        X_t = V_t·Z_t


    Parameters:
    -----------
    variant: string
        specific model; one of 'basic', 'heston', and 'sabr'.

    innov_X: string
        innovation distribution of observations; one of 'N' (for Gaussian)
        and 't' (for Student t).

    innov_V: string
        innovation distribution of volatilities; one of 'N' (for Gaussian)
        and 't' (for Student t).
    '''

    default_params = {
        'variant': 'basic',
        'innov_X': 'N',
        'innov_V': 'N'
        }

    def PX0(self):
        '''
        distribution of initial latent state, K_0(V_1)
        '''
        nu = self.nu  # long-run volatility
        xi = self.xi  # vol-of-vol

        return dists.Normal(loc=nu, scale=xi)

    def EXt(self, xp):
        '''
        conditional expectation of latent state, E[x_t|x_{t-1}]
        '''
        kappa = self.kappa  # speed of mean reversion
        nu = self.nu        # long-run volatility
        xi = self.xi        # vol-of-vol

        return xp + kappa*(nu/np.exp(xp) - 1) - 0.5*xi**2 / np.exp(xp)

    def PX(self, t, xp):
        '''
        transition kernel of latent states, K(V_t | V_{t-1})

        Note: 'xp' is log(V_{t-1}^2), 'xi' is vol-of-vol
        '''
        xp = np.minimum(50, np.maximum(-50, xp))
        kappa = self.kappa  # speed of volatility reversion
        nu = self.nu        # mean volatility level
        xi = self.xi        # vol-of-vol

        return dists.Normal(loc=self.EXt(xp),
                            scale=xi/np.exp(xp/2))

    def PY(self, t, xp, x, data):
        '''
        conditional distribution of observations, P(X_t | X_{t-1}, V_t)

        Note: 'x' is log(V_t^2), 'xp' is log(V_{t-1}^2), 'xi' is vol-of-vol
        '''
        x = np.minimum(50, np.maximum(-50, x))

        return dists.Normal(loc=0, scale=np.exp(x/2))

    def _xhat(self, xst, s, yt):
        return xst + 0.5 * s**2 * (yt ** 2 * np.exp(-xst) - 1.0)

    def proposal0(self, data):
        '''
        proposal distribution of inital latent state, nu_0(V_1)
        '''
        return dists.Normal(loc=self._xhat(0, 1, data[0]),
                            scale=1)

    def proposal(self, t, xp, data):
        '''
        conditional proposal distribution of new latent state,
        nu(V_t | V_{t-1}, X_{1:t})
        '''
        xi = self.xi
        xi = np.minimum(np.exp(50), np.maximum(-np.exp(50), xi))
        xp = np.minimum(50, np.maximum(-50, xp))

        return dists.Normal(loc=self._xhat(self.EXt(xp), xi, data[t]),
                            scale=xi)
