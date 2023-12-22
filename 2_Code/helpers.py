#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# some helpful functions for many things

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from particles import distributions
from numpy.lib.recfunctions import structured_to_unstructured  # to make structured array useful again
from itertools import chain

from scipy.optimize import brentq


def F_innov(mean=0., sd=1., **theta_F):
    '''
    Wrapper function for innovation distributions, i.e. with mean 0 sd 1.
    Reverts to simplest distribution whenever possible for performance gains:
        - if no additional parameters specified (besides mean & sd), returns
          Gaussian
        - if df specified, returns Student t with df degrees of freedom
        - if shape, tail, skew specified, returns Generalized Hyperbolic

    '''
    if theta_F['tail'] is None:
        return Normal(loc=mean, scale=sd)
    elif theta_F['shape'] is None:
        _, v = stats.t.stats(df=theta_F['tail'])
        return Student(scale=sd/np.sqrt(v), df=theta_F['tail'])
    else:
        m, v = stats.genhyperbolic.stats(p=theta_F['tail'], a=theta_F['shape'],
                                         b=theta_F['skew'])
        return GenHyp(loc=-m/np.sqrt(v), scale=sd/np.sqrt(v), tail=theta_F['tail'],
                      shape=theta_F['shape'], skew=theta_F['skew'])


def mix_cdf(x, probs, mus=0., sigmas=1., **theta_F):
    '''
    CDF of a mixture of the innovation distribution

    Parameters:
    -----------
    q: array
        quantiles to be returned.
    probs: array
        weights of mixture components.
    mus: array
        means of mixture components.
    sigmas: array
        standard deviations of mixture components.
    dfs: array
        degrees of freedoms of mixture components.

    '''

    # CDF of mixture distribution is weighted sum of individual CDFs
    return (probs * F_innov(mus, sigmas, **theta_F).cdf(x)).sum()


def abs_cdf(x, cdf):
    '''
    CDF of the absolute value of a random variable

    '''
    return cdf(x) - cdf(-x)


def sq_cdf(x, cdf):
    '''
    CDF of the square of a random variable from distribution 'dist'

    '''
    return cdf(np.sqrt(x)) - cdf(-np.sqrt(x))


def pos_cdf(x, cdf):
    '''
    CDF of positive part of a random variable, (X)_+ = max(X, 0)

    '''
    if x > 0.:
        return cdf(x)
    else:
        return 0.


def abs_mix_cdf(x, probs, mus=0., sigmas=1., **theta_F):
    ''' CDF of the absolute value of a mixture distribution '''

    return abs_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F))


def sq_mix_cdf(x, probs, mus=0., sigmas=1., **theta_F):
    ''' CDF of the square of a mixture distribution '''

    return sq_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F))


def pos_mix_cdf(x, probs, mus=0., sigmas=1., **theta_F):
    ''' CDF of the positive part of a mixture distribution '''

    return pos_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F))


def inv_cdf(cdf, p, lo, hi):
    '''
    numeric approximation of the quantile function of the distribution 'dist'

    if cdf(hi) < p or cdf(lo) > p, simply returns 2·hi resp. 2·lo (assuming
    that lo < 0)

    Parameters:
    -----------
    cdf: cdf to be inverted
        distribution whose CDF is to be inverted.
    p: list or array
        quantile levels.

    '''

    q = np.full(len(p), np.nan)
    for i in range(len(p)):
        if cdf(lo) <= p[i] <= cdf(hi):
            q[i] = brentq(f=lambda x: cdf(x) - p[i], a=lo, b=hi, xtol=1e-6)
        elif cdf(hi) < p[i]:
            q[i] = 2.0 * hi
        elif cdf(lo) > p[i]:
            q[i] = 2.0 * lo

    return q


def cap(values, floor=None, ceil=None):
    '''
    takes an array and caps each non-None entry while leaving None entries
    unchanged
    '''
    if isinstance(values, float) or values is None:
        return np.clip(values, floor, ceil) if values is not None else None
    else:
        values[values != None] = np.clip(values[values != None], floor, ceil)
        return values

# Loss functions for assessment of prediction accuracy

def sq_err(pred, truth):
    return (pred-truth)**2


def perc_err(pred, truth):
    return abs(pred - truth) / (truth + 1)


def mov_avg(x, w):
    '''
    moving average of the last 'w' values
    '''
    if x.ndim == 1:
        ret = np.cumsum(x, dtype=float)
        ret[w:] = ret[w:] - ret[:-w]
        return ret[w-1:] / w
    elif x.ndim == 2:
        ret = np.cumsum(x, axis=1, dtype=float)
        ret[:, w:] = ret[:, w:] - ret[:, :-w]
        return ret[:, w-1:] / w


# activation functions (for parameter transforms, reservoir computers, ...)

def Id(x):
    ''' R --> R '''
    return x


def sigmoid(x):
    ''' R --> [0, 1] '''
    x = cap(x, floor=-500)  # avoids overflow (returns 0 anyway)
    return 1/(1 + np.exp(-x))


def tanh(x):
    ''' R --> [-1, 1] '''
    return np.tanh(x)


def relu(x):
    ''' R --> R_+ '''
    return np.maximum(0, x)


# parameter transforms

def param_tf(theta):
    pass


# function to automatically set all priors given model specification

def set_prior(model):
    '''
    automatically sets all prior distributions given the model specification

    Parameters without constraints get a Gaussian prior with stdev = 10

    Parameters with positivity constraints (e.g., GARCH parameters) are later
        transformed with exp(); they get a Gaussian prior with stdev = 1,
        yielding a log-Normal with effective stdev ≈ 7.4;

    Parameters with [0,1] constraints (e.g., regime probabilities) are later
        transformed with the sigmoid; they get a Gaussian prior with stdev 1.5,
        which is approximately uniform after the transformation

    '''

    # s = stdev
    # s_pos = np.sqrt(s**2 / (1 - (2*stats.norm.pdf(0))**2))

    hyper = model.get('hyper')
    if hyper is not None: q = hyper.get('q')

    prior = OrderedDict()

    if model.get('regimes') in [None, 1]:

        if model['dynamics'] == 'constant':
            prior['sigma'] = Gamma(1., 0.5)

        elif model['dynamics'] == 'garch':
            if model['variant'] != 'elm':
                prior['omega'] = Gamma(1., 0.5)
                prior['alpha'] = Beta(1., 1.)
                prior['beta'] = Beta(1., 1.)
            else:
                for j in range(q+1):
                    prior['w' + str(j)] = Normal(loc=0., scale=3.)

            if model['variant'] in ['gjr', 'thr', 'exp']:
                prior['gamma'] = Normal(loc=0., scale=10.)

        elif model['dynamics'] == 'guyon':
            prior['omega'] = Gamma(1., 0.5)
            prior['alpha'] = Gamma(1., 0.5)
            prior['beta'] = Gamma(1., 0.5)
            prior['a1'] = shGamma(1., 0.5, loc=1.)
            prior['a2'] = shGamma(1., 0.5, loc=1.)
            prior['d1'] = Gamma(1., 0.5)
            prior['d2'] = Gamma(1., 0.5)

    else:  # same but for every regime
        if model['switching'] == 'mixing':
            if model['regimes'] == 2:
                prior['p_0'] = Beta(1., 1.)
            else:
                prior['p'] = Dirichlet(alpha=np.tile(1., model['regimes']))

        else:
            for k in range(model['regimes']):
                if model['regimes'] == 2:
                    prior['P_0'] = Beta(1., 1.)
                    prior['P_1'] = Beta(1., 1.)
                else:
                    prior['P_' + str(k)] = Dirichlet(alpha=np.tile(1., model['regimes']))

        for k in range(model['regimes']):
            if model['dynamics'] == 'constant':
               prior['sigma_' + str(k)] = Gamma(1., 0.5)

            elif model['dynamics'] == 'garch':
                if model['variant'] == 'elm':
                    for j in range(q+1):
                        prior['w' + str(j) + '_' + str(k)] = Normal(loc=0., scale=3.)
                elif model['variant'] == 'exp':  # no constraints
                    prior['omega_' + str(k)] = Normal(loc=0., scale=10.)
                    prior['alpha_' + str(k)] = Normal(loc=0., scale=10.)
                    prior['beta_' + str(k)] = Normal(loc=0., scale=10.)
                else:  # positivity constraints
                    prior['omega_' + str(k)] = Gamma(1., 0.5)
                    prior['alpha_' + str(k)] = Beta(1., 1.)
                    prior['beta_' + str(k)] = Beta(1., 1.)

                if model['variant'] in ['gjr', 'thr', 'exp']:
                    prior['gamma_' + str(k)] = Normal(loc=0., scale=10.)

            elif model['dynamics'] == 'guyon':
                prior['omega_' + str(k)] = Gamma(1., 0.5)
                prior['alpha_' + str(k)] = Gamma(1., 0.5)
                prior['beta_' + str(k)] = Gamma(1., 0.5)
                prior['a1_' + str(k)] = Gamma(1., 0.5)
                prior['a2_' + str(k)] = Gamma(1., 0.5)
                prior['d1_' + str(k)] = Gamma(1., 0.5)
                prior['d2_' + str(k)] = Gamma(1., 0.5)

            elif model['dynamics'] in ['elm', 't_sig', 'r_sig']:
                for j in range(model['hyper']['q']):
                    prior['w' + str(j) + '_' + str(k)] = Normal(scale=3.)

    if model['innov_X'] == 't':
        prior['df_X'] = shGamma(a=1., b=0.5, loc=2.)

    if model['jumps'] is not None:
        prior['lambda'] = Beta(1., 1.)
        prior['phi'] = Gamma(1., 0.5)

    prior = StructDist(prior)

    return prior


# kernel functions (for Guyon & Lekeufack's model)

def tspl(t, tp, alpha, delta):
    '''
    "time-shifted power law" kernel used by Guyon & Lekeufack (2023)
    '''
    alpha = alpha[:, :, np.newaxis]
    delta = delta[:, :, np.newaxis]

    tau = abs(t-tp)  # (t,)
    weights = (tau + delta)**(-alpha)  # (N,K,t)

    # normalize so that axis 2 sums to 1
    weights = weights / weights.sum(axis=2)[:, :, np.newaxis]

    return weights


def out_fct(model):  # to collect only relevant things to save memory
    '''
    function defining which quantities/variables to collect while running SMC.

    Parameters:
    -----------

    model: SMC object
        model; object which contains variables to be collected as attributes

    '''
    Z = np.array(model.summaries.logLts)
    theta = model.X.theta  # particles / posterior samples
    prior = model.fk.model.prior.laws
    W = model.W  # particle weights
    ESS = np.array(model.summaries.ESSs)
    rs_flags = np.array(model.summaries.rs_flags)
    MH_acc = np.array(model.X.shared['acc_rates']).flatten()  # MH acceptance rates
    preds = model.fk.model.predictions
    predsets = model.fk.model.predsets

    dic = {
        'Z_t': Z,
        'theta_T': theta,
        'W_T': W,
        'prior': prior,
        'ESS_t': ESS,
        'rs_flags': rs_flags,
        'MH_Acc_t': MH_acc,
        'Preds': preds,
        'PredSets': predsets
        }

    return dic
