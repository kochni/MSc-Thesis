#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# some helpful functions for many things

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from particles.distributions import *
from collections import OrderedDict
from numpy.lib.recfunctions import structured_to_unstructured  # to make structured array useful again
from itertools import chain


def cap_params(theta):
    '''
    Parameters
    ----------
    theta: Structured array

    '''
    params_names = theta.dtype.names

    # Canonical SV
    if 'alpha' in params_names:
        pass
    elif 'alpha_0' in params_names:
        pass

    # Heston model
    elif 'nu' in params_names:
        pass
    elif 'nu_0' in params_names:
        pass

    # Neural SV
    elif 'w' in params_names:
        pass


# activation functions (for parameter transforms, reservoir computers, ...)

def Id(x):
    ''' leaves input unchanged '''
    return x


def sigmoid(x):
    ''' smooth map into [0, 1] '''
    x = np.clip(x, -500., None)  # avoids overflow (returns 0 anyway)
    return 1/(1 + np.exp(-x))


def tanh(x):
    ''' maps smoothly into [-1, 1] '''
    return np.tanh(x)


def relu(x):
    ''' non-smooth map into [0,∞) '''
    return np.clip(x, 0.0, None)


def shi(x):
    ''' non-smooth map into [-1,1] '''
    return np.clip(x, -1.0, 1.0)


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
                    prior['w' + str(j)] = Laplace(loc=0., scale=3.)

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
        if model['switching'] in ['mix', 'mixing']:
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


def out_fct(smc):  # to collect only relevant things to save memory
    '''
    function defining which quantities/variables to collect after running SMC.

    Parameters:
    -----------

    smc: SMC object
        model; object which contains variables to be collected as attributes

    '''
    Z = np.array(smc.summaries.logLts)  # log of mean likelihoods
    DIC = None
    theta = smc.X.theta
    prior = smc.fk.model.prior.laws
    W = smc.W
    ESS = np.array(smc.summaries.ESSs)
    rs_flags = np.array(smc.summaries.rs_flags)
    MH_acc = np.array(smc.X.shared['acc_rates']).flatten()
    preds = smc.preds
    predsets = smc.predsets
    rand_proj = smc.rand_proj
    cpu_time = smc.cpu_time

    dic = {
        'Z': Z,
        'DIC': DIC,
        'theta': theta,
        'W': W,
        'prior': prior,
        'ESS': ESS,
        'rs_flags': rs_flags,
        'MH_Acc': MH_acc,
        'Preds': preds,
        'PredSets': predsets,
        'RandProj': rand_proj,
        'cpu_time': cpu_time
        }

    return dic
