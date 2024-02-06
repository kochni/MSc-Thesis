#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############
# Packages #
############

# basic
import numpy as np
from scipy import stats

# data
import yfinance as yfin

# SMC
from particles.core import SMC, multiSMC
from particles import distributions
from particles.smc_samplers import IBIS, SMC2
from particles.state_space_models import StateSpaceModel, Bootstrap, GuidedPF
from particles.collectors import LogLts, ESSs, Rs_flags

# utils
from statsmodels.stats.weightstats import DescrStatsW
from numpy.lib.recfunctions import structured_to_unstructured
from operator import itemgetter

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# other
from multiprocessing import cpu_count
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# own model classes
import os
# wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
# os.chdir(wd)
from det_vol import *
import pwc_sv, canon_sv, heston, elm_sv, esn_sv, randsig_sv
from helpers import *
from plots import *


# data:
ticker = '^GSPC'
S = yfin.Ticker(ticker).history(period="4y")['Close'].values  # raw prices
S = S[0:20]

# log-returns
X = 100 * np.log(S[1:]/S[:-1])

# np.savetxt("SP500.csv", data, delimiter=",")

# data = np.loadtxt("SP500.csv", delimiter=",")


# Options for predictions:
pred_opts = {
    'h': 1,           # horizon, no. of days ahead to predict / option maturity
    'M': 10**4,       # no. of simulations for MC estimations (e.g. options)
    'alpha': 0.1,     # miscoverage tolerance for prediction intervals
    'uq_method': 'naive',  # method for pred' sets ('naive', 'calibrated', or 'conformal)
    'strike': 'last'  # strike price of (European call) option
    }


###############
# IBIS MODELS #
###############

# define common parameters

fk_opts_ibis = {
    'len_chain': 20,    # length of MCMC 'move' chain
    'wastefree': False  # whether to use intermediate MCMC particles
    }

smc_opts_ibis = {  # only relevant if individual models run
    'N': 100,                    # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'ESSrmin': 0.5,              # degeneracy criterion
    'verbose': False             # whether to print ESS while fitting
    }


# collect models to be run in parallel:
models = {}


# BENCHMARK: White noise
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': 'mix',  # 'mix' or 'markov'
    'jumps': None,       # None, 'X'/'returns', or 'V'/'vol'
    'innov_X': 'N'       # 'N', 't', or 'GH'
    }
prior = OrderedDict()
prior['sigma'] = Gamma(1., 5.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_bench = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_bench, **smc_opts_ibis, **pred_opts)
# alg.run()
models['White Noise (N)'] = fk_bench


# Merton Model
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': 'mixing',
    'jumps': 'returns',
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['sigma'] = Gamma(1., .5)
prior['lambda_X'] = Beta(1., 1.)
prior['phi_X'] = Gamma(1., .5)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_merton = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_merton, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Merton (N)'] = fk_merton


# Standard GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'basic',
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 'N'
    }
# prior = set_prior(spec)
prior = OrderedDict()
prior['omega'] = Gamma(1.0, 0.5)
prior['alpha'] = Beta(1.0, 1.0)
prior['beta'] = Beta(1.0, 1.0)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_garch = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch, **smc_opts_ibis, **pred_opts)
# alg.run()
models['GARCH (N)'] = fk_garch


# Jump-Mix-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'basic',
    'regimes': 2,
    'switching': 'mixing',
    'jumps': 'returns',
    'innov_X': 'N'
    }
# prior = set_prior(spec)
prior = OrderedDict()
prior['p_0'] = Beta(1.0, 1.0)
prior['omega_0'] = Gamma(1.0, 0.5)
prior['alpha_0'] = Beta(1.0, 1.0)
prior['beta_0'] = Beta(1.0, 1.0)
prior['omega_1'] = Gamma(1.0, 0.5)
prior['alpha_1'] = Beta(1.0, 1.0)
prior['beta_1'] = Beta(1.0, 1.0)
prior['lambda_X'] = Beta(1., 1.)
prior['phi_X'] = Gamma(1., .5)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_garch = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Jump-Mix-GARCH (N)'] = fk_garch


# GJR-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'gjr',
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1.0, 0.5)
prior['alpha'] = Beta(1.0, 1.0)
prior['beta'] = Beta(1.0, 1.0)
prior['gamma'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_gjr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr, **smc_opts_ibis, **pred_opts)
# alg.run()
models['GJR-GACH (N)'] = fk_gjr


# T-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'thr',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1.0, 0.5)
prior['alpha'] = Beta(1.0, 1.0)
prior['beta'] = Beta(1.0, 1.0)
prior['gamma'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_thr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_thr, **smc_opts_ibis, **pred_opts)
# alg.run()
models['T-GACH (N)'] = fk_thr


# E-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'exp',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['omega'] = Normal(0., 10.)
prior['alpha'] = Normal(0., 10.)
prior['beta'] = Normal(0., 10.)
prior['gamma'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_exp = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp, **smc_opts_ibis, **pred_opts)
# alg.run()
models['E-GARCH (N)'] = fk_exp


# Guyon & Lekeufack
spec = {
    'dynamics': 'guyon',
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., 0.5)
prior['alpha'] = Gamma(1., 0.5)
prior['beta'] = Gamma(1., 0.5)
prior['a1'] = shGamma(1., 0.5, loc=1.)
prior['d1'] = Gamma(1., 0.5)
prior['a2'] = shGamma(1., 0.5, loc=1.)
prior['d2'] = Gamma(1., 0.5)
# prior['df'] = shGamma(1., 0.5, loc=2.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_guyon = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_guyon, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Guyon (N)'] = fk_guyon


# ELM-GARCH
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1.0,         # st. dev. of random matrix components
    'activ': relu     # activation function
                     # ('Id', 'sigmoid', 'tanh', 'relu', 'shi)
    }
spec = {
    'dynamics': 'elm',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(0., 10.)
prior['w1'] = Normal(0., 10.)
prior['w2'] = Normal(0., 10.)
prior['w3'] = Normal(0., 10.)
prior['w4'] = Normal(0., 10.)
prior['w5'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_elm = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm, **smc_opts_ibis, **pred_opts)
# alg.run()
models['ELM-GARCH (N)'] = fk_elm


# ESN
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1.0,         # st. dev. of random matrix components
    'activ': shi  # activation function
    }
spec = {
    'dynamics': 'esn',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(0., 10.)
prior['w1'] = Normal(0., 10.)
prior['w2'] = Normal(0., 10.)
prior['w3'] = Normal(0., 10.)
prior['w4'] = Normal(0., 10.)
prior['w5'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_esn = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_esn, **smc_opts_ibis, **pred_opts)
# alg.run()
models['ESN (N)'] = fk_esn


# (log-)Signature
hyper = {  # model hyperparameters
    'q': 5,  # dim. of signature
    }
spec = {
    'dynamics': 'sig',
    'variant': 'logsig',  # 'standard' or 'log'
    'hyper': hyper,
    'switching': None,
    'regimes': 1,
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(0., 10.)
prior['w1'] = Normal(0., 10.)
prior['w2'] = Normal(0., 10.)
prior['w3'] = Normal(0., 10.)
prior['w4'] = Normal(0., 10.)
prior['w5'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_sig = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_sig, **smc_opts_ibis, **pred_opts)
# alg.run()
models['log-Sig (N)'] = fk_sig


# Randomized Signature
hyper = {  # model hyperparameters
    'q': 5,           # dim. of Sig
    'sd': 1.,         # st. dev. of rSig_0
    'activ': shi  # activation function
    }
spec = {
    'dynamics': 'sig',
    'variant': 'r-sig',
    'hyper': hyper,
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(0., 10.)
prior['w1'] = Normal(0., 10.)
prior['w2'] = Normal(0., 10.)
prior['w3'] = Normal(0., 10.)
prior['w4'] = Normal(0., 10.)
prior['w5'] = Normal(0., 10.)
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_rsig = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_rsig, **smc_opts_ibis, **pred_opts)
# alg.run()
models['rand-Sig (N)'] = fk_rsig




##################
#  SMC^2 Models  #
##################

# common parameters
fk_opts_smc2 = {
    'init_Nx': 50,            # initial no. of V-particles
    'ar_to_increase_Nx': 0.5,  #
    'fk_cls': Bootstrap,       # PF type (GuidedPF or Bootstrap/None)
    'wastefree': False,        #
    'len_chain': 20,           #
    'move': None               #
    }

smc_opts_smc2 = {  # only relevant if individual models run
    'N': 100,      # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'qmc': False,   # Quasi-Monte Carlo
    'verbose': False
    }


# Piecewise Constant Volatility
spec = {
    'variant': 'pwconst',
    'regimes': 1,
    'leverage': False,
    'switching': None,
    'jumps': 'vol',
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
# prior['df_V'] = shGamma(a=1., b=0.5, loc=2.)
prior['lambda_V'] = Beta(a=1., b=1.)
prior['phi_V'] = Gamma(a=1., b=.5)
prior = StructDist(prior)
model = pwc_sv.PWConstSV
fk_pwc = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_pwc, **smc_opts_smc2, **pred_opts)
# alg.run()


# Canonical Stochastic Volatility
spec = {
    'variant': 'canonical',
    'regimes': 1,
    'leverage': False,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['omega'] = Normal(scale=2.)
prior['alpha'] = Beta(a=1., b=1.)
prior['xi'] = Gamma(a=1., b=.5)
# prior['df'] = shGamma(a=3., b=3., loc=2.)
# prior['df_X'] = shGamma(a=3., b=3., loc=2.)
# prior['df_V'] = shGamma(a=3., b=3., loc=2.)
# prior['lambda_X'] = Beta(a=1., b=1.)
# prior['phi_X'] = Gamma(a=3., b=3.)
prior = StructDist(prior)
model = canon_sv.CanonSV
fk_sv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_sv, **smc_opts_smc2, **pred_opts)
# alg.run()


# Heston Model
spec = {
    'dynamics': 'heston',
    'regimes': 1,
    'leverage': False,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['nu'] = Gamma(a=3., b=3.)
prior['kappa'] = Gamma(a=3., b=3.)
prior['xi'] = Gamma(a=3., b=3.)
# prior['df'] = shGamma(a=3., b=3., loc=2.)
# prior['df_X'] = shGamma(a=3., b=3., loc=2.)
# prior['df_V'] = shGamma(a=3., b=3., loc=2.)
prior = StructDist(prior)
model = heston.Heston
fk_hest = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_hest, **smc_opts_smc2, **pred_opts)
# alg.run()


# Bates
spec = {
    'dynamics': 'heston',
    'regimes': 1,
    'switching': None,
    'jumps': 'returns',
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['nu'] = Gamma(1., .5)
prior['kappa'] = Gamma(1., .5)
prior['xi'] = Gamma(1., .5)
prior['lambda_X'] = Beta(1., 1.)
prior['phi_X'] = Gamma(1., .5)
prior = StructDist(prior)
model = heston.Heston
fk_bates = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_bates, **smc_opts_smc2, **pred_opts)
# alg.run()


# Barndorff-Nielsen-Shephard
spec = {
    'dynamics': 'heston',
    'regimes': 1,
    'switching': None,
    'jumps': 'vol',
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['nu'] = Gamma(1., .5)
prior['kappa'] = Gamma(1., .5)
prior['xi'] = Gamma(1., .5)
prior['lambda_V'] = Beta(1., 1.)
prior['phi_V'] = Gamma(1., .5)
prior = StructDist(prior)
model = heston.Heston
fk_bns = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_bates, **smc_opts_smc2, **pred_opts)
# alg.run()


# Neural StochVol
spec = {
    'dynamics': 'neural',
    'leverage': False,
    'jumps': None,
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['v0'] = Normal(scale=1.)
prior['v1'] = Normal(scale=1.)
prior['v2'] = Normal(scale=1.)
prior['v3'] = Normal(scale=1.)
prior['v4'] = Normal(scale=1.)
prior['v5'] = Normal(scale=1.)
prior['xi'] = Gamma(a=3., b=3.)
# prior['df'] = shGamma(a=3., b=3., loc=2.)
# prior['df_X'] = shGamma(a=3., b=3., loc=2.)
# prior['df_V'] = shGamma(a=3., b=3., loc=2.)
prior = StructDist(prior)
model = elm_sv.NeuralSV
fk_neursv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_neursv, **smc_opts_smc2, **pred_opts)
# alg.run()


# Echo State StochVol
spec = {
    'dynamics': 'esn',
    'leverage': False,
    'jumps': None,
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(scale=1.)
prior['w1'] = Normal(scale=1.)
prior['w2'] = Normal(scale=1.)
prior['w3'] = Normal(scale=1.)
prior['w4'] = Normal(scale=1.)
prior['w5'] = Normal(scale=1.)
prior['xi'] = Gamma(a=3., b=3.)
# prior['df'] = shGamma(a=3., b=3., loc=2.)
# prior['df_X'] = shGamma(a=3., b=3., loc=2.)
# prior['df_V'] = shGamma(a=3., b=3., loc=2.)
prior = StructDist(prior)
model = esn_sv.EchoSV
fk_esnsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_esnsv, **smc_opts_smc2, **pred_opts)
# alg.run()


# Randomized Signature StochVol
spec = {
    'model': 'randsig_sv',
    'leverage': False,
    'jumps': None,
    'innov_X': 'N',
    'innov_V': 'N'
    }
prior = OrderedDict()
prior['w0'] = Normal(scale=1.)
prior['w1'] = Normal(scale=1.)
prior['w2'] = Normal(scale=1.)
prior['w3'] = Normal(scale=1.)
prior['w4'] = Normal(scale=1.)
prior['w5'] = Normal(scale=1.)
prior['xi'] = Gamma(a=3., b=3.)
prior = StructDist(prior)
model = randsig_sv.RandSigSV
fk_rsigsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_rsigsv, **smc_opts_smc2, **pred_opts)
# alg.run()


######################
#  MODEL COMPARISON  #
######################

multismc_opts = {
    'N': 100,       # no. of particles
    'nruns': 1,     # no. of runs (for mean & SE estimates)
    'nprocs': -2,   # - no. of CPU cores to keep idle during parallel computation
    'store_history': False,
    'collect': [    # variables collected at all times
        LogLts(),   # model evidence
        ESSs(),     # effective sample size
        Rs_flags()  # times when resampling triggered
        # Preds(),     # 1-day ahead point predictions
        # PredSets()   # 1-day ahead prediction sets
        ],
    'out_func': out_fct,  # to only store what's necessary to save memory
    'verbose': False
    }


print('Models:')
for _, m in enumerate(models):
    print(' -', m)

print('\nSettings:')
print(' - Runs:', multismc_opts['nruns'])
print(' - θ-particles:', multismc_opts['N'])
print(' - CPU cores used:', cpu_count() + multismc_opts['nprocs'])
print(' - CPU cores idle:', abs(multismc_opts['nprocs']))

print('\nData:')
print(' - Length:', len(X))

# run all models in parallel
runs = multiSMC(fk=models, **multismc_opts, **pred_opts)

# analyze SMC & posterior dist
smc_summary(runs, M0='White Noise (N)', dataset=ticker, save_plots=False)

# analyze predictions & prediction sets
truths = {}
truths['RV'] = X**2
truths['S'] = S[1:]
truths['C'] = np.clip(S[1:] - S[:-1], 0.0, None)

pred_summary(runs, 'White Noise (N)', truths)
