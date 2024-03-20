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
from particles.distributions import *
from particles.smc_samplers import IBIS, SMC2
from particles.state_space_models import StateSpaceModel, Bootstrap, GuidedPF
from particles.collectors import LogLts, ESSs, Rs_flags

# utils
from statsmodels.stats.weightstats import DescrStatsW
from numpy.lib.recfunctions import structured_to_unstructured
from operator import itemgetter

# other
from multiprocessing import cpu_count
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# own stuff
import os
# wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
# os.chdir(wd)
from det_vol import *
from pwc_sv import *
from canon_sv import *
from heston import *
from rescomp_sv import *
from helpers import *
from plots import *


X_N = np.random.normal(size=100000)
X_t = stats.t.rvs(df=5, size=100000)
S_N = 100. * np.exp(0.01*X_N)
S_t = 100. * np.exp(0.01*X_t)
C_N = np.clip(S_N-100., 0., None)
C_t = np.clip(S_t-100., 0., None)
C_N = C_N[(C_N>0.) * (C_N<6.)]
C_t = C_t[(C_t>0.) * (C_t<6.)]

plt.hist(C_N, alpha=0.5, bins=100, label="Gaussian", density=True)
plt.hist(C_t, alpha=0.5, bins=100, label="Student t", density=True)
plt.legend()

# determine data
series = 'BTC-USD'

# fetch & save data from Yahoo Finance
# S = yfin.Ticker(series).history(start='2006-01-01', end='2013-12-31')['Close'].values
# np.savetxt('data_' + series + '.csv', S, delimiter=",")

# load data
S = np.loadtxt('data/data_' + series + '.csv', delimiter=",")
# S = S[:200]
S_full = S.copy()

# log-returns
X = 100.0 * np.log(S[1:]/S[:-1])


# Options for predictions:
pred_opts = {
    'h': 1,            # horizon, no. of days ahead to predict / option maturity
    'M': 10**4,        # no. of simulations for MC estimations (e.g. options)
    'alpha': 0.1,      # miscoverage tolerance for prediction intervals
    'naive_uq': True,  # naive Bayesian pred' sets
    'cal_uq': True,    # calibrated Bayesian pred' sets
    'eta': 0.005,      # learning rate for quantile updates
    'strike': 'last'   # strike price of (European call) option
    }


###############
# IBIS MODELS #
###############

# define common parameters

fk_opts_ibis = {
    'len_chain': 3,     # length of MCMC 'move' chain
    'wastefree': False  # whether to use intermediate MCMC particles
    }

smc_opts_ibis = {  # only relevant if individual models run
    'N': 100,                   # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'ESSrmin': 0.5,              # degeneracy criterion
    'qmc': False,
    'verbose': False             # whether to print ESS while fitting
    }


# collect models to be run in parallel:
models = {}


# White noise
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': None,  # 'mix' or 'markov'
    'jumps_X': False,
    'innov_X': 't'      # 'N', 't', or 'GH'
    }
prior = OrderedDict()
prior['sigma'] = Gamma(2., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_bench = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_bench, **smc_opts_ibis, **pred_opts)
# alg.run()
models['White Noise (t)'] = fk_bench


# Merton Model
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': None,
    'jumps_X': True,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['sigma'] = Gamma(2., 1.)
prior['lambda_X'] = Beta(1., 10.)
prior['phi_X'] = Gamma(2., 1.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_merton = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_merton, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Merton (N)'] = fk_merton


# Merton Model (GH)
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': None,
    'jumps_X': True,
    'innov_X': 'GH'
    }
prior = OrderedDict()
prior['sigma'] = Gamma(2., 1.)
prior['lambda_X'] = Beta(1., 10.)
prior['phi_X'] = Gamma(2., 1.)
prior['tail_X'] = Normal(-10., 3.)
prior['shape_X'] = Normal(0., 2.)
prior['skew_X'] = Normal(0., 2.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_merton = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_merton, **smc_opts_ibis, **pred_opts)
# alg.run()
# models['Merton (GH)'] = fk_merton


# Standard GARCH (N)
spec = {
    'dynamics': 'garch',
    'variant': 'basic',
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., 1.)
prior['alpha'] = Beta(3., 1.)
prior['beta'] = Beta(1., 3.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_garch = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch, **smc_opts_ibis, **pred_opts)
# alg.run()
models['GARCH (t)'] = fk_garch


# Jump-Mix-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'basic',
    'regimes': 2,
    'switching': 'mixing',
    'jumps_X': True,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['p_0'] = Uniform(0., 1.)
prior['omega_0'] = Gamma(1., 1.)
prior['omega_1'] = Gamma(1., 1.)
prior['alpha_0'] = Beta(3., 1.)
prior['alpha_1'] = Beta(3., 1.)
prior['beta_0'] = Beta(1., 3.)
prior['beta_1'] = Beta(1., 3.)
prior['lambda_X'] = Beta(1., 10.)
prior['phi_X'] = Gamma(2., 1.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_garch = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Jump-Mix-GARCH (N)'] = fk_garch


# GJR-GARCH (N)
spec = {
    'dynamics': 'garch',
    'variant': 'gjr',
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., 1.)
prior['alpha'] = Beta(3., 1.)
prior['beta'] = Beta(1., 3.)
prior['gamma'] = Normal(0.5, 2.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_gjr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr, **smc_opts_ibis, **pred_opts)
# alg.run()
models['GJR-GARCH (t)'] = fk_gjr


# T-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'thr',
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., 1.)
prior['alpha'] = Beta(3., 1.)
prior['beta'] = Beta(1., 3.)
prior['gamma'] = Normal(0., 2.)
prior['df_X'] = shGamma(6., 1., loc=3.0)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_thr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_thr, **smc_opts_ibis, **pred_opts)
# alg.run()
models['T-GARCH (t)'] = fk_thr


# E-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'exp',
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Normal(0., 2.)
prior['alpha'] = Normal(0., 2.)
prior['beta'] = TruncNormal(0., 2., a=-1., b=1.)
prior['gamma'] = Normal(0., 2.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_exp = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp, **smc_opts_ibis, **pred_opts)
# alg.run()
models['E-GARCH (t)'] = fk_exp


# Guyon & Lekeufack (N)
spec = {
    'dynamics': 'guyon',
    'variant': 'basic',
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., 0.5)
prior['alpha'] = Gamma(1., 0.5)
prior['beta'] = Gamma(1., 0.5)
prior['a1'] = shGamma(3., 1., loc=1.)
prior['d1'] = Gamma(3., 1.)
prior['a2'] = shGamma(3., 1., loc=1.)
prior['d2'] = Gamma(3., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_guyon = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_guyon, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Guyon (t)'] = fk_guyon


# ELM-GARCH (N)
hyper = {  # model hyperparameters
    'q': 15,      # dim. of reservoir
    'L': 1,       # no. of random hidden layers
    'H': None,    # width of first hidden layer (if L>1)
    'sd': 1.0,    # st. dev. of random matrix components
    'activ': shi  # 'Id', 'sigmoid', 'tanh', 'relu', 'shi'
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'elm',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_elm = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm, **smc_opts_ibis, **pred_opts)
# alg.run()
models['ELM-GARCH (t)'] = fk_elm


# Deep ELM-GARCH (N)
hyper = {  # model hyperparameters
    'q': 10,         # dim. of reservoir
    'L': 2,
    'H': 10,
    'sd': 1.0,       # st. dev. of random matrix components
    'activ': shi     # 'Id', 'sigmoid', 'tanh', 'relu', 'shi'
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'elm',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w'+str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_elm = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Deep ELM-GARCH (t)'] = fk_elm


# ESN
hyper = {  # model hyperparameters
    'q': 10,           # dim. of reservoir
    'sd': 1.0,         # st. dev. of random matrix components
    'activ': shi  # activation function
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'esn',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_esn = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_esn, **smc_opts_ibis, **pred_opts)
# alg.run()
models['ESN DV (t)'] = fk_esn


# Barron
hyper = {  # model hyperparameters
    'q': 10,      # dim. of reservoir
    'sd': 1.0,    # st. dev. of random matrix components
    'activ': shi  # activation function
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'barron',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_bar = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_bar, **smc_opts_ibis, **pred_opts)
# alg.run()
models['Barron DV (t)'] = fk_bar


# (log-)Signature
hyper = {  # model hyperparameters
    'q': 10,  # dim. of signature
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'logsig',  # 'sig' or 'logsig'
    'hyper': hyper,
    'switching': None,
    'regimes': 1,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_sig = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_sig, **smc_opts_ibis, **pred_opts)
# alg.run()
models['log-Sig DV (t)'] = fk_sig


# Randomized Signature
hyper = {  # model hyperparameters
    'q': 7,      # dim. of Sig
    'sd': 1.,     # st. dev. of rSig_0
    'activ': shi  # activation function
    }
spec = {
    'dynamics': 'rescomp',
    'variant': 'rand-sig',
    'hyper': hyper,
    'regimes': 1,
    'switching': None,
    'jumps_X': False,
    'innov_X': 't'
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['df_X'] = shGamma(6., 1., loc=3.)
fk_opts_ibis['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
model = DetVol(spec, prior, S)
fk_rsig = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_rsig, **smc_opts_ibis, **pred_opts)
# alg.run()
models['rand-Sig DV (t)'] = fk_rsig



##################
#  SMC^2 Models  #
##################

# use only T=1'000 for StochVol models
S = S_full[0:1001]

# common parameters
fk_opts_smc2 = {
    'init_Nx': 100,            # initial no. of V-particles
    'ar_to_increase_Nx': 0.1,  # PF degeneracy criterion
    'fk_cls': Bootstrap,       # PF type (Bootstrap or GuidedPF)
    'wastefree': False,        #
    'len_chain': 3,            #
    }

smc_opts_smc2 = {  # only relevant if individual models run
    'N': 100,      # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'qmc': False,   # Quasi-Monte Carlo
    'verbose': False
    }


# Piecewise Constant Volatility Models
# changing class attributes doesn't work w multiproc!
# change in class directly
spec = {
    'regimes': 1,
    'switching': None,
    'leverage': True,
    'jumps_X': False,
    'jumps_V': True,
    'innov_X': 't',
    'innov_V': 'N'
    }
# model = PWConstSV
# model.define_model(**spec)

# Basic PWC
model = PWConstSV
prior = OrderedDict()
prior['df_X'] = shGamma(6., 1., loc=3.)
prior['lambda_V'] = Beta(1., 10.)
prior['phi_V'] = Gamma(2., 1.)
fk_opts_smc2['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
fk_pwc = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_pwc, **smc_opts_smc2, **pred_opts)
# alg.run()
models['PWC SV (t,N)'] = fk_pwc

####

# Canonical StochVol Models
# changing class attributes doesn't work w multiproc!
# change in class directly
spec = {
    'regimes': 1,
    'switching': None,
    'leverage': True,
    'jumps_X': False,
    'jumps_V': False,
    'innov_X': 'N',
    'innov_V': 'N'
    }
# CanonSV.define_model(**spec)  # doesn't work with multiSMC ???

# Canonical SV
model = CanonSV
prior = OrderedDict()
prior['omega'] = Normal(scale=2.)
prior['alpha'] = Beta(3., 1.)
prior['xi'] = Gamma(2., 2.)
# prior['rho'] = Uniform(-1., 1.)
fk_opts_smc2['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
fk_sv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_sv, **smc_opts_smc2, **pred_opts)
# alg.run()
models['Canonical SV (N,N)'] = fk_sv


# Bates w Canonical basis
model = BatesCanon
prior = OrderedDict()
prior['omega'] = Normal(scale=2.)
prior['alpha'] = Beta(3., 1.)
prior['xi'] = Gamma(2., 2.)
# prior['rho'] = Uniform(-1., 1.)
prior['lambda_X'] = Beta(1., 10.)
prior['phi_X'] = Gamma(2., 1.)
fk_opts_smc2['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
fk_sv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_sv, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['Canonical Bates (N,N)'] = fk_sv


# BNS w Canonical basis
model = BNSCanon
prior = OrderedDict()
prior['omega'] = Normal(scale=2.)
prior['alpha'] = Beta(3., 1.)
prior['xi'] = Gamma(2., 2.)
# prior['rho'] = Uniform(-1., 1.)
prior['lambda_V'] = Beta(1., 10.)
prior['phi_V'] = Gamma(2., 1.)
fk_opts_smc2['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
fk_sv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_sv, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['Canonical BNS (N,N)'] = fk_sv

####

# Heston-type Models
# changing class attributes doesn't work w multiproc!
# change in class directly
spec = {
    'regimes': 1,
    'switching': None,
    'leverage': True,
    'jumps_X': False,
    'jumps_V': False,
    'innov_X': 'N',
    'innov_V': 'N'
    }
# Heston.define_model(**spec)

# Basic Heston
model = Heston
prior = OrderedDict()
prior['nu'] = Normal(scale=2.)
prior['kappa'] = Gamma(1., 1.)
prior['xi'] = Gamma(2., 2.)
# prior['rho'] = Uniform(-1.0, 1.0)
fk_opts_smc2['len_chain'] = max(3, int(0.7*len(prior)))
prior = StructDist(prior)
fk_hest = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_hest, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['Heston (N,N)'] = fk_hest

####

# Reservoir Computers StochVol
# changing class attributes doesn't work w multiproc!
# change in class directly
hyper = {
    'q': 5,
    'sd': 1.0,
    'activ': shi
    }
spec = {
    'vol_drift': 'rescomp',
    'vol_vol': 'canonical',
    'leverage': False,
    'jumps_X': False,
    'jumps_V': False,
    'innov_X': 'N',
    'innov_V': 'N',
    }
# spec.update(hyper)
# ResCompSV.define_model(**spec)

# use only T=500 for ResComp StochVol
# S = S_full[0:500]

####

# ELM StochVol
model = ExtremeSV
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['xi'] = Gamma(a=2., b=2.)
fk_opts_smc2['len_chain'] = max(3, int(0.7 * len(prior)))
prior = StructDist(prior)
fk_elmsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_elmsv, **smc_opts_smc2, **pred_opts)
# alg.run()
models['ELM/Const SV (N,N)'] = fk_elmsv


# Echo State StochVol
model = EchoSV
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['xi'] = Gamma(a=2., b=2.)
fk_opts_smc2['len_chain'] = max(3, int(0.7 * len(prior)))
prior = StructDist(prior)
fk_esnsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_esnsv, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['ESN/Const SV (N,N)'] = fk_esnsv


# Signature StochVol
model = SigSV
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['xi'] = Gamma(a=2., b=2.)
fk_opts_smc2['len_chain'] = max(3, int(0.7 * len(prior)))
prior = StructDist(prior)
fk_sigsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_sigsv, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['log-Sig/Const SV (N,N)'] = fk_sigsv


# Randomized Signature StochVol
model = RandSigSV
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = Normal(0., 1.)
prior['xi'] = Gamma(a=2., b=2.)
fk_opts_smc2['len_chain'] = max(3, int(0.7 * len(prior)))
prior = StructDist(prior)
fk_rsigsv = SMC2(model, prior, S, **fk_opts_smc2)
# alg = SMC(fk_rsigsv, **smc_opts_smc2, **pred_opts)
# alg.run()
# models['r-Sig/Const SV (N,N)'] = fk_rsigsv



######################
#  MODEL COMPARISON  #
######################

multismc_opts = {
    'N': 500,        # no. of particles
    'nruns': 3,       # no. of runs (for mean & SE estimates)
    'nprocs': -2,     # - no. of CPU cores to keep idle during parallel computation
    'store_history': False,
    'collect': [      # variables collected at all times
        LogLts(),     # model evidence
        ESSs(),       # effective sample size
        Rs_flags()    # times when resampling triggered
        ],
    'out_func': out_fct,  # to only store what's necessary to save memory
    'verbose': False
    }

print('\nData:')
print(' - Series:', series)
print(' - Length:', len(S_full))


print('\nModels:')
for _, m in enumerate(models):
    print(' -', m)

print('\nSettings:')
print(' - Runs:', multismc_opts['nruns'])
print(' - θ-particles:', multismc_opts['N'])
print(' - x-particles (initial):', fk_opts_smc2['init_Nx'])
print(' - Particle Filter:', fk_opts_smc2['fk_cls'].__name__)
print(' - CPU cores used:', cpu_count() + multismc_opts['nprocs'])
print(' - CPU cores idle:', abs(multismc_opts['nprocs']))


# run all models in parallel
runs = multiSMC(fk=models, **multismc_opts, **pred_opts)

# add ground truths
truths = {}
truths['S'] = S_full[1:]
truths['RV'] = X**2
truths['C'] = np.clip(S_full[1:]-S_full[:-1], 0.0, None)
runs.append(truths)

# save results
import pickle

filename = 'results_' + series
with open(filename +'.pickle', 'wb') as handle:
    pickle.dump(runs, handle, protocol=pickle.HIGHEST_PROTOCOL)


