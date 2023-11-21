#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# wd & stuff
# import os
# wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
# os.chdir(wd)

# basic packages
import numpy as np
from scipy import stats

# quant finance packages
import yfinance as yfin

# SMC packages
from particles.core import SMC, multiSMC
from particles.smc_samplers import IBIS, SMC2
from particles.state_space_models import Bootstrap, GuidedPF
from particles.collectors import LogLts, ESSs, Rs_flags

# visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# other
from multiprocessing import cpu_count

# own model classes
# from static_models import *
# from dynamic_models import *
# from helpers import *


# data from Mixture Model:
T = 300
w0, a0, b0 = 0.01, 0.2, 0.6
w1, a1, b1 = 0.2, 0.6, 0.35
sigma_0 = 1
sigma_1 = 1
X = np.zeros(T)
X[0] = np.random.normal()*sigma_0
vol_true = np.full(T, np.nan)
for t in range(1, T):
    sigma_0 = np.sqrt(w0 + a0*X[t-1]**2 + b0*sigma_0**2)
    sigma_1 = np.sqrt(w1 + a1*X[t-1]**2 + b1*sigma_1**2)
    r = np.random.binomial(n=1, p=1.)
    vol_true[t] = (sigma_0, sigma_1)[r]
    if r == 0:
        X[t] = np.random.normal(scale=sigma_0)
    else:
        X[t] = np.random.normal(scale=sigma_1)


# real data:
ticker = 'MSFT'
SP500 = yfin.Ticker(ticker).history(period="2y")['Close'].values  # raw prices
plt.figure(dpi=800)
plt.plot(SP500)

# log-returns
SP500 = 100 * np.log(SP500[1:]/SP500[:-1])
plt.figure(dpi=800)
plt.plot(SP500)

# density plot
plt.figure(dpi=800)
sns.histplot(SP500, stat='density', kde=True)

# absolute returns
plt.figure(dpi=800)
plt.plot(abs(SP500))

data = X[0:200]
plt.plot(data)
# np.savetxt("SP500.csv", data, delimiter=",")

# data = np.loadtxt("SP500.csv", delimiter=",")


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


# BENCHMARK: White noise
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': None,
    'jumps': None,
    'innov_X': 't'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_bench = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_bench, **smc_opts_ibis)
# alg.run()


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
model = DetVol(spec=spec, data=data, prior=prior)
fk_const = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_const, **smc_opts_ibis)
# alg.run()


# Standard GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'basic',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
# prior['p_0'] = Beta(1., 1.)
prior['omega'] = Gamma(1., 0.5)
prior['alpha'] = Gamma(1., .5)
prior['beta'] = Beta(1., 1.)
# prior['df_X'] = shGamma(1., .5, 2.)
# prior['omega_1'] = Gamma(1., 0.5)
# prior['alpha_1'] = Gamma(1., .5)
# prior['beta_1'] = Beta(1., 1.)
# prior['lambda_X'] = Beta(1., 1.)
# prior['phi_X'] = Gamma(1., .5)
prior = StructDist(prior)
model = DetVol(spec=spec, data=data, prior=prior)
fk_garch = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch, **smc_opts_ibis)
# alg.run()


# GJR-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'gjr',
    'regimes': 2,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_gjr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr, **smc_opts_ibis)
# alg.run()


# T-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'thr',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_thr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_thr, **smc_opts_ibis)
# alg.run()


# E-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'exp',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_exp = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp, **smc_opts_ibis)
# alg.run()


# ELM-GARCH
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 3.,         # st. dev. of random matrix components
    'activ': sigmoid  # activation function
    }
spec = {
    'dynamics': 'garch',
    'variant': 'elm',
    'hyper': hyper,
    'switching': 'mixing',
    'regimes': 1,
    'jumps': None,
    'innov_X': 't'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_elm = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_elm, **smc_opts_ibis)
alg.run()

theta_hat = alg.X.theta
W = alg.wgts.W
predict(model, theta_hat, W, vol_true, data,
        s=100, M=1)


# Guyon & Lekeufack
spec = {
    'dynamics': 'guyon',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 't'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., .5)
prior['alpha'] = Gamma(1., .5)
prior['beta'] = Gamma(1., .5)
prior['a1'] = shGamma(1., .5, loc=1.)
prior['d1'] = Gamma(1., .5)
prior['a2'] = shGamma(1., .5, loc=1.)
prior['d2'] = Gamma(1., .5)
prior['df'] = shGamma(1., .5, loc=2.)
prior = StructDist(prior)
model = DetVol(spec=spec, data=data, prior=prior)
fk_guyon = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_guyon, **smc_opts_ibis)
# alg.run()


# Signature Model
hyper = {  # model hyperparameters
    'q': 5,  # dim. of signature
    }
spec = {
    'dynamics': 'sig',
    'variant': 'log',  # 'standard' or 'log'
    'hyper': hyper,
    'switching': 'mixing',
    'regimes': 1,
    'jumps': None,
    'innov_X': 't'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_sig = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_sig, **smc_opts_ibis)
alg.run()


# Randomized Signature Model
hyper = {  # model hyperparameters
    'q': 10,           # dim. of truncated/randomized Sig
    'sd': 1.,         # st. dev. of rSig_0
    'activ': sigmoid  # activation function
    }
spec = {
    'dynamics': 'sig',
    'variant': 'r_sig',
    'hyper': hyper,
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, data=data, prior=prior)
fk_rsig = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_rsig, **smc_opts_ibis)
# alg.run()


##################
#  SMC^2 Models  #
##################

# common parameters
fk_opts_smc2 = {
    'data': data,              #
    'init_Nx': 100,            # initial no. of V-particles
    'ar_to_increase_Nx': 0.1,  #
    'fk_cls': Bootstrap,       # PF type (GuidedPF or Bootstrap/None)
    'wastefree': False,        #
    'len_chain': 20,           #
    'move': None               #
    }

smc_opts_smc2 = {  # only relevant if individual models run
    'N': 100,      # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'qmc': False   # Quasi-Monte Carlo
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
model = PiecewiseConst
fk_pwc = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_pwc, **smc_opts_smc2)
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
model = StochVol
fk_sv = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_sv, **smc_opts_smc2)
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
model = Heston
fk_hest = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
alg = SMC(fk_hest, **smc_opts_smc2)
alg.run()


# Bates Model
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
model = Heston
fk_bates = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
alg = SMC(fk_bates, **smc_opts_smc2)
alg.run()


# Barndorff-Nielsen-Shephard Model
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
model = Heston
fk_bns = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_bates, **smc_opts_smc2)
# alg.run()


# Neural StochVol Model
spec = {
    'dynamics': 'neural',
    'regimes': 1,
    'leverage': False,
    'switching': 'mixing',
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
model = NeuralSV
fk_neur = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_hest, **smc_opts_smc2)
# alg.run()


######################
#  MODEL COMPARISON  #
######################

multismc_opts = {
    'N': 100,       # no. of particles
    'nruns': 2,     # no. of runs (for mean & SE estimates)
    'nprocs': -2,   # - no. of CPU cores to keep idle during parallel computation
    'store_history': False,
    'collect': [    # variables collected at all times
        LogLts(),   #  model evidence
        ESSs(),     #  effective sample size
        Rs_flags()  #  times when resampling triggered
        ],
    'out_func': out_fct,  # to only store what's necessary to save memory
    'verbose': False
    }

models = {}
models['White Noise (t)'] = fk_bench
models['GARCH (t)'] = fk_garch
# models['GJR-GARCH (t)'] = fk_gjr
# models['T-GARCH (t)'] = fk_thr
# models['E-GARCH (t)'] = fk_exp
# models['Guyon (t)'] = fk_guyon
# models['ELM-GARCH (t)'] = fk_elm
# models['Sig-GARCH (N)'] = fk_sig
# models['rSig-GARCH (N)'] = fk_rsig
# models['Pure Jump (t, t)'] = fk_pjv
# models['StochVol (N, N)'] = fk_sv
models['Heston (N, N)'] = fk_hest
# models['Merton (N)'] = fk_mert
# models['Bates (N)'] = fk_bates
# models['BNS (N)'] = fk_bns
models['Neur/Hest-SV (N, N)'] = fk_neur

print('Models:')
for _, m in enumerate(models):
    print(' -', m)

print('\nSettings:')
print(' - Runs:', multismc_opts['nruns'])
print(' - θ-particles:', multismc_opts['N'])
print(' - CPU cores used:', cpu_count() + multismc_opts['nprocs'])
print(' - CPU cores idle:', abs(multismc_opts['nprocs']))

print('\nData:')
print(' - Length:', len(data))

# run all models in parallel
runs = multiSMC(fk=models, **multismc_opts)

# summarize
smc_summary(runs, M0="t Noise", dataset="SP500",
            plot_Z=True, plot_post=True, diagnostics=True, save_plots=False)
