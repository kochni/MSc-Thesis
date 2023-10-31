#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# wd & stuff
# import os
# wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
# os.chdir(wd)

# basic packages
import numpy as np
import pandas as pd
from scipy import stats

# quant finance packages
import yfinance as yfin

# SMC packages
from particles.core import SMC, multiSMC
from particles.smc_samplers import IBIS, SMC2
from particles.state_space_models import Bootstrap, GuidedPF
from particles.distributions import StructDist, OrderedDict, IndepProd, TruncNormal, Beta
from particles.collectors import LogLts, ESSs, Rs_flags

# visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# own model classes
from static_models import *
from dynamic_models import *
from helpers import *


# synthetic data:
# T = 1000
# a0, a1, b1, g = 0.05, 0.2, 0.5, 0.5
# X = np.zeros(T)
# s2 = 1
# X[0] = np.random.normal()*np.sqrt(s2)
# for t in range(1, T):
#     s2_prev = s2
#     s2 = a0 + (a1 + g*(X[t-1]<0))*X[t-1]**2 + b1*s2_prev
#     X[t] = np.random.normal()*np.sqrt(s2)

# real data:
# ticker = '^GSPC'
# SP500 = yfin.Ticker(ticker).history(period="2y")['Close'].values
# SP500 = 100 * np.log(SP500[1:]/SP500[:-1])

# data = SP500
# plt.plot(data)
# np.savetxt("SP500.csv", data, delimiter=",")

data = np.loadtxt("SP500.csv", delimiter=",")


# DETERMINISTIC VOLATILITY MODELS

# define common parameters

# prior std of 'truncated normal' for parameters restricted to positive reals
# such that overall std is the same as for unconstrained ones
s = 10
s_pos = np.sqrt(s**2 / (1 - (2*stats.norm.pdf(0))**2))

fk_opts_ibis = {
    'len_chain': 20,    # length of MCMC 'move' chain
    'wastefree': False  # whether to use intermediate MCMC particles
    }

smc_opts_ibis = {  # only relevant if individual models run
    'N': 500,                    # no. of particles
    'resampling': 'systematic',  # resampling scheme
    'ESSrmin': 0.5,              # degeneracy criterion
    'verbose': False             # whether to print ESS while fitting
    }


# White noise (Gaussian)
prior = OrderedDict()
prior['sigma'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = WhiteNoise(innov='N', data=data, prior=prior)
fk_white_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_white_N, **smc_opts_ibis)
# alg.run()

# White noise (Student t)
prior = OrderedDict()
prior['sigma'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = WhiteNoise(innov='t', data=data, prior=prior)
fk_white_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_white_t, **smc_opts_ibis)
# alg.run()


# Standard GARCH (Gaussian)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='basic', innov='N', data=data, prior=prior)
fk_garch_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch_N, **smc_opts_ibis)
# alg.run()

# Standard GARCH (Student t)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='basic', innov='t', data=data, prior=prior)
fk_garch_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_garch_t, **smc_opts_ibis)
# alg.run()


# GJR-GARCH (Gaussian)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='gjr', innov='N', data=data, prior=prior)
fk_gjr_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr_N, **smc_opts_ibis)
# alg.run()

# GJR-GARCH (Student t)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='gjr', innov='t', data=data, prior=prior)
fk_gjr_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr_t, **smc_opts_ibis)
# alg.run()


# T-GARCH (Gaussian)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='thr', innov='N', data=data, prior=prior)
fk_thr_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_thr_N, **smc_opts_ibis)
# alg.run()

# T-GARCH (Student t)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='thr', innov='t', data=data, prior=prior)
fk_thr_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_thr_t, **smc_opts_ibis)
# alg.run()


# E-GARCH (Gaussian)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='exp', innov='N', data=data, prior=prior)
fk_exp_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp_N, **smc_opts_ibis)
# alg.run()

# E-GARCH (Student t)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['beta'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['gamma'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='exp', innov='t', data=data, prior=prior)
fk_exp_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp_t, **smc_opts_ibis)
# alg.run()


# Guyon & Lekeufack (Gaussian)
prior = OrderedDict()
prior['beta0'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['beta1'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['beta2'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['alpha1'] = TruncNormal(sigma=s_pos, a=1, b=np.exp(50))
prior['alpha2'] = TruncNormal(sigma=s_pos, a=1, b=np.exp(50))
prior['delta1'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['delta2'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='guyon', innov='N', data=data, prior=prior)
fk_guyon_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_guyon_N, **smc_opts_ibis)
# alg.run()

# Guyon & Lekeufack (Student t)
prior = OrderedDict()
prior['beta0'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['beta1'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['beta2'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['alpha1'] = TruncNormal(sigma=s_pos, a=1, b=np.exp(50))
prior['alpha2'] = TruncNormal(sigma=s_pos, a=1, b=np.exp(50))
prior['delta1'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['delta2'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='guyon', innov='t', data=data, prior=prior)
fk_guyon_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_guyon_t, **smc_opts_ibis)
# alg.run()


# ELM-GARCH (Gaussian)
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1,          # st. dev. of rSig_0
    'activ': sigmoid  # activation function
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = RCStatic(rc_type='elm', hyper=hyper, innov='N', data=data, prior=prior)
fk_elm_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm_N, **smc_opts_ibis)
# alg.run()

# ELM-GARCH (Student t)
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1,          # st. dev. of rSig_0
    'activ': sigmoid  # activation function
    }
prior = OrderedDict()
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
for j in range(hyper['q']+1):
    prior['w' + str(j)] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = RCStatic(rc_type='elm', innov='t', hyper=hyper, data=data, prior=prior)
fk_elm_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm_t, **smc_opts_ibis)
# alg.run()


# Signature GARCH (Gaussian)
pass

# Signature GARCH (Student t)
pass


# Randomized Signature GARCH (Gaussian)
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1,          # st. dev. of rSig_0
    'activ': sigmoid  # activation function
    }
prior = OrderedDict()
for j in range(hyper['q']+1):
    prior['w' + str(j)] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = RCStatic(rc_type='r_sig', hyper=hyper, innov='N',
                 data=data, prior=prior)
fk_rsig_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_rsig_N, **smc_opts_ibis)
# alg.run()

# Randomized Signature GARCH (Student t)
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1,          # st. dev. of rSig_0
    'activ': sigmoid  # activation function
    }
prior = OrderedDict()
prior['df'] = TruncNormal(sigma=s_pos, b=np.exp(50))
for j in range(hyper['q']+1):
    prior['w' + str(j)] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior = StructDist(prior)
model = RCStatic(rc_type='r_sig', hyper=hyper, innov='t',
                 data=data, prior=prior)
fk_rsig_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_rsig_t, **smc_opts_ibis)
# alg.run()


# STOCHASTIC VOLATILITY MODELS

# common parameters
fk_opts_smc2 = {
    'data': data,              #
    'init_Nx': 500,            #
    'ar_to_increase_Nx': 0.1,  #
    'fk_cls': Bootstrap,       # PF type (GuidedPF or Bootstrap/None)
    'wastefree': False,        #
    'len_chain': 10,           #
    'move': None               #
    }

smc_opts_smc2 = {  # only relevant if individual models fitted
    'N': 500,  # no. of particles
    'resampling': 'systematic', # resampling scheme
    'qmc': False  # Quasi-Monte Carlo
    }

# Mix-GARCH (Gaussian)
hyper = {
    'K': 2  # no. of regimes}
    }
prior = OrderedDict()
for j in range(hyper['K']-1):
    prior['p_' + str(j)] = Beta(a=1, b=1)
for j in range(hyper['K']):
    prior['omega_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['alpha_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['beta_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='mix', innov='N', hyper=hyper, prior=prior, data=data)
fk_mix_N = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_mix_N, **smc_opts_ibis)
# alg.run()

# Mix-GARCH (Student t)
hyper = {
    'K': 2  # no. of regimes}
    }
prior = OrderedDict()
prior['df'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
for j in range(hyper['K']-1):
    prior['p_' + str(j)] = Beta(a=1, b=1)
for j in range(hyper['K']):
    prior['omega_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['alpha_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['beta_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = GARCH(variant='mix', innov='t', hyper=hyper, prior=prior, data=data)
fk_mix_t = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_mix_t, **smc_opts_ibis)
# alg.run()


# MS-GARCH (Gaussian)
hyper = {
    'K': 2  # no. of regimes}
    }
prior = OrderedDict()
for j in range(hyper['K']):
    prior['p_' + str(j) + '0'] = Beta(a=1, b=1)
    prior['omega_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['alpha_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['beta_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = MSGARCH
fk_ms_N = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_ms_N, **smc_opts_smc2)
# alg.run()

# MS-GARCH (Student t)
hyper = {
    'K': 2  # no. of regimes}
    }
prior = OrderedDict()
for j in range(hyper['K']):
    prior['p_' + str(j) + '0'] = Beta(a=1, b=1)
    prior['omega_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['alpha_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
    prior['beta_' + str(j)] = TruncNormal(sigma=s_pos, b=np.exp(50))
prior = StructDist(prior)
model = MSGARCH
fk_ms_t = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg = SMC(fk_ms_N, **smc_opts_smc2)
# alg.run()


# Stochastic Volatility (Gaussian x2)
prior = OrderedDict()
prior['omega'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['alpha'] = TruncNormal(sigma=s, a=-np.exp(50), b=np.exp(50))
prior['xi'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior = StructDist(prior)
model = StochVol
fk_sv_N = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg_sv_N = SMC(fk_sv_N, **smc_opts_smc2)
# alg_sv_N.run()

# Stochastic Volatility (Student t x2)


# Heston Model (Gaussian x2)
prior = OrderedDict()
prior['kappa'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['nu'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior['xi'] = TruncNormal(sigma=s_pos, a=0, b=np.exp(50))
prior = StructDist(prior)
model = Heston
fk_hest_N = SMC2(ssm_cls=model, prior=prior, **fk_opts_smc2)
# alg_hest_N = SMC(fk_hest_N, **smc_opts_smc2)
# alg_hest_N.run()


# COMPARISON

multismc_opts = {
    'N': 500,       # no. of particles
    'nruns': 3,     # no. of runs (for mean & SE estimates)
    'nprocs': -2,   # - no. of CPU cores to keep idle during parallel computation
    'store_history': False,
    'collect': [    # variables collected at all times
        LogLts(),   # model evidence
        ESSs(),     # effective sample size
        Rs_flags()  # times when resampling triggered
        ],
    'out_func': out_fct,  # to only store what's necessary to save memory
    'verbose': False
    }

models = {}
models['White Noise (N)'] = fk_white_N
# models['White Noise (t)'] = fk_white_t
models['GARCH (N)'] = fk_garch_N
# models['GARCH (t)'] = fk_garch_t
models['GJR-GARCH (N)'] = fk_gjr_N
# models['GJR-GARCH (t)'] = fk_gjr_t
models['T-GARCH (N)'] = fk_thr_N
# models['T-GARCH (t)'] = fk_thr_t
models['E-GARCH (N)'] = fk_exp_N
# models['E-GARCH (t)'] = fk_exp_t
models['Guyon (N)'] = fk_guyon_N
# models['Guyon (t)'] = fk_guyon_t
models['ELM-GARCH (N)'] = fk_elm_N
# models['ELM-GARCH (t)'] = fk_elm_t
models['rSig-GARCH (N)'] = fk_rsig_N
# models['rSig-GARCH (t)'] = fk_rsig_t
models['Mix-GARCH (N)'] = fk_mix_N
# models['Mix-GARCH (t)'] = fk_mix_t
models['MS-GARCH (N)'] = fk_ms_N
# models['MS-GARCH (t)'] = fk_ms_t
models['StochVol (N)'] = fk_sv_N
# models['StochVol (t)'] = fk_sv_t
models['Heston (N)'] = fk_hest_N
# models['Heston (N)'] = fk_hest_N


runs = multiSMC(fk=models, **multismc_opts)  # run in parallel

# summarize
smc_summary(runs, M0="White Noise (N)", dataset="GJR (N)",
            plot_Z=True, plot_post=True, diagnostics=True,
            save_plots=True)





