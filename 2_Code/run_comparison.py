#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic packages
import numpy as np
from scipy import stats

# quant finance packages
import yfinance as yfin

# SMC packages
from particles.core import SMC, multiSMC
from particles import distributions
from particles.smc_samplers import IBIS, SMC2
from particles.state_space_models import Bootstrap, GuidedPF
from particles.collectors import LogLts, ESSs, Rs_flags

# visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# other
import os
from multiprocessing import cpu_count
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# own model classes
# wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
# os.chdir(wd)
# import det_vol
# import garch
# import helpers
# import plots


# simulated data:
T = 500
w0, a0, b0 = 0.01, 0.2, 0.6
w1, a1, b1 = 0.2, 0.6, 0.2
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
S = np.exp(np.cumsum(X)/100) * 100
plt.plot(S)
plt.plot(vol_true)


# real data:
ticker = '^GSPC'
S = yfin.Ticker(ticker).history(period="10y")['Close'].values  # raw prices
S = S[0:200]
plt.figure(dpi=1000)
plt.plot(S)


# log-returns
X = 100 * np.log(S[1:]/S[:-1])
plt.figure(dpi=1000)
plt.title("S&P500", fontsize="medium")
plt.xlabel("t")
plt.ylabel("100·log(S_t/S_{t-1})")
plt.plot(X)

# density plot
plt.figure(dpi=1000)
plt.suptitle("Log-Returns: " + ticker)
plt.xlabel("100·log(S_t/S_{t-1})")
X_plot = X[(X>np.quantile(X, 0.005)) * (X<-np.quantile(X, 0.005))]
sns.histplot(X_plot, stat='density', kde=False, alpha=0.4)
x = np.linspace(min(X_plot), max(X_plot), 1000)
mu, std = stats.norm.fit(X)
pdf_norm = stats.norm.pdf(x, mu, std)
df, loc, scale = stats.t.fit(X)
pdf_t = stats.t.pdf(x, df, loc, scale)
p, a, b, loc, scale = stats.genhyperbolic.fit(X)
pdf_gh = stats.genhyperbolic.pdf(x, p, a, b, loc, scale)
plt.plot(x, pdf_norm, label='Gaussian', c='lime', ls='-')
plt.plot(x, pdf_t, label="Student t", c='red', ls='-')
plt.plot(x, pdf_gh, label="Gen. Hyperbolic ", c='orange', ls='-')
plt.title("tail=" + str(round(p, 1)) + "; shape=" + str(round(a, 1)),
          fontsize='small', loc='left')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# absolute returns
plt.figure(dpi=1000)
plt.fill_between(np.arange(0, len(X), 1), abs(X), np.zeros(len(X)),
                 color='blue', alpha=0.4, lw=0)

S_train = S[0:200]
plt.plot(S_train)
# np.savetxt("SP500.csv", data, delimiter=",")

# data = np.loadtxt("SP500.csv", delimiter=",")


# Options for volatility prediction & option pricing
pred_opts = {
    'h': 1,  # horizon, no. of days ahead to predict / option maturity
    'M': 100,  # no. of simulations per particle
    'alpha': 0.1,  # error tolerance for prediction intervals
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
    'ESSrmin': .5,              # degeneracy criterion
    'verbose': False             # whether to print ESS while fitting
    }


# BENCHMARK: White noise
spec = {
    'dynamics': 'constant',
    'variant': None,
    'regimes': 1,
    'switching': 'mix',  # 'mix' or 'markov'
    'jumps': None,       # None, 'X'/'returns', or 'V'/'vol'
    'innov_X': 'N'       # 'N', 't', or 'GH'
    }
prior = set_prior(spec)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_bench = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_bench, **smc_opts_ibis)
alg.run()


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
model = DetVol(spec, data, prior)
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
# prior = set_prior(spec)
prior = OrderedDict()
prior['omega'] = Gamma(1.0, 0.5)
prior['alpha'] = Beta(1.0, 1.0)
prior['beta'] = Beta(1.0, 1.0)
# prior['tail_X'] = Normal(0.0, 5.0)
# prior['shape_X'] = Gamma(1.0, 0.5)
# prior['skew_X'] = Normal(0.0, 5.0)
prior = StructDist(prior)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_garch = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_garch, **smc_opts_ibis)
alg.run()


# GJR-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'gjr',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_gjr = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_gjr, **smc_opts_ibis)
# alg.run()


# T-GARCH
spec = {
    'dynamics': 'garch',
    'variant': 'thr',
    'regimes': 2,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec=spec, prices=S_train, prior=prior)
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
model = DetVol(spec=spec, prices=S_train, prior=prior)
fk_exp = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_exp, **smc_opts_ibis)
# alg.run()


# ELM-GARCH
hyper = {  # model hyperparameters
    'q': 5,           # dim. of reservoir
    'sd': 1.,         # st. dev. of random matrix components
    'activ': sigmoid  # activation function
    }
spec = {
    'dynamics': 'garch',
    'variant': 'elm',
    'hyper': hyper,
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = set_prior(spec)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_elm = IBIS(model, **fk_opts_ibis)
# alg = SMC(fk_elm, **smc_opts_ibis)
# alg.run()


# Guyon & Lekeufack
spec = {
    'dynamics': 'guyon',
    'regimes': 1,
    'switching': 'mixing',
    'jumps': None,
    'innov_X': 'N'
    }
prior = OrderedDict()
prior['omega'] = Gamma(1., .5)
prior['alpha'] = Gamma(1., .5)
prior['beta'] = Gamma(1., 0.5)
prior['a1'] = shGamma(1., 0.5, loc=1.)
prior['d1'] = Gamma(1., 0.5)
prior['a2'] = shGamma(1., 0.5, loc=1.)
prior['d2'] = Gamma(1., .5)
# prior['df'] = shGamma(1., .5, loc=2.)
prior = StructDist(prior)
model = DetVol(spec, prior, S_train, **pred_opts)
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
    'innov_X': 'N'
    }
# prior = set_prior(spec)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_sig = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_sig, **smc_opts_ibis)
alg.run()


# Randomized Signature Model
hyper = {  # model hyperparameters
    'q': 5,           # dim. of truncated/randomized Sig
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
# prior = set_prior(spec)
model = DetVol(spec, prior, S_train, **pred_opts)
fk_rsig = IBIS(model, **fk_opts_ibis)
alg = SMC(fk_rsig, **smc_opts_ibis)
alg.run()


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
        LogLts(),   # model evidence
        ESSs(),     # effective sample size
        Rs_flags()  # times when resampling triggered
        # Preds(),     # 1-day ahead point predictions
        # PredSets()   # 1-day ahead prediction sets
        ],
    'out_func': out_fct,  # to only store what's necessary to save memory
    'verbose': False
    }

models = {}
# models['White Noise (t)'] = fk_bench
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
# models['Heston (N, N)'] = fk_hest
# models['Merton (N)'] = fk_mert
# models['Bates (N)'] = fk_bates
# models['BNS (N)'] = fk_bns
# models['Neur/Hest-SV (N, N)'] = fk_neur

print('Models:')
for _, m in enumerate(models):
    print(' -', m)

print('\nSettings:')
print(' - Runs:', multismc_opts['nruns'])
print(' - θ-particles:', multismc_opts['N'])
print(' - CPU cores used:', cpu_count() + multismc_opts['nprocs'])
print(' - CPU cores idle:', abs(multismc_opts['nprocs']))

print('\nData:')
print(' - Length:', len(S_train)-1)

# run all models in parallel
runs = multiSMC(fk=models, **multismc_opts)

# analyze SMC & posterior dist
smc_summary(runs, M0="Gaussian Noise", dataset=ticker, save_plots=False)

# analyze predictions & prediction sets
pred_summary(runs, M0="White Noise (N)", vol_true=abs(X_train),
             X2_true=X_train**2, S_true=S_train, pay_true=S_train[1:]-S_train[:-1], pred_opts=pred_opts)
