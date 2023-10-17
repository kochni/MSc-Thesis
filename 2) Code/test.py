#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# basic packages
import numpy as np
import pandas as pd
import scipy

# SMC packages
import particles
from particles import SMC, multiSMC
from particles.smc_samplers import IBIS, SMC2
from particles import distributions as dists

# visualization packages
import matplotlib.pyplot as plt

# own model classes
import WhiteNoise
import OrnUhl
import GARCH
import SigGARCH
import StochVol


# (synthetic) data: white noise
T = 100
X1 = np.random.normal(loc=0, scale=1, size=T)
X2 = np.random.normal(loc=0, scale=3, size=T)
X = np.concatenate([X1, X2], axis=0)

plt.figure(dpi=800)
plt.plot(X)


# White noise
prior = dists.StructDist({'sigma': dists.Normal()})
white = WhiteNoise(innov='norm', prior=prior, data=X)
ibis = IBIS(model=white, len_chain=10, wastefree=False)
algo_white = SMC(fk=ibis, N=50, resampling='systematic', ESSrmin=0.5, verbose=True)
algo_white.run()


# GARCH
prior = dists.StructDist({'omega': dists.Normal(),
                          'alpha': dists.Normal(),
                          'beta':  dists.Normal(),
                          'gamma': dists.Normal()})
model = GARCH(variant='standard', innov='norm', data=X, prior=prior)
ibis = IBIS(model, len_chain=10, wastefree=False)
algo_garch = SMC(fk=ibis, N=50, resampling='systematic', ESSrmin=0.5, verbose=True)
algo_garch.run()


# Neural GARCH
def sigmoid(x): return (1+np.exp(-x))**(-1)
prior = dists.StructDist({'w1': dists.Normal(),
                          'w2': dists.Normal(),
                          'w3':  dists.Normal()})
model = GARCH(variant='neural', sd=1, activ=sigmoid, innov='norm', data=X, prior=prior)
ibis = IBIS(model, len_chain=10, wastefree=False)
algo_neurgarch = SMC(fk=ibis, N=50, resampling='systematic', ESSrmin=0.5, verbose=True)
algo_neurgarch.run()


# Stochastic Volatility
prior = dists.StructDist({'alpha': dists.Normal(),
                          'xi': dists.Normal()})
model = StochVol(innov_X='norm', innov_V='norm')
SV_args = {'ssm_cls': model,
           'prior': prior,
           'data': X}
smc2 = SMC2(**SV_args)
algo_sv = particles.SMC(fk=smc2)
algo_sv.run()


# run all in parallel (on all but 2 cores)
models = {}
models['WhiteNoise'] = algo_white
models['GARCH'] = algo_garch
runs = multiSMC(fk=models, N=10, resampling='systematic', ESSrmin=0.5,
                nprocs=1, nruns=-2, verbose=True)
