#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import os
wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
os.chdir(wd)

import plots
import pickle

# import results
series = '^GSPC'
with open('Results/results_' + series +'.pickle', 'rb') as handle:
    runs = pickle.load(handle)


# Info
plots.run_info(runs)


# CPU Times
plots.cpu_times(runs)


# Truth
plots.truth(runs)


# select models
models = 'all'

models = ['White Noise (t)', 'GARCH (t)', 'T-GARCH (t)', 'GJR-GARCH (t)',
          'Guyon (t)', 'ELM-GARCH (t)',
          'PWC SV (t,N)', 'Canonical SV (N,N)',
          'ELM/Const SV (N,N)'
          ]

# SMC Diagnostic
plots.diagnostics(runs, models)


# Posteriors
models = "GJR-GARCH (t)"
plots.posteriors(runs, models)


# Selection Criteria
plots.selection_criteria(runs, models, 'GARCH (t)', 'CHF/GBP')


# Forecasts
plots.forecast_errors(runs, models, pred_opts, 'moving', 'White Noise (t)', 500, 1000)


# Coverages
plots.coverages(runs, models, pred_opts, 'moving', 1000, None)


# Example prediction sets
models = ['T-GARCH (t)', 'ELM-GARCH (t)']
plots.exmpl_predsets(runs, models, 1250, 1750, pred_opts)


# ELM Function
plots.elm_function(runs)
plots.elmsv_function(runs)


# Predictions
plt.figure(dpi=1000)
m1 = 4
m2 = 9
m3 = 10
plt.plot(runs[-1]['C'][200:500], lw=1.0, c='black')
plt.plot(runs[:-1][m1]['Preds']['C'][200:500], label=runs[:-1][m1]['fk'])
plt.plot(runs[:-1][m2]['Preds']['C'][200:500], label=runs[:-1][m2]['fk'])
plt.plot(runs[:-1][m3]['Preds']['C'][200:500], label=runs[:-1][m3]['fk'])
plt.legend()

