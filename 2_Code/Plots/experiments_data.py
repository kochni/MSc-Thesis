#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
import yfinance as yfin
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

import os
wd = '/Users/nici/Library/CloudStorage/GoogleDrive-nikoch@ethz.ch/My Drive/1) ETH/1) Thesis/2) Code'
os.chdir(wd)

tickers = ['gjrgarch', 'canonsv', '^GSPC', 'BZ=F', 'BTC-USD', 'CHFGBP=X']
names = ['GJR-GARCH', 'Canonical SV', 'S&P500', 'Crude Oil', 'Bitcoin', 'CHF/GBP']

# (1) Prices & Log-Returns

# data
i = 5
S = np.loadtxt('Results/data_' + tickers[i] + '.csv', delimiter=",")
X = 100. * np.log(S[1:]/S[:-1])

# Log-Price
fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                        layout='constrained')
# axs[0].set_title(names[i], fontsize=18)
axs[0].plot(S, lw=1.0, c='navy')
axs[0].set_ylabel("$S_t$", fontsize=14)
axs[0].tick_params(axis='x', which='both', bottom=False, top=False,
                   labelsize=12, labelbottom=False)

# Log-Returns
axs[1].plot(X, lw=1.0, c='navy')
axs[1].set_xlabel("$t$", fontsize=14)
axs[1].set_ylabel("$X_t$", fontsize=14)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].set_xticks(np.linspace(0, 2000, 9))

fig.tight_layout()