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

# tickers = ['^GSPC', 'BZ=F', 'BTC-USD', 'CHFGBP=X']
# names = ['S&P500', 'Crude Oil', 'Bitcoin', 'CHF/GBP']
tickers = ['gjrgarch', 'canonsv', '^GSPC', 'BTC-USD']
names = ['GJR-GARCH', 'Canonical SV', 'S&P500', 'Bitcoin']

# (1) Prices & Log-Returns
fig = plt.figure(dpi=1000, figsize=(12, 8))
outer_grid = gridspec.GridSpec(2, 2, fig)

for i in range(4):
    # Create nested grids for each subplot
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[i])

    # data
    # data = yfin.Ticker(tickers[i]).history(start="2008-01-01", end="2009-12-31")['Close']
    # S = data.values
    # dates = data.index
    S = np.loadtxt('data/data_' + tickers[i] + '.csv', delimiter=",")
    X = 100. * np.log(S[1:]/S[:-1])

    # Log-Price
    ax_top = plt.Subplot(fig, inner_grid[0])
    fig.add_subplot(ax_top)
    ax_top.set_title(names[i], fontsize=18)
    # ax_top.plot(dates, np.log(S), lw=1.0, c='navy')
    ax_top.plot(S, lw=1.0, c='navy')
    ax_top.set_ylabel("$S_t$", fontsize=14)
    ax_top.tick_params(axis='x', which='both', bottom=False, top=False,
                       labelsize=12, labelbottom=False)

    # Log-Returns
    ax_bottom = plt.Subplot(fig, inner_grid[1])
    fig.add_subplot(ax_bottom)
    # ax_bottom.plot(dates[1:], X, lw=1.0, c='navy')
    ax_bottom.plot(X, lw=1.0, c='navy')
    ax_bottom.set_xlabel("$t$", fontsize=14)
    ax_bottom.set_ylabel("$X_t$", fontsize=14)
    ax_bottom.tick_params(axis='both', which='major', labelsize=12)

    # x-axis ticks
    # date_range = dates.max() - dates.min()
    # years = date_range.days / 365.25
    # interval = int(years / 5) # Aim for about 5-6 ticks
    # interval = max(1, interval) # Ensure at least 1 year interval
    # ax_bottom.xaxis.set_major_locator(mdates.YearLocator(interval))
    # ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig.tight_layout()
plt.show()


# (2) Uncond. Log-Return Density
fig, axs = plt.subplots(2, 2, dpi=1000, figsize=(12, 8))
for j, ax in enumerate(axs.flat):
    ax.set_title(names[j], fontsize=12)
    ax.set_xlabel("$X_t$" if j in [2, 3] else '', fontsize=12)
    data = yfin.Ticker(tickers[j]).history(period="20y")['Close']
    S = data.values
    X = 100 * np.log(S[1:]/S[:-1])
    X_plot = X[(X>np.quantile(X, 0.005)) * (X<-np.quantile(X, 0.005))]
    sns.histplot(X_plot, stat='density', kde=False, alpha=0.3, ax=ax,
                 legend=False).set(ylabel="Density" if j in [0, 2] else '')
    x = np.linspace(min(X_plot), max(X_plot), 1000)
    mu, std = stats.norm.fit(X)
    df, loc, scale = stats.t.fit(X)
    pdf_norm = stats.norm.pdf(x, mu, std)
    pdf_t = stats.t.pdf(x, df, loc, scale)
    p, a, b, loc, scale = stats.genhyperbolic.fit(X)
    pdf_gh = stats.genhyperbolic.pdf(x, p, a, b, loc, scale)
    ax.plot(x, pdf_norm, label='Gaussian' if j==0 else '',
            c='blue', ls='-', lw=1.7)
    ax.plot(x, pdf_t, label="Student t" if j==0 else '',
            c='forestgreen', ls='-', lw=1.7)
    ax.plot(x, pdf_gh, label="Gen. Hyperbolic" if j==0 else'',
            c='red', ls='-', lw=1.7)
    # plt.title("tail=" + str(round(p, 1)) + "; shape=" + str(round(a, 1)),
    #           fontsize='small', loc='left')
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
plt.tight_layout()


# ATM Option Payouts
plt.figure(dpi=1000)
plt.suptitle("Positive Option Payouts")
plt.ylabel("Density")
plt.xlabel("C")
C = np.maximum(S[1:] - S[:-1], 0.)
C_pos = C[C>0.]
C_plot = C_pos[(C_pos<np.quantile(C_pos, 0.99))]
# mu, std = stats.norm.fit(np.log(C_pos))
# df, loc, scale = stats.t.fit(np.log(C_pos))
# pdf_lognorm = stats.norm.pdf(np.log(x), mu, std)
# pdf_logt = stats.t.pdf(np.log(x), df, loc, scale)
x = np.linspace(min(C_plot), max(C_plot), 1000)
sns.histplot(C_plot, stat='density', kde=True, alpha=0.3)
# plt.plot(x, pdf_lognorm, label='Gaussian')
# plt.plot(x, pdf_logt, label='Student t')