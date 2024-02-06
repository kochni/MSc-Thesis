#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats
import yfinance as yfin
import matplotlib.pyplot as plt
import seaborn as sns

# plot prices
ticker = '^GSPC'
S = yfin.Ticker(ticker).history(period="10y")['Close'].values  # raw prices
plt.figure(dpi=1000)
plt.plot(S)

# plot unconditional log-returns
X = 100 * np.log(S[1:]/S[:-1])
plt.figure(dpi=1000)
plt.title("S&P500", fontsize="medium")
plt.xlabel("t")
plt.ylabel("100·log(S_t/S_{t-1})")
plt.plot(X)

# plot uncond. log-return density
plt.figure(dpi=1000)
plt.suptitle("Log-Returns: " + ticker)
plt.xlabel("100·log(S_t/S_{t-1})")
X_plot = X[(X>np.quantile(X, 0.005)) * (X<-np.quantile(X, 0.005))]
sns.histplot(X_plot, stat='density', kde=False, alpha=0.4)
x = np.linspace(min(X_plot), max(X_plot), 1000)
mu, std = stats.norm.fit(X)
df, loc, scale = stats.t.fit(X)
pdf_norm = stats.norm.pdf(x, mu, std)
pdf_t = stats.t.pdf(x, df, loc, scale)
p, a, b, loc, scale = stats.genhyperbolic.fit(X)
pdf_gh = stats.genhyperbolic.pdf(x, p, a, b, loc, scale)
plt.plot(x, pdf_norm, label='Gaussian', c='lime', ls='-')
plt.plot(x, pdf_t, label="Student t", c='red', ls='-')
plt.plot(x, pdf_gh, label="Gen. Hyperbolic ", c='orange', ls='-')
plt.title("tail=" + str(round(p, 1)) + "; shape=" + str(round(a, 1)),
          fontsize='small', loc='left')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

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