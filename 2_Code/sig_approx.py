#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from esig import stream2sig, stream2logsig

from sklearn.linear_model import LinearRegression, RidgeCV
import matplotlib.pyplot as plt
from tqdm import tqdm


# simulate path
N = 1000
dt = 1e-2
T = N*dt

kappa = 1.
mu = 1.
sigma = 2.

dW = np.random.normal(size=N)*np.sqrt(dt)
W = np.cumsum(dW)
X = np.full(N, np.nan)
X[0] = mu
for t in range(N-1):
    drift = kappa*(mu-X[t])
    vol = np.sqrt(dt)
    dXt = drift*dt + vol*dW[t]
    X[t+1] = X[t] + dXt

plt.figure(dpi=800)
plt.plot(X)

plt.figure(dpi=800)
plt.plot(W)


# train readouts from different signature types:
dim = 15  # dimensionality of signatures
W_tilde = np.vstack([W, np.linspace(0, T, N)])  # time-extended path
n = int(0.5*N)
readout = RidgeCV()

# standard signature:
M = 1
while 2*(2**M-1) + 1 < dim: M += 1  # truncation level needed to reach dim
sig_path = np.full([dim, N], np.nan)
sig_path[:, 0] = 1.
for t in tqdm(range(1, N)):
    sig_path[:, t] = stream2sig(W_tilde[:, :t].T, depth=M)[0:dim]

readout.fit(sig_path[:, :n].T, X[:n])
X_pred_sig = readout.predict(sig_path.T)


# log-signature:
M = 7
sig_path = np.full([dim, N], np.nan)
sig_path[:, 0] = 1.
for t in tqdm(range(1, N)):
    sig_path[:, t] = stream2logsig(W_tilde[:, :t].T, depth=M)[0:dim]

readout.fit(sig_path[:, :n].T, X[:n])
X_pred_logsig = readout.predict(sig_path.T)


# plot
plt.figure(dpi=800)
plt.plot(X, label='Truth', lw=0.8)
plt.plot(X_pred_sig, label='Pred Sig', lw=0.8)
plt.plot(X_pred_logsig, label='Pred log-Sig', lw=0.8)
# plt.plot(X_pred_rsig, label='Pred r-Sig', lw=0.8)
ymin, ymax = plt.ylim()
plt.vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey',
           label='end of training interval')
plt.legend()
plt.title('dim = {}'.format(dim))
