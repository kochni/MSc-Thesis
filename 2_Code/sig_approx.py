#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from esig import stream2sig, stream2logsig

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from tqdm import tqdm


# simulate path
T = 1.0
N = 2500
dt = T/N

kappa = 2.
mu = 1.
sigma = 5.

dW = np.random.normal(size=N) * np.sqrt(dt)
W = np.cumsum(dW)
X = np.full(N, np.nan)
X[0] = mu
for t in range(N-1):
    drift = kappa * (mu - X[t])
    vol = sigma * np.sqrt(dt)
    dXt = drift*dt + vol*dW[t]
    X[t+1] = X[t] + dXt

plt.figure(dpi=1000)
plt.plot(X)

# plt.figure(dpi=800)
# plt.plot(W)


# train readouts from different signature types:
q1 = 5
q2 = 30
u = np.vstack([np.linspace(0, T, N), W])  # controls (time, Brownian)
n = int(N/3)
# readout = RidgeCV()
readout = LassoCV()

# standard Signature:
M = 1
while 2*(2**M-1) + 1 < q2:
    M += 1  # truncation level needed to reach q
Sig_path_1 = np.full([q1, N], np.nan)
Sig_path_2 = np.full([q2, N], np.nan)
Sig_path_1[:, 0] = 1.
Sig_path_2[:, 0] = 1.
for t in tqdm(range(1, N)):
    Sig_path_1[:, t] = stream2sig(u[:, :t].T, depth=M)[0:q1]
    Sig_path_2[:, t] = stream2sig(u[:, :t].T, depth=M)[0:q2]

readout.fit(Sig_path_1[:, :n].T, X[:n])
X_pred_Sig_1 = readout.predict(Sig_path_1.T)
readout.fit(Sig_path_2[:, :n].T, X[:n])
X_pred_Sig_2 = readout.predict(Sig_path_2.T)


# log-Signature:
M = 7
logSig_path_1 = np.full([q1, N], np.nan)
logSig_path_2 = np.full([q2, N], np.nan)
logSig_path_1[:, 0] = 1.
logSig_path_2[:, 0] = 1.
for t in tqdm(range(1, N)):
    logSig_path_1[:, t] = stream2logsig(u[:, :t].T, depth=M)[0:q1]
    logSig_path_2[:, t] = stream2logsig(u[:, :t].T, depth=M)[0:q2]

readout.fit(logSig_path_1[:, :n].T, X[:n])
X_pred_logSig_1 = readout.predict(logSig_path_1.T)
readout.fit(logSig_path_2[:, :n].T, X[:n])
X_pred_logSig_2 = readout.predict(logSig_path_2.T)


# randomized Signature
S1 = np.random.normal(size=q1)
S2 = np.random.normal(size=q2)
A11 = np.random.normal(size=[q1, q1])
A12 = np.random.normal(size=[q2, q2])
A21 = np.random.normal(size=[q1, q1])
A22 = np.random.normal(size=[q2, q2])
b11 = np.random.normal(size=q1)
b12 = np.random.normal(size=q2)
b21 = np.random.normal(size=q1)
b22 = np.random.normal(size=q2)
rSig_path_1 = np.full([q1, N], np.nan)
rSig_path_2 = np.full([q2, N], np.nan)
for t in tqdm(range(N)):
    dS1 = (np.tanh(np.matmul(A11, S1) + b11) * dt +
            np.tanh(np.matmul(A21, S1) + b21) * dW[t])
    dS2 = (np.tanh(np.matmul(A12, S2) + b12) * dt +
            np.tanh(np.matmul(A22, S2) + b22) * dW[t])
    S1 += dS1
    S2 += dS2
    rSig_path_1[:, t] = S1 + dS1
    rSig_path_2[:, t] = S2 + dS2

readout.fit(rSig_path_1[:, :n].T, X[:n])
X_pred_rSig_1 = readout.predict(rSig_path_1.T)
readout.fit(rSig_path_2[:, :n].T, X[:n])
X_pred_rSig_2 = readout.predict(rSig_path_2.T)

# plot
plt.figure(dpi=1000)
fig, axs = plt.subplots(1, 2, figsize=(9, 5), dpi=1000,
                        layout='constrained', sharey=True)

axs[0].set_title('q = {}'.format(q1), fontsize='small', loc='left')
axs[0].set_xlabel("t")
axs[0].plot(X, label='Truth', lw=1., c='black')
axs[0].plot(X_pred_Sig_1, label='Sig', lw=0.7)
axs[0].plot(X_pred_logSig_1, label='log-Sig', lw=0.7)
axs[0].plot(X_pred_rSig_1, label='r-Sig', lw=0.7)
axs[0].set_xticks(np.linspace(0, 2500, 5), np.linspace(0, 1, 5))
ymin, ymax = plt.ylim()
axs[0].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey',
           label='end of training interval')

axs[1].set_title('q = {}'.format(q2), fontsize='small', loc='left')
axs[1].set_xlabel("t")
axs[1].plot(X, lw=1., c='black')
axs[1].plot(X_pred_Sig_2, lw=0.7)
axs[1].plot(X_pred_logSig_2, lw=0.7)
axs[1].plot(X_pred_rSig_2, lw=0.7)
axs[1].set_xticks(np.linspace(0, 2500, 5), np.linspace(0, 1, 5))
axs[1].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey')

fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
           ncol=5, fancybox=True)
