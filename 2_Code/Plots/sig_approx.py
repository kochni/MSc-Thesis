#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from esig import stream2sig, stream2logsig, sigdim, logsigdim

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers import *

# reproduceability
np.random.seed(1)

# simulate path
T = 1.0
N = 1000
dt = T/N

kappa = 3.0
mu = 2.0
sigma = 2.0

dW = np.random.normal(size=N) * np.sqrt(dt)
W = np.cumsum(dW)
X = np.full(N, np.nan)
X[0] = mu
for t in range(N-1):
    drift = kappa * (mu - X[t])
    vol = sigma
    dXt = drift*dt + vol*dW[t]
    X[t+1] = X[t] + dXt

# train readouts from different signature types:
q1 = 5
q2 = 50
u = np.vstack([np.linspace(0, T, N), W])  # controls (time, Brownian)
n = int(N/3)
# readout = LinearRegression()
readout = RidgeCV()
# readout = LassoCV()


# standard Signature:
M = 1
while sigdim(2, M) < max(q1, q2):
    M += 1  # truncation level needed to reach q
Sig_path_1 = np.full([q1, N], np.nan)
Sig_path_2 = np.full([q2, N], np.nan)
Sig_path_1[:, 0] = 1.
Sig_path_2[:, 0] = 1.
for t in tqdm(range(1, N)):
    sig = stream2sig(u[:, :t].T, depth=M)
    Sig_path_1[:, t] = sig[:q1]
    Sig_path_2[:, t] = sig[:q2]

readout.fit(Sig_path_1[:, :n].T, X[:n])
X_pred_Sig_1 = readout.predict(Sig_path_1.T)
Sig_1_coef = readout.coef_
readout.fit(Sig_path_2[:, :n].T, X[:n])
X_pred_Sig_2 = readout.predict(Sig_path_2.T)
print("Sig Coef:", Sig_1_coef[0:5])


# log-Signature:
M = 1
while logsigdim(2, M) < max(q1, q2):
    M += 1  # truncation level needed to reach q
logSig_path_1 = np.full([q1, N], np.nan)
logSig_path_2 = np.full([q2, N], np.nan)
logSig_path_1[:, 0] = 1.
logSig_path_2[:, 0] = 1.
for t in tqdm(range(1, N)):
    logsig = stream2logsig(u[:, :t].T, depth=M)
    logSig_path_1[:, t] = logsig[:q1]
    logSig_path_2[:, t] = logsig[:q2]

readout.fit(logSig_path_1[:, :n].T, X[:n])
X_pred_logSig_1 = readout.predict(logSig_path_1.T)
logSig_1_coef = readout.coef_
readout.fit(logSig_path_2[:, :n].T, X[:n])
X_pred_logSig_2 = readout.predict(logSig_path_2.T)
print("rand-Sig Coef:", logSig_1_coef[0:5])


# randomized Signature
activ = sigmoid
sd = 1.0
S1 = np.random.normal(size=q1, scale=sd)
S2 = np.random.normal(size=q2, scale=sd)
A11 = np.random.normal(size=[q1, q1], scale=sd)
A12 = np.random.normal(size=[q2, q2], scale=sd)
A21 = np.random.normal(size=[q1, q1], scale=sd)
A22 = np.random.normal(size=[q2, q2], scale=sd)
b11 = np.random.normal(size=q1, scale=sd)
b12 = np.random.normal(size=q2, scale=sd)
b21 = np.random.normal(size=q1, scale=sd)
b22 = np.random.normal(size=q2, scale=sd)
rSig_path_1 = np.full([q1, N], np.nan)
rSig_path_2 = np.full([q2, N], np.nan)
for t in tqdm(range(N)):
    dS1 = (activ(np.matmul(A11, S1) + b11) * dt
           + activ(np.matmul(A21, S1) + b21) * dW[t])
    dS2 = (activ(np.matmul(A12, S2) + b12) * dt
           + activ(np.matmul(A22, S2) + b22) * dW[t])
    S1 += dS1
    S2 += dS2
    rSig_path_1[:, t] = S1
    rSig_path_2[:, t] = S2

readout.fit(rSig_path_1[:, :n].T, X[:n])
X_pred_rSig_1 = readout.predict(rSig_path_1.T)
rSig_1_coef = readout.coef_
readout.fit(rSig_path_2[:, :n].T, X[:n])
X_pred_rSig_2 = readout.predict(rSig_path_2.T)
print("rand-Sig Coef:", rSig_1_coef[0:5])


# plot
plt.figure(dpi=1000)
fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                        layout='constrained', sharex=True, sharey=True)

axs[0].set_title('q = {}'.format(q1), fontsize=12, loc='left')
axs[0].set_ylabel("$Y_t$", fontsize=14)
axs[0].plot(X, label='Truth', lw=1.0, c='black')
axs[0].plot(X_pred_Sig_1, label='Sig', lw=1.0, c='red')
axs[0].plot(X_pred_logSig_1, label='log-Sig', lw=1.0, c='darkorange')
axs[0].plot(X_pred_rSig_1, label='rand-Sig', lw=1.0, c='tab:green')
axs[0].set_xticks(np.linspace(0, N, 5), np.linspace(0, 1, 5))
ymin, ymax = plt.ylim()
axs[0].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey',
           label='end of training interval')

axs[1].set_title('q = {}'.format(q2), fontsize=12, loc='left')
axs[1].set_ylabel("$Y_t$", fontsize=14)
axs[1].set_xlabel("$t$", fontsize=12)
axs[1].plot(X, lw=1.0, c='black')
axs[1].plot(X_pred_Sig_2, lw=1.0, c='red')
axs[1].plot(X_pred_logSig_2, lw=1.0, c='darkorange')
axs[1].plot(X_pred_rSig_2, lw=1.0, c='tab:green')
axs[1].set_xticks(np.linspace(0, N, 5), np.linspace(0, 1, 5))
axs[1].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey')

fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
           ncol=5, fancybox=True, fontsize=12)
