#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from helpers import *

# reproduceability
np.random.seed(1)

# setup
T = 1000
d = 2
q = 5
n = int(np.ceil(T/3))

# input signal
A = np.array([[0.9, -0.2], [-0.3, 0.8]])
b = np.array([0.5, 0.5]).reshape(d, 1)
A = A / (max(abs(np.linalg.eigvals(A))) + 0.1)
b = b / (max(abs(np.linalg.eigvals(A))) + 0.1)
z = np.full([d, T], np.nan)
z[:, 0] = np.random.normal(size=d)
for t in range(1, T):
    z[:, [t]] = np.matmul(A, z[:, [t-1]]) + b + np.random.normal(scale=0.1, size=[d, 1])

# truth
A = np.random.normal(size=[q, q])
A = A / max(abs(np.linalg.eigvals(A)) + 0.1)
C = np.random.normal(size=[q, d])
C = C / max(abs(np.linalg.eigvals(A)) + 0.1)
b = np.random.normal(size=[q, 1])
b = b / max(abs(np.linalg.eigvals(A)) + 0.1)
w = np.random.normal(size=[q, 1])

y = np.full([T], np.nan)
r = np.full([q, T], np.nan)
r[:, 0] = np.random.normal(size=q)
for t in range(1, T):
    r[:, [t]] = np.matmul(A, r[:, [t-1]]) + np.matmul(C, z[:, [t]]) + b

y = np.sum(w * r, axis=0)


# tested dimensionalities
q1 = 5
q2 = 50

# (1) Echo State Network
phi = sigmoid
A = np.random.normal(size=[q1, q1])
A = A / max(abs(np.linalg.eigvals(A)) + 0.1)
C = np.random.normal(size=[q1, d])
C = C / max(abs(np.linalg.eigvals(A)) + 0.1)
b = np.random.normal(size=[q1, 1])
b = b / max(abs(np.linalg.eigvals(A)) + 0.1)

r = np.full([q1, T], np.nan)
r0 = np.random.normal(size=[q1, 1])
r[:, [0]] = phi(np.matmul(A, r0) + np.matmul(C, z[:, [0]])) + b
for t in range(1, T):
    r[:, [t]] = phi(np.matmul(A, r[:, [t-1]]) + np.matmul(C, z[:, [t]])) + b

readout = RidgeCV()
readout.fit(r[:, 0:n].T, y[0:n])
y_hat_ESN_1 = readout.predict(r.T)

A = np.random.normal(size=[q2, q2])
A = A / max(abs(np.linalg.eigvals(A)) + 0.1)
C = np.random.normal(size=[q2, d])
C = C / max(abs(np.linalg.eigvals(A)) + 0.1)
b = np.random.normal(size=[q2, 1])
b = b / max(abs(np.linalg.eigvals(A)) + 0.1)

r = np.full([q2, T], np.nan)
r0 = np.random.normal(size=[q2, 1])
r[:, [0]] = phi(np.matmul(A, r0) + np.matmul(C, z[:, [0]])) + b
for t in range(1, T):
    r[:, [t]] = phi(np.matmul(A, r[:, [t-1]]) + np.matmul(C, z[:, [t]])) + b

readout = RidgeCV()
readout.fit(r[:, 0:n].T, y[0:n])
y_hat_ESN_2 = readout.predict(r.T)


# (2) Barron Functional
phi = sigmoid
A2 = np.random.normal(size=[q1, q1])
A2 = A2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
C2 = np.random.normal(size=[q1, d])
C2 = C2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
b2 = np.random.normal(size=[q1, 1])
b2 = b2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
C1 = np.random.normal(size=[q1, q1])
C1 = C1 / max(abs(np.linalg.eigvals(C1)) + 0.1)
b1 = np.random.normal(size=[q1, 1])
b1 = b1 / max(abs(np.linalg.eigvals(C1)) + 0.1)

r = np.full([q1, T], np.nan)
r0 = np.random.normal(size=[q1, 1])
res0 = phi(np.matmul(A2, r0) + np.matmul(C2, z[:, [0]])) + b2
r[:, [0]] = phi(np.matmul(C1, res0) + b1)
for t in range(1, T):
    res = phi(np.matmul(A2, r[:, [t-1]]) + np.matmul(C2, z[:, [t]])) + b2
    r[:, [t]] = phi(np.matmul(C1, res) + b1)

readout = RidgeCV()
readout.fit(r[:, 0:n].T, y[0:n])
y_hat_BAR_1 = readout.predict(r.T)

A2 = np.random.normal(size=[q2, q2])
A2 = A2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
C2 = np.random.normal(size=[q2, d])
C2 = C2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
b2 = np.random.normal(size=[q2, 1])
b2 = b2 / max(abs(np.linalg.eigvals(A2)) + 0.1)
C1 = np.random.normal(size=[q2, q2])
C1 = C1 / max(abs(np.linalg.eigvals(C1)) + 0.1)
b1 = np.random.normal(size=[q2, 1])
b1 = b1 / max(abs(np.linalg.eigvals(C1)) + 0.1)

r = np.full([q2, T], np.nan)
r0 = np.random.normal(size=[q2, 1])
res0 = phi(np.matmul(A2, r0) + np.matmul(C2, z[:, [0]])) + b2
r[:, [0]] = phi(np.matmul(C1, res0) + b1)
for t in range(1, T):
    res = phi(np.matmul(A2, r[:, [t-1]]) + np.matmul(C2, z[:, [t]])) + b2
    r[:, [t]] = phi(np.matmul(C1, res) + b1)

readout = RidgeCV()
readout.fit(r[:, 0:n].T, y[0:n])
y_hat_BAR_2 = readout.predict(r.T)


# PLOT

plt.figure(dpi=1000)
fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                        layout='constrained', sharex=True, sharey=True)

axs[0].set_title('q = {}'.format(q1), fontsize=12, loc='left')
axs[0].set_ylabel("$H(z)$", fontsize=14)
axs[0].plot(y, label='Truth', lw=1.0, c='black')
axs[0].plot(y_hat_ESN_1, label='ESN', lw=1.0, c='red')
axs[0].plot(y_hat_BAR_1, label='Barron', lw=1.0, c='tab:green')
ymin, ymax = plt.ylim()
axs[0].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey',
           label='end of training interval')

axs[1].set_title('q = {}'.format(q2), fontsize=12, loc='left')
axs[1].set_ylabel("$H(z)$", fontsize=14)
axs[1].set_xlabel("t")
axs[1].plot(y, lw=1.0, c='black')
axs[1].plot(y_hat_ESN_2, lw=1.0, c='red')
axs[1].plot(y_hat_BAR_2, lw=1.0, c='tab:green')
axs[1].vlines(x=n, ymin=ymin, ymax=ymax,
           linestyle='dashed', color='grey')

fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
           ncol=5, fancybox=True, fontsize=12)

