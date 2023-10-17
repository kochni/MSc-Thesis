#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandSigModel:

    def __init__(self):
        return None

    def reservoir(self, u, dim, sd, activation):
        '''
        Compute a random projection of the signature path of a path 'u'

        Parameters:
        ----------
        u : array
            paths of control variables.

        dim : int
            dimensionality of the randomized signature (larger ensures better
            preservation of geometry of signature components).

        sd : float
            standard deviation of the random matrix/vector components.

        activation : fun
            activation function used in the  (applied component-wise) to the
            random projections).
        '''
        N = len(u)
        du = np.diff(u)
        A = np.random.normal(loc=0, scale=sd, size=(dim, dim))
        b = np.random.normal(loc=0, scale=sd, size=dim)
        Z_path = np.zeros((N, dim))

        for t in tqdm(range(N)):
            # Z = np.random.normal(loc=0, scale=1, size=dim)  # initial value
            Z = np.zeros(dim)
            for i in range(t):
                dZ = activation(np.matmul(A, Z) + b) * du[i]
                Z += dZ

            Z_path[t, :] = Z

        return Z_path


# activation functions

def Id(x):
    return x


def sigmoid(x):
    return (1 + np.exp(-x))**(-1)


def relu(x):
    return max(0, x)


# (local) volatility function / vector field
def vol(t, X_t):
    return abs(0.1 + np.sin(X_t) + 0.2*np.sin(10*X_t)) * np.sqrt(X_t)


# simulate control & path
N = int(1e3)
dt = 1e-4
T = N*dt
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N-1)
W = np.cumsum(dW)
W = np.insert(W, 0, 0)

X = np.zeros(N)
X[0] = 1
for t in range(N-1):
    du_t = dW[t]
    dX_t = vol(t*dt, X[t]) * du_t
    X[t+1] = X[t] + dX_t

plt.plot(W)
plt.plot(X)

# compute random signature
model = RandSigModel()
dim_Z = 5
Z_path = model.reservoir(u=W, dim=dim_Z, sd=1, activation=sigmoid)

# predict path from control signature path
n = int(0.3*N)
# readout = LinearRegression()
readout = RidgeCV()
readout.fit(Z_path[:n, :], X[:n])
X_pred = readout.predict(Z_path)

# plot
plt.figure(dpi=600)
plt.plot(X, label='True', color='blue')
plt.plot(X_pred, label='Predicted', color='red')
plt.vlines(x=n, ymin=min(X), ymax=max(X),
           linestyle='dashed', color='grey',
           label='end of training interval')
plt.legend()
plt.title('dim(Z) = {}'.format(dim_Z))

readout.coef_
# readout.coef_  ==> vol(t, X_t) ?
