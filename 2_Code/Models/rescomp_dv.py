#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# signatures
from esig import stream2sig, stream2logsig, sigdim, logsigdim

# other
from operator import itemgetter


class ResCompDV:
    '''
    Reservoir computers for deterministic volatility models

    Parameters:
    -----------

    variant: string
        type of the RC approach use; one of 'elm' (for Extreme Learning
        Machine), 'esn' (for Echo State Network), 't_sig' (for truncated
        signature), 'log_sig' (for log-Signature), and 'r_sig' (for
        randomized signature);
        ELM is Markovian, while others model path-dependence.

    hyper: dict
        dictionary of hyperparameters; must include
            'q' (int): dimensionality of the reservoir.
            'sd' (float): standard deviation of the random initializations of
                the weights resp. random signature components.
            'activ' (func): activation function used in the random neural
                network resp. the generation of the randomized signature.

    K: int
        number of regimes.

    '''

    def __init__(self, variant, hyper, K):

        self.variant = variant
        self.K = K
        self.L = L = hyper.get('L')  # no. of hidden layers
        self.H = H = hyper.get('H')  # first hidden layer width
        self.q = q = hyper['q']      # last hidden layer width
        sd = hyper.get('sd')
        self.activ = hyper.get('activ')

        # draw random components
        if variant == 'elm':  # inner parameters of NN
            H = q if L == 1 else H
            self.A1 = np.random.normal(scale=sd, size=[H, 2])
            self.b1 = np.random.normal(scale=sd, size=[H, 1])
            self.A2 = np.random.normal(scale=sd, size=[q, H])
            self.b2 = np.random.normal(scale=sd, size=[q, 1])

            # to save random projection:
            if L == 1:
                self.A = self.A1
                self.b = self.b1
            else:
                self.A = [self.A1, self.A2]
                self.b = [self.b1, self.b2]

        elif variant == 'esn':
            self.A = np.random.normal(scale=sd, size=[q, q])
            self.C = np.random.normal(scale=sd, size=[q, 2])
            self.b = np.random.normal(scale=sd, size=[q, 1])
            self.res_0 = np.random.normal(scale=sd, size=[q, 1])

            spec_rad_A = max(abs(np.linalg.eigvals(self.A)))
            self.A = self.A / (spec_rad_A + 0.1)
            self.C = self.C / (spec_rad_A + 0.1)
            self.b = self.b / (spec_rad_A + 0.1)

        elif variant == 'barron':
            self.C1 = np.random.normal(scale=sd, size=[q, q])
            self.b1 = np.random.normal(scale=sd, size=[q, 1])
            self.A2 = np.random.normal(scale=sd, size=[q, q])
            self.C2 = np.random.normal(scale=sd, size=[q, 2])
            self.b2 = np.random.normal(scale=sd, size=[q, 1])
            self.res_0 = np.random.normal(scale=sd, size=[q, 1])

            spec_rad_A2 = max(abs(np.linalg.eigvals(self.A2)))
            self.A2 = self.A2 / (spec_rad_A2 + 0.1)
            self.C2 = self.C2 / (spec_rad_A2 + 0.1)
            self.b2 = self.b2 / (spec_rad_A2 + 0.1)

        elif variant == 'rand-sig':  # random matrices of Controlled ODE
            self.A1 = np.random.normal(scale=sd, size=[q, q])
            self.A2 = np.random.normal(scale=sd, size=[q, q])
            self.A3 = np.random.normal(scale=sd, size=[q, q])
            self.b1 = np.random.normal(scale=sd, size=[q, 1])
            self.b2 = np.random.normal(scale=sd, size=[q, 1])
            self.b3 = np.random.normal(scale=sd, size=[q, 1])
            self.rsig_0 = np.random.normal(scale=sd, size=[q, 1])

            # self.A1 = self.A1 / max(abs(np.linalg.eigvals(self.A1)))
            # self.A2 = self.A2 / max(abs(np.linalg.eigvals(self.A2)))
            # self.A3 = self.A3 / max(abs(np.linalg.eigvals(self.A3)))

        else:  # variant == 'sig'
            # compute minimum required truncation level to obtain enough
            # components as specified in dimensionality (q)
            self.sig_len = 1
            if variant == 'sig':
                sig_length = sigdim
            elif variant == 'logsig':
                sig_length = logsigdim
            while sig_length(2, self.sig_len) < q:
                self.sig_len += 1

    def elm_vol(self, theta, X, t):
        '''
        compute volatility by passing time index, past log-variance, and
        past log-return through a random neural network, a.k.a. "extreme
        learning machine (ELM)"

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract weights
        w_names = ['w' + str(j) for j in range(q+1)]
        w = itemgetter(*w_names)(theta)
        w = np.array(w)  # (q+1,N)
        w0 = w[-1, :]  # (q,N,)
        w = w[:-1, :]  # (N,)

        w0 = np.clip(w0, -1e20, 1e20)
        w = np.clip(w, -1e20, 1e20)

        # shape initial vol to (N,) of identical entries
        s2_0 = X[0]**2
        log_s2_j = np.full([N], np.log(s2_0))  # (N,K)
        # at t=0, loop is not entered
        for j in range(1, t+1):
            # input to reservoir: previous log-var, previous return
            # time = np.full([N], j)
            X_prev = np.full([N], X[j-1])
            Z = np.stack([X_prev, log_s2_j], axis=0)   # (2,N)
            # Z = np.stack([time, X_prev, log_s2_j], axis=0)  # (3,N)

            # first hidden layer:
            h1 = np.matmul(self.A1, Z) + self.b1
            h1 = self.activ(h1)  # (q,N)

            if self.L >= 2:
                # second hidden layer:
                h2 = np.matmul(self.A2, h1) + self.b2
                res = self.activ(h2)
            else:
                res = h1

            # linear readout
            log_s2_j = np.sum(w * res, axis=0) + w0
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def esn_vol(self, theta, X, t):
        '''
        Echo State Network

        log(σ_t^2) = w·r_t
        r_t = ϕ(A·r_{t-1} + C·z_{t-1} + b)

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract weights
        w_names = ['w' + str(j) for j in range(q+1)]
        w = itemgetter(*w_names)(theta)
        w = np.array(w)  # (q+1,N)
        w0 = w[-1, :]  # (q,N,)
        w = w[:-1, :]  # (N,)

        w0 = np.clip(w0, -1e20, 1e20)
        w = np.clip(w, -1e20, 1e20)

        # shape initial vol to (N,K) of identical entries
        s2_0 = X[0]**2
        log_s2_j = np.full([N], np.log(s2_0))  # (N,)
        res_j  = np.tile(self.res_0, [1, N])  # (q,N)
        # at t=0, loop is not entered
        for j in range(1, t+1):
            # recurrent reservoir step:
            h1 = np.matmul(self.A, res_j)  # (q,N)

            # input variable: t, previous log-var, previous return
            time = np.full([N], j)
            X_prev = np.full([N], X[j-1])
            M = np.stack([X_prev, log_s2_j], axis=0)  # (2,N)
            # M = np.stack([time, X_prev, log_s2_j], axis=0)  # (3,N)
            h2 = np.matmul(self.C, M)

            # new reservoir
            res_j = self.activ(h1 + h2 + self.b)  # (q,N)

            # get volatility from reservoir & readout:
            log_s2_j = np.sum(w * res_j, axis=0) + w0
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t


    def barron_vol(self, theta, X, t):
        '''
        Echo State Network with ELM readout

        log(σ_t^2) = w·u_t
        u_t = ϕ(C2·r_t + b2)
        r_t = ϕ(A1·r_{t-1} + C1·z_{t-1} + b1)

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract weights
        w_names = ['w' + str(j) for j in range(q+1)]
        w = itemgetter(*w_names)(theta)
        w = np.array(w)  # (q+1,N)
        w0 = w[-1, :]  # (q,N,)
        w = w[:-1, :]  # (N,)

        w0 = np.clip(w0, -1e20, 1e20)
        w = np.clip(w, -1e20, 1e20)

        # shape initial vol to (N,K) of identical entries
        s2_0 = X[0]**2
        log_s2_j = np.full([N], np.log(s2_0))  # (N)
        res_j  = np.tile(self.res_0, [1, N])  # (q,N)
        # at t=0, loop is not entered
        for j in range(1, t+1):
            # ESN reservoir:
            h1 = np.matmul(self.A2, res_j)  # (q,N)
            # time = np.full([N], j)
            X_prev = np.full([N], X[j-1])
            Z = np.stack([X_prev, log_s2_j], axis=0)  # (2,N)
            # Z = np.stack([time, X_prev, log_s2_j], axis=0)  # (3,N)
            h2 = np.matmul(self.C2, Z)
            res_j = self.activ(h1 + h2 + self.b1)  # (q,N)

            # ELM readout on ESN reservoir:
            u = np.matmul(self.C1, res_j) + self.b2
            u = self.activ(u)
            log_s2_j = np.sum(w * u, axis=0) + w0
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t


    def sig_vol(self, theta, X, t):
        '''
        compute volatility from the truncated (log-)signature of the
        time-extended path of the log-returns, (t, X_t)

        '''
        N = len(theta)
        q = self.q
        sig_len = self.sig_len

        if t > 2:
            # extract Sig version
            if self.variant == 'sig':  # standard Signature
                Sig = stream2sig
            elif self.variant == 'logsig':  # log-Signature
                Sig = stream2logsig

            # extract weights
            w_names = ['w' + str(j) for j in range(q+1)]
            w = itemgetter(*w_names)(theta)
            w = np.array(w)  # (q+1,N)
            w0 = w[-1, :]  # (q,N,)
            w = w[:-1, :]  # (N,)

            # input path: time and returns until time t-1
            time = np.arange(0, t, 1)
            X_past = X[:t]
            u = np.vstack([time, X_past]).T # (t,2)
            S = Sig(u, sig_len)[:q]  # (q,)
            S = S.reshape(-1, 1)  # (1,q)

            log_s2_t = np.sum(w * S, axis=0) + w0  # (N,)

            # extract weights:
            # if K == 1:
            #     pass
            # else:  # multi-regime
            #     Ws = np.full([q+1, N, K], np.nan)
            #     for k in range(K):
            #         w_names = ['w' + str(j) + '_' + str(k) for j in range(q+1)]
            #         Ws[:, :, k] = itemgetter(*w_names)(theta)  # (q+1,N)

            # compute volatility from weights & signature components:
            # log_s2_t = np.einsum('qNK,q->NK', Ws, sig)

            log_s2_t = np.clip(log_s2_t, -50., 50.)
            s_t = np.exp(0.5*log_s2_t)

        else:  # t == 0
            s2_0 = X[0]**2
            s_t = np.full([N], np.sqrt(s2_0))

        return s_t


    def rsig_vol(self, theta, X, t):
        '''
        compute volatility from the randomized signature of the time-extended
        path of log-returns and log-variances, (s, X_s, log(σ_s^2))_{s<t}

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract weights
        w_names = ['w' + str(j) for j in range(q+1)]
        w = itemgetter(*w_names)(theta)
        w = np.array(w)  # (q+1,N)
        w0 = w[-1, :]  # (q,N,)
        w = w[:-1, :]  # (N,)

        # cap parameters
        w0 = np.clip(w0, -1e20, 1e20)
        w = np.clip(w, -1e20, 1e20)

        s2_0 = X[0]**2
        log_s2_j = np.full([N], np.log(s2_0))  # (N)
        rsig = np.tile(self.rsig_0, [1, N])  # (q,N)
        # at t=0, loop is not entered
        for j in range(1, t):
            # increment from time:
            incr_1 = np.matmul(self.A1, rsig) + self.b1
            incr_1 = self.activ(incr_1)  # (q,N)
            # (note: p = q)

            # increment from log-returns:
            incr_2 = np.matmul(self.A2, rsig) + self.b2
            incr_2 = self.activ(incr_2)  # (q,N)
            delta_X = X[j] - X[j-1]
            incr_2 = incr_2 * delta_X  # (q,N)

            # increment from log-variance:
            incr_3 = np.matmul(self.A3, rsig) + self.b3
            incr_3 = self.activ(incr_3)  # (q,N)
            if j == 1:
                delta_logvar = log_s2_j
            else:
                delta_logvar = log_s2_j - log_s2_prev
            log_s2_prev = log_s2_j
            incr_3 = incr_3 * delta_logvar  # (q,N)

            rsig = rsig + incr_1 + incr_2 + incr_3  # (q,N)

            # get volatility from reservoir & rSig
            log_s2_j = np.sum(w * rsig, axis=0) + w0
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'elm':
            s_t = self.elm_vol(theta, X, t)
        elif self.variant == 'esn':
            s_t = self.esn_vol(theta, X, t)
        elif self.variant == 'barron':
            s_t = self.barron_vol(theta, X, t)
        elif self.variant == 'rand-sig':
            s_t = self.rsig_vol(theta, X, t)
        else:  # variant = t_sig or log_sig
            s_t = self.sig_vol(theta, X, t)

        return s_t