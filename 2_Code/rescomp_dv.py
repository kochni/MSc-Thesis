#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# basic
import numpy as np

# signatures
from esig import stream2sig, stream2logsig, sigdim, logsigdim

# other
from operator import itemgetter


class ResComp:
    '''
    Reservoir computers for deterministic volatility models

    Parameters:
    -----------

    variant: string
        type of the RC approach use; one of 'elm' (for Extreme Learning
        Machine), 'esn' (for Echo State Network), 't_sig' (for truncated
        signature), and 'r_sig' (for randomized signature);
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
        self.q = q = hyper['q']
        sd = hyper.get('sd')

        # draw random components
        if variant == 'elm':  # inner parameters of NN
            # H = 100  # hidden layer width
            self.A = np.random.normal(scale=sd, size=[q, 3, K])
            self.b = np.random.normal(scale=sd, size=[q, 1, K])
            # self.A2 = np.random.normal(scale=sd, size=[q, H, K])
            # self.b2 = np.random.normal(scale=sd, size=[q, 1, K])

        elif variant == 'esn':
            self.A = np.random.normal(scale=sd, size=[q, q, K])
            self.C = np.random.normal(scale=sd, size=[q, 3, K])
            self.b = np.random.normal(scale=sd, size=[q, 1, K])

        elif variant == 'r-sig':  # random matrices of Controlled ODE
            self.rsig_0 = np.random.normal(scale=sd, size=[q, 1, K])
            self.A1 = np.random.normal(scale=sd, size=[q, q, K])
            self.A2 = np.random.normal(scale=sd, size=[q, q, K])
            self.A3 = np.random.normal(scale=sd, size=[q, q, K])
            self.b1 = np.random.normal(scale=sd, size=[q, 1, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K])
            self.b3 = np.random.normal(scale=sd, size=[q, 1, K])

        else:  # variant == 'sig'
            # compute minimum required truncation level to obtain enough
            # components as specified in dimensionality (q)
            self.sig_len = 1
            if variant == 'sig':
                sig_length = sigdim
            elif variant == 'logsig':
                sig_length = logsigdim
            while sig_length(2, self.sig_len) < self.q + 1:
                self.sig_len += 1

        # extract repeatedly accessed hyperparameters
        self.activ = hyper.get('activ')
        self.s2_0 = np.random.gamma(1., 0.5, size=1)  # initial volatility
        self.res_0 = np.random.normal(size=[q, 1, 1])

    def elm_vol(self, theta, X, t):
        '''
        compute volatility by passing past volatility & returns through a
        random 2-layer neural network, a.k.a. "extreme learning machine (ELM)"

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract parameters (updated at every t)
        w0s = np.full([N, K], np.nan)
        Ws = np.full([q, N, K], np.nan)
        if K == 1:
            w0s[:, 0] = theta['w0']
            for j in range(q):
                Ws[j, :, 0] = theta['w' + str(j)]
        else:
            for k in range(K):
                w0s[:, k] = theta['w0_' + str(k)]
                for j in range(q):
                    Ws[j, :, k] = theta['w' + str(j) + '_' + str(k)]

        w0s = np.clip(w0s, -1e20, 1e20)
        Ws = np.clip(Ws, -1e20, 1e20)

        # shape initial vol to (N,K) of identical entries
        log_s2_j = np.full([N, K], np.log(self.s2_0))
        for j in range(1, t+1):
            log_s2_prev = log_s2_j

            # input to reservoir: t, previous log-var, previous return
            time = np.full([N, K], t)
            X_prev = np.full([N, K], X[t-1])
            M = np.stack([time, X_prev, log_s2_prev], axis=0)  # (3,N,K)
            # M = np.stack([X_prev, log_s2_prev], axis=0)  # (2,N,K)

            # hidden nodes 1:
            h1 = np.einsum('HIK,INK->HNK', self.A, M) + self.b
            h1 = self.activ(h1)  # (H,N,K)

            # hidden nodes 2:
            # h2 = np.einsum('qHK,HNK->qNK', self.A2, h1)+ self.b2
            # res = self.activ(h2)
            res = h1

            # get volatility from reservoir & readout:
            log_s2_j = np.einsum('qNK,qNK->qNK', Ws, res) + w0s
            log_s2_j = np.einsum('qNK->NK', log_s2_j)
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def esn_vol(self, theta, X, t):
        '''
        Echo State Network

        log(σ_t^2) = w·r_t
        r_t = ϕ(C·r_{t-1} + A·z_{t-1} + b)

        '''
        N = len(theta)
        K = self.K
        q = self.q

        # extract parameters (updated at every t)
        w0s = np.full([N, K], np.nan)
        Ws = np.full([q, N, K], np.nan)
        if K == 1:
            w0s[:, 0] = theta['w0']
            for j in range(q):
                Ws[j, :, 0] = theta['w' + str(j)]
        else:
            for k in range(K):
                w0s[:, k] = theta['w0_' + str(k)]
                for j in range(q):
                    Ws[j, :, k] = theta['w' + str(j) + '_' + str(k)]

        w0s = np.clip(w0s, -1e20, 1e20)
        Ws = np.clip(Ws, -1e20, 1e20)

        # shape initial vol to (N,K) of identical entries
        log_s2_j = np.full([N, K], np.log(self.s2_0))  # (N,K)
        res_j  = np.tile(self.res_0, [1, N, K])  # (1,N,K)
        for j in range(1, t+1):
            # recurrent reservoir step:
            h1 = np.einsum('qpK,qNK->qNK', self.A, res_j)  # (q,N,K)

            # input variable: t, previous log-var, previous return
            time = np.full([N, K], t)
            X_prev = np.full([N, K], X[t-1])
            M = np.stack([time, X_prev, log_s2_j], axis=0)  # (3,N,K)
            h2 = np.einsum('qIK,INK->qNK', self.C, M)

            # new reservoir
            res_j = self.activ(h1 + h2 + self.b)  # (q,N,K)

            # get volatility from reservoir & readout:
            log_s2_j = np.einsum('qNK,qNK->qNK', Ws, res_j) + w0s
            log_s2_j = np.einsum('qNK->NK', log_s2_j)
            log_s2_j = np.clip(log_s2_j, -100., 100.)

        s_t = np.exp(0.5*log_s2_j)

        return s_t


    def sig_vol(self, theta, X, t):
        '''
        compute volatility from the truncated signature of the time-extended
        path of the log-returns, (t, X_t)

        '''
        N = len(theta)
        K = self.K
        q = self.q
        sig_len = self.sig_len

        if t > 0:
            # extract Sig version
            if self.variant == 'standard': # standard Signature
                Sig = stream2sig
            elif self.variant == 'logsig':   # log-Signature
                Sig = stream2logsig

            # extract weights
            w_names = ['w' + str(j) for j in range(q+1)]
            w = itemgetter(*w_names)(theta)
            w = np.array(w)  # (q+1,N)

            # input path: time and returns until time t-1
            time = np.arange(0, t, 1)
            X_past = X[:t]
            u = np.stack([time, X_past]).T # (t,2)
            S = Sig(u, sig_len)[:q+1]  # (q+1,)
            S = S.reshape(-1, 1)  # (q+1,1)
            log_s2_t = np.sum(w * S, axis=0) + w[0, :]  # (N,)

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
            s_t = np.full([N], np.sqrt(self.s2_0))

        return s_t


    def rsig_vol(self, theta, X, t):
        '''
        compute volatility from the randomized signature of the time-extended
        path of log-returns and log-variances, (s, X_s, log(σ_s^2))_{s<t}

        '''
        N = len(theta)
        K = self.K
        q = self.q

        w = np.full([q, N, K], np.nan)
        w0 = np.full([N, K], np.nan)

        if K == 1:
            w_names = ['w' + str(j) for j in range(q+1)]
            w0_w = itemgetter(*w_names)(theta)  # (q+1,N)
            w0_w = np.array(w0_w)
            w0[:, 0] = w0_w[0, :]  # (N,)
            w[:, :, 0] = w0_w[1:, :]  # (q,N)
        else:  # multi-regime
            for k in range(K):
                w_names = ['w' + str(j) + '_' + str(k) for j in range(q+1)]
                w = itemgetter(*w_names)(theta)  # (q,N)
                w = np.array(w)
                w0[:, k] = w[0, :]
                w[:, :, k] = w[1:, :]  # (q,N)

        # cap parameters
        w0 = np.clip(w0, -1e20, 1e20)
        w = np.clip(w, -1e20, 1e20)

        if t > 0:
            log_s2_j = np.full([N, K], np.log(self.s2_0))
            rsig = np.tile(self.rsig_0, [1, N, 1])  # (q,N,K)
            for j in range(1, t):
                # increment from time:
                incr_1 = np.einsum('qpK,pNK->qNK', self.A1, rsig) + self.b1
                incr_1 = self.activ(incr_1)  # (q,N,K)
                # (note: p = q)

                # increment from log-returns:
                incr_2 = np.einsum('qpK,pNK->qNK', self.A2, rsig) + self.b2
                incr_2 = self.activ(incr_2)  # (q,N,K)
                delta_X = X[j] - X[j-1]
                incr_2 = incr_2 * delta_X  # (q,N,K)

                # increment from log-variance:
                incr_3 = np.einsum('qpK,pNK->qNK', self.A3, rsig) + self.b3
                incr_3 = self.activ(incr_2)  # (q,N,K)
                if j == 1:
                    delta_logvar = log_s2_j
                else:
                    delta_logvar = log_s2_j - log_s2_prev
                log_s2_prev = log_s2_j
                incr_3 = incr_3 * delta_logvar  # (q,N,K)

                rsig = rsig + incr_1 + incr_2 + incr_3  # (q,N,K)

                # get volatility from reservoir & rSig
                log_s2_j = np.einsum('qNK,qNK->qNK', w, rsig)
                log_s2_j = np.einsum('qNK->NK', log_s2_j) + w0  # sum over axis 0
                log_s2_j = np.clip(log_s2_j, -50., 50.)

        else:  # t = 0:
            s_t = np.full([N, K], np.sqrt(self.s2_0))
            return s_t

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'elm':
            s_t = self.elm_vol(theta, X, t)
        elif self.variant == 'esn':
            s_t = self.esn_vol(theta, X, t)
        elif self.variant == 'r-sig':
            s_t = self.rsig_vol(theta, X, t)
        else:  # variant = t_sig or log_sig
            s_t = self.sig_vol(theta, X, t)

        return s_t