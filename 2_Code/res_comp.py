#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class ResComp:
    '''
    Reservoir computers for static volatility models

    Parameters:
    -----------

    variant: string
        type of the RC approach use; one of 'elm' (for extreme learning
        machine, ELM), 't_sig' (for truncated signature), and 'r_sig'
        (for randomized signature); ELM is Markovian, while signature-based methods
        model path-dependence.

    q: int
        dimensionality of the reservoir.

    hyper: dict
        dictionary of hyperparameters; must include
            'q' (int): dimensionality of the reservoir.
            'sd' (float): standard deviation of the random initializations of
                the weights resp. random signature components.
            'activ' (func): activation function used in the random neural
                network resp. the generation of the randomized signature.

    innov: string
        innovation distribution; one of 'N' (for Gaussian) and 't' (for
         Student t)

    prior: StructDist
        prior distribution on parameters.

    data: (T,)-array
        data.
    '''

    def __init__(self, variant, hyper, K):

        self.variant = variant
        self.K = K
        self.q = q = hyper['q']
        sd = hyper.get('sd')

        # draw random components
        if variant == 'elm':  # inner parameters of NN
            H = 20  # hidden layer width
            self.A1 = np.random.normal(scale=sd, size=[H, 2, K])
            self.b1 = np.random.normal(scale=sd, size=[H, 1, K])
            self.A2 = np.random.normal(scale=sd, size=[q, H, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K])

        elif variant == 'r_sig':  # random matrices of Controlled ODE
            self.rsig_0 = np.random.normal(scale=sd, size=[q, K])
            self.A1 = np.random.normal(scale=sd, size=[q, q, K])
            self.b1 = np.random.normal(scale=sd, size=[q, 1, K])
            self.A2 = np.random.normal(scale=sd, size=[q, q, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K])

        else:
            # compute minimum required truncation level to obtain enough
            # components as specified in dimensionality (q)
            self.sig_dim = 1
            while 2*(2**self.sig_dim - 1) < self.q: self.sig_dim += 1

        # extract repeatedly accessed hyperparameters
        self.activ = hyper.get('activ')
        self.s2_0 = np.random.gamma(1., 0.5, size=1)  # initial volatility

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

        w0s = cap(w0s, -1e20, 1e20)
        Ws = cap(Ws, -1e20, 1e20)

        # shape initial vol to (N,K) of identical entries
        log_s2_j = np.tile(np.log(self.s2_0), [N, K])
        for j in range(1, t+1):
            log_s2_prev = log_s2_j

            # input to reservoir: previous log-vol & previous return
            X_prev = np.full([N, K], X[t-1])
            M = np.stack((X_prev, log_s2_prev), axis=0)  # (2,N,K)

            # hidden nodes 1:
            h1 = np.einsum('HIK,INK->HNK', self.A1, M) + self.b1
            h1 = self.activ(h1)  # (H,N,K)

            # hidden nodes 2:
            h2 = np.einsum('qHK,HNK->qNK', self.A2, h1)+ self.b2
            res = self.activ(h2)

            # get volatility from reservoir & readout:
            log_s2_j = np.einsum('qNK,qNK->qNK', Ws, res) + w0s
            log_s2_j = np.einsum('qNK->NK', log_s2_j)
            log_s2_j = cap(log_s2_j, -100, 100)

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
        sig_dim = self.sig_dim

        if t > 0:
            # time-extended path:
            X_tilde = np.vstack([X, np.arange(0, t, 1)]) # (2,t)

            if self.variant == 'standard':
                sig = stream2sig(X_tilde.T, sig_dim)[:q+1]  # (q+1,)
            else:  # log-signature
                sig = stream2sig(X_tilde.T, sig_dim)[:q+1]

            # extract weights:
            if K == 1:
                w_names = ['w' + str(j) for j in range(q+1)]
                Ws = itemgetter(*w_names)(theta)
                Ws = np.array(Ws).reshape(q+1, N, 1)  # (q+1,N,1)
            else:  # multi-regime
                Ws = np.full([q+1, N, K], np.nan)
                for k in range(K):
                    w_names = ['w' + str(j) + '_' + str(k) for j in range(q+1)]
                    Ws[:, :, k] = itemgetter(*w_names)(theta)  # (q+1,N)

            # compute volatility from weights & signature components:
            log_s2_t = np.einsum('qNK,q->NK', Ws, sig)

            log_s2_t = np.clip(log_s2_t, -50., 50.)
            s_t = np.exp(0.5*log_s2_t)

        else:  # t == 0
            s_t = np.full([N, K], np.sqrt(self.s2_0))

        return s_t


    def rsig_vol(self, theta, X, t):
        '''
        compute volatility from the randomized signature of the time-extended
        path of the log-returns, (t, X_t)
        '''
        N = len(theta)
        K = self.K
        q = self.q
        shape0 = [N, K]
        shape = [N, q, K]

        # extract weights (updated at every t during training)
        w0s = np.full([N, K], np.nan)
        Ws = np.full([q, N, K], np.nan)
        if K == 1:
            w_names = ['w' + str(j) for j in range(q+1)]
            W = itemgetter(*w_names)(theta)  # (q+1,N)
            w0s[:, 0] = np.array(W)[0, :]  # (N,)
            Ws[:, :, 0] = np.array(W)[1:, :]  # (q,N)
        else:  # multi-regime
            for k in range(K):
                w_names = ['w' + str(j) + '_' + str(k) for j in range(q+1)]
                W = itemgetter(*w_names)(theta)  # (q,N)
                w0s[:, k] = W[0, :]
                Ws[:, :, k] = np.array(W)  # (q,N)

        Ws = np.clip(Ws, -1e20, 1e20)

        if t > 0:
            log_s2_j = np.full([N, K], np.log(self.s2_0))
            rsig = np.tile(self.rsig_0.reshape(q, 1, K), [1, N, 1])  # (q,N,K)
            for j in range(1, t+1):
                log_s2_prev = log_s2_j

                # update rSig
                incr_1 = np.einsum('qpK,pNK->qNK', self.A1, rsig) + self.b1
                incr_1 = self.activ(incr_1)  # (q,N,K)
                # (note: p is same as q)

                incr_2 = np.einsum('qpK,pNK->qNK', self.A2, rsig) + self.b2
                incr_2 = self.activ(incr_2)  # (q,N,K)
                Z_prev = X[j-1] * np.exp(-0.5*log_s2_prev)  # (N,K)
                incr_2 = np.einsum('qNK,NK->qNK', incr_2, Z_prev)  # (q,N,K)

                rsig += incr_1 + incr_2  # (q,N,K)

                # get volatility from reservoir & rSig
                log_s2_j = np.einsum('qNK,qNK->qNK', Ws, rsig)
                log_s2_j = np.einsum('qNK->NK', log_s2_j) + w0s  # sum over axis 0
                log_s2_j = np.clip(log_s2_j, -50., 50.)

        else:  # t = 0:
            s_t = np.full([N, K], np.sqrt(self.s2_0))
            return s_t

        s_t = np.exp(0.5*log_s2_j)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'elm':
            s_t = self.elm_vol(theta, X, t)
        elif self.variant == 'r_sig':
            s_t = self.rsig_vol(theta, X, t)
        else:  # variant = t_sig or log_sig
            s_t = self.sig_vol(theta, X, t)

        return s_t