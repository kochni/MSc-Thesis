#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
            self.b1 = np.random.normal(scale=sd, size=[H, 1, K, 1])
            self.A2 = np.random.normal(scale=sd, size=[q, H, K])
            self.b2 = np.random.normal(scale=sd, size=[q, 1, K, 1])

        elif variant == 'r_sig':  # random matrices of Controlled ODE
            self.rsig_0 = np.random.normal(scale=sd, size=[q, K])
            self.A1 = np.random.normal(scale=sd, size=[q, q, K])
            self.b1 = np.random.normal(scale=sd, size=[q, K])
            self.A2 = np.random.normal(scale=sd, size=[q, q, K])
            self.b2 = np.random.normal(scale=sd, size=[q, K])

        else:
            # compute minimum required truncation level to obtain enough
            # components as specified in dimensionality (q)
            self.sig_dim = 1
            while 2*(2**self.sig_dim - 1) < self.q: self.sig_dim += 1

        # extract repeatedly accessed hyperparameters
        self.activ = hyper.get('activ')
        self.s2_0 = 1  # initial volatility

    def elm_vol(self, theta, X, t):
        '''
        compute volatility by passing past volatility & returns through a
        random 2-layer neural network, a.k.a. "extreme learning machine (ELM)"

        '''
        N = len(theta)
        K = self.K
        q = self.q
        shape0 = [N, K]  # shape of w0s (bias term)
        shape = [q, N, K]  # shape of Ws (linear weights)

        # during model fitting, 1 stream of returns
        # during forward simulation, N*M stream of returns
        # --> reshape X to (N,t,K,M) for stacking later
        if X.ndim == 1:  # X has shape (t,)
            X_resh = X[np.newaxis, :, np.newaxis, np.newaxis]  # (1,t,1,1)
            X_resh = np.tile(X_resh, [N, 1, K, 1])
        else:  # X has shape (N,t,M)
            X_resh = np.tile(X[:, :, np.newaxis, :], [1, 1, K, 1])

        M = X_resh.shape[3]  # no. of particle copies

        # extract parameters (updated at every t during training)
        w0s = np.full(shape0, np.nan)
        Ws = np.full(shape, np.nan)
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

        # reshape weights to
        w0s = w0s[np.newaxis, :, :, np.newaxis]

        # shape initial vol to (N,K,M) of identical entries
        log_s2_j = np.tile(np.log(self.s2_0), [N, K, M])
        for j in range(1, t+1):
            log_s2_prev = log_s2_j

            # input to reservoir is previous log-vol & previous return;
            # N*M*K copies of each
            M = np.stack((X_resh[:, t-1, :, :],
                          log_s2_prev), axis=0)  # (2,N,K,M)

            # hidden nodes 1:
            # 1x (H,2)·(2,N) mat-mul for each regime & particle copy
            # matrices differ between regimes but not particle copies
            h1 = np.einsum('HIK,INKM->HNKM', self.A1, M, optimize=True)
            h1 = h1 + self.b1
            h1 = self.activ(h1)  # (H,N,K,M)

            # hidden nodes 2:
            h2 = np.einsum('qHK,HNKM->qNKM', self.A2, h1, optimize=True)
            h2 = h2 + self.b2
            res = self.activ(h2)  # final reservoir of shape (q,N,K,M)

            # get volatility from reservoir & readout:
            # Hadamard:
            log_s2_j = np.einsum('qNK,qNKM->qNKM', Ws, res, optimize=True)
            # add bias
            log_s2_j += w0s
            # sum over axis 0:
            log_s2_j = np.einsum('qNKM->NKM', log_s2_j, optimize=True)
            log_s2_j = cap(log_s2_j, -100, 100)

        # if X is 1-dim, then must return (N,K) array for logpyt
        if X.ndim == 1: log_s2_j = log_s2_j[:, :, 0]

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

        # add axes to make X 3-dimensional --> (N,t,M)
        if X.ndim == 1:  # X.shape = (t,)
            X_resh = X[np.newaxis, :, np.newaxis]
        else:  # X has shape (N,t,M) already
            X_resh = X

        N_eff = X_resh.shape[0]  # = 1 during training, N during simulation
        M = X_resh.shape[2]      # = 1 during training, M during simulation

        if t > 0:
            # N_eff*M different paths and hence signatures, each of dim q+1
            sigs = np.full([N_eff, q+1, M], np.nan)
            for i in range(N_eff):  # esig not vectorized; can't use 'apply'
                for j in range(M):  # because output dim very different
                    # time-extended path:
                    X_tilde = np.vstack([X_resh[i, 0:t, j], np.arange(0, t, 1)])
                    # (2,t)

                    if self.variant == 'standard':
                        sigs[i, :, j] = stream2sig(X_tilde.T, sig_dim)[:q+1]
                    else:  # log-signature
                        sigs[i, :, j] = stream2sig(X_tilde.T, sig_dim)[:q+1]

            # extract weights:
            if K == 1:
                w_names = ['w' + str(j) for j in range(q+1)]
                W = itemgetter(*w_names)(theta)  # (q+1,N)
                Ws = np.array(W).T  # (N,q+1)
                Ws = Ws[:, :, np.newaxis]  # (N,q+1,1)
            else:  # multi-regime
                Ws = np.full([N, q+1, K], np.nan)
                for k in range(K):
                    w_names = ['w' + str(j) + '_' + str(k) for j in range(q+1)]
                    W = itemgetter(*w_names)(theta)  # (q+1,N)
                    Ws[:, :, k] = np.array(W).T  # (N,q+1)

            # compute volatility from weights & signature components:
            log_s2_t = np.einsum('NqK,NqM->NqKM', Ws, sigs)  # Hadamard
            log_s2_t = np.einsum('NqKM->NKM', log_s2_t)  # sum

            # if X is 1-dim, then must return 2D (N,K) array for logpyt
            if X.ndim == 1: log_s2_t = log_s2_t[:, :, 0]

            log_s2_t = cap(log_s2_t, -50., 50.)
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

        # reshape X to 3 dim (N,t,M)
        if X.ndim == 1:
            X_resh = X[np.newaxis, :, np.newaxis]  # (1,t,1)
        else:  # X has shape (N,t,K) already
            X_resh = X

        M = X_resh.shape[2]

        # K different initial signatures rSig_0 of dim q;
        # K different matrices & biases
        rsig = self.rsig_0[:, np.newaxis, :, np.newaxis]  # (q,N=1,K,M=1)
        b1 = self.b1[:, np.newaxis, :, np.newaxis]
        b2 = self.b2[:, np.newaxis, :, np.newaxis]

        # extract weights (updated at every t during training)
        if K == 1:
            w_names = ['w' + str(j) for j in range(q+1)]
            W = itemgetter(*w_names)(theta)  # (q+1,N)
            w0s = np.array(W)[-1, :]  # (N,)
            w0s = w0s[np.newaxis, :, np.newaxis]
            Ws = np.array(W)[:-1, :]  # (q,N)
            Ws = Ws[:, :, np.newaxis]
        else:  # multi-regime
            w0s = np.full([1, N, K, 1], np.nan)
            Ws = np.full([q, N, K], np.nan)
            for k in range(K):
                w0s[0, :, k, 0] = theta['w0_' + str(k)]
                w_names = ['w' + str(j) + '_' + str(k) for j in range(1, q+1)]
                W = itemgetter(*w_names)(theta)  # (q,N)
                Ws[:, :, k] = np.array(W)  # (q,N)

        Ws = cap(Ws, -1e20, 1e20)

        if t > 0:
            log_s2_j = np.full([N,K,1], np.log(self.s2_0))
            for j in range(1, t+1):
                log_s2_prev = log_s2_j  # (N,K,M) at t>0, (1,1,1) at t=1
                # print("log_s2_prev.shape:", log_s2_prev.shape)
                # print("log_s2_j.shape:", log_s2_j.shape)

                # update rSig
                incr_1 = np.einsum('qpK,pNKM->qNKM', self.A1, rsig) + b1
                incr_1 = self.activ(incr_1)  # (q,N,K,M)

                incr_2 = np.einsum('qpK,pNKM->qNKM', self.A2, rsig) + b2
                incr_2 = self.activ(incr_2)  # (q,N,K,M)

                # previous innovations (N,K,M)
                X_prev = X_resh[:, j-1, np.newaxis, :]  # X_{t-1}
                s_prev_inv = np.exp(-0.5*log_s2_prev)   # σ_{t-1}^{-1}
                Z_prev = np.einsum('NKM,NKM->NKM', X_prev, s_prev_inv) # Had'

                incr_2 = incr_2 * Z_prev[np.newaxis, :, :, :]  # (q,N,K,M)

                rsig = rsig + incr_1 + incr_2  # (q,N,K,M)

                # get volatility from reservoir & rSig
                log_s2_j = np.einsum('qNK,qNKM->qNKM', Ws, rsig)  # Hadamard
                log_s2_j = log_s2_j + w0s  #
                log_s2_j = np.einsum('qNKM->NKM', log_s2_j)  # sum over axis 0
                log_s2_j = cap(log_s2_j, -50., 50.)

            # if X is 1-dim, then must return 2D (N,K) array for logpyt
            if X.ndim == 1: log_s2_j = log_s2_j[:, :, 0]

        else:  # t = 0:
            s_t = np.full([N, K], np.sqrt(self.s2_0))
            return s_t

        s_t = np.exp(log_s2_j/2)

        return s_t

    def vol(self, theta, X, t):  # wrapper

        if self.variant == 'elm':
            s_t = self.elm_vol(theta, X, t)
        elif self.variant == 'r_sig':
            s_t = self.rsig_vol(theta, X, t)
        else:  # variant = t_sig or log_sig
            s_t = self.sig_vol(theta, X, t)

        return s_t