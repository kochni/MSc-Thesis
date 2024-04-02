"""
Core module.

Overview
========

This module defines the following core objects:

* `FeynmanKac`: the base class for Feynman-Kac models;
* `SMC`: the base class for SMC algorithms;
* `multiSMC`: a function to run a SMC algorithm several times, in
  parallel and/or with varying options.

You don't need to import this module: these objects
are automatically imported when you import the package itself::

    import particles
    help(particles.SMC)  # should work

Each of these three objects have extensive docstrings (click on the links
above if you are reading the HTML version of this file).  However, here is a
brief summary for the first two.

The FeynmanKac abstract class
=============================

A Feynman-Kac model is basically a mathematical model for the operations that
we want to perform when running a particle filter. In particular:

    * The distribution *M_0(dx_0)* says how we want to simulate the particles at
      time 0.
    * the Markov kernel *M_t(x_{t-1}, dx_t)* says how we want to simulate
      particle X_t at time t, given an ancestor X_{t-1}.
    * the weighting function *G_t(x_{t-1}, x_t)* says how we want to reweight
      at time t a particle X_t and its ancestor is X_{t-1}.

For more details on Feynman-Kac models and their properties, see Chapter 5 of
the book.

To define a Feynman-Kac model in particles, one should, in principle:

    (a) sub-class `FeynmanKac` (define a class that inherits from it)
        and define certain methods such as `M0`, `M`, `G`, see
        the documentation of `FeynmanKac` for more details;
    (b) instantiate (define an object that belongs to) that sub-class.

In many cases however, you do not need to do this manually:

    * module `state_space_models` defines classes that automatically
      generate the bootstrap, guided or auxiliary Feynman-Kac model associated
      to a given state-space model; see the documentation of that module.
    * Similarly, module `smc_samplers` defines classes that automatically
      generates `FeynmanKac` objects for SMC tempering, IBIS and so on. Again,
      check the documentation of that module.

That said, it is not terribly complicated to define manually a Feynman-Kac
model, and there may be cases where this might be useful. There is even a basic
example in the tutorials if you are interested.

SMC class
=========

`SMC` is the class that define SMC samplers. To get you started::

    import particles
    ... # define a FeynmanKac object in some way, see above
    pf = particles.SMC(fk=my_fk_model, N=100)
    pf.run()

The code above simply runs a particle filter with ``N=100`` particles for the
chosen Feynman-Kac model. When this is done, object ``pf`` contains several
attributes, such as:

    * ``X``: the current set of particles (at the final time);
    * ``W``: their weights;
    * ``cpu_time``: as the name suggests;
    * and so on.

`SMC` objects are iterators, making it possible to run the algorithm step by
step: replace the last line above by::

    next(step) # do iteration 0
    next(step) # do iteration 1
    pf.run() # do iterations 2, ... until completion (dataset is exhausted)

All options, minus ``model``, are optional. Perhaps the most important ones are:
    * ``qmc``: if set to True, runs SQMC (the quasi-Monte Carlo version of SMC)
    * ``resampling``: the chosen resampling scheme; see `resampling` module.
    * ``store_history``: whether we should store the particles at all iterations;
        useful in particular for smoothing, see `smoothing` module.

See the documentation of `SMC` for more details.

"""


import numpy as np

from particles import collectors, hilbert
from particles import resampling as rs
from particles import rqmc, smoothing, utils

# (!) modification (!)
# other packages
from particles.distributions import F_innov, inv_cdf, mix_cdf, sq_mix_cdf
from statsmodels.stats.weightstats import DescrStatsW
import itertools

err_msg_missing_trans = """
    Feynman-Kac class %s is missing method logpt, which provides the log-pdf
    of Markov transition X_t | X_{t-1}. This is required by most smoothing
    algorithms."""


class FeynmanKac:
    """Abstract base class for Feynman-Kac models.

    To actually define a Feynman-Kac model, one must sub-class FeymanKac,
    and define at least the following methods:

        * `M0(self, N)`: returns a collection of N particles generated from the
          initial distribution M_0.
        * `M(self, t, xp)`: generate a collection of N particles at time t,
           generated from the chosen Markov kernel, and given N ancestors (in
           array xp).
        * `logG(self, t, xp, x)`: log of potential function at time t.

    To implement a SQMC algorithm (quasi-Monte Carlo version of SMC), one must
    define methods:

        * `Gamma0(self, u)`: deterministic function such that, if u~U([0,1]^d),
          then Gamma0(u) has the same distribution as X_0
        * `Gamma(self, xp, u)`: deterministic function that, if U~U([0,1]^d)
          then Gamma(xp, U) has the same distribution as kernel M_t(x_{t-1}, dx_t)
          for x_{t-1}=xp

    Usually, a collection of N particles will be simply a numpy array of
    shape (N,) or (N,d). However, this is not a strict requirement, see
    e.g. module `smc_samplers` and the corresponding tutorial in the on-line
    documentation.
    """

    # by default, we mutate at every time t

    def __init__(self, T):
        self.T = T

    def _error_msg(self, meth):
        cls_name = self.__class__.__name__
        return f'method/property {meth} missing in class {cls_name}'

    def M0(self, N):
        """Sample N times from initial distribution M_0 of the FK model"""
        raise NotImplementedError(self._error_msg("M0"))

    def M(self, t, xp):
        """Generate X_t according to kernel M_t, conditional on X_{t-1}=xp"""
        raise NotImplementedError(self._error_msg("M"))

    def logG(self, t, xp, x):
        """Evaluates log of function G_t(x_{t-1}, x_t)"""
        raise NotImplementedError(self._error_msg("logG"))

    def Gamma0(self, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M0."""
        raise NotImplementedError(self._error_msg("Gamma0"))

    def Gamma(self, t, xp, u):
        """Deterministic function that transform a uniform variate of dimension
        d_x into a random variable with the same distribution as M(self, t, xp).
        """
        raise NotImplementedError(self._error_msg("Gamma"))

    def logpt(self, t, xp, x):
        """Log-density of X_t given X_{t-1}."""
        raise NotImplementedError(err_msg_missing_trans % self.__class__.__name__)

    @property
    def isAPF(self):
        """Returns true if model is an APF"""
        return "logeta" in dir(self)

    def done(self, smc):
        """Time to stop the algorithm"""
        return smc.t >= self.T

    def time_to_resample(self, smc):
        """When to resample."""
        return smc.aux.ESS < smc.N * smc.ESSrmin

    def default_moments(self, W, X):
        """Default moments (see module ``collectors``).

        Computes weighted mean and variance (assume X is a Numpy array).
        """
        return rs.wmean_and_var(W, X)

    def summary_format(self, smc):
        return "t=%i: resample:%s, ESS (end of iter)=%.2f" % (
            smc.t,
            smc.rs_flag,
            smc.wgts.ESS,
        )


class SMC:
    """Metaclass for SMC algorithms.

       Parameters
       ----------
       fk : FeynmanKac object
           Feynman-Kac model which defines which distributions are
           approximated
       N : int, optional (default=100)
           number of particles
       qmc : bool, optional (default=False)
           if True use the Sequential quasi-Monte Carlo version (the two
           options resampling and ESSrmin are then ignored)
       resampling : {'multinomial', 'residual', 'stratified', 'systematic', 'ssp'}
           the resampling scheme to be used (see `resampling` module for more
           information; default is 'systematic')
       ESSrmin : float in interval [0, 1], optional
           resampling is triggered whenever ESS / N < ESSrmin (default=0.5)
       store_history : bool, int or callable (default=False)
           whether and when history should be saved; see module `smoothing`
       verbose : bool, optional
           whether to print basic info at every iteration (default=False)
       collect : list of collectors, or 'off' (for turning off summary collections)
           see module ``collectors``

    Attributes
    ----------

       t : int
          current time step
       X : typically a (N,) or (N, d) ndarray (but see documentation)
           the N particles
       A : (N,) ndarray (int)
          ancestor indices: A[n] = m means ancestor of X[n] has index m
       wgts : `Weights` object
           An object with attributes lw (log-weights), W (normalised weights)
           and ESS (the ESS of this set of weights) that represents
           the main (inferential) weights
       aux : `Weights` object
           the auxiliary weights (for an auxiliary PF, see FeynmanKac)
       cpu_time : float
           CPU time of complete run (in seconds)
       hist : `ParticleHistory` object (None if option history is set to False)
           complete history of the particle system; see module `smoothing`
       summaries : `Summaries` object (None if option summaries is set to False)
           each summary is a list of estimates recorded at each iteration. The
           summaries computed by default are ESSs, rs_flags, logLts.
           Extra summaries may also be computed (such as moments and online
           smoothing estimates), see module `collectors`.

       Methods
       -------
       run()
           run the algorithm until completion
       step()
           run the algorithm for one step (object self is an iterator)
    """

    def __init__(
        self,
        fk=None,
        N=100,
        qmc=False,
        resampling="systematic",
        ESSrmin=0.5,
        store_history=False,
        verbose=False,
        collect=None,

        # (!) modification (!)
        # variables for prediction
        h=None,
        M=None,
        naive_uq=None,
        cal_uq=None,
        alpha=None,
        eta=None,
        strike=None
    ):

        self.fk = fk
        self.N = N
        self.qmc = qmc
        self.resampling = resampling
        self.ESSrmin = ESSrmin
        self.verbose = verbose

        # initialisation
        self.t = 0
        self.rs_flag = False  # no resampling at time 0, by construction
        self.logLt = 0.0
        self.DIC = 0.0  # (!) modification (!)
        self.wgts = rs.Weights()
        self.aux = None
        self.X, self.Xp, self.A = None, None, None

        # (!) modification (!)
        self.preds = {}
        self.predsets = {}
        self.h = h
        self.M = M
        self.naive_uq = naive_uq
        self.cal_uq = cal_uq
        self.alpha = alpha
        self.alpha_star_RV = alpha
        self.alpha_star_S = alpha
        self.alpha_star_C = alpha
        self.eta = eta
        self.strike = strike
        self.rand_proj = {}

        # summaries computed at every t
        if collect == "off":
            self.summaries = None
        else:
            self.summaries = collectors.Summaries(collect)
        self.hist = smoothing.generate_hist_obj(store_history, self)

    def __str__(self):
        return self.fk.summary_format(self)

    @property
    def W(self):
        return self.wgts.W

    def reset_weights(self):
        """ Reset weights after a resampling step."""
        if self.fk.isAPF:
            lw = rs.log_mean_exp(self.logetat, W=self.W) - self.logetat[self.A]
            self.wgts = rs.Weights(lw=lw)
        else:
            self.wgts = rs.Weights()

    def setup_auxiliary_weights(self):
        """ Compute auxiliary weights (for APF)."""
        if self.fk.isAPF:
            self.logetat = self.fk.logeta(self.t - 1, self.X)
            self.aux = self.wgts.add(self.logetat)
        else:
            self.aux = self.wgts

    def generate_particles(self):
        if self.qmc:
            u = rqmc.sobol(self.N, self.fk.du).squeeze()
            # squeeze: must be (N,) if du=1
            self.X = self.fk.Gamma0(u)
        else:
            self.X = self.fk.M0(self.N)

    def reweight_particles(self):
        self.wgts = self.wgts.add(self.fk.logG(self.t, self.Xp, self.X))

    def resample_move(self):
        self.rs_flag = self.fk.time_to_resample(self)
        if self.rs_flag:  # if resampling
            self.A = rs.resampling(self.resampling, self.aux.W, M=self.N)
            # we always resample self.N particles, even if smc.X has a
            # different size (example: waste-free)
            self.Xp = self.X[self.A]
            self.reset_weights()
        else:
            self.A = np.arange(self.N)
            self.Xp = self.X
        self.X = self.fk.M(self.t, self.Xp)

    def resample_move_qmc(self):
        self.rs_flag = True  # we *always* resample in SQMC
        u = rqmc.sobol(self.N, self.fk.du + 1)
        tau = np.argsort(u[:, 0])
        self.h_order = hilbert.hilbert_sort(self.X)
        self.A = self.h_order[rs.inverse_cdf(u[tau, 0], self.aux.W[self.h_order])]
        self.Xp = self.X[self.A]
        v = u[tau, 1:].squeeze()
        #  v is (N,) if du=1, (N,d) otherwise
        self.reset_weights()
        self.X = self.fk.Gamma(self.t, self.Xp, v)

    # (!) modification (!)
    def theta_capped(self, K):

        theta = self.X.theta  # (N,n_params), structured array
        N = len(theta)

        # (1) Model Parameters
        # GARCH:
        if 'alpha' in theta.dtype.names and 'beta' in theta.dtype.names:
            theta['omega'] = np.clip(theta['omega'], 0., 100.)
            theta['alpha'] = np.clip(theta['alpha'], 0., 1.)
            theta['beta'] = np.clip(theta['beta'], 0., 1.)
        elif 'alpha_0' in theta.dtype.names and 'beta_0' in theta.dtype.names:
            theta['p_0'] = np.clip(theta['p_0'], 0., 1.)
            theta['omega_0'] = np.clip(theta['omega_0'], 0., None)
            theta['omega_1'] = np.clip(theta['omega_1'], 0., None)
            theta['alpha_0'] = np.clip(theta['alpha_0'], 0., 1.)
            theta['alpha_1'] = np.clip(theta['alpha_1'], 0., 1.)
            theta['beta_0'] = np.clip(theta['beta_0'], 0., 1.)
            theta['beta_1'] = np.clip(theta['beta_1'], 0., 1.)
        # Canonical SV:
        elif 'alpha' in theta.dtype.names and 'xi' in theta.dtype.names:
            theta['alpha'] = np.clip(theta['alpha'], -1., 1.)
            theta['xi'] = np.clip(theta['xi'], 0., None)
        if 'alpha_0' in theta.dtype.names and 'xi_0' in theta.dtype.names:
            theta['p_0'] = np.clip(theta['p_0'], 0., 1.)
            theta['alpha_0'] = np.clip(theta['alpha_0'], -1., 1.)
            theta['alpha_1'] = np.clip(theta['alpha_1'], -1., 1.)
            theta['xi_0'] = np.clip(theta['xi_0'], 0., None)
            theta['xi_1'] = np.clip(theta['xi_1'], 0., None)
       # Heston:
        elif 'kappa' in theta.dtype.names:
            theta['kappa'] = np.clip(theta['kappa'], 0., None)
            theta['nu'] = np.clip(theta['nu'], 0., None)
            theta['xi'] = np.clip(theta['xi'], 0., None)
        elif 'kappa_0' in theta.dtype.names:
            theta['kappa_0'] = np.clip(theta['kappa_0'], 0., None)
            theta['kappa_1'] = np.clip(theta['kappa_1'], 0., None)
            theta['nu_0'] = np.clip(theta['nu_0'], 0., None)
            theta['nu_1'] = np.clip(theta['nu_1'], 0., None)
            theta['xi_0'] = np.clip(theta['xi_0'], 0., None)
            theta['xi_1'] = np.clip(theta['xi_1'], 0., None)
        # Neural:

        # (2) Distribution parameters
        # Return distribution:
        if 'df_X' in theta.dtype.names:
            theta['df_X'] = np.clip(theta['df_X'], 3.0, 500.)
        elif 'tail_X' in theta.dtype.names:
            theta['tail_X'] = np.clip(theta['tail_X'], -50., 50.)
            skew_X = np.clip(theta['skew_X'], -50., 50.)
            theta['skew_X'] = skew_X
            theta['shape_X'] = np.clip(theta['shape_X'],
                                     abs(skew_X)+0.01, 50.01)

        # Volatility distribution
        if 'df_V' in theta.dtype.names:
            theta['df_V'] = np.clip(theta['df_V'], 3.0, 500.)
        elif 'tail_V' in theta.dtype.names:
            theta['tail_V'] = np.clip(theta['tail_V'], -50., 50.)
            skew_V = np.clip(theta['skew_V'], -50., 50.)
            theta['skew_V'] = skew_V
            theta['shape_V'] = np.clip(theta['shape_V'],
                                       abs(skew_V)+0.01, 50.01)

        # (3) Jump parameters
        # Jumps in returns:
        if 'lambda_X' in theta.dtype.names:
            theta['lambda_X'] = np.clip(theta['lambda_X'], 0., 1.)
            theta['phi_X'] = np.clip(theta['phi_X'], 0., 1e10)
        # Jumps in volatility
        if 'jumps_V' in theta.dtype.names:
            theta['lambda_V'] = np.clip(theta['lambda_V'], 0., 1.)
            theta['phi_V'] = np.clip(theta['phi_V'], 0., 50.)

        return theta

    # (!) modification (!)
    def probs(self, theta, K):
        ''' probabilities of regimes and jumps for SMC^2 models '''

        N = len(theta)

        # regime probabilities (N,K)
        if K == 1:
            p_regimes = np.ones([N, 1])  # (N,1)
        elif K == 2:
            p_0 = theta['p_0']
            p_regimes = np.vstack([p_0, 1.0-p_0]).T  # (N,2)
        elif K == 3:
            p_0 = theta['p_0']
            p_1 = theta['p_1']
            p_regimes = np.vstack([p_0, p_1, 1.0-p_0-p_1]).T

        # jump probabilities (N,J) where J=1 if no jumps and J=2 if yes
        if 'lambda_X' not in theta.dtype.names:
            p_jumps_X = np.ones([N, 1])
        else:
            lambda_X = theta['lambda_X']
            p_jumps_X = np.vstack([1.0-lambda_X, lambda_X]).T

        if 'lambda_V' not in theta.dtype.names:
            p_jumps_V = np.ones([N, 1])
        else:
            lambda_V = theta['lambda_V']
            p_jumps_V = np.vstack([1.0-lambda_V, lambda_V]).T

        return p_regimes, p_jumps_X, p_jumps_V

    # (!) modification (!)
    def predict_IBIS(self):
        '''
        produce point predictions and prediction sets for IBIS models

        Note: first datapoint is actually X_1 = log(S_1/S_0); last datapoint
              is log(S_T/S_{T-1}), where T is length of price series.
              Hence, quantities being predicted are
              - X_{t+1} = 100·log(S_t/S_{t-1}) = X[t]
              - S_{t+1} = S_t·exp(X_t/100)
              - C_{t+1} = (S_{t+1} - S_t)_+

        '''

        t = self.t  # current time step
        returns = self.fk.model.data
        S = self.fk.model.S  # prices
        S_t = S[t]
        M = self.M  # no. of returns sampled;

        K = self.fk.model.K

        # θ-particles & weights
        N = self.N  # no. of θ-particles
        W = self.W  # weights
        theta = self.theta_capped(K)  # capped particles

        # volatilities & probabilities
        vols = self.fk.model.s_t   # (N,K,J)
        probs = self.fk.model.p_t  # (N,K,J)

        # innovation distribution
        innov_X = self.fk.model.innov_X

        # parameters of innov distr
        theta_FX = {}
        if innov_X == 't':
            theta_FX['df'] = theta['df_X'].reshape(N, 1, 1)
        elif innov_X == 'GH':
            theta_FX['shape'] = theta['shape_X'].reshape(N, 1, 1)
            theta_FX['tail'] = theta['tail_X'].reshape(N, 1, 1)
            theta_FX['skew'] = theta['skew_X'].reshape(N, 1, 1)

        # jumps
        jumps_X = self.fk.model.jumps_X

        ####################
        # Simulate Returns #
        ####################
        M = self.M  # no. of simulations; (!) why tuple??

        # gather all possible combinations of indices of stochastic components
        i_set = np.arange(0, N, 1)    # θ-particle indices
        k_set = np.arange(0, K, 1)    # regime indices
        JX_set = np.arange(0, jumps_X+1, 1)  # X-jump indicators
        I = itertools.product(i_set, k_set, JX_set)  # generator
        I = np.array(list(I))  # array of all possible index combn's

        # probabilities: probs above are not yet normalized over θ-particles
        probs = np.einsum('N,NKJ->NKJ', W, probs)

        # sample M volatilities
        n_distinct = len(I)  # no. of distinct index samples
        ind_set_ind = np.random.choice(np.arange(n_distinct),
                                       p=probs.flatten(), size=M)

        i_ind = I[ind_set_ind, 0]  # sampled θ-particle indices
        k_ind = I[ind_set_ind, 1]  # ...
        JX_ind = I[ind_set_ind, 2]

        # sampled volatilities
        vol_sim = vols[i_ind, k_ind, JX_ind]  # (M,)

        # sample 1 return for each sampled volatility
        theta_FX_smpl = {k: v[i_ind, k_ind, 0] for k, v in theta_FX.items()}  # (M,)
        X_sim = F_innov(0., vol_sim, **theta_FX_smpl).rvs(M)  # (M,)
        X_sim = np.clip(X_sim, -50., 50.)

        #####################
        # Point Predictions #
        #####################
        # (1) Realized Variance
        RV_pred = np.einsum('NKJ,NKJ->...', vols**2, probs)

        # (2) Price
        # no point prediction (assumed martingale)

        # (3) Option Payout
        strike = S_t if self.strike == 'last' else self.strike
        C_sim = np.clip(np.exp(0.01*X_sim) * S_t - strike, 0., None)
        C_pred = np.mean(C_sim)

        ###################
        # Prediction Sets #
        ###################

        # naive quantile levels:
        q_S = [0.5*self.alpha, 1.-0.5*self.alpha]
        q_RV = [1.-self.alpha]
        q_C = [1.-self.alpha]  # for upper bound / VaR intervals

        # calibrated quantile levels:
        if self.cal_uq is True:
            q_S += [0.5*self.alpha_star_S, 1.-0.5*self.alpha_star_S]
            q_RV += [1.-self.alpha_star_RV]
            q_C += [1.-self.alpha_star_C]

        # for non-GH innovations, get "closed form" (i.e., quantiles of
        # induced mixture distribution) for variance reduction
        if innov_X != 'GH':
            # (1) Realized Variance
            RV_predset = inv_cdf(cdf=lambda x: sq_mix_cdf(x, probs, 0., vols,
                                                          **theta_FX),
                                 p=q_RV, lo=0., hi=100.)
            RV_predset = [0.0, RV_predset[0], 0.0, RV_predset[1]]

            # (2) Price
            X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., vols,
                                                      **theta_FX),
                                p=q_S, lo=-50., hi=50.)
            S_predset = np.exp(0.01*X_predset) * S_t

            # (3) Option Payout
            X_predset = inv_cdf(cdf=lambda x: mix_cdf(x, probs, 0., vols,
                                                      **theta_FX),
                                p=q_C, lo=0., hi=50.)
            C_predset = np.exp(0.01*X_predset) * S_t - strike
            C_predset = [0.0, C_predset[0], 0.0, C_predset[1]]

        # GH CDF too intensive to evaluate, inversion not feasible
        # -> use empirical quantiles from simulated returns
        else:
            # (1) Realized Variance
            RV_predset = np.quantile(X_sim**2, q_RV)
            RV_predset = [0.0, RV_predset[0], 0.0, RV_predset[1]]

            # (2) Price
            X_predset = np.quantile(X_sim, q_S)
            S_predset = np.exp(0.01*X_predset) * S_t

            # (3) Option Payouts
            C_predset = np.quantile(C_sim, q_C)
            C_predset = [0.0, C_predset[0], 0.0, C_predset[1]]

        # append pred's & pred' sets
        if t == 0:
            self.preds['RV'] = RV_pred
            self.preds['C'] = C_pred

            self.predsets['RV'] = RV_predset
            self.predsets['S'] = S_predset
            self.predsets['C'] = C_predset

        else:
            self.preds['RV'] = np.append(self.preds['RV'], RV_pred)
            self.preds['C'] = np.append(self.preds['C'], C_pred)

            self.predsets['RV'] = np.vstack([self.predsets['RV'], RV_predset])
            self.predsets['S'] = np.vstack([self.predsets['S'], S_predset])
            self.predsets['C'] = np.vstack([self.predsets['C'], C_predset])

        # check coverage & calibrate quantile level:
        S_next = S[t+1]
        RV_next = returns[t]**2
        C_next = np.clip(S_next - S_t, 0.0, None)  # option payouts

        RV_err = 1.0 - (RV_predset[2] <= RV_next <= RV_predset[3])
        S_err = 1.0 - (S_predset[2] <= S_next <= S_predset[3])
        C_err = 1.0 - (C_predset[2] <= C_next <= C_predset[3])

        self.alpha_star_RV += self.eta * (self.alpha - RV_err)
        self.alpha_star_S += self.eta * (self.alpha - S_err)
        self.alpha_star_C += self.eta * (self.alpha - C_err)

        self.alpha_star_RV = np.clip(self.alpha_star_RV, 0.001, 0.999)
        self.alpha_star_S = np.clip(self.alpha_star_S, 0.001, 0.999)
        self.alpha_star_C = np.clip(self.alpha_star_C, 0.001, 0.999)

    # (!) modification (!)
    def predict_SMC2(self):
        '''
        produce point predictions and prediction sets for SMC^2 models

        Note: first datapoint is actually X_1 = log(S_1/S_0).
              Hence X_t = X[t-1], and  quantities being predicted are
              - X_{t+1} = 100·log(S_t/S_{t-1}) = X[t]
              - S_{t+1} = S_t·exp(X_t/100)
              - C_{t+1} = (S_{t+1} - S_t)_+

        '''
        # prepare commonly used variables
        t = self.t  # current time step
        K = self.X.pfs[0].fk.ssm.K
        S = self.fk.S
        S_t = S[t]
        returns = self.fk.data
        T = len(S)

        # θ-particles & weights
        theta = self.theta_capped(K)  # particles
        N = self.N  # no. of particles
        W = self.W  # weights

        # x-particles & weights
        Nx = self.X.pfs[0].N  # no. of x-particles
        x = np.full([N, Nx], np.nan)   # x-particles
        Wx = np.full([N, Nx], np.nan)  # weights
        for i in range(N):
            x[i, :] = self.X.pfs[i].X
            Wx[i, :] = self.X.pfs[i].W
        x = np.clip(x, -50., 50.)

        # reservoirs (for ResComps)
        rc_names = ['Extreme', 'Echo', 'Barron', 'Sig', 'RandSig']
        if any(s in str(self.X.pfs[0].fk.ssm) for s in rc_names):
            q = self.X.pfs[0].fk.ssm.q  # reservoir dimensionality
            res = np.full([q, N, Nx], np.nan)  # reservoirs
            for i in range(N):
                res[:, i, :] = self.X.pfs[i].fk.ssm.res
        else:
            res = np.full([1, N, Nx], np.nan)  # reservoirs

        # innovations
        innov_X = self.X.pfs[0].fk.ssm.innov_X
        innov_V = self.X.pfs[0].fk.ssm.innov_V

        # probabilities
        p_regimes, p_jumps_X, p_jumps_V = self.probs(theta, K)

        # whether jumps specified or not
        jumps_X = self.X.pfs[0].fk.ssm.jumps_X
        jumps_V = self.X.pfs[0].fk.ssm.jumps_V

        ####################
        # Simulate Returns #
        ####################
        M = self.M  # no. of simulations; (!) why tuple??

        # gather all possible combinations of indices of stochastic components
        i_set = np.arange(0, N, 1)   # θ-particle indices
        j_set = np.arange(0, Nx, 1)  # V-particle indices
        k_set = np.arange(0, K, 1)   # regime indices
        JX_set = np.arange(0, jumps_X+1, 1)  # X-jump indicators
        JV_set = np.arange(0, jumps_V+1, 1)  # V-jump indicators
        I = itertools.product(i_set, j_set, k_set, JX_set, JV_set)  # generator
        I = np.array(list(I))  # array of all possible index combn's

        # probability of each combination
        probs = np.einsum('N,NS,NK,NJ,NL->NSKJL', W, Wx, p_regimes,
                          p_jumps_X, p_jumps_V)
        # (!) are probabilities in correct order? (!)

        # sample indices
        n_distinct = len(I)  # no. of different index combinations
        ind_set_ind = np.random.choice(np.arange(n_distinct),
                                       p=probs.flatten(),
                                       size=M)

        i_ind = I[ind_set_ind, 0]  # sampled θ-particle indices
        j_ind = I[ind_set_ind, 1]  # ...
        k_ind = I[ind_set_ind, 2]
        JX_ind = I[ind_set_ind, 3]
        JV_ind = I[ind_set_ind, 4]

        # (1) sample M volatilities:
        # E[log(V_{t+1}^2)]
        ElogV2 = self.X.pfs[0].fk.ssm.EXt(theta[i_ind],
                                          x[i_ind, j_ind],
                                          returns[t-1],
                                          k_ind,
                                          res[:, i_ind, j_ind])

        # SD(log(V_{t+1}^2)) (w/o jump!)
        SDlogV2 = self.X.pfs[0].fk.ssm.SDXt(theta[i_ind],
                                            x[i_ind, j_ind],
                                            returns[t-1],
                                            k_ind,
                                            res[:, i_ind, j_ind])

        if jumps_V is True:
            # increase vol-olf-vol where V-jump occured:
            SDlogV2[JV_ind==1] = np.sqrt(SDlogV2[JV_ind==1]**2 + theta['phi_V'][i_ind[JV_ind==1]]**2)

        theta_FV = {}
        if innov_V == 't':
            theta_FV['df'] = theta['df_V'][i_ind]
        elif innov_V == 'GH':
            theta_FV['shape'] = theta['shape_V'][i_ind]
            theta_FV['tail'] = theta['tail_V'][i_ind]
            theta_FV['skew'] = theta['skew_V'][i_ind]

        logV2_sim = F_innov(ElogV2, SDlogV2, **theta_FV, a=-20., b=20.).rvs(M)
        logV2_sim = np.clip(logV2_sim, -20., 20.)  # truncation not applied if Gaussian
        V2_sim = np.exp(logV2_sim)
        vol_sim = np.exp(0.5*logV2_sim)

        # (2) sample 1 return for each sampled volatility
        if jumps_X is True:
            # increase volatility where X-jump occured:
            vol_sim[JX_ind==1] = np.sqrt(vol_sim[JX_ind==1] + theta['phi_X'][i_ind[JX_ind==1]]**2)

        theta_FX = {}
        if innov_X == 't':
            theta_FX['df'] = theta['df_X'][i_ind]
        elif innov_X == 'GH':
            theta_FX['shape'] = theta['shape_X'][i_ind]
            theta_FX['tail'] = theta['tail_X'][i_ind]
            theta_FX['skew'] = theta['skew_X'][i_ind]

        X_sim = F_innov(0., vol_sim, **theta_FX).rvs(M)
        X_sim = np.clip(X_sim, -50., 50.)

        #####################
        # Point Predictions #
        #####################
        # (1) Realized Variance
        RV_pred = np.mean(V2_sim)

        # (2) Price
        # no point prediction (assumed martingale)

        # (3)) Option Payout
        strike = S_t if self.strike == 'last' else self.strike
        C_sim = np.clip(np.exp(0.01*X_sim) * S_t - strike, 0., None)
        C_pred = np.mean(C_sim)

        ###################
        # Prediction Sets #
        ###################

        # naive quantile levels:
        q_S = [0.5*self.alpha, 1.-0.5*self.alpha]
        q_RV = [1.-self.alpha]
        q_C = [1.-self.alpha]  # for upper bound / VaR intervals

        # calibrated quantile levels:
        if self.cal_uq is True:
            q_S += [0.5*self.alpha_star_S, 1.-0.5*self.alpha_star_S]
            q_RV += [1.-self.alpha_star_RV]
            q_C += [1.-self.alpha_star_C]

        # (1) Realized Variance
        RV_sim = X_sim ** 2
        RV_predset = np.quantile(RV_sim, q_RV)
        RV_predset = [0.0, RV_predset[0], 0.0, RV_predset[1]]

        # (2) Price
        X_predset = np.quantile(X_sim, q_S)
        S_predset = np.exp(0.01*X_predset) * S_t

        # (3) Option Payout
        C_predset = np.quantile(C_sim, q_C)
        C_predset = [0.0, C_predset[0], 0.0, C_predset[1]]

        # append pred's & pred' sets
        if t == 0:
            self.preds['RV'] = RV_pred
            self.preds['C'] = C_pred

            self.predsets['RV'] = RV_predset
            self.predsets['S'] = S_predset
            self.predsets['C'] = C_predset

        else:
            self.preds['RV'] = np.append(self.preds['RV'], RV_pred)
            self.preds['C'] = np.append(self.preds['C'], C_pred)

            self.predsets['RV'] = np.vstack([self.predsets['RV'], RV_predset])
            self.predsets['S'] = np.vstack([self.predsets['S'], S_predset])
            self.predsets['C'] = np.vstack([self.predsets['C'], C_predset])

        # check coverage & calibrate quantile level:
        S_next = S[t+1]
        RV_next = returns[t]**2
        C_next = np.clip(S_next - S_t, 0.0, None)  # option payouts

        RV_err = 1.0 - (RV_predset[2] <= RV_next <= RV_predset[3])
        S_err = 1.0 - (S_predset[2] <= S_next <= S_predset[3])
        C_err = 1.0 - (C_predset[2] <= C_next <= C_predset[3])

        self.alpha_star_RV += self.eta * (self.alpha - RV_err)
        self.alpha_star_S += self.eta * (self.alpha - S_err)
        self.alpha_star_C += self.eta * (self.alpha - C_err)

        self.alpha_star_RV = np.clip(self.alpha_star_RV, 0.001, 0.999)
        self.alpha_star_S = np.clip(self.alpha_star_S, 0.001, 0.999)
        self.alpha_star_C = np.clip(self.alpha_star_C, 0.001, 0.999)

    def compute_summaries(self):
        if self.t > 0:
            prec_log_mean_w = self.log_mean_w
        self.log_mean_w = self.wgts.log_mean  # log mean unnormalized weight
        if self.t == 0 or self.rs_flag:
            self.loglt = self.log_mean_w
        else:
            self.loglt = self.log_mean_w - prec_log_mean_w
        self.logLt += self.loglt

        # (!) modification (!)
        # self.DIC

        if self.verbose:
            print(self)
        if self.hist:
            self.hist.save(self)
        # must collect summaries *after* history, because a collector (e.g.
        # FixedLagSmoother) may needs to access history
        if self.summaries:
            self.summaries.collect(self)

    def __next__(self):
        """ One step of a particle filter. """
        if self.fk.done(self):
            # (!) modification (!)
            # print that model done
            if 'IBIS' in str(self.fk):
                model = str(self.fk.model.dynamics) + " (" + str(self.fk.model.variant) + ")"
                print(model + " done!")
            elif 'SMC2' in str(self.fk):
                model = str(self.fk.ssm_cls.__name__)
                print(model + " done!")

            raise StopIteration

        if self.t == 0:
            # (!) modification (!)
            # in ResComp models, draw new set of random matrices
            if 'SMC2' in str(self.fk):
                rc_names = ['Extreme', 'Echo', 'Barron', 'Sig', 'RandSig']
                if any(s in str(self.fk.ssm_cls) for s in rc_names):
                    self.fk.ssm_cls.generate_matrices()
                    if 'Extreme' in str(self.fk.ssm_cls):
                        self.rand_proj['A'] = self.fk.ssm_cls.A
                        self.rand_proj['b'] = self.fk.ssm_cls.b
                        # self.rand_proj['activ'] = shi

            elif 'IBIS' in str(self.fk):
                self.fk.model.generate_matrices()
                # if ELM, save inner weights to later check learned function:
                if self.fk.model.variant == 'elm':
                    self.rand_proj['A'] = self.fk.model.model.A
                    self.rand_proj['b'] = self.fk.model.model.b
                    self.rand_proj['activ'] = self.fk.model.model.activ

            self.generate_particles()

        else:
            self.setup_auxiliary_weights()  # APF
            if self.qmc:
                self.resample_move_qmc()
            else:
                self.resample_move()

        self.reweight_particles()

        # (!) modification (!)
        # when particle filter step complete (SMC.X is ThetaParticles object),
        # produce predictions & prediction sets
        if 'IBIS' in str(self.fk):
            self.predict_IBIS()
        elif 'ThetaParticles' in str(self.X):
            self.predict_SMC2()

        self.compute_summaries()
        self.t += 1

    def next(self):
        return self.__next__()  #  Python 2 compatibility

    def __iter__(self):
        return self

    @utils.timer
    def run(self):
        """Runs particle filter until completion.

        Note: this class implements the iterator protocol. This makes it
        possible to run the algorithm step by step::

            pf = SMC(fk=...)
            next(pf)  # performs one step
            next(pf)  # performs one step
            for _ in range(10):
                next(pf)  # performs 10 steps
            pf.run()  # runs the remaining steps

        In that case, attribute `cpu_time` records the CPU cost of the last
        command only.
        """
        for _ in self:
            pass


####################################################


class _picklable_f:

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, **kwargs):
        pf = SMC(**kwargs)
        pf.run()
        return self.fun(pf)


@_picklable_f
def _identity(x):
    return x


def multiSMC(nruns=10, nprocs=0, out_func=None, collect=None, **args):
    """Run SMC algorithms in parallel, for different combinations of parameters.

    `multiSMC` relies on the `multiplexer` utility, and obeys the same logic.
    A basic usage is::

        results = multiSMC(fk=my_fk_model, N=100, nruns=20, nprocs=0)

    This runs the same SMC algorithm 20 times, using all available CPU cores.
    The output, ``results``, is a list of 20 dictionaries; a given dict corresponds
    to a single run, and contains the following (key, value) pairs:

        + ``'run'``: a run identifier (a number between 0 and nruns-1)

        + ``'output'``: the corresponding SMC object (once method run was completed)

    Since a `SMC` object may take a lot of space in memory (especially when
    the option ``store_history`` is set to True), it is possible to require
    `multiSMC` to store only some chosen summary of the SMC runs, using option
    `out_func`. For instance, if we only want to store the estimate
    of the log-likelihood of the model obtained from each particle filter::

        of = lambda pf: pf.logLt
        results = multiSMC(fk=my_fk_model, N=100, nruns=20, out_func=of)

    It is also possible to vary the parameters. Say::

        results = multiSMC(fk=my_fk_model, N=[100, 500, 1000])

    will run the same SMC algorithm 30 times: 10 times for N=100, 10 times for
    N=500, and 10 times for N=1000. The number 10 comes from the fact that we
    did not specify nruns, and its default value is 10. The 30 dictionaries
    obtained in results will then contain an extra (key, value) pair that will
    give the value of N for which the run was performed.

    It is possible to vary several arguments. Each time a list must be
    provided. The end result will amount to take a *cartesian product* of the
    arguments::

        results = multiSMC(fk=my_fk_model, N=[100, 1000], resampling=['multinomial',
                           'residual'], nruns=20)

    In that case we run our algorithm 80 times: 20 times with N=100 and
    resampling set to multinomial, 20 times with N=100 and resampling set to
    residual and so on.

    Finally, if one uses a dictionary instead of a list, e.g.::

        results = multiSMC(fk={'bootstrap': fk_boot, 'guided': fk_guided}, N=100)

    then, in the output dictionaries, the values of the parameters will be replaced
    by corresponding keys; e.g. in the example above, {'fk': 'bootstrap'}. This is
    convenient in cases such like this where the parameter value is some non-standard
    object.

    Parameters
    ----------
    * nruns : int, optional
        number of runs (default is 10)
    * nprocs : int, optional
        number of processors to use; if negative, number of cores not to use.
        Default value is 1 (no multiprocessing)
    * out_func : callable, optional
        function to transform the output of each SMC run. (If not given, output
        will be the complete SMC object).
    * collect : list of collectors, or 'off'
        this particular argument of class SMC may be a list, hence it is "protected"
        from Cartesianisation
    * args : dict
        arguments passed to SMC class (except collect)

    Returns
    -------
    A list of dicts

    See also
    --------
    `utils.multiplexer`: for more details on the syntax.
    """
    f = _identity if out_func is None else _picklable_f(out_func)
    return utils.multiplexer(
        f=f,
        nruns=nruns,
        nprocs=nprocs,
        seeding=True,
        protected_args={"collect": collect},
        **args
    )
