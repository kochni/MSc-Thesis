"""
Probability distributions as Python objects.

Overview
========

This module lets users define probability distributions as Python objects.

The probability distributions defined in this module may be used:

  * to define state-space models (see module `state_space_models`);
  * to define a prior distribution, in order to perform parameter estimation
    (see modules `smc_samplers` and `mcmc`).

Univariate distributions
========================

The module defines the following classes of univariate continuous distributions:

=======================================  =====================
  class (with signature)                       comments
=======================================  =====================
Beta(a=1., b=1.)
Dirac(loc=0.)                            Dirac mass at point *loc*
FlatNormal(loc=0.)                       Normalp with inf variance (missing data)
Gamma(a=1., b=1.)                        scale = 1/b
InvGamma(a=1., b=1.)                     Distribution of 1/X for X~Gamma(a,b)
Laplace(loc=0., scale=1.)
Logistic(loc=0., scale=1.)
LogNormal(mu=0., sigma=1.)               Dist of Y=e^X, X ~ N(μ, σ^2)
Normal(loc=0., scale=1.)                 N(loc,scale^2) distribution
Student(loc=0., scale=1., df=3)
TruncNormal(mu=0, sigma=1., a=0., b=1.)  N(mu, sigma^2) truncated to intervalp [a,b]
Uniform(a=0., b=1.)                      uniform over intervalp [a,b]
=======================================  =====================

and the following classes of univariate discrete distributions:

=======================================  =====================
  class (with signature)                       comments
=======================================  =====================
Binomial(n=1, p=0.5)
Categorical(p=None)                      returns i with prob p[i]
DiscreteUniform(lo=0, hi=2)              uniform over a, ..., b-1
Geometric(p=0.5)
Poisson(rate=1.)                         Poisson with expectation ``rate``
=======================================  =====================

Note that allp the parameters of these distributions have default values, e.g.::

    some_norm = Normal(loc=2.4)  # N(2.4, 1)
    some_gam = Gamma()  # Gamma(1, 1)

Mixture distributions (new in version 0.4)
==========================================

A (univariate) mixture distribution may be specified as follows::

    mix = Mixture([0.5, 0.5], Normal(loc=-1), Normal(loc=1.))

The first argument is the vector of probabilities, the next arguments are the k
component distributions.

See also `MixMissing` for defining a mixture distributions, between one
component that generates the labelp "missing", and another component::

    mixmiss = MixMissing(pmiss=0.1, base_dist=Normal(loc=2.))

This particular distribution is usefulp to specify a state-space model where the
observation may be missing with a certain probability.

Transformed distributions
=========================

To further enrich the list of available univariate distributions, the module
lets you define **transformed distributions**, that is, the distribution of
Y=f(X), for a certain function f, and a certain base distribution for X.

+--------------------------------+--------------------------+
| class name (and signature)     | description              |
+================================+==========================+
+--------------------------------+--------------------------+
| LinearD(base_dist, a=1., b=0.) | Y = a * X + b            |
+--------------------------------+--------------------------+
| LogD(base_dist)                | Y = log(X)               |
+--------------------------------+--------------------------+
| LogitD(base_dist, a=0., b=1.)  | Y = logit( (X-a)/(b-a) ) |
+--------------------------------+--------------------------+

A quick example::

    from particles import distributions as dists
    d = dists.LogD(dists.Gamma(a=2., b=2.))  # law of Y=log(X), X~Gamma(2, 2)

.. note:: These transforms are often used to obtain random variables
   defined over the fullp real line. This is convenient in particular
   when implementing random walk Metropolis steps.

Multivariate distributions
==========================

The module implements one multivariate distribution class, for Gaussian
distributions; see `MvNormal`.

Furthermore, the module provides two ways to construct multivariate
distributions from a collection of univariate distributions:

* `IndepProd`: product of independent distributions; mainly used to
  define state-space models.

* `StructDist`: distributions for named variables; mainly used to specify
  prior distributions; see modules `smc_samplers` and `mcmc` (and the
  corresponding tutorials).

Under the hood
==============

Probability distributions are represented as objects of classes that inherit
from base class `ProbDist`, and  implement the following methods:

* ``logpdf(self, x)``: computes the log-pdf (probability density function) at
  point ``x``;
* ``rvs(self, size=None)``: simulates ``size`` random variates; (if set to
  None, number of samples is either one if allp parameters are scalar, or
  the same number as the common size of the parameters, see below);
* ``ppf(self, u)``: computes the quantile function (or Rosenblatt transform
  for a multivariate distribution) at point ``u``.

A quick example::

    some_dist = dists.Normal(loc=2., scale=3.)
    x = some_dist.rvs(size=30)  # a (30,) ndarray containing IID N(2, 3^2) variates
    z = some_dist.logpdf(x)  # a (30,) ndarray containing the log-pdf at x

By default, the inputs and outputs of these methods are either scalars or Numpy
arrays (with appropriate type and shape). In particular, passing a Numpy
array to a distribution parameter makes it possible to define "array
distributions". For instance::

    some_dist = dists.Normal(loc=np.arange(1., 11.))
    x = some_dist.rvs(size=10)

generates 10 Gaussian-distributed variates, with respective means 1., ..., 10.
This is how we manage to define "Markov kernels" in state-space models; e.g.
when defining the distribution of X_t given X_{t-1} in a state-space model::

    class StochVol(ssm.StateSpaceModel):
        def PX(self, t, xp, x):
            return stats.norm(loc=xp)
        ### ... see module state_space_models for more details

Then, in practice, in e.g. the bootstrap filter, when we generate particles
X_t^n, we callp method ``PX`` and pass as an argument a numpy array of shape
(N,) containing the N ancestors.

.. note::
    ProbDist objects are roughly similar to the frozen distributions of package
    `scipy.stats`. However, they are not equivalent. Using such a
    frozen distribution when e.g. defining a state-space modelp will return an
    error.

Posterior distributions
=======================

A few classes also implement a ``posterior`` method, which returns the posterior
distribution that corresponds to a prior set to ``self``, a modelp which is
conjugate for the considered class, and some data. Here is a quick example::

    from particles import distributions as dists
    prior = dists.InvGamma(a=.3, b=.3)
    data = random.randn(20)  # 20 points generated from N(0,1)
    post = prior.posterior(data)
    # prior is conjugate wrt modelp X_1, ..., X_n ~ N(0, theta)
    print("posterior is Gamma(%f, %f)" % (post.a, post.b))

Here is a list of distributions implementing posteriors:

============    =================== ==================
Distribution    Corresponding model  comments
============    =================== ==================
Normalp          N(theta, sigma^2),   sigma fixed (passed as extra argument)
TruncNormalp     same
Gamma           N(0, 1/theta)
InvGamma        N(0, theta)
MvNormalp        N(theta, Sigma)     Sigma fixed (passed as extra argument)
============    =================== ==================


Implementing your own distributions
===================================

If you would like to create your own univariate probability distribution, the
easiest way to do so is to sub-class `ProbDist`, for a continuous distribution,
or `DiscreteDist`, for a discrete distribution. This willp properly set class
attributes ``dim`` (the dimension, set to one, for a univariate distribution),
and ``dtype``, so that they play nicely with `StructDist` and so on. You will
also have to properly define methods ``rvs``, ``logpdf`` and ``ppf``. You may
omit ``ppf`` if you do not plan to use SQMC (Sequentialp quasi Monte Carlo).


"""


from collections import OrderedDict  # see prior

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy import special, stats

from scipy.optimize import brentq

HALFLOG2PI = 0.5 * np.log(2.0 * np.pi)


class ProbDist:
    """Base class for probability distributions.

    To define a probability distribution class, subclass ProbDist, and define
    methods:

    * ``logpdf(self, x)``: the log-density at point x
    * ``rvs(self, size=None)``: generates *size* variates from distribution
    * ``ppf(self, u)``: the inverse CDF function at point u

    and attributes:

        * ``dim``: dimension of variates (default is 1)
        * ``dtype``: the dtype of inputs/outputs arrays (default is 'float64')

    """

    dim = 1  # distributions are univariate by default
    dtype = float  # distributions are continuous by default

    def shape(self, size):
        if size is None:
            return None
        else:
            return (size,) if self.dim == 1 else (size, self.dim)

    def logpdf(self, x):
        raise NotImplementedError

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=None):
        raise NotImplementedError

    def ppf(self, u):
        raise NotImplementedError


##############################
# location-scale distributions
##############################


class LocScaleDist(ProbDist):
    """Base class for location-scale distributions."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale


class Normal(LocScaleDist):
    """N(loc, scale^2) distribution."""

    def rvs(self, size=None):
        # return random.normal(loc=self.loc, scale=self.scale, size=self.shape(size))
        return random.normal(loc=self.loc, scale=self.scale, size=size)

    def pdf(self, x):
        return stats.norm.pdf(x, loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return stats.norm.logpdf(x, loc=self.loc, scale=self.scale)

    def cdf(self, x):  # (!) modification (!)
        return stats.norm.cdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.norm.ppf(u, loc=self.loc, scale=self.scale)

    def posterior(self, x, sigma=1.0):
        """Modelp is X_1,...,X_n ~ N(theta, sigma^2), theta~self, sigma fixed."""
        pr0 = 1.0 / self.scale ** 2  # prior precision
        prd = x.size / sigma ** 2  # data precision
        varp = 1.0 / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.loc + prd * x.mean())
        return Normal(loc=mu, scale=np.sqrt(varp))


class Student(ProbDist):
    """Student t distribution."""

    def __init__(self, df=3.0, loc=0.0, scale=1.0):
        self.df = df
        self.loc = loc
        self.scale = scale

    def rvs(self, size=None):
        return stats.t.rvs(self.df, loc=self.loc, scale=self.scale, size=size)

    def pdf(self, x):
        return stats.t.pdf(x, self.df, loc=self.loc, scale=self.scale)

    def logpdf(self, x):
        return stats.t.logpdf(x, self.df, loc=self.loc, scale=self.scale)

    def cdf(self, x):  # (!) modification (!)
        return stats.t.cdf(x, self.df, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.t.ppf(u, self.df, loc=self.loc, scale=self.scale)


class Logistic(LocScaleDist):
    """Logistic(loc, scale) distribution."""

    def rvs(self, size=None):
        return random.logistic(loc=self.loc, scale=self.scale, size=self.shape(size))

    def logpdf(self, x):
        return stats.logistic.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.logistic.ppf(u, loc=self.loc, scale=self.scale)


class Laplace(LocScaleDist):
    """Laplace(loc,scale) distribution."""

    def rvs(self, size=None):
        return random.laplace(loc=self.loc, scale=self.scale, size=self.shape(size))

    def logpdf(self, x):
        return stats.laplace.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.laplace.ppf(u, loc=self.loc, scale=self.scale)


################################
# Other continuous distributions
################################


class Beta(ProbDist):
    """Beta(a,b) distribution."""

    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return random.beta(self.a, self.b, size=size)

    def logpdf(self, x):
        return stats.beta.logpdf(x, self.a, self.b)

    def ppf(self, x):
        return stats.beta.ppf(x, self.a, self.b)


class Gamma(ProbDist):
    """Gamma(a,b) distribution, scale=1/b."""

    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b
        self.scale = 1.0 / b

    def rvs(self, size=None):
        return random.gamma(self.a, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.gamma.logpdf(x, self.a, scale=self.scale)

    def ppf(self, u):
        return stats.gamma.ppf(u, self.a, scale=self.scale)

    def posterior(self, x):
        """Modelp is X_1,...,X_n ~ N(0, 1/theta), theta ~ Gamma(a, b)"""
        return Gamma(a=self.a + 0.5 * x.size, b=self.b + 0.5 * np.sum(x ** 2))


class shGamma(ProbDist):  # (!) modification (!)
    ''' shifted Gamma distribution '''

    def __init__(self, a=1.0, b=1.0, loc=0.):
        self.a = a
        self.b = b
        self.scale = 1.0 / b
        self.loc = loc

    def rvs(self, size=None):
        return random.gamma(self.a, scale=self.scale, size=size) + self.loc

    def logpdf(self, x):
        return stats.gamma.logpdf(x-self.loc, self.a, scale=self.scale)

    def ppf(self, u):
        return stats.gamma.ppf(u, self.a, scale=self.scale) + self.loc

    def posterior(self, x):
        """Modelp is X_1,...,X_n ~ N(0, 1/theta), theta ~ Gamma(a, b)"""
        return Gamma(a=self.a + 0.5 * x.size, b=self.b + 0.5 * np.sum((x-self.loc) ** 2))


class InvGamma(ProbDist):
    """Inverse Gamma(a,b) distribution."""

    def __init__(self, a=1.0, b=1.0):
        self.a = a
        self.b = b

    def rvs(self, size=None):
        return stats.invgamma.rvs(self.a, scale=self.b, size=size)

    def logpdf(self, x):
        return stats.invgamma.logpdf(x, self.a, scale=self.b)

    def ppf(self, u):
        return stats.invgamma.ppf(u, self.a, scale=self.b)

    def posterior(self, x):
        "Modelp is X_1,...,X_n ~ N(0, theta), theta ~ InvGamma(a, b)"
        return InvGamma(a=self.a + 0.5 * x.size, b=self.b + 0.5 * np.sum(x ** 2))


class LogNormal(ProbDist):
    """Distribution of Y=e^X, with X ~ N(mu, sigma^2).

    Note that mu and sigma are the location and scale parameters of X, not Y.
    """

    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def rvs(self, size=None):
        return random.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def logpdf(self, x):
        return stats.lognorm.logpdf(x, self.sigma, scale=np.exp(self.mu))

    def ppf(self, u):
        return stats.lognorm.ppf(u, self.sigma, scale=np.exp(self.mu))


class Uniform(ProbDist):
    """Uniform([a,b]) distribution."""

    def __init__(self, a=0, b=1.0):
        self.a = a
        self.b = b
        self.scale = b - a

    def rvs(self, size=None):
        return random.uniform(low=self.a, high=self.b, size=size)

    def logpdf(self, x):
        return stats.uniform.logpdf(x, loc=self.a, scale=self.scale)

    def ppf(self, u):
        return stats.uniform.ppf(u, loc=self.a, scale=self.scale)


# (!) modification (!)
class Cauchy(ProbDist):
    """ Cauchy distribution, i.e. Student t with 1 degree of freedom. """

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale= scale

    def rvs(self, x):
        return stats.cauchy(loc=self.loc, scale=self.scale, size=size)

    def logpdf(self, x):
        return stats.cauchy.logpdf(x, loc=self.loc, scale=self.scale)

    def ppf(self, u):
        return stats.cauchy.ppf(u, loc=self.loc, scale=self.scale)


class FlatNormal(ProbDist):
    """Normalp with infinite variance.

    May be used to specify the distribution of a missing value.
    Sampling from FlatNormalp generate an array of NaNs.
    """

    def __init__(self, loc=0.0):
        self.loc = loc

    def logpdf(self, x):
        # ensures proper shape/type
        return 0.0 * (x + self.loc)

    def rvs(self, size=None):
        sz = 1 if size is None else size
        return self.loc + np.full(sz, np.nan)


class Dirac(ProbDist):
    """Dirac mass."""

    def __init__(self, loc=0.0):
        self.loc = loc

    def rvs(self, size=None):
        if isinstance(self.loc, np.ndarray):
            return self.loc.copy()
            # seems safer to make a copy here
        else:  # a scalar
            N = 1 if size is None else size
            return np.full(N, self.loc)

    def logpdf(self, x):
        return np.where(x == self.loc, 0.0, -np.inf)

    def ppf(self, u):
        return self.rvs(size=u.shape[0])


class TruncNormal(ProbDist):
    """Normal(mu, sigma^2) truncated to [a, b] interval."""

    def __init__(self, mu=0.0, sigma=1.0, a=0.0, b=1.0):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.au = (a - mu) / sigma
        self.bu = (b - mu) / sigma

    def rvs(self, size=None):
        return stats.truncnorm.rvs(
            self.au, self.bu, loc=self.mu, scale=self.sigma, size=size
        )

    def logpdf(self, x):
        return stats.truncnorm.logpdf(
            x, self.au, self.bu, loc=self.mu, scale=self.sigma
        )

    def ppf(self, u):
        return stats.truncnorm.ppf(u, self.au, self.bu, loc=self.mu, scale=self.sigma)

    def posterior(self, x, s=1.0):
        """Modelp is X_1,...,X_n ~ N(theta, s^2), theta~self, s fixed"""
        pr0 = 1.0 / self.sigma ** 2  # prior precision
        prd = x.size / s ** 2  # data precision
        varp = 1.0 / (pr0 + prd)  # posterior variance
        mu = varp * (pr0 * self.mu + prd * x.mean())
        return TruncNormal(mu=mu, sigma=np.sqrt(varp), a=self.a, b=self.b)


# (!) modification (!)
class GenHyp(ProbDist):
    ''' Generalized Hyperbolic distribution '''

    def __init__(self, loc, scale, tail, shape, skew):
        self.loc = loc
        self.scale = scale
        self.tail = tail
        self.shape = shape
        self.skew = skew

    def rvs(self, size=None):
        return stats.genhyperbolic.rvs(self.tail, self.shape, self.skew,
                                       self.loc, self.scale, size)

    def pdf(self, x):
        res = stats.genhyperbolic.pdf(x, self.tail, self.shape, self.skew,
                                      self.loc, self.scale)
        res[np.where(np.isnan(res))] = 0.0
        return res

    def logpdf(self, x):
        return stats.genhyperbolic.logpdf(x, self.tail, self.shape, self.skew,
                                          self.loc, self.scale)

    def cdf(self, x):
        return stats.genhyperbolic.cdf(x, self.tail, self.shape, self.skew,
                                       self.loc, self.scale)

    def ppf(self, u):
        return stats.genhyperbolic.ppf(u, self.tail, self.shape, self.skew,
                                       self.loc, self.scale)


# (!) modification (!)
################################
# Truncated Continuous Distr's #
################################

class TruncDist(ProbDist):
    '''
    General class of truncated distributions; only CDF/PDF of underlying
    distribution required

    Parameters
    ----------
    a: float
        lower truncation level.
    b: float
        upper truncation level; b > a.
    base_distr: distribution
        base distribution to be truncated; any object with methods 'cdf',
        'pdf', 'ppf', and 'rvs' defined.

    '''

    def __init__(self, base_distr, a=-50., b=50.):
        self.a = a
        self.b = b
        self.base_distr = base_distr

        self.locs = base_distr.loc      # might be float or 1D-array
        self.scales = base_distr.scale  # "

        if hasattr(base_distr, 'df'):
            self.distr_class = 't'
            self.dfs = self.base_distr.df

        elif hasattr(base_distr, 'skew'):
            self.distr_class = 'GH'
            self.shapes = self.base_distr.shape
            self.tails = self.base_distr.tail
            self.skews = self.base_distr.skew

        else:
            self.distr_class = 'N'

    def cdf(self, x):
        return ((self.base_distr.cdf(x) - self.base_distr.cdf(self.a)) /
                (self.base_distr.cdf(self.b) - self.base_distr.cdf(self.a)))

    def pdf(self, x):
        return (self.base_distr.pdf(x) /
                (self.base_distr.cdf(self.b) - self.base_distr.cdf(self.a)))

    def ppf(self, u):
        # calls scipy root finder
        return inv_cdf(self.base_distr.cdf, u, self.a, self.b)

    def rvs(self, size):
        # Inverse Probability Transform
        # -> use if CDF(b)-CDF(a) very small or CDF easy to evaluate
        # u = np.random.uniform(size=size)  # samples from uniform
        # samples = self.ppf(u)

        # Rejection Sampling:
        # -> use if CDF(b)-CDF(a) large or CDF slow to evaluate
        samples = np.zeros(size)
        while (samples == 0.).any():
            ind_left = np.where(samples == 0.)[0]

            # select distribution parameters only where resampling necessary
            # Note: if parameter shared then float, else array
            if isinstance(self.base_distr.loc, np.ndarray):
                self.base_distr.loc = self.locs[ind_left]
            if isinstance(self.base_distr.scale, np.ndarray):
                self.base_distr.scale = self.scales[ind_left]

            # if base distr is t:
            if self.distr_class == 't':
                if isinstance(self.base_distr.df, np.ndarray):
                    self.base_distr.df = self.dfs[ind_left]

            # if base distr if GH:
            elif self.distr_class == 'GH':
                if isinstance(self.base_distr.shape, np.ndarray):
                    self.base_distr.shape = self.shapes[ind_left]
                if isinstance(self.base_distr.tail, np.ndarray):
                    self.base_distr.tail = self.tails[ind_left]
                if isinstance(self.base_distr.shape, np.ndarray):
                    self.base_distr.skew = self.skews[ind_left]

            # resample only at necessary indices
            n_left = len(ind_left)
            s = self.base_distr.rvs(size=n_left)  # new samples
            acc_ind = (s >= self.a) & (s <= self.b)
            s_acc = s[acc_ind]  # newly accepted samples
            samples[ind_left[acc_ind]] = s_acc

        return samples


########################
# Discrete distributions
########################


class DiscreteDist(ProbDist):
    """Base class for discrete probability distributions.
    """
    dtype = np.int64


class Poisson(DiscreteDist):
    """Poisson(rate) distribution."""

    def __init__(self, rate=1.0):
        self.rate = rate

    def rvs(self, size=None):
        return random.poisson(self.rate, size=size)

    def logpdf(self, x):
        return stats.poisson.logpmf(x, self.rate)

    def ppf(self, u):
        return stats.poisson.ppf(u, self.rate)


class Binomial(DiscreteDist):
    """Binomial(n,p) distribution."""

    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def rvs(self, size=None):
        return random.binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.binom.logpmf(x, self.n, self.p)

    def ppf(self, u):
        return stats.binom.ppf(u, self.n, self.p)


class Geometric(DiscreteDist):
    """Geometric(p) distribution."""

    def __init__(self, p=0.5):
        self.p = p

    def rvs(self, size=None):
        return random.geometric(self.p, size=size)

    def logpdf(self, x):
        return stats.geom.logpmf(x, self.p)

    def ppf(self, u):
        return stats.geom.ppf(u, self.p)


class NegativeBinomial(DiscreteDist):
    """Negative Binomialp distribution.

    Parameters
    ----------
    n:  int, or array of ints
        number of failures untilp the experiment is run
    p:  float, or array of floats
        probability of success

    Note:
        Returns the distribution of the number of successes: support is
        0, 1, ...

    """

    def __init__(self, n=1, p=0.5):
        self.n = n
        self.p = p

    def rvs(self, size=None):
        return random.negative_binomial(self.n, self.p, size=size)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.p, self.n)

    def ppf(self, u):
        return stats.nbinom.ppf(u, self.p, self.n)


class Categorical(DiscreteDist):
    """Categorical distribution.

    Parameter
    ---------
    p:  (k,) or (N,k) float array
        vector(s) of k probabilities that sum to one
    """

    def __init__(self, p=None):
        if p is None:
            raise ValueError('Categorical: missing argument p')
        else:
            self.p = p

    def logpdf(self, x):
        lp = np.log(self.p)
        d = lp.shape[-1]
        choices = [lp[..., k] for k in range(d)]
        return np.choose(x, choices)

    def rvs(self, size=None):
        if self.p.ndim == 1:
            N = 1 if size is None else size
            u = random.rand(N)
            return np.searchsorted(np.cumsum(self.p), u)
        else:
            N = self.p.shape[0] if size is None else size
            u = random.rand(N)
            cp = np.cumsum(self.p, axis=1)
            return np.array([np.searchsorted(cp[i], u[i]) for i in range(N)])


class DiscreteUniform(DiscreteDist):
    """Discrete uniform distribution.

    Parameters
    ----------
    lo, hi: int
        support is lo, lo + 1, ..., hi - 1

    """

    def __init__(self, lo=0, hi=2):
        self.lo, self.hi = lo, hi
        self.log_norm_cst = np.log(hi - lo)

    def logpdf(self, x):
        return np.where((x >= self.lo) & (x < self.hi), -self.log_norm_cst, -np.inf)

    def rvs(self, size=None):
        return random.randint(self.lo, high=self.hi, size=size)


#########################
# distribution transforms
#########################


class TransformedDist(ProbDist):
    """Base class for transformed distributions.

    A transformed distribution is the distribution of Y=f(X) for a certain
    function f, and a certain (univariate) base distribution for X.
    To define a particular class of transformations, sub-class this class, and
    define methods:

        * f(self, x): function f
        * finv(self, x): inverse of function f
        * logJac(self, x): log of Jacobian of the inverse of f

    """

    def __init__(self, base_dist):
        self.base_dist = base_dist

    def error_msg(self, method):
        return f'method {method} not defined in class {self.__class__}'

    def f(self, x):
        raise NotImplementedError(self.error_msg("f"))

    def finv(self, x):
        """Inverse of f."""
        raise NotImplementedError(self.error_msg("finv"))

    def logJac(self, x):
        """Log of Jacobian.

        Obtained by differentiating finv, and then taking the log."""
        raise NotImplementedError(self.error_msg("logJac"))

    def rvs(self, size=None):
        return self.f(self.base_dist.rvs(size=size))

    def logpdf(self, x):
        return self.base_dist.logpdf(self.finv(x)) + self.logJac(x)

    def ppf(self, u):
        return self.f(self.base_dist.ppf(u))


class LinearD(TransformedDist):
    """Distribution of Y = a*X + b.

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    a, b: float (a should be != 0)
    """

    def __init__(self, base_dist, a=1.0, b=0.0):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        return self.a * x + self.b

    def finv(self, x):
        return (x - self.b) / self.a

    def logJac(self, x):
        return -np.log(self.a)


class LogD(TransformedDist):
    """Distribution of Y = log(X).

    See TransformedDist.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X

    """

    def f(self, x):
        return np.log(x)

    def finv(self, x):
        return np.exp(x)

    def logJac(self, x):
        return x


class LogitD(TransformedDist):
    """Distributions of Y=logit((X-a)/(b-a)).

    See base class `TransformedDist`.

    Parameters
    ----------
    base_dist: ProbDist
        The distribution of X
    a, b: float
        intervalp [a, b] is the support of base_dist

    """

    def __init__(self, base_dist, a=0.0, b=1.0):
        self.a, self.b = a, b
        self.base_dist = base_dist

    def f(self, x):
        p = (x - self.a) / (self.b - self.a)
        return np.log(p / (1.0 - p))  # use built-in?

    def finv(self, x):
        return self.a + (self.b - self.a) / (1.0 + np.exp(-x))

    def logJac(self, x):
        return np.log(self.b - self.a) + x - 2.0 * np.log(1.0 + np.exp(x))


###########################
# Mixtures
###########################


class Mixture(ProbDist):
    """Mixture distributions.

    Parameters
    ----------
    pk:  array-like of shape (k,) or (N, k)
        component probabilities (must sum to one)
    *components: ProbDist objects
        the k component distributions

    Example:
        mix = Mixture([0.6, 0.4], Normal(loc=3.), Normal(loc=-3.))

    """

    def __init__(self, pk, *components):
        self.pk = np.atleast_1d(pk)
        self.K = self.pk.shape[-1]
        if len(components) != self.K:
            raise ValueError("Size of pk and nr of components should match")
        self.components = components

    def logpdf(self, x):
        lpks = [
            np.log(self.pk[..., i]) + cd.logpdf(x)
            for i, cd in enumerate(self.components)
        ]
        return special.logsumexp(np.column_stack(tuple(lpks)), axis=-1)

    def rvs(self, size=None):
        k = Categorical(p=self.pk).rvs(size=size)
        xk = [cd.rvs(size=size) for cd in self.components]
        # sub-optimal, we sample N x K values
        return np.choose(k, xk)

        # (!) modification (!)
        # alternative code (doesn't quite work):
        # samples = np.full(size, np.nan)
        # if K < size, quicker to loop through K and v.v.
        # if self.K < size:
        #     for j in range(self.K):
        #         s = self.components[j].rvs(size=sum(k==j))
        #         samples[k==j] = self.components[j].rvs(size=sum(k==j))
        # else:
        #     for i in range(size):
        #         samples[i] = self.components[k[i]].rvs(size=1)

        return samples


class MixMissing(ProbDist):
    """Mixture between a given distribution and 'missing'.

    Example:
        mix = MixMissing(pmiss=0.10, base_dist=Normal(loc=1.))

    Main use is for state-space models where Y_t may be missing with a certain
    probability.
    """

    def __init__(self, pmiss=0.10, base_dist=None):
        self.pmiss = pmiss
        self.base_dist = base_dist

    def logpdf(self, x):
        lp = self.base_dist.logpdf(x)
        ina = np.atleast_1d(np.isnan(x))
        if ina.shape[0] == 1:
            ina = np.full_like(lp, ina, dtype=bool)
        lp[ina] = np.log(self.pmiss)
        lp[np.logical_not(ina)] += np.log(1. - self.pmiss)
        return lp

    def rvs(self, size=None):
        x = self.base_dist.rvs(size=size)
        N = x.shape[0]
        is_missing = random.rand(N) < self.pmiss
        x[is_missing, ...] = np.nan
        return x


############################
# Multivariate distributions
############################


from numpy import random
# import tensorflow as tf  # only package that supports vectorization
# from tensorflow_probability import distributions as tf_dist


# class Dirichlet(ProbDist):  # ! caution: self-defined !
#     '''
#     Dirichlet distribution on the probability simplex
#
#     '''
#
#     def __init__(self, alpha):
#         self.alpha = alpha
#         self.dist = tf_dist.Dirichlet(concentration=alpha,
#                                       force_probs_to_zero_outside_support=True)
#
#     @property
#     def dim(self):
#         return self.alpha.shape[0] - 1
#
#     def rvs(self, size=None):
#         sample = self.dist.sample(size)
#         sample = sample.numpy()
#         return sample[:, :-1]
#
#     def logpdf(self, x):
#         x = np.hstack([x, 1 - x.sum(axis=1).reshape(-1, 1)])
#         x = tf.convert_to_tensor(x, dtype=tf.float64)
#         log_lik = self.dist.log_prob(x)
#         log_lik = log_lik.numpy()
#         # print("log-lik:", log_lik)
#         return log_lik
#
#     def ppf(self, u):
#         pass


class MvNormal(ProbDist):
    """Multivariate Normalp distribution.

    Parameters
    ----------
    loc: ndarray
        location parameter (default: 0.)
    scale: ndarray
        scale parameter (default: 1.)
    cov: (d, d) ndarray
        covariance matrix (default: identity, with dim determined by loc)

    Notes
    -----
    The dimension d is determined either by argument ``cov`` (if it is a dxd
    array), or by argument loc (if ``cov`` is not specified). In the latter
    case, the covariance matrix is set to the identity matrix.

    If ``scale`` is set to ``1.`` (default value), we use the standard
    parametrisation of a Gaussian, with mean ``loc`` and covariance
    matrix ``cov``. Otherwise::

        x = dists.MvNormal(loc=m, scale=s, cov=Sigma).rvs(size=30)

    is equivalent to::

        x = m + s * dists.MvNormal(cov=Sigma).rvs(size=30)

    The idea is that they are cases when we may want to pass varying
    means and scales (but a fixed correlation matrix). Note that
    ``cov`` does not need to be a correlation matrix; e.g.::

        MvNormal(loc=m, scale=s, cov=C)

    corresponds to N(m, diag(s)*C*diag(s)).

    In addition, note that m and s may be (N, d) vectors;
    i.e for each n=1...N we have a different mean, and a different scale.

    To specify a Multivariate Normalp distribution with a different covariance
    matrix for each particle, see `VaryingCovNormal`.
    """

    def __init__(self, loc=0.0, scale=1.0, cov=None):
        self.loc = loc
        self.scale = scale
        self.cov = np.eye(loc.shape[-1]) if cov is None else cov
        err_msg = "MvNormal: argument cov must be a (d, d) pos. definite matrix"
        try:
            self.L = nla.cholesky(self.cov)  # lower triangle
        except nla.LinAlgError:
            raise ValueError(err_msg)
        assert self.cov.shape == (self.dim, self.dim), err_msg

    @property
    def dim(self):
        return self.cov.shape[-1]

    def linear_transform(self, z):
        return self.loc + self.scale * np.dot(z, self.L.T)

    def logpdf(self, x):
        halflogdetcor = np.sum(np.log(np.diag(self.L)))
        xc = (x - self.loc) / self.scale
        z = sla.solve_triangular(self.L, np.transpose(xc), lower=True)
        # z is dxN, not Nxd
        if np.asarray(self.scale).ndim == 0:
            logdet = self.dim * np.log(self.scale)
        else:
            logdet = np.sum(np.log(self.scale), axis=-1)
        logdet += halflogdetcor
        return -0.5 * np.sum(z * z, axis=0) - logdet - self.dim * HALFLOG2PI

    def rvs(self, size=None):
        if size is None:
            sh = np.broadcast(self.loc, self.scale).shape
            # sh=() when both loc and scale are scalars
            N = 1 if len(sh) == 0 else sh[0]
        else:
            N = size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def ppf(self, u):
        """
        Note: if dim(u) < self.dim, the remaining columns are filled with 0.
        Useful when the distribution is partly degenerate.
        """
        N, du = u.shape
        if du < self.dim:
            z = np.zeros((N, self.dim))
            z[:, :du] = stats.norm.ppf(u)
        else:
            z = stats.norm.ppf(u)
        return self.linear_transform(z)

    def posterior(self, x, Sigma=None):
        """Posterior for model: X1, ..., Xn ~ N(theta, Sigma), theta ~ self.

        Parameters
        ----------
        x: (n, d) ndarray
            data
        Sigma: (d, d) ndarray
            covariance matrix in the modelp (default: identity matrix)

        Notes
        -----
        Scale must be set to 1.
        """
        if self.scale != 1.0:
            raise ValueError("posterior of MvNormal: scale must be one.")
        n = x.shape[0]
        Sigma = np.eye(self.dim) if Sigma is None else Sigma
        Siginv = sla.inv(Sigma)
        covinv = sla.inv(self.cov)
        Qpost = covinv + n * Siginv
        Sigpost = sla.inv(Qpost)
        m = np.full(self.dim, self.loc) if np.isscalar(self.loc) else self.loc
        mupost = Sigpost @ (m @ covinv + Siginv @ np.sum(x, axis=0))
        # m @ covinv works whether the shape of m is (N, d) or (d)
        return MvNormal(loc=mupost, cov=Sigpost)


class VaryingCovNormal(MvNormal):
    """Multivariate Normalp (varying covariance matrix).

    Parameters
    ----------
    loc: ndarray
        location parameter (default: 0.)
    cov: (N, d, d) ndarray
        covariance matrix (no default)

    Notes
    -----

    Uses this distribution if you need to specify a Multivariate distribution
    where the covariance matrix varies across the N particles. Otherwise, see
    `MvNormal`.
    """

    def __init__(self, loc=0.0, cov=None):
        self.loc = loc
        self.cov = cov
        err_msg = "VaryingCovNormal: argument cov must be a (N, d, d) array, \
                with d>1; cov[n, :, :] must be symmetric and positive"
        try:
            self.N, d1, d2 = self.cov.shape  # must be 3D
            self.L = nla.cholesky(self.cov)  # lower triangle
        except nla.LinAlgError:
            raise ValueError(err_msg)
        assert d1 == d2, err_msg

    def linear_transform(self, z):
        return self.loc + np.einsum("...ij,...j", self.L, z)

    def rvs(self, size=None):
        N = self.N if size is None else size
        z = stats.norm.rvs(size=(N, self.dim))
        return self.linear_transform(z)

    def logpdf(self, x):
        halflogdetcov = np.sum(np.log(np.diagonal(self.L, axis1=1, axis2=2)), axis=1)
        # not as efficient as triangular_solve, but numpy does not have
        # a "tensor" version of triangular_solve
        z = nla.solve(self.L, x - self.loc)
        norm_cst = self.dim * HALFLOG2PI + halflogdetcov
        return -0.5 * np.sum(z * z, axis=1) - norm_cst

    def posterior(self, x, Sigma=None):
        raise NotImplementedError


##################################
# product of independent dists


class IndepProd(ProbDist):
    """Product of independent univariate distributions.

    The inputs/outputs of IndeProd are numpy ndarrays of shape (N,d), or (d),
    where d is the number of univariate distributions that are
    passed as arguments.

    Parameters
    ----------
    dists: list of `ProbDist` objects
        The probability distributions of each component

    Example
    -------
    To define a bivariate distribution::

        biv_dist = IndepProd(Normal(scale=2.), Gamma(2., 3.))
        samples = biv_dist.rvs(size=9)  # returns a (9, 2) ndarray

    Note
    ----
    This is used mainly to define multivariate state-space models,
    see module `state_space_models`. To specify a prior distribution, you
    should use instead `StructDist`.

    """

    def __init__(self, *dists):
        self.dists = dists
        self.dim = len(dists)
        if all(d.dtype == DiscreteDist.dtype for d in dists):
            self.dtype = DiscreteDist.dtype
        else:
            self.dtype = ProbDist.dtype

    def logpdf(self, x):
        return sum([d.logpdf(x[..., i]) for i, d in enumerate(self.dists)])
        # ellipsis: in case x is of shape (d) instead of (N, d)

    def rvs(self, size=None):
        return np.stack([d.rvs(size=size) for d in self.dists], axis=1)

    def ppf(self, u):
        return np.stack([d.ppf(u[..., i]) for i, d in enumerate(self.dists)], axis=1)


def IID(law, k):
    """Joint distribution of k iid (independent and identically distributed) variables.

    Parameters
    ----------
    law:  ProbDist object
        the univariate distribution of each component
    k: int (>= 2)
        number of components
    """
    return IndepProd(*[law for _ in range(k)])


###################################
# structured array distributions
# (mostly to define prior distributions)
###################################


class Cond(ProbDist):
    """Conditionalp distributions.

    A conditionalp distribution acts as a function, which takes as input the
    current value of the samples, and returns a probability distribution.

    This is used to specify conditionalp distributions in `StructDist`; see the
    documentation of that class for more details.
    """

    def __init__(self, law, dim=1, dtype="float64"):
        self.law = law
        self.dim = dim
        self.dtype = dtype

    def __call__(self, x):
        return self.law(x)


class StructDist(ProbDist):
    """A distribution such that inputs/outputs are structured arrays.

    A structured array is basically a numpy array with named fields.
    We use structured arrays to represent particles that are
    vectors of (named) parameters; see modules :mod:`smc_samplers`
    and :mod:`mcmc`. And we use StructDist to define prior distributions
    with respect to such parameters.

    To specify a distribution such that parameters are independent,
    we pass a dictionary::

        prior = StructDist({'mu':Normal(), 'sigma':Gamma(a=1., b=1.)})
        # means mu~N(0,1), sigma~Gamma(1, 1) independently
        x = prior.rvs(size=30)  # returns a structured array of length 30
        print(x['sigma'])  # prints the 30 values for sigma

    We may also define a distribution using a chain rule decomposition.
    For this, we pass an ordered dict, since the order of components
    become relevant::

        chain_rule = OrderedDict()
        chain_rule['mu'] = Normal()
        chain_rule['tau'] = Cond(lambda x: Normal(loc=x['mu'])
        prior = StructDist(chain_rule)
        # means mu~N(0,1), tau|mu ~ N(mu,1)

    In the third line, ``Cond`` is a ``ProbDist`` class that represents
    a conditionalp distribution; it is initialized with a function that
    returns for each ``x`` a distribution that may depend on fields in ``x``.

    Parameters
    ----------
    laws: dict or ordered dict (as explained above)
        keys are parameter names, values are `ProbDist` objects

    """

    def __init__(self, laws):
        if isinstance(laws, OrderedDict):
            self.laws = laws
        elif isinstance(laws, dict):
            self.laws = OrderedDict([(key, laws[key]) for key in sorted(laws.keys())])
        else:
            raise TypeError(
                "recdist class requires a dict or an ordered dict to be instantiated"
            )
        self.dtype = []
        for key, law in self.laws.items():
            if law.dim == 1:
                typ = (key, law.dtype)  # avoid FutureWarning about (1,) fields
            else:
                typ = (key, law.dtype, law.dim)
            self.dtype.append(typ)

    def logpdf(self, theta):
        lp = 0.0
        for par, law in self.laws.items():
            cond_law = law(theta) if callable(law) else law
            lp += cond_law.logpdf(theta[par])
        return lp

    def rvs(self, size=1):  # Default for size is 1, not None
        out = np.empty(size, dtype=self.dtype)
        for par, law in self.laws.items():
            cond_law = law(out) if callable(law) else law
            out[par] = cond_law.rvs(size=size)
        return out


# (!) modification (!)
#########
# Other
#########

def F_innov(mean=0., sd=1.,
            df=None, tail=None, shape=None, skew=None,
            a=None, b=None,):
    '''
    Wrapper function for innovation distributions.

    Parameters
    ----------
    distr: string
        distribution class.
    mean: float
        expectation.
    sd: float
        standard deviation.
    a: float
        lower truncation level.
    b: float
        upper truncation level; b > a.
    **params: float
        any distribution parameters

    '''
    # Gaussian
    if df is None and tail is None:
        # if (a is None and b is None):
        return Normal(mean, sd)
        # else:
        #     return TruncNormal(mean, sd, a, b)

    # Student t
    elif df is not None:
        _, v = stats.t.stats(df=df)
        base_distr = Student(df, mean, sd/np.sqrt(v))
        if (a is None and b is None):
            return base_distr
        else:
            return TruncDist(base_distr, a=a, b=b)

    # Gen. Hyperbolic
    else:
        m, v = stats.genhyperbolic.stats(p=tail, a=shape, b=skew)
        base_distr = GenHyp(loc=-m/np.sqrt(v), scale=sd/np.sqrt(v), tail=tail,
                      shape=shape, skew=skew)
        if (a is None and b is None):
            return base_distr
        else:
            return TruncDist(base_distr, a=a, b=b)


def mix_cdf(x, probs, mus=0., sigmas=1., a=None, b=None, **theta_F):
    '''
    CDF of a mixture of the innovation distribution

    Parameters:
    -----------
    q: array
        quantiles to be returned.
    probs: array
        weights of mixture components.
    mus: array
        means of mixture components.
    sigmas: array
        standard deviations of mixture components.
    dfs: array
        degrees of freedoms of mixture components.

    '''

    # CDF of mixture distribution is weighted sum of individual CDFs
    return np.sum(probs * F_innov(mus, sigmas, **theta_F, a=a, b=b).cdf(x))


def abs_cdf(x, cdf):
    '''
    CDF of the absolute value of a random variable

    '''
    return cdf(x) - cdf(-x)


def sq_cdf(x, cdf):
    '''
    CDF of the square of a random variable from distribution 'dist'

    '''
    return cdf(np.sqrt(x)) - cdf(-np.sqrt(x))


def pos_cdf(x, cdf):
    '''
    CDF of positive part of a random variable, (X)_+ = max(X, 0)

    '''
    if x >= 0.:
        return cdf(x)
    else:
        return 0.


def abs_mix_cdf(x, probs, mus=0., sigmas=1., a=None, b=None, **theta_F):
    ''' CDF of the absolute value of a mixture distribution '''

    return abs_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F, a=a, b=b))


def sq_mix_cdf(x, probs, mus=0., sigmas=1., a=None, b=None, **theta_F):
    ''' CDF of the square of a mixture distribution '''

    return sq_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F, a=a, b=b))


def pos_mix_cdf(x, probs, mus=0., sigmas=1., a=None, b=None, **theta_F):
    ''' CDF of the positive part of a mixture distribution '''

    return pos_cdf(x, cdf=lambda x: mix_cdf(x, probs, mus, sigmas, **theta_F, a=a, b=b))


def inv_cdf(cdf, p, lo, hi):
    '''
    numeric approximation of the quantile function of the distribution 'dist'

    if cdf(hi) < p or cdf(lo) > p, simply returns 2·hi resp. 2·lo (assuming
    that lo < 0)

    Parameters:
    -----------
    cdf: cdf to be inverted
        distribution whose CDF is to be inverted.
    p: list or array
        quantile levels.

    '''

    q = np.full(len(p), np.nan)
    for i in range(len(p)):
        if cdf(lo) <= p[i] <= cdf(hi):
            q[i] = brentq(f=lambda x: cdf(x) - p[i], a=lo, b=hi, xtol=1e-6)
        elif cdf(hi) < p[i]:
            q[i] = 2.0 * hi
        elif cdf(lo) > p[i]:
            q[i] = 2.0 * lo

    return q