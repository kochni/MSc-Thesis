#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# some helpful functions for many things

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from particles.distributions import StructDist, OrderedDict, Normal, TruncNormal, Student, Cauchy, Beta, Gamma, shGamma, Dirichlet
from numpy.lib.recfunctions import structured_to_unstructured  # to make structured array useful again
from itertools import chain

from seaborn import kdeplot  # for kernel density plots


def cap(values, floor=-np.inf, ceil=np.inf):
    if isinstance(values, float) or values is None:
        return min(ceil, max(floor, values)) if values is not None else None
    else:
        values[values != None] = np.minimum(ceil, np.maximum(floor, values[values != None]))
        return values


def F_innov(mean=0., sd=1., df=None,
            shape=None, tail=None, skew=None):
    '''
    Wrapper function for innovation distributions, i.e. with mean 0 sd 1.
    Reverts to simplest distribution whenever possible for performance gains:
        - if no additional parameters specified (besides mean & sd), returns
          Gaussian
        - if df specified, returns Student t with df degrees of freedom
        - if shape, tail, skew specified, returns Generalized Hyperbolic

    '''
    if shape is not None:
        m, v = stats.genhyperbolic.stats(a=shape, p=tail, b=skew)
        return GenHyp(loc=-m/np.sqrt(v), scale=sd/np.sqrt(v), tail=tail,
                      shape=shape, skew=skew)
    elif df is not None:
        _, v = stats.t.stats(df=df)
        return Student(scale=sd/np.sqrt(v), df=df)
    else:
        return Normal(loc=mean, scale=sd)


# activation functions (for parameter transforms, reservoir computers, ...)

def Id(x):
    ''' R --> R '''
    return x


def sigmoid(x):
    ''' R --> [0, 1] '''
    x = cap(x, floor=-500)  # avoids overflow (returns 0 anyway)
    return 1/(1 + np.exp(-x))


def tanh(x):
    ''' R --> [-1, 1] '''
    return np.tanh(x)


def relu(x):
    ''' R --> R_+ '''
    return np.maximum(0, x)


# parameter transforms

def param_tf(theta):
    pass


# function to automatically set all priors given model specification

def set_prior(model):
    '''
    automatically sets all prior distributions given the model specification

    Parameters without constraints get a Gaussian prior with stdev = 10

    Parameters with positivity constraints (e.g., GARCH parameters) are later
        transformed with exp(); they get a Gaussian prior with stdev = 1,
        yielding a log-Normal with effective stdev ≈ 7.4;

    Parameters with [0,1] constraints (e.g., regime probabilities) are later
        transformed with the sigmoid; they get a Gaussian prior with stdev 1.5,
        which is approximately uniform after the transformation

    '''

    # s = stdev
    # s_pos = np.sqrt(s**2 / (1 - (2*stats.norm.pdf(0))**2))

    hyper = model.get('hyper')
    if hyper is not None: q = hyper.get('q')

    prior = OrderedDict()

    if model.get('regimes') in [None, 1]:

        if model['dynamics'] == 'constant':
            prior['sigma'] = Gamma(1., 0.5)

        elif model['dynamics'] == 'garch':
            if model['variant'] != 'elm':
                prior['omega'] = Gamma(1., 0.5)
                prior['alpha'] = Gamma(1., 0.5)
                prior['beta'] = Gamma(1., 0.5)
            else:
                for j in range(q+1):
                    prior['w' + str(j)] = Normal(loc=0., scale=3.)

            if model['variant'] in ['gjr', 'thr', 'exp']:
                prior['gamma'] = Normal(loc=0., scale=10.)

        elif model['dynamics'] == 'guyon':
            prior['omega'] = Gamma(1., 0.5)
            prior['alpha'] = Gamma(1., 0.5)
            prior['beta'] = Gamma(1., 0.5)
            prior['a1'] = shGamma(1., 0.5, loc=1.)
            prior['a2'] = shGamma(1., 0.5, loc=1.)
            prior['d1'] = Gamma(1., 0.5)
            prior['d2'] = Gamma(1., 0.5)

    else:  # same but for every regime
        if model['switching'] == 'mixing':
            if model['regimes'] == 2:
                prior['p_0'] = Beta(1., 1.)
            else:
                prior['p'] = Dirichlet(alpha=np.tile(1., model['regimes']))

        else:
            for k in range(model['regimes']):
                if model['regimes'] == 2:
                    prior['P_0'] = Beta(1., 1.)
                    prior['P_1'] = Beta(1., 1.)
                else:
                    prior['P_' + str(k)] = Dirichlet(alpha=np.tile(1., model['regimes']))

        for k in range(model['regimes']):
            if model['dynamics'] == 'constant':
               prior['sigma_' + str(k)] = Gamma(1., 0.5)

            elif model['dynamics'] == 'garch':
                if model['variant'] == 'elm':
                    for j in range(q+1):
                        prior['w' + str(j) + '_' + str(k)] = Normal(loc=0., scale=3.)
                elif model['variant'] == 'exp':  # no constraints
                    prior['omega_' + str(k)] = Normal(loc=0., scale=10.)
                    prior['alpha_' + str(k)] = Normal(loc=0., scale=10.)
                    prior['beta_' + str(k)] = Normal(loc=0., scale=10.)
                else:  # positivity constraints
                    prior['omega_' + str(k)] = Gamma(1., 0.5)
                    prior['alpha_' + str(k)] = Gamma(1., 0.5)
                    prior['beta_' + str(k)] = Gamma(1., 0.5)

                if model['variant'] in ['gjr', 'thr', 'exp']:
                    prior['gamma_' + str(k)] = Normal(loc=0., scale=10.)

            elif model['dynamics'] == 'guyon':
                prior['omega_' + str(k)] = Gamma(1., 0.5)
                prior['alpha_' + str(k)] = Gamma(1., 0.5)
                prior['beta_' + str(k)] = Gamma(1., 0.5)
                prior['a1_' + str(k)] = Gamma(1., 0.5)
                prior['a2_' + str(k)] = Gamma(1., 0.5)
                prior['d1_' + str(k)] = Gamma(1., 0.5)
                prior['d2_' + str(k)] = Gamma(1., 0.5)

            elif model['dynamics'] in ['elm', 't_sig', 'r_sig']:
                for j in range(model['hyper']['q']):
                    prior['w' + str(j) + '_' + str(k)] = Normal(scale=3.)

    if model['innov_X'] == 't':
        prior['df_X'] = shGamma(a=1., b=0.5, loc=2.)

    if model['jumps'] is not None:
        prior['lambda'] = Beta(1., 1.)
        prior['phi'] = Gamma(1., 0.5)

    prior = StructDist(prior)

    return prior


# kernel functions (for Guyon & Lekeufack's model)

def tspl(t, tp, alpha, delta):
    '''
    "time-shifted power law" kernel used by Guyon & Lekeufack (2023)
    '''
    tau = abs(t-tp)
    weights = (tau + delta)**(-alpha)

    # normalize so that axis 0 sums to 1
    weights = weights / weights.sum(axis=2)[:, :, np.newaxis]

    return weights


def out_fct(model):  # to collect only relevant things to save memory
    Z = np.array(model.summaries.logLts)
    theta = model.X.theta  # particles / posterior samples
    prior = model.fk.model.prior.laws
    W = model.W  # particle weights
    ESS = np.array(model.summaries.ESSs)
    rs_flags = np.array(model.summaries.rs_flags)
    MH_acc = np.array(model.X.shared['acc_rates']).flatten()  # MH acceptance rates

    dic = {
        'Z_t': Z,
        'theta_T': theta,
        'W_T': W,
        'prior': prior,
        'ESS_t': ESS,
        'rs_flags': rs_flags,
        'MH_Acc_t': MH_acc
        }

    return dic


def smc_summary(runs, plot_post=False, plot_Z=False,
                M0=None, dataset=None, diagnostics=False, save_plots=False):
    '''
    summarize and visualize the results of an analysis of models fitted using
    SMC

    Parameters:
    -----------
    runs: SMC object
        SMC object of type 'SMC' to which 'run()' has been applied

    plot_post: bool
        whether to plot the posterior parameter distributions.

    plot_Z: bool
        whether to plot the on-line evidence estimates.

    dataset: string
        name or description of the dataset used for model estimation.

    M0: string
        if plot_Z == True: base model against which log-evidence of remaining models is
        compared; must be contained in runs[j]['fk'] for some j.
    '''

    # preliminaries
    n_runs = runs[-1]['run'] + 1
    T = len(runs[0]['Z_t'])
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!
    x = list(range(T))

    # plot log Bayes Factors over time
    if plot_Z == True:

        alpha = 0.3
        plt.figure(dpi=800)

        Z = np.zeros([n_models, n_runs, T])

        # compute mean & variance of evidence over runs
        for m in range(n_models):
            Z[m, :, :] = 2.*np.array([runs[j]['Z_t'] for j in range(len(runs)) if runs[j]['fk']==model_names[m]])
            Z_mean = np.mean(Z, axis=1)
            Z_var = np.var(Z, axis=1)

        # binary model comparison
        for m in range(n_models):
            score = Z_mean - Z_mean[model_names.index(M0), :]  # mean excess log evidence over that of base model
            score_sd = np.sqrt(Z_var/n_runs + Z_var[model_names.index(M0), :]/n_runs)  # SE of difference of means

            if m != model_names.index(M0):
                plt.plot(score[m, :], label=model_names[m])
                plt.fill_between(x, score[m, :]+2*score_sd[m, :], score[m, :]-2*score_sd[m, :], alpha=alpha)

        plt.axhline(y=0, c='black', ls='--', lw=0.5)
        # plt.axhline(y=2, c='green', ls='--', lw=0.25)
        # plt.axhline(y=-2, c='red', ls='--', lw=0.25)
        # plt.axhline(y=6, c='green', ls='--', lw=0.25)
        # plt.axhline(y=-6, c='red', ls='--', lw=0.25)
        plt.axhline(y=10, c='green', ls='--', lw=0.25)
        plt.axhline(y=-10, c='red', ls='--', lw=0.25)
        plt.xlabel("t")
        plt.ylabel("2·Excess log-evidence")
        plt.suptitle(dataset)
        plt.title("M0: " + M0, {'fontsize': 10})
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                   ncol=3, fancybox=True)
        if save_plots == True: plt.savefig('Plots/Evidence.png')
        # plt.close()

    # plot prior vs posterior parameter distributions
    if plot_post == True:

        for m in range(n_models):
            # take the particles & weights of each model's 1st run
            theta = runs[m]['theta_T']
            weights = runs[m]['W_T']

            priors = runs[m]['prior']

            pars_names = theta.dtype.names
            n_pars = len(pars_names)
            theta = structured_to_unstructured(theta)

            # if switching model w >=3 regimes, regime probabilities are stored
            # in multidimensional variable 'p'
            K = theta.shape[1] - n_pars + 2
            if K > 2:
                # rename multivariate parameter 'p' to 'p_0', 'p_1', ...
                new_names = ()
                for j in range(theta.shape[1]-n_pars+1):
                    new_names += ('p_' + str(j),)
                ind = pars_names.index('p')
                pars_names = pars_names[:ind] + new_names + pars_names[ind+1:]
                n_pars = len(pars_names)
                # remove Dirichlet prior with all marginals
                # (marginals of Dirichlet are Beta)
                del runs[m]['prior']['p']
                for k in range(K-1):
                    runs[m]['prior']['p_' + str(k)] = Beta(a=1, b=K-1)

            n_pars = min(12, n_pars)  # plot at most the first 9 parameters
            # create grid which is as square as possible, but always add new
            # rows first
            n_row = int(np.ceil(np.sqrt(n_pars)))
            n_col = int(np.ceil(n_pars/n_row))
            fig, axs = plt.subplots(n_row, n_col, dpi=800, figsize=(10, 8))
            fig.suptitle(model_names[m], fontsize=16)

            if n_pars == 1: axs = np.array([[axs]])  # else silly error
            if n_col*n_row != n_pars:
                for j in range(1, n_col*n_row-n_pars+1):
                    fig.delaxes(axs[-1, -j])

            for k, ax in enumerate(axs.flat):
                if k >= n_pars: break

                # draw kernel density of posterior parameter samples
                kdeplot(x=theta[:, k], weights=weights, fill=True,
                        label='Posterior', ax=ax)

                # marks samples on x-axis
                y0, y1 = ax.get_ylim()
                h = 0.04*(y1-y0)
                ax.vlines(theta[:, k], ymin=0, ymax=h)

                # draw prior density
                x0, x1 = ax.get_xlim()
                x_pdf = np.linspace(x0, x1)
                ax.plot(x_pdf, priors[pars_names[k]].pdf(x_pdf),
                        ls='dashed', label='Prior')
                ax.set_title(pars_names[k])
                ax.legend()

                # Remove axis labels
                ax.set_xlabel('')
                ax.set_ylabel('')

            fig.supylabel('Density (weighted)')
            plt.tight_layout()  # Add space between subplots
            plt.show()
            if save_plots == True: plt.savefig('Plots/Posterior_' + model_names[m] + '.png')
        # plt.close()

    #  SMC diagnostic checks
    if diagnostics == True:

        n_row = int(np.ceil(np.sqrt(n_models)))
        n_col = int(np.ceil(n_models/n_row))

        # plot ESS over time
        plt.figure(dpi=800)
        fig, axs = plt.subplots(n_row, n_col, figsize=(9, 5), dpi=800,
                                layout='constrained', sharex=True, sharey=True)
        if n_col*n_row != n_models:
            for j in range(1, n_col*n_row-n_models+1):
                fig.delaxes(axs[-1,-j])
        if n_models == 1: axs = np.array([[axs]])  # else silly error

        for m, ax in enumerate(axs.flat):
            if m >= n_models: break

            # mean time between resampling
            rs_flags = np.array(
                [runs[j]['rs_flags'] for j in range(len(runs)) if runs[j]['fk']==model_names[m]]
                )  # use all runs to compute mean time between resampling
            rs = np.where(rs_flags)[1]
            gaps = np.diff(rs)
            gaps = gaps[gaps>0]
            mean_gap = gaps.mean()
            mean_gap = round(mean_gap, 1)

            # effective sample size
            ESS = runs[m]['ESS_t']  # plot ESSs of 1st run of each model

            line, = ax.plot(x, ESS, lw=1)
            ax.set_title(model_names[m], fontsize='small', loc='left')
            ax.set_title("Δt = " + str(mean_gap), fontsize='small', loc='right')

        fig.suptitle('Effective Sample Size', fontsize=12)
        fig.supxlabel('t')
        fig.supylabel('ESS')
        plt.show()
        if save_plots == True: plt.savefig('Plots/ESS.png')
        plt.close()

        # plot MH acceptance rates over time
        plt.figure(dpi=800)
        fig, axs = plt.subplots(n_row, n_col, figsize=(9, 5), dpi=800,
                                layout='constrained', sharey=True)
        if n_col*n_row != n_models:
            for j in range(1, n_col*n_row-n_models+1):
                fig.delaxes(axs[-1,-j])
        if n_models == 1: axs = np.array([[axs]])  # else silly error

        for m, ax in enumerate(axs.flat):
            if m >= n_models: break

            acc_rates = runs[m]['MH_Acc_t']
            ax.plot(acc_rates)
            ax.axhline(y=0.2, c='black', ls='--', lw=0.5)
            ax.axhline(y=0.4, c='black', ls='--', lw=0.5)
            ax.set_title(model_names[m], fontsize='small', loc='left')

        fig.suptitle('MH Acceptance Rates', fontsize=12)
        fig.supxlabel('Step')
        fig.supylabel('Acceptance Rate')
        plt.show()
        if save_plots == True: plt.savefig('Plots/MH_Acceptance.png')
        # plt.close()


def predict(model, theta_hat, W, truth, train_data,
            s, M, lt_vol=None):

    T = len(data)
    vol_pred, reg_vols = model.predict(theta_hat, W, s=100, M=5)

    # plot over time
    plt.figure(dpi=800)
    plt.plot(truth, lw=0.8)
    # plt.plot(reg_vols[:, 800:].T, ls="--", lw=0.75)
    plt.plot(vol_pred, color="red", lw=0.8)
    plt.ylabel("Volatility")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.vlines(x=T, color="red", lw=0.5, ls="--",
               label="End of train sample",
               ymin=ymin, ymax=ymax)
    plt.hlines(y=data.std(), color='green', lw=0.5, ls="--",
               label='Train SD',
               xmin=xmin, xmax=xmax)
    if lt_vol is not None:
        plt.hlines(y=lt_vol, color='blue', lw=0.5, ls="--",
                   label='LT Vol',
                   xmin=xmin, xmax=xmax)
    plt.legend()

    # pred vs. truth scatter plot
    plt.figure(dpi=800)
    plt.scatter(x=truth[0:T], y=vol_pred[0:T], label="in-sample", marker='x')
    plt.scatter(x=truth[T:], y=vol_pred[T:], label="out-sample",
                c=np.arange(s))
    plt.colorbar(label="s-days ahead prediction")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([xmin, xmax], [ymin, ymax], ls="--", lw=0.8, color='grey')
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.legend()
