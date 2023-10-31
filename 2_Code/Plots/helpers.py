#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# some helpful functions for many things

import numpy as np
import matplotlib.pyplot as plt

from numpy.lib.recfunctions import structured_to_unstructured  # to make structured array useful again
from itertools import chain

from seaborn import kdeplot  # for kernel density plots


# activation functions (for neural models and randomized signatures)

def Id(x):
    return x


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# kernel functions (for Guyon & Lekeufack's model)
# should have the following properties:
# 1) k(t1, t2) decreasing in t2
# 2) k(t1, t2) -> 0 for t2 -> inf
# 3) sum_{t2 = 1}^t1 k(t1, t2) = 1

def tspl(t, tp, alpha, delta):
    '''
    "time-shifted power law" kernel used by Guyon & Lekeufack (2023)
    '''
    tau = abs(t-tp)
    tau = np.tile(tau, (len(delta), 1))

    alpha = alpha.reshape(-1, 1)
    delta = delta.reshape(-1, 1)

    weights = (tau + delta)**(-alpha)
    weights = weights / weights.sum(axis=1)[:, np.newaxis]  # normalize so that rows sum to 1

    return weights


def cap_and_log(values, warn_cat, t, ceil=np.inf, floor=-np.inf):
    if (values > ceil).any() | (values < floor).any():

        # update overflow logs fle
        with open("Overflow_Logs.txt", "a") as file:
            n_violations = (values > ceil).sum() + (values < floor).sum()
            n_possible = len(values)
            warn_msg = warn_cat + ": " + str(n_violations) + "/" + str(n_possible) + " over-/underflows in volatility at t=" + str(t) + "\n"
            file.write(warn_msg)

        # return capped value
        values = np.minimum(ceil, np.maximum(floor, values))

    return values


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
        'MH_Acc_t': MH_acc,}

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
            Z[m, :, :] = np.array([runs[j]['Z_t'] for j in range(len(runs)) if runs[j]['fk']==model_names[m]])
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
        plt.xlabel("t")
        plt.ylabel("Excess log-evidence")
        plt.suptitle(dataset)
        plt.title("M0: " + M0, {'fontsize': 10})
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                   ncol=3, fancybox=True)
        if save_plots == True: plt.savefig('Plots/Evidence.png')
        plt.close()

    # plot prior vs posterior parameter distributions
    if plot_post == True:

        for m in range(n_models):
            theta = runs[m]['theta_T']  # take the particles of each model's 1st run
            weights = runs[m]['W_T']
            pars_names = theta.dtype.names
            n_pars = len(pars_names)
            theta = structured_to_unstructured(theta)
            priors = runs[m]['prior']

            n_pars = min(9, n_pars)  # plot at most the first 9 parameters
            n_row = int(np.ceil(np.sqrt(n_pars)))
            n_col = int(np.ceil(n_pars/n_row))

            fig, axs = plt.subplots(n_row, n_col, dpi=800, figsize=(10, 8))
            fig.suptitle(model_names[m], fontsize=16)

            if n_pars == 1: axs = np.array([[axs]])
            if n_col*n_row != n_pars:
                for j in range(1, n_col*n_row-n_pars+1):
                    fig.delaxes(axs[-1,-j])

            for k, ax in enumerate(axs.flat):
                if k >= n_pars: break

                # draw kernel density of posterior parameter samples
                kdeplot(x=theta[:, k], weights=weights, fill=True, label='Posterior', ax=ax)

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
        plt.close()

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
        plt.close()

