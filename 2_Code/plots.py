#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from seaborn import kdeplot  # kernel density plots

from particles.distributions import Beta
from numpy.lib.recfunctions import structured_to_unstructured


def smc_summary(runs, M0=None, dataset=None, save_plots=False):
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
    T = len(runs[0]['Z'])
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    #############
    # Evidences #
    #############
    plt.figure(dpi=800)
    plt.suptitle(dataset)
    plt.title("M0: " + M0, {'fontsize': 10})
    plt.axhline(y=0, c='black', ls='--', lw=0.5)
    # plt.axhline(y=2, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-2, c='red', ls='--', lw=0.25)
    # plt.axhline(y=6, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-6, c='red', ls='--', lw=0.25)
    plt.axhline(y=10, c='green', ls='--', lw=0.25)
    plt.axhline(y=-10, c='red', ls='--', lw=0.25)
    plt.xlabel("t")
    plt.ylabel("2·Excess log-evidence")

    # collect evidences of models over time for all runs
    Z = np.zeros([n_models, n_runs, T])
    for m in range(n_models):
        Z[m, :, :] = 2.0 * np.array([runs[j]['Z'] for j in range(len(runs)) if runs[j]['fk']==model_names[m]])

    # binary model comparison
    scores = (Z - Z[model_names.index(M0), :, :]).mean(axis=1)  # excess log evidence over that of base model
    scores_sd = (Z - Z[model_names.index(M0), :, :]).std(axis=1)

    # plot
    for m in [j for j in range(n_models) if runs[j]['fk'] != M0]:
        plt.plot(scores[m, :], label=model_names[m])
        plt.fill_between(np.arange(0, T, 1), scores[m, :]+2*scores_sd[m, :],
                         scores[m, :]-2*scores_sd[m, :], alpha=0.25)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)
    if save_plots is True:
        plt.savefig('Plots/Evidence.png')
    # plt.close()

    ##############
    # Posteriors #
    ##############
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

    ####################
    # Algo Diagnostics #
    ####################
    n_row = int(np.ceil(np.sqrt(n_models)))
    n_col = int(np.ceil(n_models/n_row))

    # plot ESS over time
    plt.figure(dpi=1000)
    fig, axs = plt.subplots(n_row, n_col, figsize=(9, 5), dpi=1000,
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
        ESS = runs[m]['ESS']  # plot ESSs of 1st run of each model

        line, = ax.plot(np.arange(0, T, 1), ESS, lw=1)
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
            fig.delaxes(axs[-1, -j])
    if n_models == 1: axs = np.array([[axs]])  # else silly error

    for m, ax in enumerate(axs.flat):
        if m >= n_models: break

        acc_rates = runs[m]['MH_Acc']
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


# Loss functions for assessment of prediction accuracy

def sq_err(pred, truth):
    return (pred-truth)**2


def perc_err(pred, truth):
    return abs(pred - truth) / (truth + 1)


def mov_avg(x, w):
    '''
    moving average of the last 'w' values
    '''
    if x.ndim == 1:
        ret = np.cumsum(x, dtype=float)
        ret[w:] = ret[w:] - ret[:-w]
        return ret[w-1:] / w
    elif x.ndim == 2:
        ret = np.cumsum(x, axis=1, dtype=float)
        ret[:, w:] = ret[:, w:] - ret[:, :-w]
        return ret[:, w-1:] / w


def run_avg(x):
    '''
    running average (moving average with window size t)
    '''
    if x.ndim == 1:
        return np.cumsum(x) / np.arange(1, len(x)+1)
    elif x.ndim == 2:
        return np.cumsum(x, axis=1) / np.arange(1, x.shape[1]+1).reshape(1, -1)


def plot_pred_errs(runs, truths, avg_method):
    n_runs = runs[-1]['run'] + 1
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]
    # first 50 time steps not displayed
    T = len(runs[0]['Z']) - 50
    w = int(np.ceil(0.05*T))

    for Y in runs[0]['Preds'].keys():
        plt.figure(dpi=1000)
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                                layout='constrained', sharex=True)
        fig.suptitle("Prediction Error: " + Y)
        fig.supxlabel("t")
        fig.supylabel("L(M)")
        for m in [j for j in range(n_models)]:
            ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
            preds = np.array([runs[j]['Preds'][Y][50:] for j in ind])
            sq_errs = sq_err(preds, truths[Y][50:])
            pc_errs = perc_err(preds, truths[Y][50:])
            if avg_method == 'moving':
                mean_sq_err = np.mean(mov_avg(sq_errs, w), axis=0)
                mean_pc_err = np.mean(mov_avg(pc_errs, w), axis=0)
                std_sq_err = np.std(mov_avg(sq_errs, w), axis=0)
                std_pc_err = np.std(mov_avg(pc_errs, w), axis=0)
            elif avg_method == 'full':
                w = 1
                mean_sq_err = np.mean(run_avg(sq_errs), axis=0)
                mean_pc_err = np.mean(run_avg(pc_errs), axis=0)
                std_sq_err = np.std(run_avg(sq_errs), axis=0)
                std_pc_err = np.std(run_avg(pc_errs), axis=0)

            axs[0].set_title("Squared Error", fontsize='small', loc='left')
            axs[0].plot(mean_sq_err, label=model_names[m])
            axs[0].set_yscale('log')
            axs[0].fill_between(np.arange(0, T-w+1),
                                mean_sq_err+2.0*std_sq_err,
                                mean_sq_err-2.0*std_sq_err,
                                alpha=0.25)
            axs[1].set_title("Percentage Error", fontsize='small', loc='left')
            axs[1].plot(mean_pc_err)
            axs[1].set_yscale('log')
            axs[1].fill_between(np.arange(0, T-w+1),
                                mean_pc_err+2.0*std_pc_err,
                                mean_pc_err-2.0*std_pc_err,
                                alpha=0.25)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
                   ncol=3, fancybox=True)


def plot_coverages(runs, truths, avg_method):
    n_runs = runs[-1]['run'] + 1
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]
    T = len(runs[0]['Z'])
    w = int(np.ceil(0.05*T))

    for Y in runs[0]['PredSets'].keys():
        plt.figure(dpi=1000)
        fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                                layout='constrained', sharex=True)
        fig.suptitle("Prediction Sets: " + Y, fontsize=12)
        fig.supxlabel("t")
        axs[0].set_title("Coverage", fontsize='small', loc='left')
        axs[0].axhline(y=1.0-alpha, color='black', ls='--', lw=0.8)
        if avg_method == 'moving':
            axs[0].set_title("w=" + str(w), fontsize='small', loc='right')
            conf_bound_hi = (stats.binom(n=w*n_runs, p=1.0-alpha
                                         ).ppf(q=0.975)
                             / w / n_runs)
            conf_bound_lo = (stats.binom(n=w*n_runs, p=1.0-alpha
                                         ).ppf(q=0.025)
                             / w / n_runs)
            # axs[0].axhline(y=conf_bound_hi, ls='--', lw=0.8, color='red')
            # axs[0].axhline(y=conf_bound_lo, ls='--', lw=0.8, color='red')
        else:
            w = 1
            conf_bound_hi = (stats.binom(n=np.arange(1, T)*n_runs, p=1.0-alpha
                                         ).ppf(q=0.975)
                             / np.arange(1, T) / n_runs)
            conf_bound_lo = (stats.binom(n=np.arange(1, T)*n_runs, p=1.0-alpha
                                         ).ppf(q=0.025)
                             / np.arange(1, T) / n_runs)
            # axs[0].plot(conf_bound_hi, ls='--', lw=0.8, color='red')
            # axs[0].plot(conf_bound_lo, ls='--', lw=0.8, color='red')

        axs[1].set_title("Width", fontsize='small', loc='left')

        for m in range(n_models):
            ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
            predsets = np.array([runs[j]['PredSets'][Y] for j in ind])
            cover = (np.greater_equal(truths[Y], predsets[:, :, 0]) *
                     np.less_equal(truths[Y], predsets[:, :, 1]))
            width = np.diff(predsets, axis=2)[:, :, 0]
            if avg_method == 'moving':
                mean_cover = mov_avg(cover, w).mean(axis=0)
                sd_cover = mov_avg(cover, w).std(axis=0)
                mean_width = mov_avg(width, w).mean(axis=0)
                sd_width = mov_avg(width, w).std(axis=0)
            else:
                mean_cover = run_avg(cover).mean(axis=0)
                sd_cover = run_avg(cover).std(axis=0)
                mean_width = run_avg(width).mean(axis=0)
                sd_width = run_avg(width).std(axis=0)

            axs[0].plot(mean_cover, label=model_names[m])
            axs[0].fill_between(np.arange(0, T-w+1, 1),
                                mean_cover+2.0*sd_cover,
                                mean_cover-2.0*sd_cover,
                                alpha=0.25)
            axs[1].plot(mean_width)
            axs[1].set_yscale('log')
            axs[1].fill_between(np.arange(0, T-w+1, 1),
                                mean_width+2.0*sd_width,
                                mean_width-2.0*sd_width,
                                alpha=0.25)

        # add red area in empirical coverage
        y_min, y_max = axs[0].get_ylim()
        axs[0].fill_between(np.arange(0, T-w+1, 1),
                            conf_bound_hi, y_max,
                            color='red', lw=0.0, alpha=0.1)
        axs[0].fill_between(np.arange(0, T-w+1, 1),
                            conf_bound_lo, y_min,
                            color='red', lw=0.0, alpha=0.1)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
                   ncol=3, fancybox=True)


def pred_summary(runs, M0, truths):
    '''
    plots summarizing prediction errors, prediction sets, and
    coverages for quantities of interest

    '''
    n_runs = runs[-1]['run'] + 1
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]
    T = len(runs[0]['Z'])
    h = pred_opts['h']
    M = pred_opts['M']
    alpha = pred_opts['alpha']
    strike = S[T] if pred_opts['strike'] == 'last' else pred_opts.get('strike')

    ####################
    # Prediction Error #
    ####################
    # (1) Realized Variance


    # (3) Option Payouts
    plt.figure(dpi=1000)
    fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
                            layout='constrained', sharex=True)
    fig.suptitle("Prediction Error: Option Payout")
    fig.supxlabel("t")
    fig.supylabel("-log L(M)/L(M0)")
    axs[0].set_title("M0: " + M0, fontsize='small')
    axs[0].axhline(y=0, c='black', ls='--', lw=0.5)
    axs[1].axhline(y=0, c='black', ls='--', lw=0.5)
    # error of reference model
    ind_M0 = [j for j in range(len(runs)) if runs[j]['fk'] == M0]
    preds_M0 = np.array([runs[j]['Preds']['C'] for j in ind_M0])
    sq_errs_M0 = sq_err(preds_M0, C_true)
    pc_errs_M0 = perc_err(preds_M0, C_true)
    for m in [j for j in range(n_models) if runs[j]['fk'] != M0]:
        ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
        preds = np.array([runs[j]['Preds']['C'] for j in ind])**2
        sq_errs = sq_err(preds, C_true)
        pc_errs = perc_err(preds, C_true)
        mean_sq_err = (-np.log(run_avg(sq_errs) / run_avg(sq_errs_M0))).mean(axis=0)
        mean_pc_err = (-np.log(run_avg(pc_errs) / run_avg(pc_errs_M0))).mean(axis=0)
        std_sq_err = (-np.log(run_avg(sq_errs) / run_avg(sq_errs_M0))).std(axis=0)
        std_pc_err = (-np.log(run_avg(pc_errs) / run_avg(pc_errs_M0))).std(axis=0)
        axs[0].set_title("Squared Error", fontsize='small', loc='left')
        axs[0].plot(mean_sq_err, label=model_names[m])
        axs[0].fill_between(np.arange(0, T+1), mean_sq_err+2.0*std_sq_err,
                            mean_sq_err-2.0*std_sq_err, alpha=0.25)
        axs[1].set_title("Percentage Error", fontsize='small', loc='left')
        axs[1].plot(mean_pc_err, label=model_names[m])
        axs[1].fill_between(np.arange(0, T+1), mean_pc_err+2.0*std_pc_err,
                            mean_pc_err-2.0*std_pc_err, alpha=0.25)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

    ###################
    # Prediction Sets #
    ###################

    # mean coverages over all past observations
    plot_coverages(runs, truths, 'full', w=30)

    # mean coverages over moving window
    plot_coverages(runs, truths, 'moving')

    # mean coverages over top 20% of outcomes

    # Example prediction sets:
    # (1) Realized Variance
    plt.figure(dpi=1000)
    plt.suptitle("Prediction Set: Realized Variance")
    plt.title(model_names[0], fontsize='small')
    plt.xlabel("t")
    plt.plot(truths['RV'], color='black', label="Truth")
    plt.fill_between(np.arange(0, T, 1),
                     runs[0]['PredSets']['RV'][:, 0],
                     runs[0]['PredSets']['RV'][:, 1],
                     label=str(int((1-alpha)*100)) + "% prediction set",
                     color='blue', alpha=0.25, lw=0)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=2, fancybox=True)

    # (2) Prices
    plt.figure(dpi=1000)
    plt.suptitle("Prediction Set: Price")
    plt.title(model_names[0], fontsize='small')
    plt.xlabel("t")
    plt.plot(truths['S'], color='black', label="Truth")
    plt.fill_between(np.arange(0, T, 1),
                     runs[0]['PredSets']['S'][:, 0],
                     runs[0]['PredSets']['S'][:, 1],
                     label=str(int((1-alpha)*100)) + "% prediction set",
                     color='blue', alpha=0.25, lw=0)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=2, fancybox=True)

    # (3) Option Payouts
    plt.figure(dpi=1000)
    plt.suptitle("Prediction Set: Option Payout")
    plt.title(model_names[0], fontsize='small')
    plt.xlabel("t")
    plt.plot(truths['C'], color='black', label="Truth")
    plt.fill_between(np.arange(0, T, 1),
                     runs[0]['PredSets']['C'][:, 0],
                     runs[0]['PredSets']['C'][:, 1],
                     label=str(int((1-alpha)*100)) + "% prediction set",
                     color='blue', alpha=0.25, lw=0)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
               ncol=2, fancybox=True)


    #################
    # Option Prices #
    #################
    # (6) Running prediction set for option price
    # plt.figure(dpi=1000)
    # plt.suptitle("Running Option Price Prediction Set")
    # mean payout by particle
    # payouts = (np.maximum(S_sim[:, T:, :], strike)).mean(axis=2)
    # payouts_q_lo = np.quantile(payouts, q=0.5*alpha, axis=0)
    # payouts_q_hi = np.quantile(payouts, q=1-0.5*alpha, axis=0)
    # plt.plot(np.arange(T, T+h, 1), S_mean.T, color='lime', alpha=2/N)
    # plt.fill_between(np.arange(T, T+h, 1), payouts_q_lo, payouts_q_hi,
    #                  color='blue', alpha=0.3, lw=0)
    # plt.plot(S, color='black', label="Realized payout")
    # plt.vlines(x=T, label='End of train sample',
    #            color='grey', lw=0.7, ls='--', ymin=ymin, ymax=ymax)
    # if strike > 0.:  # plot strike price
    #     plt.hlines(y=strike, color='grey', lw=0.7, ls='--',
    #                xmin=T, xmax=T+h)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))

    # (7) histogram of option prices (weighted)
    # plt.figure(dpi=1000)
    # plt.title("Distribution of Option Prices")
    # payouts = np.maximum(S_sim[:, -1, :] - strike, 0.)  # (N,M)
    # Cs = payouts.mean(axis=1)
    # sns.histplot(x=Cs, weights=W, bins=30, stat='density', kde=True)
    # plt.xlabel("C")
    # plt.ylabel("Density")
