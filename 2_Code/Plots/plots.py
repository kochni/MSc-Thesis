#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from particles.distributions import Beta
from rescomp_dv import ResCompDV
from helpers import shi
from numpy.lib.recfunctions import structured_to_unstructured
from tabulate import tabulate


def run_info(runs):
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    T = len(runs[0]['Z'])
    N = len(runs[0]['theta'])

    print("Runs:", n_runs)
    print("Length:", T)
    print("θ-particles:", N)


def truth(runs):
    truths = runs[-1]
    S = truths['S']
    X = 100.0 * np.log(S[1:]/S[:-1])

    # Plot log-returns
    plt.figure(dpi=1000)
    plt.suptitle("Returns")
    plt.xlabel("t")
    plt.plot(X)

    # Plot price process
    plt.figure(dpi=1000)
    plt.suptitle("Prices")
    plt.xlabel("t")
    plt.plot(S)


def cpu_times(runs):
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]
    table = [[np.nan]*3]*(n_models+1)
    table[0] = ['Model:', 'Mean (min):', 'Min:', 'Max:']
    for m in range(n_models):
        ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
        model = model_names[m]
        time_mean = np.round(np.mean([runs[j]['cpu_time']/60 for j in ind]), 2)
        time_min = np.round(np.min([runs[j]['cpu_time']/60 for j in ind]), 2)
        time_max = np.round(np.max([runs[j]['cpu_time']/60 for j in ind]), 2)
        table[m+1] = [model, time_mean, time_min, time_max]

    print("\nCPU Times")
    print(tabulate(table))


def posteriors(runs, models):
    # preliminaries
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    T = len(runs[0]['Z'])
    if models != 'all':
        # filter runs of selected models
        runs = [runs[j] for j in np.where([runs[j]['fk'] in models for j in range(len(runs))])[0]]
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    for m in range(len(runs)):
        # take the particles & weights of each model's 1st run
        theta = runs[m]['theta']
        weights = runs[m]['W']
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

        n_row = int(np.ceil(n_pars/4))
        n_col = int(np.ceil(n_pars/n_row))
        fig, axs = plt.subplots(n_row, n_col, dpi=1000, figsize=(16, 5))
        # fig.suptitle(model_names[m], fontsize=16)
        fig.supylabel('Density (weighted)', fontsize=12)

        if n_pars == 1: axs = np.array([[axs]])  # else silly error
        if n_col*n_row != n_pars:
            for j in range(1, n_col*n_row-n_pars+1):
                fig.delaxes(axs[-1, -j])

        for k, ax in enumerate(axs.flat):
            if k >= n_pars: break

            # draw kernel density of posterior parameter samples
            sns.kdeplot(x=theta[:, k], weights=weights, fill=True,
                        label='Posterior' if k==0 else '',
                        ax=ax)

            # marks samples on x-axis
            y0, y1 = ax.get_ylim()
            h = 0.04*(y1-y0)
            ax.vlines(theta[:, k], ymin=0, ymax=h)

            # draw prior density
            x0, x1 = ax.get_xlim()
            x_pdf = np.linspace(x0, x1)
            ax.plot(x_pdf, priors[pars_names[k]].pdf(x_pdf),
                    ls='dashed',
                    label='Prior' if k==0 else '')
            ax.set_title(pars_names[k])

            # Remove axis labels
            ax.set_xlabel('')
            ax.set_ylabel('')

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
                   ncol=2, fancybox=True)
        plt.tight_layout()  # Add space between subplots
        plt.show()


def diagnostics(runs, models):
    # preliminaries
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    T = len(runs[0]['Z'])
    N = len(runs[0]['theta'])
    if models != 'all':
        # filter runs of selected models
        runs = [runs[j] for j in np.where([runs[j]['fk'] in models for j in range(len(runs))])[0]]
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    #######
    # ESS #
    #######
    n_row = int(np.ceil(np.sqrt(n_models))) + 1
    n_col = int(np.ceil(n_models/n_row))

    plt.figure(dpi=1000)
    fig, axs = plt.subplots(n_row, n_col, figsize=(9, 7), dpi=1000,
                            layout='constrained', sharex=False, sharey=True)
    fig.supxlabel('t', fontsize=14)
    fig.supylabel('ESS', fontsize=14)

    if n_col*n_row != n_models:
        for j in range(1, n_col*n_row-n_models+1):
            fig.delaxes(axs[-1,-j])
    if n_models == 1:
        axs = np.array([[axs]])  # else silly error

    for m, ax in enumerate(axs.flat):
        if m >= n_models: break

        T = runs[m]['Z'].shape[0]
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

        line, = ax.plot(np.arange(T), ESS, lw=1)
        ax.set_title(model_names[m], fontsize=12, loc='left')
        ax.set_title("Δt = " + str(mean_gap), fontsize=12, loc='right')
        ax.set_yticks(np.linspace(0, N, 3))
        ax.set_xticks([])

    plt.show()

    #################
    # MH Acceptance #
    #################
    plt.figure(dpi=1000)
    fig, axs = plt.subplots(n_row, n_col, figsize=(9, 7), dpi=800,
                            layout='constrained', sharey=True)
    fig.supxlabel('Step', fontsize=14)
    fig.supylabel('MH Acceptance Rate', fontsize=14)

    if n_col*n_row != n_models:
        for j in range(1, n_col*n_row-n_models+1):
            fig.delaxes(axs[-1, -j])
    if n_models == 1: axs = np.array([[axs]])  # else silly error

    for m, ax in enumerate(axs.flat):
        if m >= n_models: break

        acc_rates = runs[m]['MH_Acc']
        ax.plot(acc_rates)
        ax.fill_between(np.arange(len(acc_rates)),
                        0.2, 0.4,
                        color='green', lw=0.0, alpha=0.1)
        ax.set_title(model_names[m], fontsize=12, loc='left')
        ax.set_xticks([])

    plt.show()


def selection_criteria(runs, models, M0, dataset_name):
    # preliminaries
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    T = len(runs[0]['Z'])
    if models != 'all':
        # filter runs of selected models
        runs = [runs[j] for j in np.where([runs[j]['fk'] in models for j in range(len(runs))])[0]]
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    #############
    # Evidences #
    #############
    plt.figure(dpi=1000, figsize=(12, 5))
    plt.title("$\mathcal{M}_0$: " + M0, loc='left', fontsize=12)
    plt.title("Dataset: " + dataset_name, loc='right', fontsize=12)
    plt.axhline(y=0, c='black', ls='--', lw=0.5)
    # plt.axhline(y=2, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-2, c='red', ls='--', lw=0.25)
    # plt.axhline(y=6, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-6, c='red', ls='--', lw=0.25)
    plt.axhline(y=5, c='green', ls='--', lw=0.7)
    plt.axhline(y=-5, c='red', ls='--', lw=0.7)
    plt.xlabel("t")
    plt.ylabel("Excess log-evidence", fontsize=12)

    # collect evidences of models over time for all runs
    Z = np.full([n_models, n_runs, T], np.nan)
    for m in range(n_models):
        model_ind = np.where(np.array([runs[j]['fk'] for j in range(len(runs))]) == model_names[m])[0]
        T = runs[model_ind[0]]['Z'].shape[0]
        Z[m, :, 0:T] = np.array([runs[j]['Z'] for j in model_ind])

    # binary model comparison
    scores = (Z - Z[model_names.index(M0), :, :]).mean(axis=1)  # excess log evidence over that of base model
    scores_min = (Z - Z[model_names.index(M0), :, :]).min(axis=1)
    scores_max = (Z - Z[model_names.index(M0), :, :]).max(axis=1)

    # plot
    for m in [j for j in range(n_models) if runs[j]['fk'] != M0]:
        plt.plot(scores[m, :], label=model_names[m])
        model_ind = np.where(np.array([runs[j]['fk'] for j in range(n_models)]) == model_names[m])[0][0]
        T = runs[model_ind]['Z'].shape[0]
        plt.fill_between(np.arange(T),
                         scores_min[m, 0:T],
                         scores_max[m, 0:T],
                         alpha=0.25)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=4)

    #######
    # DIC #
    #######
    # plt.figure(dpi=1000)
    # plt.suptitle("DIC", fontsize=12)
    # plt.title(dataset, loc='left')
    # plt.title("M0: " + M0, loc='right')
    # plt.axhline(y=0, c='black', ls='--', lw=0.5)
    # plt.axhline(y=2, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-2, c='red', ls='--', lw=0.25)
    # plt.axhline(y=6, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-6, c='red', ls='--', lw=0.25)
    # plt.axhline(y=10, c='green', ls='--', lw=0.25)
    # plt.axhline(y=-10, c='red', ls='--', lw=0.25)
    # plt.xlabel("t")
    # plt.ylabel("Excess DIC")

    # collect evidences of models over time for all runs
    # DIC = np.zeros([n_models, n_runs, T])
    # for m in range(n_models):
    #     DIC[m, :, :] = np.array([runs[j]['DIC'] for j in range(len(runs)) if runs[j]['fk']==model_names[m]])

    # binary model comparison
    # scores = (DIC - DIC[model_names.index(M0), :, :]).mean(axis=1)  # excess log evidence over that of base model
    # scores_sd = (DIC - DIC[model_names.index(M0), :, :]).std(axis=1)

    # plot
    # for m in [j for j in range(n_models) if runs[j]['fk'] != M0]:
        # plt.plot(scores[m, :], label=model_names[m])
        # plt.fill_between(np.arange(0, T, 1), scores[m, :]+2*scores_sd[m, :],
        #                  scores[m, :]-2*scores_sd[m, :], alpha=0.25)
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3)


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


def forecast_errors(runs, models, pred_opts, avg_method, M0, start, end):
    # preliminaries
    truths = runs[-1]
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    if models != 'all':
        # filter runs of selected models
        runs = [runs[j] for j in np.where([runs[j]['fk'] in models for j in range(len(runs))])[0]]
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    T_full = len(runs[0]['Z'])
    end = T_full if end is None else end
    alpha = pred_opts['alpha']
    w = int(np.ceil(0.05*T_full))

    # table of overall mean square errors
    table = [[np.nan]*3]*(n_models+1)
    table[0] = ['Model:', 'Mean:', 'SD:']

    for Y in runs[0]['Preds'].keys():
        truth = truths[Y][start:end]
        M0_ind = np.where(np.array([runs[j]['fk'] for j in range(len(runs))]) == M0)[0]
        preds = np.array([runs[j]['Preds'][Y] for j in M0_ind])
        preds = preds[:, start:end]
        sq_errs_M0 = sq_err(preds, truth)
        # pc_errs_M0 = perc_err(preds, truths[Y])

        plt.figure(dpi=1000, figsize=(9, 5))
        # fig, axs = plt.subplots(2, 1, figsize=(9, 5), dpi=1000,
        #                         layout='constrained', sharex=True)
        plt.suptitle("Prediction Error: " + Y, fontsize=16)
        plt.xlabel("$t$")
        plt.ylabel("$\mathcal{L}(\mathcal{M}_0) - \mathcal{L}(\mathcal{M})$", fontsize=12)
        plt.xticks(np.linspace(0, end-start-w, 5), np.linspace(start+w, end, 5, dtype=int))

        for m in range(n_models):
            T = len(runs[m]['Z'])
            model_ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
            preds = np.full([n_runs, T_full], np.nan)
            preds[:, 0:T] = np.array([runs[j]['Preds'][Y] for j in model_ind])
            preds = preds[:, start:end]
            sq_errs = sq_err(preds, truth)
            # pc_errs = perc_err(preds, truths[Y][:T])
            sq_err_diff = sq_errs_M0 - sq_errs
            # pc_err_diff = pc_errs_M0 - pc_errs

            if avg_method == 'moving':
                mean_sq_err_diff = np.mean(mov_avg(sq_err_diff, w), axis=0)
                # mean_pc_err_diff = np.mean(mov_avg(pc_err_diff, w), axis=0)
                std_sq_err_diff = np.std(mov_avg(sq_err_diff, w), axis=0)
                # std_pc_err_diff = np.std(mov_avg(pc_err_diff, w), axis=0)
            elif avg_method == 'full':
                w = 1
                mean_sq_err_diff = np.mean(run_avg(sq_err_diff), axis=0)
                # mean_pc_err_diff = np.mean(run_avg(pc_err_diff), axis=0)
                std_sq_err_diff = np.std(run_avg(sq_err_diff), axis=0)
                # std_pc_err_diff = np.std(run_avg(pc_err_diff), axis=0)

            plt.title("Squared Error", fontsize='small', loc='left')
            plt.plot(mean_sq_err_diff, label=model_names[m])
            # axs[0].set_yscale('log')
            plt.fill_between(np.arange(end-start-w+1),
                                mean_sq_err_diff + 2.0*std_sq_err_diff,
                                mean_sq_err_diff - 2.0*std_sq_err_diff,
                                alpha=0.25)
            # axs[1].set_title("Percentage Error", fontsize='small', loc='left')
            # axs[1].plot(mean_pc_err_diff)
            # axs[1].set_yscale('log')
            # axs[1].fill_between(np.arange(T-burnin-w+1),
            #                     mean_pc_err_diff + 2.0*std_pc_err_diff,
            #                     mean_pc_err_diff - 2.0*std_pc_err_diff,
            #                     alpha=0.25)

            rmse = np.sqrt(sq_errs.mean(axis=1))
            rmse_mean = np.round(rmse.mean(), 5)
            rmse_sd = np.round(rmse.std(), 5)
            table[m+1] = [model_names[m], rmse_mean, rmse_sd]

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
                   ncol=3, fancybox=True)

        print("\nRMSE (t=" + str(start) + " to t=" + str(end) +"): " + Y)
        print(tabulate(table))


def coverages(runs, models, pred_opts, avg_method, start, end):
    # preliminaries
    truths = runs[-1]
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    if models != 'all':
        # filter runs of selected models
        runs = [runs[j] for j in np.where([runs[j]['fk'] in models for j in range(len(runs))])[0]]
    n_models = int(len(runs)/n_runs)
    model_names = [runs[j]['fk'] for j in range(n_models)]  # need to preserve ordering!

    T_full = len(runs[0]['Z'])
    end = T_full if end is None else end
    alpha = pred_opts['alpha']
    w = int(np.ceil(0.05*(end-start)))

    # table of overall mean square errors
    table = [[np.nan]*3]*(n_models+1)
    table[0] = ['Model:', 'Mean:', 'SD:']

    for Y in truths.keys():
        truth = truths[Y][start:end]

        # plt.figure(dpi=1000)
        # fig, axs = plt.subplots(2, 2, figsize=(12, 5), dpi=1000,
        #                         layout='constrained', sharex=True, sharey=False)
        # axs[0, 0].set_title("Naive", fontsize=12, loc='left')
        # axs[0, 1].set_title("Calibrated", fontsize=12, loc='left')
        # axs[0, 0].set_ylabel("Coverage", fontsize=12)
        # axs[1, 0].set_ylabel("Width", fontsize=12)
        # axs[1, 0].set_xlabel("t", fontsize=12)
        # axs[1, 1].set_xlabel("t", fontsize=12)
        # axs[0, 0].axhline(y=1.0-alpha, color='black', ls='--', lw=0.8)
        # axs[0, 1].axhline(y=1.0-alpha, color='black', ls='--', lw=0.8)
        # axs[0, 0].set_xticks(np.linspace(0, end-start-w, 5), np.linspace(start+w, end, 5, dtype=int))
        # axs[0, 1].set_xticks(np.linspace(0, end-start-w, 5), np.linspace(start+w, end, 5, dtype=int))
        # axs[0, 1].set_yticks([])
        # axs[1, 1].set_yticks([])

        # if avg_method == 'moving':
        #     axs[0, 1].set_title("w=" + str(w), fontsize=8, loc='right')
        #     conf_bound_hi = (stats.binom(n=w*n_runs, p=1.0-alpha
        #                                 ).ppf(q=0.975)
        #                      / w / n_runs)
        #     conf_bound_lo = (stats.binom(n=w*n_runs, p=1.0-alpha
        #                                  ).ppf(q=0.025)
        #                      / w / n_runs)
            # axs[0].axhline(y=conf_bound_hi, ls='--', lw=0.8, color='red')
            # axs[0].axhline(y=conf_bound_lo, ls='--', lw=0.8, color='red')
        # else:
        #     w = 1
        #     conf_bound_hi = (stats.binom(n=np.arange(T)*n_runs, p=1.0-alpha
        #                                  ).ppf(q=0.975)
        #                      / np.arange(1, T) / n_runs)
        #     conf_bound_lo = (stats.binom(n=np.arange(T)*n_runs, p=1.0-alpha
        #                                 ).ppf(q=0.025)
        #                      / np.arange(1, T) / n_runs)
            # axs[0].plot(conf_bound_hi, ls='--', lw=0.8, color='red')
            # axs[0].plot(conf_bound_lo, ls='--', lw=0.8, color='red')

        # y0_min = 1.0
        # y0_max = 0.0
        # y1_min = 1e10
        # y1_max = 0.0
        for m in range(n_models):
            T = len(runs[m]['Z'])
            model_ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
            predsets_naive = np.full([n_runs, T_full, 2], np.nan)
            predsets_calib = np.full([n_runs, T_full, 2], np.nan)
            predsets_naive[:, 0:T, :] = np.array([runs[j]['PredSets'][Y][:, 0:2] for j in model_ind])
            predsets_calib[:, 0:T, :] = np.array([runs[j]['PredSets'][Y][:, 2:4] for j in model_ind])

            for i in range(2):  # 0=naive; 1=calibrated
                predsets = predsets_naive if i==0 else predsets_calib
                cover = (np.greater_equal(truth, predsets[:, start:end, 0]) *
                         np.less_equal(truth, predsets[:, start:end, 1]))
                width = np.diff(predsets[:, start:end, :], axis=2)[:, :, 0]

                if avg_method == 'moving':
                    mean_cover = mov_avg(cover, w)
                    mean_width = mov_avg(width, w)
                else:
                    mean_cover = run_avg(cover)
                    mean_width = run_avg(width)

                # axs[0, i].plot(mean_cover.mean(axis=0), lw=1.0,
                #                label=model_names[m] if i==0 else'')
                # axs[0, i].fill_between(np.arange(T-burnin-w+1),
                #                        mean_cover + 2.0*sd_cover,
                #                        mean_cover - 2.0*sd_cover,
                #                        alpha=0.25)
                # axs[1, i].plot(mean_width.mean(axis=0))
                # axs[1, i].set_yscale('log')
                # axs[1, i].fill_between(np.arange(T-w+1),
                #                       mean_width + 2.0*sd_width,
                #                       mean_width - 2.0*sd_width,
                #                       alpha=0.25)
                if i == 0:
                    MAD = 100. * abs(mean_cover - (1.-alpha)).mean(axis=1)
                    MAD_mean = round(MAD.mean(), 3)
                    MAD_sd = round(MAD.std(), 3)
                    table[m+1] = [model_names[m], MAD_mean, MAD_sd]

                # y0_min = min(min(mean_cover.mean(axis=0)), y0_min)
                # y0_max = max(max(mean_cover.mean(axis=0)), y0_max)
                # y1_min = min(min(mean_width.mean(axis=0)), y1_min)
                # y1_max = max(max(mean_width.mean(axis=0)), y1_max)

        # y0_min = y0_min / 1.01
        # y0_max = y0_max * 1.01
        # y1_min = y1_min / 1.01
        # y1_max = y1_max * 1.01

        # axs[0, 0].set_ylim(y0_min, y0_max)
        # axs[0, 1].set_ylim(y0_min, y0_max)
        # axs[1, 0].set_ylim(y1_min, y1_max)
        # axs[1, 1].set_ylim(y1_min, y1_max)

        # add red area in empirical coverage
        # T = len(runs[0]['Z'])
        # axs[0, 0].fill_between(np.arange(0, end-start-w+1, 1),
        #                        conf_bound_hi, y0_max,
        #                        color='red', lw=0.0, alpha=0.1)
        # axs[0, 0].fill_between(np.arange(0, end-start-w+1, 1),
        #                        conf_bound_lo, y0_min,
        #                        color='red', lw=0.0, alpha=0.1)
        # axs[0, 1].fill_between(np.arange(0, end-start-w+1, 1),
        #                        conf_bound_hi, y0_max,
        #                        color='red', lw=0.0, alpha=0.1)
        # axs[0, 1].fill_between(np.arange(0, end-start-w+1, 1),
        #                        conf_bound_lo, y0_min,
        #                        color='red', lw=0.0, alpha=0.1)

        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.03),
        #            ncol=4, fancybox=True, fontsize=12)
        # plt.tight_layout()

        print("\nCoverage (t=" + str(start) + " to t=" + str(end) +"): " + Y)
        print(tabulate(table))

        ###############################
        # Coverage conditional on "x" #
        ###############################
        # X2 = truths['RV'][burnin:]
        # bins = 10
        # quantiles = np.linspace(0, 1, bins+1)
        # X2_bins = np.quantile(X2, quantiles)
        # X2_bins[0] = 0.0
        # X2_bins[-1] = np.inf

        # plt.figure(dpi=1000)
        # fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=1000,
        #                         layout='constrained', sharex=True, sharey=True)
        # fig.suptitle("Conditional Coverage: " + Y, fontsize=12)
        # fig.supxlabel("X_{t-1}^2")
        # axs[0].axhline(y=1.0-alpha, color='black', ls='--', lw=0.8)
        # axs[1].axhline(y=1.0-alpha, color='black', ls='--', lw=0.8)

        # cond_cov = np.full([n_runs, n_models, bins, 2], np.nan)
        # for m in range(n_models):
        #     ind = [j for j in range(len(runs)) if runs[j]['fk'] == model_names[m]]
        #     predsets_all = np.array([runs[j]['PredSets'][Y] for j in ind])
        #     for i in range(2):  # 0=naive; 1=calibrated
        #         predsets = predsets_all[:, burnin:T, (2*i):((i+1)*2)]
        #         for j in range(bins):
        #             mask = (np.greater_equal(X2, X2_bins[j]) *
        #                     np.less_equal(X2, X2_bins[j+1]))
        #             truths_bin = truths[Y][mask]
        #             predsets_bin = predsets[:, mask, :]
        #             cond_cov[:, m, j, i] = np.mean((np.greater_equal(truths_bin, predsets_bin[:, :, 0]) *
        #                                             np.less_equal(truths_bin, predsets_bin[:, :, 1])))
        #             cond_cov_mean = cond_cov.mean(axis=0)
        #             cond_cov_sd = cond_cov.std(axis=0)
        #
        #         axs[i].plot(cond_cov_mean[m, :, i],
        #                     label=model_names[m] if i==0 else '')
        #         x_min, x_max = axs[i].get_xlim()
        #         axs[i].fill_between(np.linspace(x_min, x_max+1, bins),
        #                             cond_cov_mean[m, :, i]+2.0*cond_cov_sd[m, :, i],
        #                             cond_cov_mean[m, :, i]-2.0*cond_cov_sd[m, :, i])
        #         axs[i].set_xlim(x_min, x_max)

        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0),
        #            ncol=3, fancybox=True)


def exmpl_predsets(runs, models, start, end, pred_opts):
    # preliminaries
    truths = runs[-1]
    Y = list(truths.keys())
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1
    n_models = int(len(runs)/n_runs)
    length = 500
    alpha = pred_opts['alpha']

    #############
    # Coverages #
    #############
    fig, axs = plt.subplots(3, 2, figsize=(14, 8), dpi=1000,
                            layout='constrained',
                            sharex=True, sharey=False)

    for m in range(len(models)):
        model_ind = np.where(np.array([runs[j]['fk'] for j in range(n_models)]) == models[m])[0][0]
        T = len(runs[model_ind]['Z'])

        axs[0, m].set_title(models[m], fontsize=12, loc='left')
        axs[-1, 0].set_xlabel("t", fontsize=12)
        axs[-1, 1].set_xlabel("t", fontsize=12)

        for j in range(len(Y)):
            axs[j, 0].set_ylabel(Y[j], fontsize=12)
            axs[j, 1].set_yticks([])
            truth = truths[Y[j]][start:end]
            axs[j, m].plot(truth, color='black', lw=0.7, label="Truth")
            y_min, y_max = axs[j, m].get_ylim()
            axs[j, m].set_ylim([y_min, y_max])

            predset_naive = [runs[model_ind]['PredSets'][Y[j]][start:end, 0],
                             runs[model_ind]['PredSets'][Y[j]][start:end, 1]]
            predset_calib = [runs[model_ind]['PredSets'][Y[j]][start:end, 2],
                             runs[model_ind]['PredSets'][Y[j]][start:end, 3]]

            axs[j, m].fill_between(np.arange(length),
                             np.maximum(predset_naive[0],
                                        predset_calib[0]),
                             np.minimum(predset_naive[1],
                                        predset_calib[1]),
                             alpha=0.4, lw=0, fc='royalblue')
            axs[j, m].fill_between(np.arange(length),
                             predset_naive[0], predset_calib[0],
                             alpha=0.4, lw=0, fc='green',
                             where=predset_naive[0] > predset_calib[0])
            axs[j, m].fill_between(np.arange(length),
                             predset_naive[0], predset_calib[0],
                             alpha=0.4, lw=0, fc='red',
                             where=predset_naive[0] < predset_calib[0])
            axs[j, m].fill_between(np.arange(length),
                             predset_naive[1], predset_calib[1],
                             alpha=0.4, lw=0, fc='green',
                             where=predset_naive[1] < predset_calib[1])
            axs[j, m].fill_between(np.arange(length),
                             predset_naive[1], predset_calib[1],
                             alpha=0.4, lw=0, fc='red',
                             where=predset_naive[1] > predset_calib[1])

            axs[j, m].set_xticks(np.linspace(0, end-start, 5), np.linspace(start, end, 5, dtype=int))

            cov_naive = np.mean((np.greater_equal(truth, predset_naive[0]) *
                                 np.less_equal(truth, predset_naive[1])))
            cov_calib = np.mean((np.greater_equal(truth, predset_calib[0]) *
                                 np.less_equal(truth, predset_calib[1])))

            txt = (str(round(100*cov_naive, 1)) + " / "
                   + str(round(100*cov_calib, 1)) + "%")
            if m==0 and j==0:
                txt = "Coverage (naive / calibr.): " + txt
            axs[j, m].set_title(txt, fontsize=12, loc='right')

        fig.tight_layout()


def elm_function(runs):
    # preliminaries
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1

    d = 100  # grid density
    X_grid = np.linspace(-10., 10., d)
    logV2_grid = np.linspace(-10., 10., d)

    for r in range(n_runs):
        ###############
        # Shallow ELM #
        ###############
        elm_ind = np.where([runs[j]['fk'] == 'ELM-GARCH (t)' for j in range(len(runs))])[0][r]
        A = runs[elm_ind]['RandProj']['A']  # (q,2)
        b = runs[elm_ind]['RandProj']['b']  # (q,1)
        activ = runs[elm_ind]['RandProj']['activ']

        q_shallow = A.shape[0]
        W = runs[elm_ind]['W']  # particle weights
        w = np.array([runs[elm_ind]['theta']['w' + str(j)] for j in range(1, q_shallow+1)])  # (q,N)
        w0 = runs[elm_ind]['theta']['w0']  # (N,)

        def elm_vol(X, logV2):
            z = np.array([X, logV2]).reshape(2, 1)  # (2,1)
            r = activ(np.matmul(A, z) + b)  # (q,1)
            y = np.sum(w * r, axis=0) + w0  # (N,)
            y = np.sum(W * y)
            return y

        logV2_elm = np.zeros((d, d))
        for j, X in enumerate(X_grid):
            for i, logV2 in enumerate(logV2_grid):
                logV2_elm[i, j] = elm_vol(X, logV2)

        ############
        # Deep ELM #
        ############
        delm_ind = np.where(['Deep' in runs[j]['fk'] for j in range(len(runs))])[0][r]
        A = runs[delm_ind]['RandProj']['A']
        A1 = A[0]  # (H,d)
        A2 = A[1]  # (q,H)
        b = runs[delm_ind]['RandProj']['b']
        b1 = b[0]  # (H,1)
        b2 = b[1]  # (q,1)
        activ = runs[delm_ind]['RandProj']['activ']

        q_deep = A2.shape[0]
        H = A1.shape[0]
        W = runs[delm_ind]['W']
        w = np.array([runs[delm_ind]['theta']['w' + str(j)] for j in range(1, q_deep+1)])  # (q,N)
        w0 = runs[delm_ind]['theta']['w0']  # (N,)

        def deep_elm_vol(X, logV2):
            z = np.array([X, logV2]).reshape(2, 1)
            h1 = activ(np.matmul(A1, z) + b1)
            r = activ(np.matmul(A2, h1) + b2)
            y = np.sum(w * r, axis=0) + w0
            y = np.sum(W * y)
            return y

        logV2_delm = np.zeros((d, d))
        for j, X in enumerate(X_grid):
            for i, logV2 in enumerate(logV2_grid):
                logV2_delm[i, j] = deep_elm_vol(X, logV2)

        #########
        # Truth #
        #########
        omega = 0.25
        alpha = 0.1
        beta = 0.5
        gamma = 0.3

        def true_vol(X, logV2):
            logV2_next = np.log(omega + alpha*(X**2) + gamma*(X**2)*(X<0) + beta*np.exp(logV2))
            return logV2_next

        logV2_true = np.zeros((d, d))
        for j, X in enumerate(X_grid):
            for i, logV2 in enumerate(logV2_grid):
                logV2_true[i, j] = true_vol(X, logV2)

        ########
        # PLOT #
        ########
        min_value = min(logV2_true.min(), logV2_elm.min(), logV2_delm.min())
        max_value = max(logV2_true.max(), logV2_elm.max(), logV2_delm.max())
        levels = np.linspace(min_value, max_value, 50)

        fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=1000, layout='constrained', sharex=True, sharey=True)

        cs = axs[0].contourf(X_grid, logV2_grid, logV2_true, levels=levels, cmap='viridis')
        axs[0].set_title('True', fontsize=14)
        axs[0].set_xlabel('$X_t$', fontsize=14)
        axs[0].set_ylabel('$\log \sigma_t^2$', fontsize=14)

        axs[1].contourf(X_grid, logV2_grid, logV2_elm, levels=levels, cmap='viridis')
        axs[1].set_title('ELM', fontsize=14)
        axs[1].set_title('q = ' + str(q_shallow), loc='right')
        axs[1].set_xlabel('$X_t$', fontsize=14)

        axs[2].contourf(X_grid, logV2_grid, logV2_delm, levels=levels, cmap='viridis')
        axs[2].set_title('Deep ELM', fontsize=14)
        axs[2].set_title('q = ' + str(q_deep) + '; H = ' + str(H), loc='right')
        axs[2].set_xlabel('$X_t$', fontsize=14)

        cbar = fig.colorbar(cs, ax=axs, orientation='vertical', pad=0.02)
        cbar.set_label('$\log \sigma_{t+1}^2$', fontsize=14)


def elmsv_function(runs):
    # preliminaries
    runs = runs[:-1]
    n_runs = runs[-1]['run'] + 1

    d = 100  # grid density
    X_grid = np.linspace(-10., 10., d)
    logV2_grid = np.linspace(-10., 10., d)

    for r in range(n_runs):
        #######
        # ELM #
        #######
        elm_ind = np.where(['ELM/' in runs[j]['fk'] for j in range(len(runs))])[0][r]
        A = runs[elm_ind]['RandProj']['A']  # (q,2)
        b = runs[elm_ind]['RandProj']['b']  # (q,1)
        activ = shi

        q_shallow = A.shape[0]
        W = runs[elm_ind]['W']  # particle weights
        w = np.array([runs[elm_ind]['theta']['w' + str(j)] for j in range(1, q_shallow+1)])  # (q,N)
        w0 = runs[elm_ind]['theta']['w0']  # (N,)

        def elm_vol(X, logV2):
            z = np.array([X, logV2]).reshape(2, 1)  # (2,1)
            r = activ(np.matmul(A, z) + b)  # (q,1)
            y = np.sum(w * r, axis=0) + w0  # (N,)
            y = np.sum(W * y)
            return y

        logV2_elm = np.zeros((d, d))
        for j, X in enumerate(X_grid):
            for i, logV2 in enumerate(logV2_grid):
                logV2_elm[i, j] = elm_vol(X, logV2)

        #########
        # Truth #
        #########
        omega = 0.5
        alpha = 0.8

        def true_vol(X, logV2):
            logV2_next = omega*(1-alpha) + alpha*logV2
            return logV2_next

        logV2_true = np.zeros((d, d))
        for j, X in enumerate(X_grid):
            for i, logV2 in enumerate(logV2_grid):
                logV2_true[i, j] = true_vol(X, logV2)

        ########
        # PLOT #
        ########
        min_value = min(logV2_true.min(), logV2_elm.min())
        max_value = max(logV2_true.max(), logV2_elm.max())
        levels = np.linspace(min_value, max_value, 50)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=1000, layout='constrained', sharex=True, sharey=True)

        cs = axs[0].contourf(X_grid, logV2_grid, logV2_true, levels=levels, cmap='viridis')
        axs[0].set_title('True', fontsize=14)
        axs[0].set_xlabel('$X_t$', fontsize=14)
        axs[0].set_ylabel('$\log V_t^2$', fontsize=14)

        axs[1].contourf(X_grid, logV2_grid, logV2_elm, levels=levels, cmap='viridis')
        axs[1].set_title('ELM', fontsize=14)
        axs[1].set_title('q = ' + str(q_shallow), loc='right')
        axs[1].set_xlabel('$X_t$', fontsize=14)

        cbar = fig.colorbar(cs, ax=axs, orientation='vertical', pad=0.02)
        cbar.set_label('$\mathbb{E} \, \log V_{t+1}^2$', fontsize=14)


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
