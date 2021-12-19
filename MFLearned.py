#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets.LastFM import LastFM
from RecSysExp import dict_rec_classes, dataset_kwargs
from RunBestParameters import dict_rec_classes, load_URMs


contextRc = {
    'axes.grid': True,
    'xtick.labelsize': 50,
    'ytick.labelsize': 50,
    'legend.fontsize': 30,
    'grid.linewidth': 2
}


def latent_factors_study():
    with plt.style.context(['default', contextRc]):
        start_path = 'latent_factors'
        metric = 'MAP'
        cutoff = 5
        datasets = ['1M', 'LastFM', 'hetrec2011']
        num_factors = [10, 30, 50, 100, 150, 250]
        algos = ['PureSVD', 'ALS', 'GANMF-u', 'GANMF-i']
        for d in datasets:
            marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.set_xlim([0, max(num_factors) + 5])
            ax.set_xticks(num_factors)
            ax.set_xticklabels([str(x) for x in num_factors])
            ax.locator_params(axis='x', nbins=len(num_factors))
            for algo in algos:
                train_mode = ''
                if algo.startswith('GANMF'):
                    algo, train_mode = algo.split('-')
                    train_mode = 'user' if train_mode == 'u' else 'item'

                scores = []
                for k in num_factors:
                    results_path = os.path.join(start_path, dict_rec_classes[algo].RECOMMENDER_NAME + '_' + train_mode + '_' + d + '_' + str(k), 'test_results.pkl')
                    with open(results_path, 'rb') as f:
                        results_dict = pickle.load(f)
                        scores.append(results_dict[cutoff][metric])
                if algo == 'GANMF':
                    algo = 'GANMF-i' if train_mode == 'item' else 'GANMF-u'
                ax.plot(num_factors, scores, label='WRMF' if algo == 'ALS' else algo, marker=next(marker), linewidth=5, ms=15)
            ax.set_xlabel('K', fontsize=50)
            ax.set_ylabel(metric + '@' + str(cutoff), fontsize=50)
            ax.legend(loc='best')
            fig.savefig(os.path.join('latent_factors', 'latent_factors' + ''.join(algos) + '_' + d), bbox_inches='tight')


def mf_qualitative_study():
    with plt.style.context(['seaborn-paper', contextRc]):
        metric = 'MAP'
        cutoff = 20
        datasets = ['1M', 'hetrec2011', LastFM]
        algorithms = ['PureSVD', 'ALS', 'GANMF-u', 'GANMF-i']
        user_masks = {
            '1M': [25, 100, 500, 1000],
            'hetrec2011': [25, 100, 500, 1000],
            'LastFM': [10, 20, 30, 40]
        }

        for d in datasets:
            dataset_name = d if isinstance(d, str) else d.DATASET_NAME
            URM_train, URM_test, _, _, _ = load_URMs(d, dataset_kwargs)
            count_ratings = (URM_train + URM_test).sum(axis=1).A1
            df = pd.DataFrame({'uid': [], 'algo': [], 'key': [], metric: []})
            for algo in algorithms:
                training_mode = ''
                if algo.startswith('GANMF'):
                    algo, training_mode = algo.split('-')
                    training_mode = 'user' if training_mode == 'u' else 'item'
                    model = dict_rec_classes[algo](URM_train, mode=training_mode, is_experiment=True, verbose=True)
                else:
                    model = dict_rec_classes[algo](URM_train)

                sim = 'cosine' if algo == 'ItemKNN' else ''

                save_path = os.path.join('test_results', model.RECOMMENDER_NAME + '_' + training_mode + sim + '_' + dataset_name)
                model.loadModel(save_path)

                def build_xticks():
                    xticks = []
                    for i, val in enumerate(user_masks[dataset_name]):
                        if i == 0:
                            xticks.append('<' + str(val))
                        elif i == len(user_masks[dataset_name])-1:
                            xticks.append('>=' + str(val))
                        else:
                            lbound = user_masks[dataset_name][i-1]
                            xticks.append('>=' + str(lbound) + ', <' + str(val))
                    return xticks

                def apply_key(u):
                    no_ratings = count_ratings[u]
                    for i, ubound in enumerate(user_masks[dataset_name]):
                        if no_ratings < ubound:
                            if i == 0:
                                return '<' + str(ubound)
                            else:
                                lbound = user_masks[dataset_name][i-1]
                                return '>=' + str(lbound) + ', <' + str(ubound)
                        else:
                            if i == len(user_masks[dataset_name])-1:
                                return '>=' + str(ubound)

                def fast_eval(usersToEvaluate):
                    from Base.Evaluation.metrics import average_precision

                    scores = []
                    recommended_items, _ = model.recommend(usersToEvaluate, remove_seen_flag=True, cutoff=cutoff, return_scores=True)

                    for u in usersToEvaluate:
                        relevant_items = URM_test.indices[URM_test.indptr[u]: URM_test.indptr[u+1]]
                        recommendation_list = recommended_items[u]
                        is_relevant = np.in1d(recommendation_list, relevant_items, assume_unique=True)
                        scores.append(average_precision(is_relevant[:cutoff], relevant_items))
                    return scores

                userid = list(range(URM_train.shape[0]))
                keys = [apply_key(u) for u in userid]
                if training_mode == 'item':
                    model.URM_train = model.URM_train.T.tocsr()
                ress = fast_eval(np.array(userid))
                if training_mode == 'item':
                    model.URM_train = model.URM_train.T.tocsr()
                
                if training_mode != '':
                    training_mode = '-u' if training_mode == 'user' else '-i'
                df = df.append(pd.DataFrame({'algo': [algo + training_mode] * len(keys), 'key': keys, metric: ress}))
            
            fig, ax = plt.subplots(figsize=(20, 10))
            ax = sns.barplot(data=df, x='key', y=metric, hue='algo', ci=None, ax=ax, order=build_xticks())
            ax.set_ylabel(metric + '@' + str(cutoff), fontsize=50)
            ax.set_xlabel('item interactions per user', fontsize=50)
            ax.legend().set_title('')
            fig.savefig(os.path.join('qualitative_study', '_'.join(algorithms) + '_' + dataset_name), bbox_inches='tight')


if __name__ == '__main__':
    latent_factors_study()
    mf_qualitative_study()
