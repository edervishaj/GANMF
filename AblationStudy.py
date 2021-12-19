#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import pickle
import itertools
import subprocess
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from GANRec.GANMF import GANMF
from RecSysExp import load_URMs, dataset_kwargs, all_datasets, name_datasets

seed = 1337

contextRc = {
    'axes.grid': True,
    'xtick.labelsize': 50,
    'ytick.labelsize': 50,
    'legend.fontsize': 50,
    'grid.linewidth': 2
}


def feature_matching_coefficient(arguments):
    start_path = 'feature_matching'
    end_path = 'feature_matching'
    values = ['00', '02', '04', '06', '08', '10']
    cutoff = 5
    range_coeff = [0, 0.2, 0.4, 0.6, 0.8, 1]

    mode = 'item'
    dataset = None
    for arg in arguments:
        if arg in name_datasets and dataset is None:
            dataset = arg

    map = []
    ndcg = []
    for val in values:
        p = os.path.join(start_path, 'GANMF_' + mode + '_' + dataset + '_' + val, 'GANMF_' + mode + '_' + dataset, 'test_results.pkl')
        with open(p, 'rb') as f:
            d = pickle.load(f)
            map.append(d[cutoff]['MAP'])
            ndcg.append(d[cutoff]['NDCG'])

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])

    with plt.style.context(['default', contextRc]):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_xlabel('\u03B1', fontsize=50)
        ax.plot(range_coeff, map, label='MAP@5', marker=next(marker), linewidth=5, ms=15)
        ax.plot(range_coeff, ndcg, label='NDCG@5', marker=next(marker), linewidth=5, ms=15)
        ax.legend(loc='best')
        fig.savefig(os.path.join(end_path, 'GANMF_' + mode + '_' + dataset + '_feature_matching_effect.png'), bbox_inches='tight')


def feature_matching_cos_sim(arguments):
    plt.style.use('fivethirtyeight')
    sns.set_context('paper', font_scale=5)

    start_path = 'feature_matching'
    end_path = os.path.join(start_path, 'cosine_similarities')

    mode = 'user'
    dataset = None
    for arg in arguments:
        if arg in name_datasets and dataset is None:
            dataset = arg
        # if arg in ['--user', '--item'] and mode is None:
        #     mode = arg[2:]

    URM_train, _, _, _, _ = load_URMs(all_datasets[name_datasets.index(dataset)], dataset_kwargs)

    no_feature_matching_params_dir = os.path.join(start_path, 'GANMF_' + mode + '_' + dataset + '_00', 'GANMF_' + mode + '_' + dataset)

    model = GANMF(URM_train, mode=mode, is_experiment=True)
    model.loadModel(no_feature_matching_params_dir)

    all_preds = model._compute_item_score(user_id_array=np.array(range(URM_train.shape[0])))
    similarity = cosine_similarity(all_preds)

    mean = np.mean(similarity)
    std = np.std(similarity)

    fig, ax = plt.subplots(figsize=(20, 10))
    with sns.axes_style('darkgrid', {'font.scale': 5}):
        sns.heatmap(similarity, vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=ax)
        ax.tick_params(left=False, bottom=False)

    hm_save_path = os.path.join(end_path, 'GANMF_' + mode + '_' + dataset + '_wo_fm.png')
    stats_save_path = os.path.join(end_path, 'GANMF_' + mode + '_' + dataset + '_wo_fm.txt')
    fig.savefig(hm_save_path, bbox_inches="tight")

    with open(stats_save_path, 'w') as f:
        f.write('Mean: ' + str(mean))
        f.write('\n')
        f.write('Std: ' + str(std))

    best_params_dir = os.path.join('test_results', 'GANMF_' + mode + '_' + dataset)

    model = GANMF(URM_train=URM_train, mode=mode, is_experiment=True)
    model.loadModel(best_params_dir)

    all_preds = model._compute_item_score(user_id_array=np.array(range(URM_train.shape[0])))
    similarity = cosine_similarity(all_preds)

    mean = np.mean(similarity)
    std = np.std(similarity)

    fig, ax = plt.subplots(figsize=(20, 10))
    with sns.axes_style('darkgrid', {'font.scale': 5}):
        sns.heatmap(similarity, vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=ax)
        ax.tick_params(left=False, bottom=False)

    hm_save_path = os.path.join(end_path, 'GANMF_' + mode + '_' + dataset + '_with_fm.png')
    stats_save_path = os.path.join(end_path, 'GANMF_' + mode + '_' + dataset + '_with_fm.txt')
    fig.savefig(hm_save_path, bbox_inches="tight")

    with open(stats_save_path, 'w') as f:
        f.write('Mean: ' + str(mean))
        f.write('\n')
        f.write('Std: ' + str(std))


def run_binGANMF(arguments):
    mode = None
    dataset = None
    for arg in arguments:
        if arg in name_datasets and dataset is None:
            dataset = arg
        if arg in ['--user', '--item'] and mode is None:
            mode = arg

    if dataset is not None:
        subprocess.run(['python', 'RecSysExp.py', dataset, 'DisGANMF', mode])
        subprocess.run(['python', 'RunBestParameters.py', dataset, 'DisGANMF', mode])


if __name__ == '__main__':
    """
    Run this script as:
    
    python AblationStudy.py <dataset-name> [binGANMF | feature-matching [--user | --item]]
    """

    assert len(sys.argv) >= 2, 'Number of arguments must be greater than 2, given {:d}'.format(len(sys.argv))
    arguments = sys.argv[1:]

    if 'binGANMF' in arguments:
        run_binGANMF(arguments)

    if 'feature-matching' in arguments:
        feature_matching_coefficient(arguments)
        feature_matching_cos_sim(arguments)
