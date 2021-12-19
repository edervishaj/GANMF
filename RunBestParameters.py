#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import time
import pickle
from datetime import timedelta
from Base.Evaluation.Evaluator import EvaluatorHoldout
from RecSysExp import dict_rec_classes, all_datasets, set_seed, load_URMs, dataset_kwargs, all_recommenders,\
    name_datasets, similarities

seed = 1337


def load_best_params(path, dataset, recommender, training_mode='', similarity=''):
    params_path = os.path.join(path, recommender + '_' + training_mode + similarity + '_' + dataset, 'best_params.pkl')
    if os.path.exists(params_path):
        with open(params_path, 'rb') as f:
            return pickle.load(f)
    else:
        return {}


def main(arguments):
    test_results_path = 'test_results'
    if not os.path.exists(test_results_path):
        os.makedirs(test_results_path, exist_ok=False)

    exp_path = 'experiments'
    dataset = None
    algo = None
    sim = ''
    train_mode = ''
    cutoffs = [5, 10, 20, 50]
    force = False
    best_params_dir = None

    if '--user' in arguments and train_mode == '':
        train_mode = 'user'
        arguments.remove('--user')

    if '--item' in arguments and train_mode == '':
        train_mode = 'item'
        arguments.remove('--item')

    if '--force' in arguments:
        force = True
        arguments.remove('--force')

    for idx, arg in enumerate(arguments):
        if arg in all_recommenders and algo is None:
            algo = arg
        if arg in similarities and sim == '':
            sim = arg
        if arg in name_datasets and dataset is None:
            dataset = all_datasets[name_datasets.index(arg)]
        if arg == '--bp':
            best_params_dir = arguments[idx+1]

    dname = dataset if isinstance(dataset, str) else dataset.DATASET_NAME
    if best_params_dir is not None:
        with open(os.path.join(best_params_dir, 'best_params.pkl'), 'rb') as f:
            best_params = pickle.load(f)
    else:
        best_params = load_best_params(exp_path, dname, dict_rec_classes[algo].RECOMMENDER_NAME, train_mode, sim)
    print(best_params)

    save_path = os.path.join(test_results_path if best_params_dir is None else best_params_dir,
                             dict_rec_classes[algo].RECOMMENDER_NAME + '_' + train_mode + sim + '_' + dname)

    results_text = os.path.join(save_path, 'test_results.txt')
    results_pkl = os.path.join(save_path, 'test_results.pkl')

    if not os.path.exists(results_pkl) or force:
        set_seed(seed)

        URM_train, URM_test, _, _, _ = load_URMs(dataset, dataset_kwargs)
        test_evaluator = EvaluatorHoldout(URM_test, cutoffs, exclude_seen=True)
        if algo in ['GANMF', 'DisGANMF', 'CFGAN', 'CAAE']:
            model = dict_rec_classes[algo](URM_train, mode=train_mode, seed=seed, is_experiment=True)
            train_start_time = time.time()
            model.fit(validation_set=None, sample_every=None, validation_evaluator=None, **best_params)
        else:
            model = dict_rec_classes[algo](URM_train)
            train_start_time = time.time()
            model.fit(**best_params)
        train_end_time = time.time()
        print(f'Training time: {str(timedelta((train_end_time - train_start_time) / 1000))}')

        testing_start_time = time.time()
        results_dict, results_str = test_evaluator.evaluateRecommender(model)
        testing_end_time = time.time()

        print(results_str)
        print(f'Testing time: {str(timedelta((testing_end_time - testing_start_time) / 1000))}')

        os.makedirs(save_path, exist_ok=force)
        with open(os.path.join(save_path, 'test_results.txt'), 'w') as f:
            f.write(results_str)
            f.write(f'Training time: {str(timedelta((train_end_time - train_start_time) / 1000))}')
            f.write(f'Testing time: {str(timedelta((testing_end_time - testing_start_time) / 1000))}')
        with open(os.path.join(save_path, 'test_results.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)

        model.saveModel(save_path)
    else:
        with open(results_text, 'r') as f:
            print(f.readlines())


if __name__ == '__main__':
    """
    Run this script as 
    
    python RunBestParameters.py <dataset-name> <recommender-name> [--user | --item] [<similarity-type>] [--force] [--bp <best-params-dir>]
    """

    assert len(sys.argv) >= 2, 'Number of arguments must be greater than 2, given {:d}'.format(len(sys.argv))
    args = sys.argv[1:]
    main(args)
