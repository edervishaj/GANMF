#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import json
import time
import psutil
import pickle
import shutil
import random
import platform
import datetime
import warnings
import subprocess
import numpy as np
import tensorflow as tf
import scipy.sparse as sps
import multiprocessing as mp

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.logging.set_verbosity(tf.logging.ERROR)

# Supress Tensorflow logs
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import skopt
from skopt.space.space import Real, Integer, Categorical
from skopt import gp_minimize, dummy_minimize

from datasets.Jester import Jester
from datasets.LastFM import LastFM
from datasets.CiaoDVD import CiaoDVD
from datasets.Movielens import Movielens
from datasets.Delicious import Delicious
from datasets.BookCrossing import BookCrossing
from datasets.AmazonMusic import AmazonMusic

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.NonPersonalizedRecommender import TopPop, Random

import GANRec as gans
from GANRec.GANMF import GANMF
from GANRec.CFGAN import CFGAN

from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython

# Seed for reproducibility of results and consistent initialization of weights/splitting of dataset
seed = 1337
# random.seed(seed)
# np.random.seed(seed)
# tf.set_random_seed(seed)

# Generic parameters for each dataset
dataset_kwargs = {}
dataset_kwargs['use_local'] = True
dataset_kwargs['force_rebuild'] = True
dataset_kwargs['implicit'] = True
dataset_kwargs['save_local'] = False
dataset_kwargs['verbose'] = False
dataset_kwargs['split'] = True
dataset_kwargs['split_ratio'] = [0.8, 0.2, 0]
dataset_kwargs['min_ratings'] = 2

URM_suffixes = ['_URM_train.npz', '_URM_test.npz', '_URM_validation.npz', '_URM_train_small.npz', '_URM_early_stop.npz']
all_datasets = [LastFM, CiaoDVD, Delicious, '100K', '1M', '10M', AmazonMusic]
early_stopping_algos = [IALSRecommender, MatrixFactorization_BPR_Cython]

train_mode = ''

exp_path = os.path.join('experiments', 'datasets')
if not os.path.exists(exp_path):
    os.makedirs(exp_path, exist_ok=False)


def set_seed(seed):
    # Seed for reproducibility of results and consistent initialization of weights/splitting of dataset
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_dataset(dataset, specs):
    set_seed(seed)  # Need to set this so each dataset is created the same in any machine/order selected

    if isinstance(dataset, str) and dataset in Movielens.urls.keys():
        reader = Movielens(version=dataset, **specs)
    else:
        reader = dataset(**specs)

    sets = []
    URM_train = reader.get_URM_train()
    URM_test = reader.get_URM_test()
    URM_for_train, _, URM_validation = reader.split_urm(
        URM_train.tocoo(), split_ratio=[0.75, 0, 0.25], save_local=False, verbose=False, min_ratings=1)
    URM_train_small, _, URM_early_stop = reader.split_urm(
        URM_for_train.tocoo(), split_ratio=[0.85, 0, 0.15], save_local=False, verbose=False, min_ratings=1)

    sets.extend([URM_train, URM_test, URM_validation, URM_train_small, URM_early_stop])

    for suf, urm in zip(URM_suffixes, sets):
        sps.save_npz(os.path.join(exp_path, reader.DATASET_NAME + suf), urm, compressed=True)

    return sets


def load_URMs(dataset, specs):
    sets = []
    dataset_name = dataset if isinstance(dataset, str) else dataset.DATASET_NAME
    urm_to_load = [os.path.join(exp_path, dataset_name + x) for x in URM_suffixes]
    all_exist = np.array([os.path.isfile(path) for path in urm_to_load]).all()
    if all_exist:
        for urm in urm_to_load:
            sets.append(sps.load_npz(urm))
    else:
        sets = make_dataset(dataset, specs)
    return tuple(sets)


class RecSysExp:
    def __init__(self, recommender_class, dataset, fit_param_names=[], metric='MAP',
                 method='bayesian', at=5, verbose=True, seed=1234):

        # Seed for reproducibility of results and consistent initialization of weights/splitting of dataset
        set_seed(seed)

        self.recommender_class = recommender_class
        self.dataset = dataset
        self.dataset_name = self.dataset if isinstance(self.dataset, str) else self.dataset.DATASET_NAME
        self.fit_param_names = fit_param_names
        self.metric = metric
        self.method = method
        self.at = at
        self.verbose = verbose
        self.seed = seed
        self.isGAN = False

        # if isinstance(self.dataset, str) and self.dataset in Movielens.urls.keys():
        #     self.reader = Movielens(version=self.dataset, **dataset_kwargs)
        # else:
        #     self.reader = self.dataset(**dataset_kwargs)

        # self.logsdir = os.path.join('experiments', self.recommender_class.RECOMMENDER_NAME + '_' + self.reader.DATASET_NAME)
        self.logsdir = os.path.join('experiments',
                self.recommender_class.RECOMMENDER_NAME + '_' + train_mode + '_' + self.dataset_name)

        if not os.path.exists(self.logsdir):
            os.makedirs(self.logsdir, exist_ok=False)

        # with open(os.path.join(self.logsdir, 'dataset_config.txt'), 'w') as f:
        #     json.dump(self.reader.config, f, indent=4)

        codesdir = os.path.join(self.logsdir, 'code')
        os.makedirs(codesdir, exist_ok=True)
        shutil.copy(os.path.abspath(sys.modules[self.__module__].__file__), codesdir)
        shutil.copy(os.path.abspath(sys.modules[self.recommender_class.__module__].__file__), codesdir)

        # self.URM_train, self.URM_test, self.URM_validation = self.reader.split_urm(split_ratio=[0.6, 0.2, 0.2], save_local=False, verbose=False)
        # self.URM_train = self.reader.get_URM_train()
        # self.URM_test = self.reader.get_URM_test()
        # self.URM_for_train, _, self.URM_validation = self.reader.split_urm(
        #         self.URM_train.tocoo(), split_ratio=[0.75, 0, 0.25], save_local=False, verbose=False)
        # self.URM_train_small, _, self.URM_early_stop = self.reader.split_urm(self.URM_for_train.tocoo(), split_ratio=[0.85, 0, 0.15], save_local=False, verbose=False)

        # del self.URM_for_train

        self.URM_train, self.URM_test, self.URM_validation, self.URM_train_small, self.URM_early_stop = load_URMs(
            dataset, dataset_kwargs)

        self.evaluator_validation = EvaluatorHoldout(self.URM_validation, [self.at], exclude_seen=True)
        self.evaluator_earlystop = EvaluatorHoldout(self.URM_early_stop, [self.at], exclude_seen=True)
        self.evaluatorTest = EvaluatorHoldout(self.URM_test, [self.at, 10, 20], exclude_seen=True, minRatingsPerUser=2)

        self.fit_params = {}

        modules = getattr(self.recommender_class, '__module__', None)
        if modules and modules.split('.')[0] == gans.__name__:
            self.isGAN = True

        # EARLY STOPPING from Maurizio's framework for baselines
        self.early_stopping_parameters = {
            'epochs_min': 0,
            'validation_every_n': 5,
            'stop_on_validation': True,
            'validation_metric': self.metric,
            'lower_validations_allowed': 5,
            'evaluator_object': self.evaluator_earlystop
        }

        # EARYL STOPPING for GAN-based recommenders
        self.my_early_stopping = {
            'allow_worse': 5,
            'freq': 5,
            'validation_evaluator': self.evaluator_earlystop,
            'validation_set': None,
            'sample_every': None,
        }

    def build_fit_params(self, params):
        for i, val in enumerate(params):
            param_name = self.dimension_names[i]
            if param_name in self.fit_param_names:
                self.fit_params[param_name] = val
            elif param_name == 'epochs' and self.recommender_class in early_stopping_algos:
                self.fit_params[param_name] = val

    def save_best_params(self, additional_params=None):
        d = dict(self.fit_params)
        if additional_params is not None:
            d.update(additional_params)
        with open(os.path.join(self.logsdir, 'best_params.pkl'), 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

    def load_best_params(self):
        with open(os.path.join(self.logsdir, 'best_params.pkl'), 'rb') as f:
            return pickle.load(f)

    def obj_func(self, params):
        """
        Black-box objective function.

        Parameters
        ----------
        params: list
            Ranges of hyperparameters to consider. List of skopt.space.space.Dimension.

        Returns
        -------
        obj_func_value: float
            Value of the objective function as denoted by the experiment metric.
        """

        # print('Optimizing for', self.reader.DATASET_NAME)
        print('Optimizing', self.recommender_class.RECOMMENDER_NAME, 'for', self.dataset_name)

        # Split the parameters into build_params and fit_params
        self.build_fit_params(params)

        # Create the model and fit it.
        try:
            if self.isGAN:
                model = self.recommender_class(self.URM_train_small, mode=train_mode, is_experiment=True)
                model.logsdir = self.logsdir
                fit_early_params = dict(self.fit_params)
                fit_early_params.update(self.my_early_stopping)
                last_epoch = model.fit(**fit_early_params)

                # Save the right number of epochs that produces the current model
                if last_epoch != self.fit_params['epochs']:
                    self.fit_params['epochs'] = last_epoch - \
                                                self.my_early_stopping['allow_worse'] * self.my_early_stopping['freq']

            else:
                model = self.recommender_class(self.URM_train_small)
                if self.recommender_class in early_stopping_algos:
                    fit_early_params = dict(self.fit_params)
                    fit_early_params.update(self.early_stopping_parameters)
                    model.fit(**fit_early_params)
                else:
                    model.fit(**self.fit_params)

            results_dic, results_run_string = self.evaluator_validation.evaluateRecommender(model)
            fitness = -results_dic[self.at][self.metric]
        except tf.errors.ResourceExhaustedError:
            fitness = 0

        try:
            if fitness < self.best_res:
                self.best_res = fitness
                self.save_best_params(additional_params=dict(epochs=model.epochs_best) if self.recommender_class in early_stopping_algos else None)
        except AttributeError:
            self.best_res = fitness
            self.save_best_params(additional_params=model.get_early_stopping_final_epochs_dict() if self.recommender_class in early_stopping_algos else None)

        with open(os.path.join(self.logsdir, 'results.txt'), 'a') as f:
            d = self.fit_params
            if self.recommender_class in early_stopping_algos:
                d.update(model.get_early_stopping_final_epochs_dict()) 
            d_str = json.dumps(d)
            f.write(d_str)
            f.write('\n')
            f.write(results_run_string)
            f.write('\n\n')

        return fitness

    def tune(self, params, evals=10, init_config=None, seed=None):
        """
        Runs the hyperparameter search using Gaussian Process as surrogate model or Random Search,
        saves the results of the trials and print the best found parameters.

        Parameters
        ----------
        params: list
            List of skopt.space.space.Dimensions to be searched.

        evals: int
            Number of evaluations to perform.

        init_config: list, default None
            An initial parameter configuration for seeding the Gaussian Process

        seed: int, default None
            Seed for random_state of `gp_minimize` or `dummy_minimize`.
            Set to a fixed integer for reproducibility.
        """

        msg = 'Started ' + self.recommender_class.RECOMMENDER_NAME + ' ' + self.dataset_name
        subprocess.run(['telegram-send', msg])


        U, I = self.URM_test.shape

        if self.recommender_class == GANMF:
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='emb_dim', dtype=int))
            self.fit_param_names.append('emb_dim')

        if self.recommender_class == CFGAN:
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='d_nodes', dtype=int))
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='g_nodes', dtype=int))
            self.fit_param_names.append('d_nodes')
            self.fit_param_names.append('g_nodes')

        self.dimension_names = [p.name for p in params]

        '''
        Need to make sure that the max. value of `num_factors` parameters must be lower than
        the max(U, I)
        '''
        try:
            idx = self.dimension_names.index('num_factors')
            maxval = params[idx].bounds[1]
            if maxval > min(U, I):
                params[idx] = Integer(1, min(U, I), name='num_factors', dtype=int)
        except ValueError:
            pass

        if len(params) > 0:

            t_start = int(time.time())

            if seed is None:
                seed = self.seed


            if self.method == 'bayesian':
                results = gp_minimize(self.obj_func, params, n_calls=evals, x0=init_config, random_state=seed, verbose=True)
            else:
                results = dummy_minimize(self.obj_func, params, n_calls=evals, random_state=seed, verbose=True)

            t_end = int(time.time())

        # Save best parameters of this experiment
        # best_params = dict(zip(self.dimension_names, results.x))
        # with open(os.path.join(self.logsdir, 'best_params.pkl'), 'wb') as f:
        #     pickle.dump(best_params, f, pickle.HIGHEST_PROTOCOL)

            best_params = self.load_best_params()

            with open(os.path.join(self.logsdir, 'results.txt'), 'a') as f:
                f.write('Experiment ran for {}\n'.format(str(datetime.timedelta(seconds=t_end - t_start))))
                f.write('Best {} score: {}. Best result found at: {}\n'.format(self.metric, results.fun, best_params))

            if self.recommender_class in [IALSRecommender, MatrixFactorization_BPR_Cython]:
                self.dimension_names.append('epochs')
            self.build_fit_params(best_params.values())

        # Retrain with all training data
        if self.isGAN:
            model = self.recommender_class(self.URM_train, mode=train_mode, is_experiment=True)
            model.logsdir = self.logsdir
            model.fit(**self.fit_params)
            # load_models(model, save_dir='best_model', all_in_folder=True)

        else:
            model = self.recommender_class(self.URM_train)
            model.fit(**self.fit_params)
            # model.loadModel(os.path.join(self.logsdir, 'best_model'))

        _, results_run_string = self.evaluatorTest.evaluateRecommender(model)

        print('\n\nResults on test set:')
        print(results_run_string)
        print('\n\n')

        with open(os.path.join(self.logsdir, 'result_test.txt'), 'w') as f:
            f.write(results_run_string)

        msg = 'Finished ' + self.recommender_class.RECOMMENDER_NAME + ' ' + self.dataset_name
        subprocess.run(['telegram-send', msg])


def run_exp(experiment, dimensions, evals, init_config=None):
    experiment.tune(dimensions, evals, init_config)


def set_affinity_on_worker():
    """When a new worker process is created, the affinity is set to all CPUs"""
    if platform.system() == 'Linux':
        print("I'm the process %d, setting affinity to all CPUs." % os.getpid())
        os.system("taskset -p 0xf %d" % os.getpid())


def main(arguments):
    global train_mode
    EVALS = 50
    use_mp = True
    run_all = False
    selected_exp = []
    selected_datasets = []

    list_experiments = ['TopPop', 'Random', 'PureSVD', 'ALS', 'NMF', 'BPR', 'IRGAN', 'CFGAN', 'GANMF']
    list_datasets = [d if isinstance(d, str) else d.DATASET_NAME for d in all_datasets]

    if '--build_datasets' in arguments:
        print('Building all necessary datasets required for the experiments. Disregarding other arguments! ' +
        'You will need to run this script again without --build_datasets in order to run experiments!')
        # Make all datasets
        for d in all_datasets:
            load_URMs(d, dataset_kwargs)
        return

    if '--no_mp' in arguments:
        print('No multiprocessing requested! Falling back to serial execution of experiments!')
        use_mp = False
        arguments.remove('--no_mp')

    if '--run_all' in arguments:
        print('All datasets selected for each algorithm!')
        selected_datasets = all_datasets
        run_all = True

    if '--user' in arguments:
        train_mode = 'user'

    if '--item' in arguments:
        train_mode = 'item'

    for arg in arguments:
        if not run_all and arg in list_datasets:
            selected_datasets.append(all_datasets[list_datasets.index(arg)])
        if arg in list_experiments:
            selected_exp.append(arg)


    dict_rec_classes = {}
    dict_dimensions = {}
    dict_fit_params = {}
    dict_init_configs = {}


    # Experiment parameters
    puresvd_dimensions = [
        Integer(1, 250, name='num_factors', dtype=int)
    ]
    puresvd_fit_params = [d.name for d in puresvd_dimensions]



    ials_dimensions = [
        Integer(1, 250, name='num_factors', dtype=int),
        Categorical(["linear", "log"], name='confidence_scaling'),
        Real(low=1e-3, high=50, prior='log-uniform', name='alpha', dtype=float),
        Real(low=1e-5, high=1e-2, prior='log-uniform', name='reg', dtype=float),
        Real(low=1e-3, high=10.0, prior='log-uniform', name='epsilon', dtype=float)
    ]
    ials_fit_params = [d.name for d in ials_dimensions]



    bpr_dimensions = [
        Categorical([1500], name='epochs'),
        Integer(1, 250, name='num_factors', dtype=int),
        Categorical([128, 256, 512, 1024], name='batch_size'),
        Categorical(["adagrad", "adam"], name='sgd_mode'),
        Real(low=1e-12, high=1e-3, prior='log-uniform', name='positive_reg'),
        Real(low=1e-12, high=1e-3, prior='log-uniform', name='negative_reg'),
        Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
    ]
    bpr_fit_params = [d.name for d in bpr_dimensions]



    nmf_dimensions = [
        Integer(1, 500, name='num_factors', dtype=int),
        Real(low=1e-5, high=1, prior='log-uniform', name='l1_ratio', dtype=float),
        Categorical(['coordinate_descent', 'multiplicative_update'], name='solver'),
        Categorical(['nndsvda'], name='init_type'),
        Categorical(['frobenius', 'kullback-leibler'], name='beta_loss')
    ]
    nmf_fit_params = ['num_factors', 'l1_ratio', 'solver', 'init_type', 'beta_loss']



    cfgan_dimensions = [
        Categorical([300], name='epochs'),
        Integer(1, 5, prior='uniform', name='d_steps', dtype=int),
        Integer(1, 5, prior='uniform', name='g_steps', dtype=int),
        Integer(1, 5, prior='uniform', name='d_layers', dtype=int),
        Integer(1, 5, prior='uniform', name='g_layers', dtype=int),
        Categorical(['linear', 'tanh', 'sigmoid'], name='d_hidden_act'),
        Categorical(['linear', 'tanh', 'sigmoid'], name='g_hidden_act'),
        Categorical(['ZR', 'PM', 'ZP'], name='scheme'),
        Categorical([64, 128, 256, 512, 1024], name='d_batch_size'),
        Categorical([64, 128, 256, 512, 1024], name='g_batch_size'),
        Real(low=0, high=1, prior='uniform', name='zr_ratio', dtype=float),
        Real(low=0, high=1, prior='uniform', name='zp_ratio', dtype=float),
        Real(low=0, high=1, prior='uniform', name='zr_coefficient', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='d_lr', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='g_lr', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='d_reg', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='g_reg', dtype=float),
    ]
    cfgan_fit_params = [d.name for d in cfgan_dimensions]



    ganmf_dimensions = [
        Categorical([300], name='epochs'),
        Integer(1, 250, name='num_factors', dtype=int),
        Categorical([64, 128, 256, 512, 1024], name='batch_size'),
        Integer(1, 10, name='m', dtype=int),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='d_lr', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='g_lr', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='d_reg', dtype=float),
        Real(low=1e-2, high=0.5, prior='uniform', name='recon_coefficient', dtype=float),
        # Integer(5, 400, name='emb_dim', dtype=int),
        # Integer(1, 10, name='d_steps', dtype=int),
        # Integer(1, 10, name='g_steps', dtype=int),
        # Real(low=1e-6, high=1e-4, prior='log-uniform', name='g_reg', dtype=float),
    ]
    ganmf_fit_params = [d.name for d in ganmf_dimensions]


    dict_rec_classes['TopPop'] = TopPop
    dict_rec_classes['Random'] = Random
    dict_rec_classes['PureSVD'] = PureSVDRecommender
    dict_rec_classes['BPR'] = MatrixFactorization_BPR_Cython
    dict_rec_classes['ALS'] = IALSRecommender
    dict_rec_classes['NMF'] = NMFRecommender
    dict_rec_classes['GANMF'] = GANMF
    dict_rec_classes['CFGAN'] = CFGAN

    dict_dimensions['TopPop'] = []
    dict_dimensions['Random'] = []
    dict_dimensions['PureSVD'] = puresvd_dimensions
    dict_dimensions['BPR'] = bpr_dimensions
    dict_dimensions['ALS'] = ials_dimensions
    dict_dimensions['NMF'] = nmf_dimensions
    dict_dimensions['GANMF'] = ganmf_dimensions
    dict_dimensions['CFGAN'] = cfgan_dimensions

    dict_fit_params['TopPop'] = []
    dict_fit_params['Random'] = []
    dict_fit_params['PureSVD'] = puresvd_fit_params
    dict_fit_params['BPR'] = bpr_fit_params
    dict_fit_params['ALS'] = ials_fit_params
    dict_fit_params['NMF'] = nmf_fit_params
    dict_fit_params['GANMF'] = ganmf_fit_params
    dict_fit_params['CFGAN'] = cfgan_fit_params

    # dict_init_configs['GANMF'] = [300, 10, 128, 3, 1e-3, 1e-3, 1e-4, 0.05, 128]

    pool_list_experiments = []
    pool_list_dimensions = []

    for exp in selected_exp:
        for d in selected_datasets:
            new_exp = RecSysExp(dict_rec_classes[exp], dataset=d, fit_param_names=dict_fit_params[exp],
                                method='bayesian', seed=seed)
            if use_mp:
                pool_list_experiments.append(new_exp)
                pool_list_dimensions.append(dict_dimensions[exp])
            else:
                new_exp.tune(dict_dimensions[exp], evals=EVALS,
                             init_config=dict_init_configs[exp] if exp in dict_init_configs else None)

    if use_mp:
        # Need to turn off MKL's own threading mechanism in order to use MP
        # https://github.com/joblib/joblib/issues/138
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        
        pool = mp.Pool(initializer=set_affinity_on_worker)
        pool.starmap_async(run_exp, zip(pool_list_experiments, pool_list_dimensions, [EVALS]*len(pool_list_experiments)))
        pool.close()
        pool.join()


if __name__ == '__main__':
    # Run this script as `python RecSysExp.py [--build_datasets] [experiment_name] [--run_all] [dataset_name] [--no_mp]`
    assert len(sys.argv) >= 2, 'Number of arguments must be greater than 2, given {:d}'.format(len(sys.argv))
    arguments = sys.argv[1:]
    main(arguments)
