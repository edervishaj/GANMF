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
import pickle
import shutil
import random
import datetime
import warnings
import subprocess
import numpy as np
import tensorflow as tf
import scipy.sparse as sps

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
tf.logging.set_verbosity(tf.logging.ERROR)

# Supress Tensorflow logs
os.environ['KMP_WARNINGS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import skopt
from skopt.callbacks import CheckpointSaver
from skopt import gp_minimize, dummy_minimize
from skopt.space.space import Real, Integer, Categorical

from datasets.LastFM import LastFM
from datasets.CiaoDVD import CiaoDVD
from datasets.Movielens import Movielens
from datasets.Delicious import Delicious
from datasets.AmazonMusic import AmazonMusic

from Base.Evaluation.Evaluator import EvaluatorHoldout
from Base.NonPersonalizedRecommender import TopPop, Random

import GANRec as gans
from GANRec.GANMF import GANMF
from GANRec.CFGAN import CFGAN
from GANRec.DisGANMF import DisGANMF
from GANRec.DeepGANMF import DeepGANMF

from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from MatrixFactorization.NMFRecommender import NMFRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

seed = 1337

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
all_datasets = [LastFM, '1M']
name_datasets = [d if isinstance(d, str) else d.DATASET_NAME for d in all_datasets]
all_recommenders = ['TopPop', 'Random', 'PureSVD', 'ALS', 'BPR', 'SLIMBPR', 'SLIMELASTIC', 'CFGAN', 'GANMF', 'DisGANMF',
                    'DeepGANMF', 'fullGANMF', 'ItemKNN']
early_stopping_algos = [IALSRecommender, MatrixFactorization_BPR_Cython, SLIM_BPR_Cython]
similarities = ['cosine', 'jaccard', 'tversky', 'dice', 'euclidean', 'asymmetric']
similarity_algos = ['ItemKNN']

train_mode = ''
similarity_mode = ''

exp_path = os.path.join('experiments', 'datasets')
if not os.path.exists(exp_path):
    os.makedirs(exp_path, exist_ok=False)


def set_seed(seed):
    # Seed for reproducibility of results and consistent initialization of weights/splitting of dataset
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_similarity_params(dimensions, similarity):
    if similarity == 'asymmetric':
        dimensions.append(Real(low=0, high=2, prior='uniform', dtype=float, name='asymmetric_alpha'))
        dimensions.append(Categorical([True], name='normalize'))

    elif similarity == 'tversky':
        dimensions.append(Real(low=0, high=2, prior='uniform', dtype=float, name='tversky_alpha'))
        dimensions.append(Real(low=0, high=2, prior='uniform', dtype=float, name='tversky_beta'))
        dimensions.append(Categorical([True], name='normalize'))

    elif similarity == 'euclidean':
        dimensions.append(Categorical([True, False], name='normalize'))
        dimensions.append(Categorical([True, False], name='normalize_avg_row'))
        dimensions.append(Categorical(['lin', 'log', 'exp'], name='similarity_from_distance_mode'))

    return dimensions


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
    dataset_name = ('Movielens' + dataset) if isinstance(dataset, str) else dataset.DATASET_NAME
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
        self.logsdir = os.path.join('experiments',
                self.recommender_class.RECOMMENDER_NAME + '_' + train_mode + similarity_mode + '_' + self.dataset_name)

        if not os.path.exists(self.logsdir):
            os.makedirs(self.logsdir, exist_ok=False)

        codesdir = os.path.join(self.logsdir, 'code')
        os.makedirs(codesdir, exist_ok=True)
        shutil.copy(os.path.abspath(sys.modules[self.__module__].__file__), codesdir)
        shutil.copy(os.path.abspath(sys.modules[self.recommender_class.__module__].__file__), codesdir)

        self.URM_train, self.URM_test, self.URM_validation, self.URM_train_small, self.URM_early_stop = load_URMs(
            dataset, dataset_kwargs)

        self.evaluator_validation = EvaluatorHoldout(self.URM_validation, [self.at], exclude_seen=True)
        self.evaluator_earlystop = EvaluatorHoldout(self.URM_early_stop, [self.at], exclude_seen=True)
        self.evaluatorTest = EvaluatorHoldout(self.URM_test, [self.at, 10, 20, 50], exclude_seen=True, minRatingsPerUser=2)

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

        # EARLY-STOPPING for GAN-based recommenders
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

        print('Optimizing', self.recommender_class.RECOMMENDER_NAME, 'for', self.dataset_name)

        # Split the parameters into build_params and fit_params
        self.build_fit_params(params)

        # Create the model and fit it.
        try:
            if self.isGAN:
                model = self.recommender_class(self.URM_train_small, mode=train_mode, seed=seed, is_experiment=True)
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
            return 0

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

        if self.recommender_class == CFGAN or self.recommender_class == DeepGANMF:
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='d_nodes', dtype=int))
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='g_nodes', dtype=int))
            self.fit_param_names.append('d_nodes')
            self.fit_param_names.append('g_nodes')

        if self.recommender_class == DisGANMF:
            params.append(Integer(4, int(I * 0.75) if I <= 1024 else 1024, name='d_nodes', dtype=int))
            self.fit_param_names.append('d_nodes')

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

            # Check if there is already a checkpoint for this experiment
            checkpoint_path = os.path.join(self.logsdir, 'checkpoint.pkl')
            checkpoint_exists = True if os.path.exists(checkpoint_path) else False
            checkpoint_saver = CheckpointSaver(os.path.join(self.logsdir, 'checkpoint.pkl'), compress=3)

            if seed is None:
                seed = self.seed

            t_start = int(time.time())

            if checkpoint_exists:
                previous_run = skopt.load(checkpoint_path)
                if self.method == 'bayesian':
                    results = gp_minimize(self.obj_func, params, n_calls=evals - len(previous_run.func_vals),
                                          x0=previous_run.x_iters, y0=previous_run.func_vals, n_random_starts=0,
                                          random_state=seed, verbose=True, callback=[checkpoint_saver])
                else:
                    results = dummy_minimize(self.obj_func, params, n_calls=evals - len(previous_run.func_vals),
                                             x0=previous_run.x_iters, y0=previous_run.func_vals, random_state=seed,
                                             verbose=True, callback=[checkpoint_saver])
            else:
                if self.method == 'bayesian':
                    results = gp_minimize(self.obj_func, params, n_calls=evals, random_state=seed, verbose=True,
                                          callback=[checkpoint_saver])
                else:
                    results = dummy_minimize(self.obj_func, params, n_calls=evals, random_state=seed, verbose=True,
                                          callback=[checkpoint_saver])

            t_end = int(time.time())

            best_params = self.load_best_params()

            with open(os.path.join(self.logsdir, 'results.txt'), 'a') as f:
                f.write('Experiment ran for {}\n'.format(str(datetime.timedelta(seconds=t_end - t_start))))
                f.write('Best {} score: {}. Best result found at: {}\n'.format(self.metric, results.fun, best_params))

            if self.recommender_class in [IALSRecommender, MatrixFactorization_BPR_Cython]:
                self.dimension_names.append('epochs')
            self.build_fit_params(best_params.values())

        # Save best parameters as text file
        with open(os.path.join(self.logsdir, 'best_params.pkl'), 'rb') as g:
            d = pickle.load(g)
            with open(os.path.join(self.logsdir, 'best_params.txt'), 'w') as f:
                f.write(json.dumps(d))

        msg = 'Finished ' + self.recommender_class.RECOMMENDER_NAME + ' ' + self.dataset_name
        subprocess.run(['telegram-send', msg])


def run_exp(experiment, dimensions, evals, init_config=None):
    experiment.tune(dimensions, evals, init_config)


def main(arguments):
    global train_mode, similarity_mode
    EVALS = 50
    algo = None
    sim = None
    dataset = None

    if '--build-datasets' in arguments:
        print('Building all necessary datasets required for the experiments. Disregarding other arguments! ' +
        'You will need to run this script again without --build_datasets in order to run experiments!')
        # Make all datasets
        for d in all_datasets:
            load_URMs(d, dataset_kwargs)
        return

    if '--user' in arguments and train_mode == '':
        train_mode = 'user'
        arguments.remove('--user')

    if '--item' in arguments and train_mode == '':
        train_mode = 'item'
        arguments.remove('--item')

    for arg in arguments:
        if arg in all_recommenders and algo is None:
            algo = arg
        if arg in similarities and sim is None:
            sim = arg
            similarity_mode = sim
        if arg in name_datasets and dataset is None:
            dataset = all_datasets[name_datasets.index(arg)]

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



    slimbpr_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Categorical([1500], name='epochs'),
        Categorical([True, False], name='symmetric'),
        Categorical(["sgd", "adagrad", "adam"], name='sgd_mode'),
        Real(low=1e-9, high=1e-3, prior='log-uniform', name='lambda_i', dtype=float),
        Real(low=1e-9, high=1e-3, prior='log-uniform', name='lambda_j', dtype=float),
        Real(low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate', dtype=float)
    ]
    slimbpr_fit_names = [d.name for d in slimbpr_dimensions]



    slimelastic_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Real(low=1e-5, high=1.0, prior='log-uniform', name='l1_ratio', dtype=float),
        Real(low=1e-3, high=1.0, prior='uniform', name='alpha')
    ]
    slimelastic_fit_names = [d.name for d in slimelastic_dimensions]



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
        Integer(low=1, high=250, name='num_factors', dtype=int),
        Categorical([64, 128, 256, 512, 1024], name='batch_size'),
        Integer(low=1, high=10, name='m', dtype=int),
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



    disgan_dimensions = [
        Categorical([300], name='epochs'),
        Categorical(['linear', 'tanh', 'relu', 'sigmoid'], name='d_hidden_act'),
        Integer(low=1, high=5, prior='uniform', name='d_layers', dtype=int),
        Integer(low=1, high=250, name='num_factors', dtype=int),
        Categorical([64, 128, 256, 512, 1024], name='batch_size'),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='d_lr', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='g_lr', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='d_reg', dtype=float),
        Real(low=1e-2, high=0.5, prior='uniform', name='recon_coefficient', dtype=float)
    ]
    disgan_fit_params = [d.name for d in disgan_dimensions]



    deepganmf_dimensions = [
        Categorical([300], name='epochs'),
        Categorical(['linear', 'tanh', 'relu', 'sigmoid'], name='d_hidden_act'),
        Categorical(['linear', 'tanh', 'relu', 'sigmoid'], name='g_hidden_act'),
        Categorical(['linear', 'tanh', 'relu', 'sigmoid'], name='g_output_act'),
        Categorical([1, 3, 5], name='d_layers'),
        Categorical([1, 2, 3, 4, 5], name='g_layers'),
        Categorical([64, 128, 256, 512, 1024], name='batch_size'),
        Integer(low=1, high=10, name='m', dtype=int),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='d_lr', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='g_lr', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='d_reg', dtype=float),
        Real(low=1e-2, high=0.5, prior='uniform', name='recon_coefficient', dtype=float),
    ]
    deepganmf_fit_params = [d.name for d in deepganmf_dimensions]



    itemknn_dimensions = [
        Integer(low=5, high=1000, name='topK', dtype=int),
        Integer(low=0, high=1000, name='shrink', dtype=int),
        Categorical([True, False], name='normalize')
    ]
    itemknn_fit_params = [d.name for d in itemknn_dimensions]



    dict_rec_classes['TopPop'] = TopPop
    dict_rec_classes['Random'] = Random
    dict_rec_classes['PureSVD'] = PureSVDRecommender
    dict_rec_classes['BPR'] = MatrixFactorization_BPR_Cython
    dict_rec_classes['ALS'] = IALSRecommender
    dict_rec_classes['GANMF'] = GANMF
    dict_rec_classes['CFGAN'] = CFGAN
    dict_rec_classes['DisGANMF'] = DisGANMF
    dict_rec_classes['SLIMBPR'] = SLIM_BPR_Cython
    dict_rec_classes['SLIMELASTIC'] = SLIMElasticNetRecommender
    dict_rec_classes['DeepGANMF'] = DeepGANMF
    dict_rec_classes['ItemKNN'] = ItemKNNCFRecommender

    dict_dimensions['TopPop'] = []
    dict_dimensions['Random'] = []
    dict_dimensions['PureSVD'] = puresvd_dimensions
    dict_dimensions['BPR'] = bpr_dimensions
    dict_dimensions['ALS'] = ials_dimensions
    dict_dimensions['GANMF'] = ganmf_dimensions
    dict_dimensions['CFGAN'] = cfgan_dimensions
    dict_dimensions['DisGANMF'] = disgan_dimensions
    dict_dimensions['SLIMBPR'] = slimbpr_dimensions
    dict_dimensions['SLIMELASTIC'] = slimelastic_dimensions
    dict_dimensions['DeepGANMF'] = deepganmf_dimensions
    dict_dimensions['ItemKNN'] = itemknn_dimensions

    dict_fit_params['TopPop'] = []
    dict_fit_params['Random'] = []
    dict_fit_params['PureSVD'] = puresvd_fit_params
    dict_fit_params['BPR'] = bpr_fit_params
    dict_fit_params['ALS'] = ials_fit_params
    dict_fit_params['GANMF'] = ganmf_fit_params
    dict_fit_params['CFGAN'] = cfgan_fit_params
    dict_fit_params['DisGANMF'] = disgan_fit_params
    dict_fit_params['SLIMBPR'] = slimbpr_fit_names
    dict_fit_params['SLIMELASTIC'] = slimelastic_fit_names
    dict_fit_params['DeepGANMF'] = deepganmf_fit_params
    dict_fit_params['ItemKNN'] = itemknn_fit_params

    if algo in similarity_algos:
        if sim is not None:
            dict_dimensions[algo].append(Categorical([sim], name='similarity'))
            dict_dimensions[algo] = get_similarity_params(dict_dimensions[algo], sim)
            dict_fit_params[algo] = [d.name for d in dict_dimensions[algo]]
        else:
            raise ValueError(f'{algo} selected but no similarity specified!')

    new_exp = RecSysExp(dict_rec_classes[algo], dataset=dataset, fit_param_names=dict_fit_params[algo],
                        method='bayesian', seed=seed)
    new_exp.tune(dict_dimensions[algo], evals=EVALS,
                 init_config=dict_init_configs[algo] if algo in dict_init_configs else None)


if __name__ == '__main__':
    """
    Run this script as:
    
    python RecSysExp.py [--build-datasets] <algorithm_name> [--user | --item] <dataset_name> [<similarity_type>]
    """

    assert len(sys.argv) >= 2, f'Number of arguments must be greater than 2, given {len(sys.argv)}'
    args = sys.argv[1:]
    main(args)
