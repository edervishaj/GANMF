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
from datasets.Yelp2015 import Yelp2015
from datasets.Movielens import Movielens
from datasets.AmazonMusic import AmazonMusic
from datasets.AmazonMoviesTV import AmazonMoviesTV
from datasets.PinterestNeuMF import PinterestNeuMF

from Base.Evaluation.Evaluator import EvaluatorHoldout

import GANRec as gans
from GANRec.CAAE import CAAE
from GANRec.GANMF import GANMF
from GANRec.CFGAN import CFGAN
from GANRec.DisGANMF import DisGANMF
from GANRec.DeepGANMF import DeepGANMF

from EASE_R.EASE_R_Recommender import EASE_R_Recommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop, Random
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from GraphBased.P3alphaRecommender import P3alphaRecommender
from MatrixFactorization.IALSRecommender import IALSRecommender
from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython

seed = 1337

# Generic parameters for each dataset
dataset_kwargs = {
    'use_local': True,
    'force_rebuild': True,
    'implicit': True,
    'save_local': False,
    'verbose': False,
    'split': True,
    'split_ratio': [0.8, 0.2, 0],
    'min_ratings_user': 2,
    # 'min_ratings_item': 5,
    # 'header': True,
}

URM_suffixes = ['_URM_train.npz', '_URM_test.npz', '_URM_validation.npz', '_URM_train_small.npz', '_URM_early_stop.npz']
all_datasets = ['1M', 'hetrec2011', LastFM, AmazonMoviesTV, AmazonMusic, Yelp2015, PinterestNeuMF, '100K']
name_datasets = [d if isinstance(d, str) else d.DATASET_NAME for d in all_datasets]
all_recommenders = ['TopPop', 'PureSVD', 'ALS', 'BPR', 'SLIMBPR', 'SLIMELASTIC', 'ItemKNN', 'P3Alpha', 'EASER', 'CFGAN',
                    'CAAE', 'GANMF', 'DisGANMF', 'DeepGANMF']
early_stopping_algos = [IALSRecommender, MatrixFactorization_BPR_Cython, SLIM_BPR_Cython]
similarities = ['cosine', 'jaccard', 'tversky', 'dice', 'euclidean', 'asymmetric']
similarity_algos = ['ItemKNN']

train_mode = ''
similarity_mode = ''

dict_rec_classes = {
    # GAN-based
    'CAAE': CAAE,
    'CFGAN': CFGAN,
    'GANMF': GANMF,
    'DisGANMF': DisGANMF,
    'DeepGANMF': DeepGANMF,

    # Non-personalized
    'TopPop': TopPop,
    'Random': Random,

    # MF
    'ALS': IALSRecommender,
    'PureSVD': PureSVDRecommender,
    'BPR': MatrixFactorization_BPR_Cython,

    # KNN
    'SLIMBPR': SLIM_BPR_Cython,
    'EASER': EASE_R_Recommender,
    'P3Alpha': P3alphaRecommender,
    'ItemKNN': ItemKNNCFRecommender,
    'SLIMELASTIC': SLIMElasticNetRecommender,
}

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
        dimensions.append(Real(low=0, high=2, prior='uniform', name='asymmetric_alpha', dtype=float))
        dimensions.append(Categorical([True], name='normalize'))

    elif similarity == 'tversky':
        dimensions.append(Real(low=0, high=2, prior='uniform', name='tversky_alpha', dtype=float))
        dimensions.append(Real(low=0, high=2, prior='uniform', name='tversky_beta', dtype=float))
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
    URM_for_train, _, URM_validation = reader.split_urm(URM_train.tocoo(), split_ratio=[0.75, 0, 0.25],
                                                        save_local=False, min_ratings_user=1, verbose=False)
    URM_train_small, _, URM_early_stop = reader.split_urm(URM_for_train.tocoo(), split_ratio=[0.85, 0, 0.15],
                                                          save_local=False, min_ratings_user=1, verbose=False)

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

        print('Optimizing', self.recommender_class.RECOMMENDER_NAME, train_mode, similarity_mode, 'for', self.dataset_name)

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

    def tune(self, params, evals=10, seed=None):
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

        msg = 'Started ' + self.recommender_class.RECOMMENDER_NAME + train_mode + similarity_mode + ' ' + self.dataset_name
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

        msg = 'Finished ' + self.recommender_class.RECOMMENDER_NAME + train_mode + similarity_mode + ' ' + self.dataset_name
        subprocess.run(['telegram-send', msg])


def main(arguments):
    global train_mode, similarity_mode
    EVALS = 50
    algo = None
    sim = None
    dataset = None
    build_dataset = False

    if '--build-dataset' in arguments:
        build_dataset = True

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

    if build_dataset:
        dataset_str = dataset if isinstance(dataset, str) else dataset.DATASET_NAME
        print('Building ' + dataset_str + '. Skipping other arguments! You need to run this script without --build-datasets to run experiments!')
        load_URMs(dataset, dataset_kwargs)
        return

    # Experiment parameters
    puresvd_dimensions = [
        Integer(1, 250, name='num_factors', dtype=int)
    ]

    ials_dimensions = [
        Integer(1, 250, name='num_factors', dtype=int),
        Categorical(["linear", "log"], name='confidence_scaling'),
        Real(low=1e-3, high=50, prior='log-uniform', name='alpha', dtype=float),
        Real(low=1e-5, high=1e-2, prior='log-uniform', name='reg', dtype=float),
        Real(low=1e-3, high=10.0, prior='log-uniform', name='epsilon', dtype=float)
    ]

    bpr_dimensions = [
        Categorical([1500], name='epochs'),
        Integer(1, 250, name='num_factors', dtype=int),
        Categorical([128, 256, 512, 1024], name='batch_size'),
        Categorical(["adagrad", "adam"], name='sgd_mode'),
        Real(low=1e-12, high=1e-3, prior='log-uniform', name='positive_reg'),
        Real(low=1e-12, high=1e-3, prior='log-uniform', name='negative_reg'),
        Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate'),
    ]

    slimbpr_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Categorical([1500], name='epochs'),
        Categorical([True, False], name='symmetric'),
        Categorical(["sgd", "adagrad", "adam"], name='sgd_mode'),
        Real(low=1e-9, high=1e-3, prior='log-uniform', name='lambda_i', dtype=float),
        Real(low=1e-9, high=1e-3, prior='log-uniform', name='lambda_j', dtype=float),
        Real(low=1e-4, high=1e-1, prior='log-uniform', name='learning_rate', dtype=float)
    ]

    slimelastic_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Real(low=1e-5, high=1.0, prior='log-uniform', name='l1_ratio', dtype=float),
        Real(low=1e-3, high=1.0, prior='uniform', name='alpha', dtype=float)
    ]

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

    caae_dimensions = [
        Categorical([300], name='epochs'),
        Categorical([5, 10, 15, 20], name='d_steps'),
        Categorical([5, 10, 15, 20], name='g_steps'),
        Categorical([5, 10, 15, 20], name='gpr_steps'),
        Categorical([1, 2, 3, 4, 5], name='g_layers'),
        Categorical([1, 2, 3, 4, 5], name='gpr_layers'),
        Categorical([20, 50, 100, 150, 200], name='g_units'),
        Categorical([20, 50, 100, 150, 200], name='gpr_units'),
        Integer(low=1, high=250, name='num_factors', dtype=int),
        Categorical([32, 64, 128, 256], name='m_batch'),
        Categorical([1024 * i for i in range(1, 11)], name='d_bsize'),
        Categorical([1e-4, 5e-4, 1e-3, 5e-3], name='lr'),
        Categorical([1e-4, 1e-3, 1e-2, 1e-1], name='beta'),
        Categorical([i / 10 for i in range(1, 10)], name='S'),
        Categorical([i / 10 for i in range(1, 10)], name='lmbda')
    ]

    ganmf_dimensions = [
        Categorical([300], name='epochs'),
        Integer(low=1, high=250, name='num_factors', dtype=int),
        Categorical([64, 128, 256, 512, 1024], name='batch_size'),
        Integer(low=1, high=10, name='m', dtype=int),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='d_lr', dtype=float),
        Real(low=1e-4, high=1e-2, prior='log-uniform', name='g_lr', dtype=float),
        Real(low=1e-6, high=1e-4, prior='log-uniform', name='d_reg', dtype=float),
        Categorical([0.8], name='recon_coefficient')
        # Real(low=1e-2, high=0.5, prior='uniform', name='recon_coefficient', dtype=float),
        # Integer(5, 400, name='emb_dim', dtype=int),
        # Integer(1, 10, name='d_steps', dtype=int),
        # Integer(1, 10, name='g_steps', dtype=int),
        # Real(low=1e-6, high=1e-4, prior='log-uniform', name='g_reg', dtype=float),
    ]

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

    itemknn_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Integer(low=0, high=1000, prior='uniform', name='shrink', dtype=int),
        Categorical([True, False], name='normalize')
    ]

    p3alpha_dimensions = [
        Integer(low=5, high=1000, prior='uniform', name='topK', dtype=int),
        Real(low=0, high=2, prior='uniform', name='alpha', dtype=float),
        Categorical([True, False], name='normalize_similarity')
    ]

    easer_dimensions = [
        Categorical([None], name='topK'),
        Categorical([False], name='normalize_matrix'),
        Real(low=1e0, high=1e7, prior='log-uniform', name='l2_norm', dtype=float)
    ]

    dict_dimensions = {
        'TopPop': [],
        'Random': [],
        'PureSVD': puresvd_dimensions,
        'BPR': bpr_dimensions,
        'ALS': ials_dimensions,
        'SLIMBPR': slimbpr_dimensions,
        'SLIMELASTIC': slimelastic_dimensions,
        'ItemKNN': itemknn_dimensions,
        'P3Alpha': p3alpha_dimensions,
        'EASER': easer_dimensions,
        'CFGAN': cfgan_dimensions,
        'CAAE': caae_dimensions,
        'GANMF': ganmf_dimensions,
        'DisGANMF': disgan_dimensions,
        'DeepGANMF': deepganmf_dimensions
    }

    if algo in similarity_algos:
        if sim is not None:
            dict_dimensions[algo].append(Categorical([sim], name='similarity'))
            dict_dimensions[algo] = get_similarity_params(dict_dimensions[algo], sim)
        else:
            raise ValueError(f'{algo} selected but no similarity specified!')

    new_exp = RecSysExp(dict_rec_classes[algo], dataset=dataset,
                        fit_param_names=[d.name for d in dict_dimensions[algo]],
                        method='bayesian', seed=seed)
    new_exp.tune(dict_dimensions[algo], evals=EVALS)


if __name__ == '__main__':
    """
    Run this script as:
    
    python RecSysExp.py [--build-dataset] <dataset_name> <algorithm_name> [--user | --item] [<similarity_type>]
    """

    assert len(sys.argv) >= 2, f'Number of arguments must be greater than 2, given {len(sys.argv)}'
    args = sys.argv[1:]
    main(args)
