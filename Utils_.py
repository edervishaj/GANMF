#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import pickle
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
plt.style.use('fivethirtyeight')

from sklearn.metrics.pairwise import cosine_similarity


CONSTANTS = dict(root_dir=os.path.dirname(os.path.abspath(__file__)))


class EarlyStoppingScheduler(object):
    """Performs early stopping mechanism according to a fixed number of worse evaluations on a validation set."""
    def __init__(self, model, evaluator, metrics=['PRECISION', 'RECALL', 'MAP', 'NDCG'], freq=1, allow_worse=5, after=0):
        """Constructor

        Parameters
        ----------
        model: BaseRecommender
            Implements _compute_item_score() TODO: change the base interface for models

        evaluator: Evaluator
            Initialized with the validation set.

        metrics: list[str], default ['PRECISION', 'RECALL', 'MAP', 'NDCG']
            List of metrics present in the evaluator for which early stopping will be evaluated.

        freq: int, default 1
            Frequency in epochs when to perform evaluation on validation set.

        allow_worse: int, default 5
            Allowed number of bad results on all metrics.

        after: int, default 0
            Start early stopping after this epoch.
        """

        self.model = model
        self.evaluator = evaluator
        self.metrics = metrics
        self.freq = freq
        self.best_scores = np.zeros(len(metrics))
        self.allow_worse = allow_worse
        self.worse_left = allow_worse
        self.after = after
        self.scores = []

    def score(self, epoch):
        if epoch % self.freq == 0:
            results_dic, _ = self.evaluator.evaluateRecommender(self.model) #TODO: dependent on recommender interface
            curr_scores = np.array([results_dic[5][m] for m in self.metrics])
            self.scores.append(curr_scores)
            if np.all(np.less_equal(curr_scores, self.best_scores)):
                if self.worse_left > 0:
                    self.worse_left -= 1
                else:
                    self.model.stop_fit()
                    self.model.load_model()
            else:
                self.best_scores = curr_scores
                self.worse_left = self.allow_worse
                self.model.save_current_model()

    def reset(self):
        self.worse_left = self.allow_worse

    def __call__(self, epoch):
        if epoch > self.after:
            self.score(epoch)

    def load_best(self):
        self.model.load_model()

    def get_scores(self):
        return self.scores


def cos_sim(list_vec1, list_vec2):
    """ Element-wise cosine similarity between two lists of vectors """
    sim = np.array([])
    for vec1, vec2 in zip(list_vec1, list_vec2):
        sim = np.append(sim, cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1)).flatten())
    return np.mean(sim)


def cosine_sim(matrix):
    similarity = np.dot(matrix, np.transpose(matrix))
    inv_squared_magnitude = 1.0 / np.diag(similarity)
    inv_squared_magnitude[np.isinf(inv_squared_magnitude)] = 0.0
    sqrt_inv_mag = np.sqrt(inv_squared_magnitude)
    cos_similarity = similarity * sqrt_inv_mag
    cosine = cos_similarity.T * sqrt_inv_mag
    return cosine


def plot_loss_acc(model, dict_values, xlabel='epochs', ylabel=None, scale='linear'):
    """
    Plots training loss and accuracy values for Discriminator and Generator.

    Parameters
    ----------
    model:
        Recommendation model used (must be GAN-based).

    dict_values: dict
        Dictionary where each key is to be used in the legend.

    xlabel: str, default `epochs`
        Label to use for the x-axis.
    
    ylabel: str, default None
        Label to use for the y-axis.

    scale: str, default `linear`
        Scale to use for plotting. Options are `linear` and `log`.
    """

    if scale != 'log':
        scale = 'linear'

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(dict_values.keys())
    epochs = len(dict_values[keys[0]])
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        if scale == 'log':
            plt.plot(range(epochs), np.log(dict_values[k]), label=k, linestyle='-', alpha=0.8, marker=next(marker))
        else:
            plt.plot(range(epochs), dict_values[k], label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper right')

    title = 'Loss function of model ' + model.RECOMMENDER_NAME + '\n'
    title += '{'

    config_list = ['d_nodes', 'g_nodes', 'g1_nodes', 'g2_nodes', 'd_hidden_act', 'g_hidden_act', 'g_output_act',
                   'use_dropout', 'use_batchnorm', 'dropout', 'batch_mom', 'epochs', 'sgd_var', 'adam_var', 'sgd_mom', 'beta1']

    for c in model.config.keys():
        if c in config_list:
            title += c + ':' + str(model.config[c]) + ', '

    title = title[:-2]
    title += '}'

    plt.title(title)
    
    save_path = os.path.join(model.logsdir, 'loss' + '_epochs_' + str(epochs) + '.png')
    fig.savefig(save_path, bbox_inches="tight")


def plot_generator_ratings(ratings, rec, neg=False):
    '''
    Plots the mean and std of the fake ratings of batch as received by the generator
    during training.

    :param ratings: List of fake ratings in form [[batch_size, 1], [batch_size, 1], ...]
    :param rec: GAN Model that generated the ratings
    '''

    data = pd.DataFrame(columns=['epoch', 'rating'])
    for e, r in enumerate(ratings):
        epoch_data = (np.ones(r.shape[0], dtype=np.int32) * e).tolist()
        rating_data = r.flatten().tolist()
        tmp_df = pd.DataFrame([[x[0], x[1]] for x in zip(epoch_data, rating_data)], columns=['epoch', 'rating'])
        data = data.append(tmp_df, ignore_index=True)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.relplot(x='epoch', y='rating', data=data, ci='sd', kind='line', ax=ax)

    if neg:
        save_path = os.path.join(rec.logsdir, 'fake_ratings_neg.png')
    else:
        save_path = os.path.join(rec.logsdir, 'fake_ratings.png')
    fig.savefig(save_path, bbox_inches="tight")


def plot_gradients(gradients):
    """
    Ridgeplot of gradients over training epochs

    Parameters
    ----------
    gradients: np.ndarray of elements (epoch_number, layer, node_gradient)
        Array of gradients
    """

    pal = sns.cubehelix_palette(n_colors=16, start=0.3, rot=-0.5, light=.7)

    # We have to create a pd.DataFrame in order to use Seaborn.FacetGrid for the ridgeplot.
    epochs = np.unique(gradients[:, 0])
    layers = np.unique(gradients[:, 1])
    fig, ax = plt.subplots(1, len(layers), figsize=(20, 10))
    df = pd.DataFrame(gradients, columns=['epochs', 'layer', 'gradients'])
    for i, l in enumerate(layers):
        g = sns.FacetGrid(df.iloc[:, df.layer == l], row='epochs', hue='epochs', aspect=15, height=.5, palette=pal, ax=ax[0,i])
        g.map(sns.kdeplot, 'gradients', clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)
        g.fig.subplots_adjust(hspace=-.25)
        g.set_titles('')
        g.set(yticks=[])
        g.despine(bottom=True, left=True)


    pass


def plot(feed, title, save_dir, xlabel='epochs', ylabel=None):
    """
    Plots the dictionary provided. Each key is considered a separate line.

    Parameters
    ----------
    feed: dict
        Keys of the dictionary are used in the legend of the plot.

    title: str
        Title of the plot. Also the filename of the plot.

    save_dir: str
        Directory where to save the plot.

    xlabel: str
        Label to be used for the x-axis of the plot.

    ylabel: str, default None
        Label to be used for the y-axis of the plot.
    """

    marker = itertools.cycle(['o', '^', 's', 'p', '1', 'D', 'P', '*'])
    keys = list(feed.keys())
    fig = plt.figure(figsize=(20, 10))
    plt.xlabel(xlabel)
    if isinstance(ylabel, str):
        plt.ylabel(ylabel)
    plt.grid(True)

    for k in keys:
        data = feed[k]
        plt.plot(range(1, len(data)+1), data, label=k, linestyle='-', alpha=0.8, marker=next(marker))

    plt.legend(keys, loc='upper left')

    plt.title(title)

    save_path = os.path.join(save_dir, title + '.png')
    fig.savefig(save_path, bbox_inches="tight")


def gini(array):
    """ From https://github.com/oliviaguest/gini"""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def dense_spmatrix(matrix, dtype=np.int8):
    """
    Produces a dense 2D numpy matrix of `dtype` from a scipy.sparse matrix.
    """

    matrix = matrix.tocoo()
    dense = np.zeros(matrix.shape, dtype=dtype)
    dense[matrix.row, matrix.col] = matrix.data
    return dense


def save_weights(sess, frm, to):
    for idx, var in enumerate(frm):
        sess.run(to[idx].assign(var))


def saveWeights(model, save_dir):
    from Base.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender
    from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

    if isinstance(model, BaseSimilarityMatrixRecommender):
        model.save_model(folder_path=save_dir)
    elif isinstance(model, BaseMatrixFactorizationRecommender):
        model.saveModel(folder_path=save_dir)
    elif model.__class__.__name__ in ['CAAE', 'GANMF', 'CFGAN', 'DisGANMF', 'DeepGANMF']:
        params = {}
        for k in model.params:
            params[k] = [model.sess.run(p) for p in model.params[k]]
        with open(os.path.join(save_dir, 'weights.pkl'), 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)
