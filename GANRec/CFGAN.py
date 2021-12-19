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
import tqdm
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from Base.BaseRecommender import BaseRecommender
from Utils_ import EarlyStoppingScheduler, save_weights
from GANRec.Cython.cython_utils import get_non_interactions, compute_masks

class CFGAN(BaseRecommender):
    RECOMMENDER_NAME = 'CFGAN'

    def __init__(self, URM_train, mode='user', seed=1234, verbose=False, is_experiment=False):

        self.mode = mode
        if self.mode == 'item':
            self.URM_train = URM_train.T.tocsr()
        else:
            self.URM_train = URM_train
        self.num_users, self.num_items = self.URM_train.shape
        self.config = None
        self.seed = seed
        self.verbose = verbose
        self.logsdir = os.path.join('plots', self.RECOMMENDER_NAME, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.is_experiment = is_experiment

        if not os.path.exists(self.logsdir) and not self.is_experiment:
            os.makedirs(self.logsdir, exist_ok=False)

        if not self.is_experiment:
            # Save this file inside logsdir/code
            codesdir = os.path.join(self.logsdir, 'code')
            os.makedirs(codesdir, exist_ok=False)
            shutil.copy(os.path.abspath(sys.modules[self.__module__].__file__), codesdir)

    def build(self, d_nodes=32, d_layers=1, g_nodes=32, g_layers=1, d_hidden_act='linear', g_hidden_act='linear'):
        bias_init = tf.random_uniform_initializer(-0.01, 0.01)

        ##########################
        # DISCRIMINATOR FUNCTION #
        ##########################
        def discriminator(condition, input_data):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                d = tf.concat([condition, input_data], axis=1)
                for l in range(d_layers):
                    scale = np.sqrt(6 / (d.shape.as_list()[1] + d_nodes))
                    w_init = tf.random_uniform_initializer(-scale, scale)
                    i = str(l)
                    d = tf.layers.dense(d, units=d_nodes, activation=None, name='D_hidden_' + i,
                                        bias_initializer=bias_init,
                                        kernel_initializer=w_init)
                    if d_hidden_act == 'tanh':
                        d = tf.nn.tanh(d, name='D_H_act_' + i)
                    elif d_hidden_act == 'sigmoid':
                        d = tf.nn.sigmoid(d, name='D_H_act_' + i)
                    elif d_hidden_act == 'relu':
                        d = tf.nn.relu(d, name='D_H_act_' + i)
                    elif d_hidden_act == 'LeakyReLU':
                        d = tf.nn.leaky_relu(d, name='D_H_act_' + i)
                    else:
                        d = tf.identity(d, name='D_H_act_' + i)
                scale = np.sqrt(6 / (d.shape.as_list()[1] + 1))
                w_init = tf.random_uniform_initializer(-scale, scale)
                d_output = tf.layers.dense(d, units=1, activation=None, name='D_output', bias_initializer=bias_init,
                                           kernel_initializer=w_init)
                return d_output

        ######################
        # GENERATOR FUNCTION #
        ######################
        def generator(condition):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                g = condition
                for l in range(g_layers):
                    scale = np.sqrt(6 / (g.shape.as_list()[1] + g_nodes))
                    w_init = tf.random_uniform_initializer(-scale, scale)
                    i = str(l)
                    g = tf.layers.dense(g, units=g_nodes, activation=None, name='G_hidden_' + i,
                                        bias_initializer=bias_init,
                                        kernel_initializer=w_init)
                    if g_hidden_act == 'tanh':
                        g = tf.nn.tanh(g, name='G_H_act_' + i)
                    elif g_hidden_act == 'sigmoid':
                        g = tf.nn.sigmoid(g, name='G_H_act_' + i)
                    elif g_hidden_act == 'relu':
                        g = tf.nn.relu(g, name='G_H_act_' + i)
                    elif g_hidden_act == 'LeakyReLU':
                        g = tf.nn.leaky_relu(g, name='G_H_act_' + i)
                    else:
                        g = tf.identity(g, name='G_H_act_' + i)
                scale = np.sqrt(6 / (g.shape.as_list()[1] + self.num_items))
                w_init = tf.random_uniform_initializer(-scale, scale)
                g_output = tf.layers.dense(g, units=self.num_items, activation=None, name='G_output',
                                           bias_initializer=bias_init, kernel_initializer=w_init)
                return g_output

        self.discriminator, self.generator = discriminator, generator

    def fit(self, d_nodes=32, g_nodes=32, d_layers=1, g_layers=1, scheme='ZR', d_hidden_act='linear',
            g_hidden_act='linear', epochs=300, d_lr=1e-5, g_lr=1e-5, d_reg=0, g_reg=0, d_steps=1, g_steps=1,
            d_batch_size=32, g_batch_size=32, zr_ratio=0., zp_ratio=0., zr_coefficient=0., allow_worse=5, freq=5,
            after=0, metrics=['MAP'], validation_evaluator=None, sample_every=None, validation_set=None):

        # Construct the model config
        self.config = dict(locals())
        del self.config['self']

        # First clear the session to save GPU memory
        tf.reset_default_graph()
        # Set fixed seed for the TF graph
        tf.set_random_seed(self.seed)

        self.build(d_nodes, d_layers, g_nodes, g_layers, d_hidden_act, g_hidden_act)

        # Create optimizers
        opt_gen = tf.train.AdamOptimizer(learning_rate=g_lr)
        opt_disc = tf.train.AdamOptimizer(learning_rate=d_lr)

        self.condition = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        real_data = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        train_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        zr_mask = tf.placeholder(dtype=np.float32, shape=[None, self.num_items])

        # Generator operations
        self.fake_profile = self.generator(self.condition)
        g_out = self.fake_profile * train_mask

        # Discriminator operations
        d_real = self.discriminator(self.condition, real_data)
        d_fake = self.discriminator(self.condition, g_out)

        # D losses
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        # Get trainable variables
        self.params = {}
        self.params['D'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.params['G'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.best_params = {}
        for p in self.params:
            self.best_params[p] = []
            for idx, var in enumerate(self.params[p]):
                self.best_params[p].append(tf.get_variable(p + '_best_params_' + str(idx), shape=var.get_shape(),
                                                           trainable=False))

        # G losses
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))

        # final losses
        d_l2 = tf.add_n([tf.nn.l2_loss(var) for var in self.params['D']])
        d_loss = d_loss_real + d_loss_fake + d_reg * d_l2

        g_l2 = tf.add_n([tf.nn.l2_loss(var) for var in self.params['G']])
        zr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.fake_profile - 0) * zr_mask, 1, keepdims=True))
        g_loss = gen_loss + g_reg * g_l2 + zr_coefficient * zr_loss

        # Define training operations
        d_train_op = tf.train.AdamOptimizer(learning_rate=d_lr).minimize(d_loss, var_list=self.params['D'])
        g_train_op = tf.train.AdamOptimizer(learning_rate=g_lr).minimize(g_loss, var_list=self.params['G'])

        # DO NOT allocate all GPU memory to this process
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        eps = 1e-10

        self._stop_training = False
        if allow_worse is not None:
            early_stop = EarlyStoppingScheduler(self, evaluator=validation_evaluator, allow_worse=allow_worse,
                                                freq=freq, metrics=metrics, after=after)

        train_g_loss = []
        train_d_loss = []

        all_users = np.array(range(self.num_users), dtype=np.int32)

        not_selected = get_non_interactions(all_users, self.URM_train)

        d_step = d_batch_size
        g_step = g_batch_size

        if self.verbose:
            print('Starting training...')

        t_start = time.time()
        e_start = time.time()

        epoch = 1

        pbar = tqdm.tqdm(total=epochs, initial=1)

        while not self._stop_training and epoch < epochs + 1:
            batch_d_loss = []
            batch_g_loss = []

            # Sample zero reconstruction and partial masking samples
            # zr_sample_indices = {}
            # pm_sample_indices = {}

            # for u in all_users:
            #     if scheme == 'ZP' or scheme == 'ZR':

            #         selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zr_ratio),
            #                                     replace=False)
            #         zr_sample_indices[u] = selected

            #     if scheme == 'ZP' or scheme == 'PM':
            #         selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zp_ratio),
            #                                     replace=False)
            #         pm_sample_indices[u] = selected
            zr_sample_indices, pm_sample_indices = compute_masks(all_users, not_selected, scheme, zr_ratio, zp_ratio)

            for _ in range(d_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + d_step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()

                    np_train_mask = np.empty_like(real_histories)
                    for i, u in enumerate(uids):
                        if scheme == 'ZP' or scheme == 'PM':
                            tmp = np.copy(real_histories[i])
                            tmp[pm_sample_indices[u]] = 1
                            np_train_mask[i] = tmp
                        else:
                            np_train_mask[i] = real_histories[i]

                    _, dloss = self.sess.run([d_train_op, d_loss],
                                             {self.condition: real_histories, real_data: real_histories,
                                              train_mask: np_train_mask})
                    batch_d_loss.append(dloss)
                    start_idx = end_idx

            for _ in range(g_steps):
                start_idx = 0
                while start_idx < len(all_users):
                    end_idx = start_idx + g_step
                    if end_idx > len(all_users):
                        end_idx = len(all_users)

                    uids = all_users[start_idx: end_idx]
                    real_histories = self.URM_train[uids].toarray()

                    np_train_mask = np.empty_like(real_histories)
                    np_zr_mask = np.zeros_like(real_histories)
                    for i, u in enumerate(uids):
                        if scheme == 'ZP' or scheme == 'PM':
                            tmp = np.copy(real_histories[i])
                            tmp[pm_sample_indices[u]] = 1
                            np_train_mask[i] = tmp
                        else:
                            np_train_mask[i] = real_histories[i]

                        tmp = np.zeros_like(real_histories[0])
                        if scheme == 'ZP' or scheme == 'ZR':
                            tmp[zr_sample_indices[u]] = 1
                            np_zr_mask[i] = tmp
                        else:
                            np_zr_mask[i] = tmp

                    _, gloss = self.sess.run([g_train_op, g_loss],
                                             {self.condition: real_histories, real_data: real_histories,
                                              train_mask: np_train_mask, zr_mask: np_zr_mask})
                    batch_g_loss.append(gloss)
                    start_idx = end_idx

            mean_epoch_g_loss = np.mean(batch_g_loss)
            mean_epoch_d_loss = np.mean(batch_d_loss)

            train_g_loss.append(mean_epoch_g_loss)
            train_d_loss.append(mean_epoch_d_loss)

            if validation_set is not None and sample_every is not None and epoch % sample_every == 0:
                t_end = time.time()
                total = t_end - e_start
                print('Epoch : {:d}. Total: {:.2f} secs, {:.2f} secs/epoch.'.format(epoch, total, total / sample_every))
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                _, results_run_string = validation_evaluator.evaluateRecommender(self)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                print(results_run_string)
                e_start = time.time()

            if validation_evaluator is not None:
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()
                early_stop(epoch)
                if self.mode == 'item':
                    self.URM_train = self.URM_train.T.tocsr()

                if self._stop_training:
                    print('Training stopped, epoch:', epoch)

            epoch += 1
            pbar.update()
        pbar.close()

        t_end = time.time()
        if self.verbose:
            print('Training took {:.2f} seconds'.format(t_end - t_start))

        if self.mode == 'item':
            self.URM_train = self.URM_train.T.tocsr()

        return epoch-1 if self._stop_training else epoch

    def stop_fit(self):
        self._stop_training = True

    def save_current_model(self):
        for model in self.params:
            save_weights(self.sess, self.params[model], self.best_params[model])

    def load_model(self):
        for model in self.best_params:
            save_weights(self.sess, self.best_params[model], self.params[model])

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        step = 128
        start_idx = 0
        if self.mode == 'item':
            out = np.empty((0, self.num_items))
            tmp_URM = self.URM_train.T.tocsr()
            while start_idx < self.num_users:
                end_idx = start_idx + step
                if end_idx > self.num_users:
                    end_idx = self.num_users

                predictions = self.sess.run(self.fake_profile, {self.condition: tmp_URM[start_idx: end_idx].toarray()})
                out = np.vstack((out, predictions))
                start_idx = end_idx
            return out.transpose()[user_id_array]
        else:
            out = np.empty((0, self.num_items))
            while start_idx < len(user_id_array):
                end_idx = start_idx + step
                if end_idx > len(user_id_array):
                    end_idx = len(user_id_array)

                predictions = self.sess.run(self.fake_profile, {
                    self.condition: self.URM_train[user_id_array[start_idx: end_idx]].toarray()})
                out = np.vstack((out, predictions))
                start_idx = end_idx
            return out

    def saveModel(self, folder_path, file_name):
        all_params = [var for k in self.params.keys() for var in self.params[k]]
        tf.train.Saver(all_params, max_to_keep=1).save(self.sess, os.path.join(folder_path, file_name), write_meta_graph=False, write_state=False)
