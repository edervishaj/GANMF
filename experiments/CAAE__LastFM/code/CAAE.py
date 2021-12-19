#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

import os
import sys
import time
import tqdm
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
from Base.BaseRecommender import BaseRecommender
from Utils_ import EarlyStoppingScheduler, dense_spmatrix, save_weights
from GANRec.Cython.cython_utils import get_non_interactions, random_choice


class CAAE(BaseRecommender):
    RECOMMENDER_NAME = 'CAAE'

    def __init__(self, URM_train, verbose=False, mode='user', seed=1234, is_experiment=False):
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

    def build(self, num_factors=10, g_layers=1, g_units=20, gpr_layers=1, gpr_units=20, beta=1e-4, lmbda=0.5):
        glorot_uniform = tf.glorot_uniform_initializer()

        ##############
        # D FUNCTION #
        ##############
        def disc(user_id, real_item, fake_item):
            with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
                user_embeddings = tf.get_variable(shape=[self.num_users, num_factors], trainable=True,
                                                  name='disc_user_embeddings', initializer=glorot_uniform,
                                                  dtype=tf.float32)

                item_embeddings = tf.get_variable(shape=[self.num_items, num_factors], trainable=True,
                                                  name='disc_item_embeddings', initializer=glorot_uniform,
                                                  dtype=tf.float32)

                item_bias = tf.get_variable(shape=[self.num_items], trainable=True, name='disc_item_bias')

            user_lookup = tf.nn.embedding_lookup(user_embeddings, user_id)
            real_item_lookup = tf.nn.embedding_lookup(item_embeddings, real_item)
            fake_item_lookup = tf.nn.embedding_lookup(item_embeddings, fake_item)
            real_item_bias = tf.gather(item_bias, real_item)
            fake_item_bias = tf.gather(item_bias, fake_item)

            pre_logits = tf.reduce_sum(tf.multiply(user_lookup, real_item_lookup - fake_item_lookup), 1) + (
                        real_item_bias - fake_item_bias)

            disc_loss = -tf.reduce_mean(tf.log(tf.sigmoid(pre_logits))) + beta * (tf.nn.l2_loss(user_lookup) +
                                                                                  tf.nn.l2_loss(real_item_lookup) +
                                                                                  tf.nn.l2_loss(fake_item_lookup) +
                                                                                  tf.nn.l2_loss(real_item_bias) +
                                                                                  tf.nn.l2_loss(fake_item_bias))

            reward_logits = tf.reduce_sum(tf.multiply(user_lookup, fake_item_lookup), 2) + fake_item_bias
            g_reward = tf.log(tf.sigmoid(reward_logits - 1))
            gpr_reward = tf.log(tf.sigmoid(1 - reward_logits))

            return disc_loss, g_reward, gpr_reward

        ##############
        # G FUNCTION #
        ##############
        def gen(user_profile, e_mask, reward, fake_item_id):
            input_len = user_profile.get_shape().as_list()[1]
            with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
                g = user_profile
                for l in range(g_layers):
                    g = tf.layers.dense(g, units=g_units, kernel_initializer=glorot_uniform, activation='sigmoid',
                                        name='g_layer_' + str(l))
                reconstruction = tf.layers.dense(g, units=input_len, kernel_initializer=glorot_uniform,
                                                 activation='sigmoid', name='g_reconstruction')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

            ae_loss = tf.reduce_sum(tf.reduce_sum(tf.square((reconstruction - user_profile) * e_mask), 1))
            fake_item_prob = tf.reshape(tf.gather_nd(tf.nn.softmax(reconstruction), fake_item_id), tf.shape(reward))
            gen_loss = -lmbda * tf.reduce_mean(tf.log(fake_item_prob) * reward) + (1 - lmbda) * ae_loss + \
                       beta * tf.add_n([tf.nn.l2_loss(var) for var in g_vars])
            return reconstruction, gen_loss

        ###############
        # G' FUNCTION #
        ###############
        def gen_prime(user_profile, reward, fake_item_id):
            input_len = user_profile.get_shape().as_list()[1]
            with tf.variable_scope('G_prime', reuse=tf.AUTO_REUSE):
                g = user_profile
                for l in range(gpr_layers):
                    g = tf.layers.dense(g, units=gpr_units, kernel_initializer=glorot_uniform, activation='sigmoid',
                                        name='gpr_layer_' + str(l))
                reconstruction = tf.layers.dense(g, units=input_len, kernel_initializer=glorot_uniform,
                                                 activation='sigmoid', name='gpr_reconstruction')
            gpr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_prime')
            fake_item_prob = tf.reshape(tf.gather_nd(tf.nn.softmax(reconstruction), fake_item_id), tf.shape(reward))
            gen_prime_loss = -tf.reduce_mean(tf.log(fake_item_prob) * reward) + \
                             beta * tf.add_n([tf.nn.l2_loss(var) for var in gpr_vars])
            return reconstruction, gen_prime_loss

        self.D, self.G, self.G_prime = disc, gen, gen_prime

    def fit(self, epochs=300, d_steps=1, g_steps=1, gpr_steps=1, g_layers=1, g_units=20, gpr_layers=1, gpr_units=20,
            num_factors=10, d_bsize=1024, m_batch=32, lmbda=0.5, beta=1e-4, lr=1e-4, S=0.3, allow_worse=None, freq=None,
            after=0, metrics=['MAP'], sample_every=None, validation_evaluator=None, validation_set=None):

        # Construct the model config
        self.config = dict(locals())
        del self.config['self']

        # First clear the session to save GPU memory
        tf.reset_default_graph()
        # Set fixed seed for the TF graph
        tf.set_random_seed(self.seed)

        # self.build(num_factors, g_layers, g_units, gpr_layers, gpr_units, beta, lmbda)
        self.build(num_factors, g_layers, g_units, g_layers, g_units, beta, lmbda)

        # Optimizers
        g_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gprime_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        d_opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

        # placeholders
        self.real_profile = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        ae_mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])
        user_id = tf.placeholder(dtype=tf.int32)
        real_item_id = tf.placeholder(dtype=tf.int32)
        fake_item_id = tf.placeholder(dtype=tf.int32)
        reward = tf.placeholder(dtype=tf.float32)

        # Forward ops
        dloss, g_reward, gpr_reward = self.D(user_id, real_item_id, fake_item_id)
        self.g_reconstruction, g_loss = self.G(self.real_profile, ae_mask, reward, fake_item_id)
        gpr_reconstruction, gprime_loss = self.G_prime(self.real_profile, reward, fake_item_id)

        # model parameters
        self.params = {}
        self.params['D'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        self.params['G'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
        self.params['G_prime'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_prime')

        self.best_params = {}
        for p in self.params:
            self.best_params[p] = []
            for idx, var in enumerate(self.params[p]):
                self.best_params[p].append(tf.get_variable(p + '_best_params_' + str(idx), shape=var.get_shape(),
                                                           trainable=False))

        # update ops
        dtrain = d_opt.minimize(dloss, var_list=self.params['D'])
        gtrain = g_opt.minimize(g_loss, var_list=self.params['G'])
        gprime_train = gprime_opt.minimize(gprime_loss, var_list=self.params['G_prime'])

        ##################
        # START TRAINING #
        ##################

        # DO NOT allocate all GPU memory to this process
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())

        self._stop_training = False
        if validation_evaluator is not None:
            early_stop = EarlyStoppingScheduler(self, evaluator=validation_evaluator, allow_worse=allow_worse,
                                                freq=freq, metrics=metrics, after=after)

        all_users = np.arange(self.num_users, dtype=np.int32)
        all_items = np.arange(self.num_items, dtype=np.int32)

        all_interactions = [(u, i) for u in all_users
                            for i in self.URM_train.indices[self.URM_train.indptr[u]: self.URM_train.indptr[u + 1]]]
        non_interactions = get_non_interactions(all_users, self.URM_train)

        median_interactions = np.median(np.ediff1d(self.URM_train.indptr)).astype(np.int32)

        user_profiles = dense_spmatrix(self.URM_train)

        train_d_loss = []
        train_pg_loss = []
        train_ng_loss = []

        if self.verbose:
            print('Starting training...')

        t_start = time.time()
        e_start = time.time()

        epoch = 1

        pbar = tqdm.tqdm(total=epochs, initial=1)

        while not self._stop_training and epoch < epochs + 1:
            batch_d_loss = []
            batch_pg_loss = []
            batch_ng_loss = []

            np.random.shuffle(all_interactions)

            interaction_users, interaction_items = list(zip(*all_interactions))

            # Twice as many users and positive items since D needs triples for items from G and G'
            users = np.array(list(interaction_users))
            positive_items = np.array(list(interaction_items))

            g_profiles = self.sess.run(self.g_reconstruction, feed_dict={self.real_profile: user_profiles})
            gprime_profiles = self.sess.run(gpr_reconstruction, feed_dict={self.real_profile: user_profiles})

            # Compute CDF for item sampling from G
            exp_g_profiles = np.exp(g_profiles)
            exp_g_sum = exp_g_profiles.sum(axis=1)
            g_item_prob = exp_g_profiles / exp_g_sum[:, np.newaxis]
            g_item_cdf = np.cumsum(g_item_prob, axis=1)

            # Compute CDF for item sampling from G'
            exp_gprime_profiles = np.exp(gprime_profiles)
            exp_gprime_sum = exp_gprime_profiles.sum(axis=1)
            gprime_item_prob = exp_gprime_profiles / exp_gprime_sum[:, np.newaxis]
            gprime_item_cdf = np.cumsum(gprime_item_prob, axis=1)

            for _ in range(d_steps):
                g_neg_items, gpr_neg_items = random_choice(g_item_cdf, gprime_item_cdf, custom_ordered_rows=users)

                start_idx = 0
                while start_idx < len(all_interactions):
                    end_idx = start_idx + d_bsize
                    if end_idx > len(all_interactions):
                        end_idx = len(all_interactions)

                    u = users[start_idx: end_idx]
                    pos_i = positive_items[start_idx: end_idx]

                    # Train D with negative items sampled from G
                    neg_i = g_neg_items[start_idx: end_idx]
                    _, _dloss = self.sess.run([dtrain, dloss],
                                              feed_dict={user_id: u, real_item_id: pos_i, fake_item_id: neg_i})
                    batch_d_loss.append(_dloss)

                    # Train D with negative items sampled from G'
                    neg_i = gpr_neg_items[start_idx: end_idx]
                    _, _dloss = self.sess.run([dtrain, dloss],
                                              feed_dict={user_id: u, real_item_id: pos_i, fake_item_id: neg_i})
                    batch_d_loss.append(_dloss)

                    start_idx = end_idx

            for _ in range(g_steps):
                uids = np.random.choice(all_users, size=m_batch, replace=False)

                # Get real profiles and AE masks
                u_profiles = user_profiles[uids]
                e_masks = user_profiles[uids]

                # For each user
                for i, u in enumerate(uids):
                    u_neg_items = non_interactions[u]
                    sum_probs = gprime_item_prob[u, u_neg_items].sum()
                    Nu = np.random.choice(u_neg_items, size=int(len(u_neg_items) * S),
                                          p=gprime_item_prob[u, u_neg_items] / sum_probs, replace=False)

                    # Update both real profile and AE mask
                    # u_profiles[i][Nu] = 0     # This is not necessary since non-interactions are already zero
                    e_masks[i][Nu] = 1

                # Sample 2 * median_interactions from G for user
                g_recon_prof = self.sess.run(self.g_reconstruction, feed_dict={self.real_profile: u_profiles})
                exp_profiles = np.exp(g_recon_prof)
                exp_sum = exp_profiles.sum(axis=1)
                prob = exp_profiles / exp_sum[:, np.newaxis]
                cdf = np.cumsum(prob, axis=1)
                g_items = random_choice(cdf, size=2*median_interactions).reshape(m_batch, -1)

                # Compute reward
                reward_val = self.sess.run(g_reward, feed_dict={user_id: uids.reshape(-1, 1), fake_item_id: g_items})

                # Convert negative items to indices for tf.gather_nd
                g_items = np.concatenate((
                                    np.tile(np.arange(m_batch).reshape(-1, 1), 2*median_interactions).reshape(-1, 1),
                                    g_items.reshape(-1, 1)), axis=1).reshape(-1, 2)

                # Update G
                _, _gen_loss = self.sess.run([gtrain, g_loss], feed_dict={self.real_profile: u_profiles,
                                                                          ae_mask: e_masks,
                                                                          reward: reward_val,
                                                                          fake_item_id: g_items})

                batch_pg_loss.append(_gen_loss)

            for _ in range(gpr_steps):
                uids = np.random.choice(all_users, size=m_batch)
                u_profiles = user_profiles[uids]

                # Sample 2 * median_interactions from G' for user
                gpr_recon_prof = self.sess.run(gpr_reconstruction, feed_dict={self.real_profile: u_profiles})
                exp_profiles = np.exp(gpr_recon_prof)
                exp_sum = exp_profiles.sum(axis=1)
                prob = exp_profiles / exp_sum[:, np.newaxis]
                cdf = np.cumsum(prob, axis=1)
                gpr_items = random_choice(cdf, size=2*median_interactions).reshape(m_batch, -1)

                # Compute reward
                reward_val = self.sess.run(gpr_reward,
                                           feed_dict={user_id: uids.reshape(-1, 1), fake_item_id: gpr_items})

                # Convert negative items to indices for tf.gather_nd
                gpr_items = np.concatenate((
                                np.tile(np.arange(m_batch).reshape(-1, 1), 2*median_interactions).reshape(-1, 1),
                                gpr_items.reshape(-1, 1)), axis=1).reshape(-1, 2)

                # Update G
                _, _gen_prime_loss = self.sess.run([gprime_train, gprime_loss], feed_dict={self.real_profile: u_profiles,
                                                                                           reward: reward_val,
                                                                                           fake_item_id: gpr_items})

                batch_ng_loss.append(_gen_prime_loss)

            mean_epoch_pg_loss = np.mean(batch_pg_loss)
            mean_epoch_ng_loss = np.mean(batch_ng_loss)
            mean_epoch_d_loss = np.mean(batch_d_loss)

            train_pg_loss.append(mean_epoch_pg_loss)
            train_ng_loss.append(mean_epoch_ng_loss)
            train_d_loss.append(mean_epoch_d_loss)

            if validation_set is not None and sample_every is not None and epoch % sample_every == 0:
                t_end = time.time()
                total = t_end - e_start
                print('Epoch : {:d}. Total: {:.2f} secs, {:.2f} secs/epoch.'.format(epoch, total, total / sample_every))
                _, results_run_string = validation_evaluator.evaluateRecommender(self)
                print(results_run_string)
                e_start = time.time()

            if validation_evaluator is not None:
                early_stop(epoch)

                if self._stop_training:
                    print('Training stopped, epoch:', epoch)

            epoch += 1
            pbar.update()
        pbar.close()

        t_end = time.time()
        if self.verbose:
            print('Training took {:.2f} seconds'.format(t_end - t_start))

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
        out = np.empty((0, self.num_items))
        step = 512
        start_idx = 0
        while start_idx < len(user_id_array):
            end_idx = start_idx + step
            if end_idx > len(user_id_array):
                end_idx = len(user_id_array)

            reconstruction = self.sess.run(self.g_reconstruction,
                                           {self.real_profile: self.URM_train[start_idx: end_idx].toarray()})
            out = np.vstack((out, reconstruction))
            start_idx = end_idx
        return out
