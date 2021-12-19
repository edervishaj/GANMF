"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

import cython
from cython.parallel import prange
from libc.math cimport exp, log, pow
from libc.stdlib cimport malloc, free

import numpy as np
from GANRec.Cython.cython_utils cimport cdf, argmax_slice, random_choice_nogil, my_memview_slice, get_unobserved


@cython.cdivision(True)
cdef inline float sigmoid(float x):
    return 1.0 / (1.0 + exp(-x))

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline float l2_loss(float[:] x):
    cdef int k
    cdef float loss = 0.0
    for k in range(x.shape[0]):
        loss += pow(x[k], 2)
    return loss


# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.initializedcheck(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
cdef class IRGAN_Cython:

    cdef:
        int num_users, num_items, num_factors, seed
        float[:, :] gen_user_factors, gen_item_factors, dis_user_factors, dis_item_factors
        float[:] gen_item_bias, dis_item_bias
        object early_stopper

    def __init__(self, int num_users, int num_items, int num_factors=10, init_delta=0.05, int seed=1234, early_stopper=None):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.seed = seed
        self.early_stopper = early_stopper

        self.gen_user_factors = np.random.uniform(-init_delta, init_delta, (self.num_users, self.num_factors)).astype(np.float32)
        self.gen_item_factors = np.random.uniform(-init_delta, init_delta, (self.num_items, self.num_factors)).astype(np.float32)
        self.gen_item_bias = np.zeros(self.num_items, dtype=np.float32)

        self.dis_user_factors = np.random.uniform(-init_delta, init_delta, (self.num_users, self.num_factors)).astype(np.float32)
        self.dis_item_factors = np.random.uniform(-init_delta, init_delta, (self.num_items, self.num_factors)).astype(np.float32)
        self.dis_item_bias = np.zeros(self.num_items, dtype=np.float32)

    def fit(self, URM_train, early_stopper, int epochs=300, int pre_train_epochs=100, int batch_size=32, int DNS_K=5,
        float DNS_lr=0.05, float D_lr=1e-4, float G_lr=1e-4, int d_steps=1, int g_steps=1, float temperature=0.2,
        float disc_reg=1e-4, float gen_reg=1e-4):

        cdef:
            int epoch
            int[:] indptr = URM_train.indptr, indices = URM_train.indices
        
        ######################
        # PRETRAIN GENERATOR #
        ######################

        for epoch in range(pre_train_epochs):
            # dynamic negative sampling
            users, pos, neg = self.dynamic_negative_sample(indptr, indices, DNS_K)

        
    def dynamic_negative_sample(self, int[:] indptr, int[:] indices, int DNS_K=1):
        cdef:
            int[:] users = np.empty(indices.shape[0], dtype=np.int32)
            int[:] positive_items = np.empty(indices.shape[0], dtype=np.int32)
            int[:] neg_items = np.empty(indices.shape[0], dtype=np.int32)

            int u, i, idx, neg_size, start_idx=0
            int[:] pos, unobserved_indices
            int[:] selection = np.empty(DNS_K, dtype=np.int32)
            float[:] _cdf, user_ratings
            float[:, :] all_ratings = self.compute_scores()

        for u in range(indptr.shape[0] - 1):
            pos = indices[indptr[u]: indptr[u + 1]]
            user_ratings = all_ratings[u]
            unobserved_indices = np.empty(self.num_items - pos.shape[0], dtype=np.int32)
            get_unobserved(unobserved_indices, pos, self.num_items)
            _cdf = np.empty(unobserved_indices.shape[0], dtype=np.float32)
            my_memview_slice(_cdf, user_ratings, unobserved_indices)
            for idx in range(pos.shape[0]):
                random_choice_nogil(selection, _cdf, DNS_K, self.seed)
                users[start_idx + idx] = u
                positive_items[start_idx + idx] = pos[idx]
                neg_items[start_idx + idx] = argmax_slice(user_ratings, selection)
            start_idx += pos.shape[0]

        return (np.asarray(users), np.asarray(positive_items), np.asarray(neg_items))

    def dns_update_step(self, int u, int i, int j, float DNS_lr, float reg, str model='generator'):
        """
        The update step is:

        w = w + learning_rate * \
                \delta log(
                    sigmoid(
                        sum_{k in num_factors}(
                            user_factors[u,k] * (item_factors[i, k] - item_factors[j, k])
                        ) + (item_bias[i] - item_bias[j])
                    ) + || user_factors[u] ||^2 + || item_factors[i] ||^2 + || item_factors[j] ||^2 + item_bias[i]^2 + item_bias[j]^2
                ) / \delta w
        
        for triple u, i, j
        """

        cdef:
            int k
            float x_uij = 0.0, sig_xuij, curr_user_factor, curr_pos_factor, curr_neg_factor, curr_pos_bias, curr_neg_bias, loss = 0.0

        assert model in ('generator', 'discriminator'), model + ' not in (generator, discriminator)'

        # compute 
        if model == 'discriminator':
            for k in range(self.num_factors):
                x_uij += self.dis_user_factors[u, k] * (self.dis_item_factors[i, k] - self.dis_item_factors[j, k]) + \
                        self.dis_item_bias[i] - self.dis_item_bias[j]
            sig_xuij = sigmoid(-x_uij)

            loss += -sig_xuij

            # update the latent factors
            for k in range(self.num_factors):
                # copy the current weights
                curr_user_factor = self.dis_user_factors[u, k]
                curr_pos_factor = self.dis_item_factors[i, k]
                curr_neg_factor = self.dis_item_factors[j, k]
                curr_pos_bias = self.dis_item_bias[i]
                curr_neg_bias = self.dis_item_bias[j]

                self.dis_user_factors[u, k] += DNS_lr * ((1 - sig_xuij) * (curr_pos_factor - curr_neg_factor) + 2 * reg * curr_user_factor)
                self.dis_item_factors[i, k] += DNS_lr * ((1 - sig_xuij) * curr_user_factor + 2 * reg * curr_pos_factor)
                self.dis_item_factors[j, k] += DNS_lr * ((1 - sig_xuij) * -curr_user_factor + 2 * reg * curr_neg_factor)
                self.dis_item_bias[i] += DNS_lr * ((1 - sig_xuij) + 2 * reg * curr_pos_bias)
                self.dis_item_bias[j] += DNS_lr * ((1 - sig_xuij) + 2 * reg * curr_neg_bias)

        else:
            for k in range(self.num_factors):
                x_uij += self.gen_user_factors[u, k] * (self.gen_item_factors[i, k] - self.gen_item_factors[j, k]) + \
                        self.gen_item_bias[i] - self.gen_item_bias[j]
            sig_xuij = sigmoid(-x_uij)

            loss += -sig_xuij

            # update the latent factors
            for k in range(self.num_factors):
                # copy the current weights
                curr_user_factor = self.gen_user_factors[u, k]
                curr_pos_factor = self.gen_item_factors[i, k]
                curr_neg_factor = self.gen_item_factors[j, k]
                curr_pos_bias = self.gen_item_bias[i]
                curr_neg_bias = self.gen_item_bias[j]

                self.gen_user_factors[u, k] += DNS_lr * ((1 - sig_xuij) * (curr_pos_factor - curr_neg_factor) + 2 * reg * curr_user_factor)
                self.gen_item_factors[i, k] += DNS_lr * ((1 - sig_xuij) * curr_user_factor + 2 * reg * curr_pos_factor)
                self.gen_item_factors[j, k] += DNS_lr * ((1 - sig_xuij) * -curr_user_factor + 2 * reg * curr_neg_factor)
                self.gen_item_bias[i] += DNS_lr * ((1 - sig_xuij) + 2 * reg * curr_pos_bias)
                self.gen_item_bias[j] += DNS_lr * ((1 - sig_xuij) + 2 * reg * curr_neg_bias)

        return loss


    def compute_scores(self, model='generator'):
        cdef:
            cdef int u, i, k
            cdef float[:, :] scores = np.zeros((self.num_users, self.num_items), dtype=np.float32)

        assert model in ('generator', 'discriminator'), model + ' not in (generator, discriminator)'

        if model == 'discriminator':
            for u in prange(self.num_users, nogil=True):
                for i in range(self.num_items):
                    for k in range(self.num_factors):
                        scores[u, i] += self.dis_user_factors[u, k] * self.dis_item_factors[i, k]
                    scores[u, i] += self.dis_item_bias[i]
        else:
            for u in prange(self.num_users, nogil=True):
                for i in range(self.num_items):
                    for k in range(self.num_factors):
                        scores[u, i] += self.gen_user_factors[u, k] * self.gen_item_factors[i, k]
                    scores[u, i] += self.gen_item_bias[i]
        
        return scores
    