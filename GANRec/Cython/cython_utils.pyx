"""
@author Ervin Dervishaj
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

import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free, rand, srand, RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
def get_non_interactions(int[:] all_users, URM_train):
    not_selected = {}
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t max_users = all_users.shape[0]
    cdef int u
    for i in range(max_users):
        u = all_users[i]
        not_selected[u] = np.nonzero(URM_train[u].toarray()[0] == 0)[0]
    return not_selected



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def compute_masks(int[:] all_users, dict not_selected, str scheme, float zr_ratio, float zp_ratio):
    zr_sample_indices = {}
    pm_sample_indices = {}
    cdef Py_ssize_t i = 0
    cdef int u
    cdef Py_ssize_t max_users = all_users.shape[0]
    for i in range(max_users):
        u = all_users[i]
        if scheme == 'ZP' or scheme == 'ZR':
            selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zr_ratio),
                                        replace=False)
            zr_sample_indices[u] = selected

        if scheme == 'ZP' or scheme == 'PM':
            selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zr_ratio),
                                        replace=False)
            pm_sample_indices[u] = selected

    return zr_sample_indices, pm_sample_indices



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def random_choice(float[:, :] cdf, float[:, :] cdf2=None, int size=1, int[:] custom_ordered_rows=None):
    cdef int i, r, len_rows
    cdef float a
    cdef int[:] chosen_items, chosen_items2, rows
    cdef float[:] samples

    if cdf2 is not None:
        assert cdf.shape[0] == cdf2.shape[0]

    if custom_ordered_rows is None:
        rows = np.repeat(np.arange(cdf.shape[0]), size).astype(np.int32)
    else:
        rows = custom_ordered_rows

    len_rows = rows.shape[0]
    samples = np.random.random(len_rows if cdf2 is None else len_rows * 2).astype(np.float32)

    chosen_items = np.zeros(len_rows, dtype=np.int32)
    chosen_items2 = np.zeros(len_rows, dtype=np.int32)

    for i in prange(len_rows, nogil=True):
        a = samples[i]
        r = rows[i]
        chosen_items[i] = binarysearch(cdf[r], a)
        if cdf2 is not None:
            a = samples[i + len_rows]
            chosen_items2[i] = binarysearch(cdf2[r], a)

    if cdf2 is not None:
        return np.ascontiguousarray(chosen_items), np.ascontiguousarray(chosen_items2)
    return np.ascontiguousarray(chosen_items)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_unobserved(int[:] out, int[:] pos_indices, int num_items) nogil:
    cdef:
        int i, j = 0

    for i in range(num_items):
        if i < pos_indices[0] or i > pos_indices[pos_indices.shape[0] - 1]:
            out[j] = i
            j += 1
        else:
            if binarysearch_int(pos_indices, i) == -1:
                out[j] = i
                j += 1



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void random_choice_nogil(int[:] out, float[:] cdf, int size=1, int seed=1234) nogil:
    cdef:
        float sample
        int idx

    srand(seed)
    for idx in range(size):
        sample = rand()
        out[idx] = binarysearch(cdf, sample)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int binarysearch_int(int[:] arr, int value) nogil:
    cdef int start_idx = 0
    cdef int end_idx = arr.shape[0]
    cdef int mid_idx

    while start_idx < end_idx:
        mid_idx = (start_idx + end_idx) / 2
        if arr[mid_idx] == value:
            return mid_idx
        
        if arr[mid_idx] < value:
            start_idx = mid_idx + 1
        else:
            end_idx = mid_idx
    
    return -1



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int binarysearch(float[:] cdf, float value) nogil:
    cdef int start_idx = 0
    cdef int end_idx = cdf.shape[0]
    cdef int mid_idx

    while start_idx < end_idx:
        mid_idx = (start_idx + end_idx) / 2
        if cdf[mid_idx] < value:
            start_idx = mid_idx + 1
        else:
            end_idx = mid_idx
    
    if start_idx >= cdf.shape[0]:
        return cdf.shape[0] - 1
    else:
        return start_idx



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void my_memview_slice(float[:] out, float[:] arr, int[:] selection) nogil:
    cdef:
        int i

    for i in range(selection.shape[0]):
        out[i] = arr[selection[i]]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef int argmax_slice(float[:] arr, int[:] selection) nogil:
    cdef:
        int idx, jdx, max_idx=0

    for idx in range(1, selection.shape[0]):
        if arr[selection[idx]] > arr[selection[max_idx]]:
            max_idx = idx

    return selection[max_idx]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef float sum(float[:] arr) nogil:
    cdef:
        int idx
        float _sum = 0
    
    for idx in range(arr.shape[0]):
        _sum += arr[idx]

    return _sum



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void cdf(float[:] out, float[:] arr) nogil:
    cdef:
        int idx
        float p
        float _sum = sum(arr)

    out[0] = arr[0] / _sum
    for idx in range(1, arr.shape[0]):
        p = arr[idx] / _sum
        out[idx] = out[idx - 1] + p
