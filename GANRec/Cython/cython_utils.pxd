"""
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
"""

cdef void cdf(float[:] out, float[:] arr) nogil
cdef int argmax_slice(float[:] arr, int[:] selection) nogil
cdef void my_memview_slice(float[:] out, float[:] arr, int[:] selection) nogil
cdef void random_choice_nogil(int[:] out, float[:] cdf, int size=*, int seed=*) nogil
cdef void get_unobserved(int[:] out, int[:] pos_indices, int num_items) nogil