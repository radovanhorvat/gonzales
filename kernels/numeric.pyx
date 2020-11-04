cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
from simulator.utils import timing
cimport openmp

ctypedef np.float64_t DTYPE_t


# ---------------------------------------------------
# Numerical kernels
# ---------------------------------------------------

@timing
def calc_com_old(pos_vec, mass_vec):
    n, m = pos_vec.shape
    center_of_mass = np.zeros((n, m))
    for i in range(m):
        center_of_mass[:, i] = pos_vec[:, i] * mass_vec
    return center_of_mass.sum(axis=0) / mass_vec.sum()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_com(DTYPE_t [:, :] r, DTYPE_t [:] m):
    """
        Calculate center of mass.
    """
    cdef int n = r.shape[0]
    cdef DTYPE_t [:] com = np.zeros(3)
    cdef int i
    cdef double mass = 0

    for i in range(n):
        mass += m[i]
        com[0] += m[i] * r[i, 0]
        com[1] += m[i] * r[i, 1]
        com[2] += m[i] * r[i, 2]
    com[0] /= mass
    com[1] /= mass
    com[2] /= mass
    return np.asarray(com)


# ---------------------------------------------------
# Wrappers
# ---------------------------------------------------

@timing
def calc_com_wrap(r, m):
    return calc_com(r, m)
