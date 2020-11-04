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

# -------- old stuff

@timing
def calc_com_old(pos_vec, mass_vec):
    n, m = pos_vec.shape
    center_of_mass = np.zeros((n, m))
    for i in range(m):
        center_of_mass[:, i] = pos_vec[:, i] * mass_vec
    return center_of_mass.sum(axis=0) / mass_vec.sum()


@timing
def calc_pe_old(pos_vec, mass_vec, epsilon, G=6.67e-11):
    n, m = pos_vec.shape
    interaction_matrix = np.zeros((n, n))
    for i in range(n - 1):
        R_1 = np.ones((n - (i + 1), m)) * pos_vec[i]
        R_2 = pos_vec[i + 1:]
        R = R_2 - R_1
        interaction_matrix[i, i + 1:] = 1 / (np.linalg.norm(R, axis=1) + epsilon)
    return -G * np.dot(interaction_matrix @ mass_vec, mass_vec)

@timing
def calc_ke_old(vel_vec, mass_vec):
    return 0.5 * (vel_vec ** 2).sum(axis=1) @ mass_vec

# -------- new stuff

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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_ke(DTYPE_t [:, :] v, DTYPE_t [:] m):
    """
        Calculate kinetic energy.
    """
    cdef int n = v.shape[0]
    cdef int i
    cdef double ke = 0

    for i in range(n):
        ke += m[i] * (v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1] + v[i, 2] * v[i, 2])
    return 0.5 * ke


@cython.boundscheck(False)
@cython.wraparound(False)
cdef calc_pe(DTYPE_t [:, :] r, DTYPE_t [:] m, double G, double eps):
    """
        Calculate potential energy.
    """
    cdef int n = r.shape[0]
    cdef int i, j
    cdef double dx, dy, dz, ds, d
    cdef double pe = 0

    for i in prange(n, nogil=True):
        for j in range(i + 1, n):
            dx = r[j, 0] - r[i, 0]
            dy = r[j, 1] - r[i, 1]
            dz = r[j, 2] - r[i, 2]
            ds = dx * dx + dy * dy + dz * dz
            d = sqrt(ds)
            pe += m[i] * m[j] / (d + eps)
    return - G * pe


# ---------------------------------------------------
# Wrappers
# ---------------------------------------------------

@timing
def calc_com_wrap(r, m):
    return calc_com(r, m)


@timing
def calc_ke_wrap(v, m):
    return calc_ke(v, m)


@timing
def calc_pe_wrap(r, m, G, eps):
    return calc_pe(r, m, G, eps)