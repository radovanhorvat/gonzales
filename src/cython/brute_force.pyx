cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cimport openmp

ctypedef np.float64_t DTYPE_t


# ---------------------------------------------------
# Cython brute force
# ---------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _calculate_accs_pp(DTYPE_t [:, :] r, DTYPE_t[:] m, double G, double eps):
	cdef int n = r.shape[0]
	cdef int k = r.shape[1]
	cdef DTYPE_t [:, :] accs = np.zeros([n, k])
	cdef int i, j, p
	cdef double d, f, dx, dy, dz, ds, k1, k2
	cdef int n_threads = openmp.omp_get_max_threads()
	cdef int start_ind, stop_ind
	cdef int chunk_size = int(n / n_threads)
	cdef DTYPE_t [:, :, :] temp = np.zeros([n, 6, k])

	for p in prange(n_threads, nogil=True):
		start_ind = p * chunk_size
		stop_ind = (p + 1) * chunk_size if p < n_threads - 1 else n
		for i in range(start_ind, stop_ind):
			k1 = m[i] * G
			for j in range(i + 1, n):
				k2 = m[j] * G
				dx = r[j, 0] - r[i, 0]
				dy = r[j, 1] - r[i, 1]
				dz = r[j, 2] - r[i, 2]
				ds = dx * dx + dy * dy + dz * dz
				d = sqrt(ds)
				f = 1.0 / (d * d * d + eps)
				# update accs
				accs[i, 0] += dx * f * k2
				accs[i, 1] += dy * f * k2
				accs[i, 2] += dz * f * k2
				# update temporary array - we need this to avoid thread race
				temp[j, p, 0] -= dx * f * k1
				temp[j, p, 1] -= dy * f * k1
				temp[j, p, 2] -= dz * f * k1

	accs += np.sum(temp, axis=1)

	return np.asarray(accs)


# ---------------------------------------------------
# Wrappers
# ---------------------------------------------------

def calculate_accs_pp(r, m, G, eps):
	"""
	Parallel brute-force calculation of accelerations.

	:param r: N x 3 numpy array, position vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:param G: gravitational constant
	:param eps: gravitational softening
	:return: N x 3 numpy array, acceleration vectors of particles
	"""
	return _calculate_accs_pp(r, m, G, eps)
