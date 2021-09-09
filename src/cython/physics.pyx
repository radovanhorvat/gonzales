cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from cython.parallel import prange
cimport openmp


ctypedef np.float64_t DTYPE_t


# ---------------------------------------------------
# Physics functions
# ---------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _calc_com(DTYPE_t [:, :] r, DTYPE_t [:] m):
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
cdef _calc_ke(DTYPE_t [:, :] v, DTYPE_t [:] m):
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
cdef _calc_pe(DTYPE_t [:, :] r, DTYPE_t [:] m, double G, double eps):
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _calc_te(DTYPE_t [:, :] r, DTYPE_t [:, :] v, DTYPE_t [:] m, double G, double eps):
	"""
		Calculates total energy.
	"""
	cdef double te
	te = _calc_pe(r, m, G, eps) + _calc_ke(v, m)
	return te


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _calc_ang_momentum(DTYPE_t [:, :] r, DTYPE_t [:, :] v, DTYPE_t [:] m):
	"""
		Calculates angular momentum.
	"""
	cdef int n = r.shape[0]
	cdef DTYPE_t [:] angm = np.zeros(3)
	cdef int i

	for i in range(n):
		angm[0] += m[i] * (r[i, 1] * v[i, 2] - r[i, 2] * v[i, 1])
		angm[1] += m[i] * (r[i, 2] * v[i, 0] - r[i, 0] * v[i, 2])
		angm[2] += m[i] * (r[i, 0] * v[i, 1] - r[i, 1] * v[i, 0])
	return angm


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _advance_r(DTYPE_t [:, :] r, DTYPE_t [:, :] v, DTYPE_t [:, :] accs, double dt):
	"""
		Updates positions for time step dt.
	"""
	cdef int n = accs.shape[0]
	cdef int i

	for i in range(n):
		r[i, 0] +=  v[i, 0] * dt + 0.5 * accs[i, 0] * dt * dt
		r[i, 1] +=  v[i, 1] * dt + 0.5 * accs[i, 1] * dt * dt
		r[i, 2] +=  v[i, 2] * dt + 0.5 * accs[i, 2] * dt * dt


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _advance_v(DTYPE_t [:, :] v, DTYPE_t [:, :] accs, DTYPE_t [:, :] new_accs, double dt):
	"""
		Updates velocities for time step dt.
	"""
	cdef int n = accs.shape[0]
	cdef int i

	for i in range(n):
		v[i, 0] +=  0.5 * (accs[i, 0] + new_accs[i, 0]) * dt
		v[i, 1] +=  0.5 * (accs[i, 1] + new_accs[i, 1]) * dt
		v[i, 2] +=  0.5 * (accs[i, 2] + new_accs[i, 2]) * dt


# ---------------------------------------------------
# Wrappers
# ---------------------------------------------------

def calc_com(r, m):
	"""
	Calculates center of mass of a set of particles.

	:param r: N x 3 numpy array, position vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:return: 1 x 3 numpy array, center of mass
	"""
	return _calc_com(r, m)


def calc_ke(v, m):
	"""
	Calculates kinetic energy of a system of particles.

	:param v: N x 3 numpy array, velocity vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:return: double, kinetic energy of the system
	"""
	return _calc_ke(v, m)


def calc_pe(r, m, G, eps):
	"""
	Calculates potential energy of a system of particles.

	:param r: N x 3 numpy array, position vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:param G: gravitational constant
	:param eps: gravitational softening
	:return: double, potential energy of the system
	"""
	return _calc_pe(r, m, G, eps)


def calc_te(r, v, m, G, eps):
	"""
	Calculates total mechanical energy of a system of particles.

	:param r: N x 3 numpy array, position vectors of the particles
	:param v: N x 3 numpy array, velocity vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:param G: gravitational constant
	:param eps: gravitational softening
	:return: double, total mechanical energy of the system
	"""
	return _calc_te(r, v, m, G, eps)


def calc_ang_mom(r, v, m):
	"""
	Calculates angular momentum of a system of particles.

	:param r: N x 3 numpy array, position vectors of the particles
	:param v: N x 3 numpy array, velocity vectors of the particles
	:param m: N x 1 numpy array, mass vector of the particles
	:return: 1 x 3 numpy array, angular momentum vector
	"""
	return _calc_ang_momentum(r, v, m)


def advance_r(r, v, accs, dt):
	_advance_r(r, v, accs, dt)


def advance_v(v, accs, new_accs, dt):
	_advance_v(v, accs, new_accs, dt)
