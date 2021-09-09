cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt

ctypedef np.float64_t DTYPE_t


cdef extern from "octnode.h":
	double* calc_accs_wrap(int n, double* points, double* masses, double G, double eps, double theta, double root_width, double root_x, double root_y, double root_z)


cdef _calc_accs(int n, DTYPE_t [:, :] r, DTYPE_t [:] m, double G, double eps, double theta, double w, double r_x, double r_y, double r_z):
	cdef double* accs
	accs = calc_accs_wrap(n, &r[0, 0], &m[0], G, eps, theta, w, r_x, r_y, r_z) 
	accs_py = np.asarray(<DTYPE_t[:n, :3]> accs)
	return accs_py


def calc_accs_octree(w, r_x, r_y, r_z, r, m, G, eps, theta):
	return _calc_accs(r.shape[0], r, m, G, eps, theta, w, r_x, r_y, r_z)
