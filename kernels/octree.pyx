cimport cython
cimport numpy as np
import numpy as np
from simulator.utils import timing
from libc.math cimport sqrt

ctypedef np.float64_t DTYPE_t


cdef extern from "octree.h" namespace "octree":
    cdef cppclass Octree:

        Octree(double x, double y, double z, double w, double u_G, double u_eps, double u_theta, double* points,
		       double* masses, int n_points)
        void build()
        void calculate_accs()
        void calculate_accs_st()
        void calculate_accs_st_parallel()
        double a_x, a_y, a_z
        int num_nodes, num_leaves, n
        double* accs


cdef class CPPOctree:
    cdef Octree *_thisptr

    def __cinit__(self, double x, double y, double z, double w, double u_G, double u_eps, double u_theta, DTYPE_t [:, :] points,
		          DTYPE_t [:] masses, int n_points):
        self._thisptr = new Octree(x, y, z, w, u_G, u_eps, u_theta, &points[0, 0], &masses[0], n_points)

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    cdef info(self):
        return self._thisptr.num_nodes, self._thisptr.num_leaves

    cdef accs(self):
        x = np.asarray(<DTYPE_t[:self._thisptr.n, :3]> self._thisptr.accs)
        return x

    cdef void build(self):
        self._thisptr.build()

    cdef void calculate_accs(self):
        self._thisptr.calculate_accs()

    cdef void calculate_accs_st(self):
        self._thisptr.calculate_accs_st()

    cdef void calculate_accs_st_parallel(self):
        self._thisptr.calculate_accs_st_parallel()


cdef CPPOctree build_tree(double w, double x, double y, double z, DTYPE_t [:, :] r, DTYPE_t [:] m, double G, double eps, double theta):
    cdef CPPOctree tree
    tree = CPPOctree(x, y, z, w, G, eps, theta, r, m, r.shape[0])
    tree.build()
    #print(tree.info())
    return tree


cdef calc_accs(CPPOctree tree):
    #tree.calculate_accs()
    #tree.calculate_accs_st()
    tree.calculate_accs_st_parallel()
    return tree.accs()


# ---------------------------------------------------
# Wrappers
# ---------------------------------------------------

# for timing:

@timing
def build_tree_wrap(w, x, y, z, r, m, G, eps, theta):
    return build_tree(w, x, y, z, r, m, G, eps, theta)

@timing
def calc_accs_wrap(tree):
    return calc_accs(tree)

@timing
def calc_accs_octree(w, x, y, z, r, m, G, eps, theta):
    tree = build_tree_wrap(w, x, y, z, r, m, G, eps, theta)
    return calc_accs_wrap(tree)

# for simulation:

def calc_accs_octree_wrap(w, x, y, z, r, m, G, eps, theta):
    tree = build_tree(w, x, y, z, r, m, G, eps, theta)
    return calc_accs(tree)
