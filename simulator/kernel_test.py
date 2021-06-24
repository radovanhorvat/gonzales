import numpy as np
import functools
import timeit

from simulator.space import Space
import kernels.brute_force as kernbf
import kernels.octree as kernoct
import kernels.numeric as kernum
import kernels.octree_c as kernoct_c


if __name__ == '__main__':
    def vel_func(pos_vec):
        return np.array((1, 2, 3))


    def mass_func(pos_vec):
        return 1.0

    n = 1000000
    cube_length = 2.0
    G = 1.0
    eps = 1.0e-3
    theta = 0.75

    #ofp = r'D:\Python_Projects\input_files\cuboid_10_6.hdf5'
    space = Space()
    #space.from_hdf5(ofp)
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    #accs3 = kernbf.calculate_accs_pp_wrap(space.r, space.m, G, eps)
    #accs3 = kernoct.calc_accs_octree_wrap(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)

    #t = timeit.Timer(functools.partial(kernoct.calc_accs_octree_wrap, cube_length, 0., 0., 0., space.r, space.m, G, eps, theta))
    
    t = timeit.Timer(functools.partial(kernoct_c.calc_accs_wrap_wrap_c, n, space.r, space.m, G, eps, theta, cube_length, 0., 0., 0))
    
    print(t.timeit(1))
