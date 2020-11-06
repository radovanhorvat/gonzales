import numpy as np

from simulator.space import Space
import kernels.brute_force as kernbf
import kernels.octree as kernoct
import kernels.numeric as kernum


if __name__ == '__main__':
    def vel_func(pos_vec):
        return np.array((1, 2, 3))


    def mass_func(pos_vec):
        return 1.0

    n = 10000
    cube_length = 1.0
    G = 1.0
    eps = 1.0e-3
    theta = 0.75

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)


    #accs3 = kernbf.calculate_accs_pp_wrap(space.r, space.m, G, eps)
    accs3 = kernoct.calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)

    x = kernum.advance_pp_old(space, 0.5, accs3, space.m, G, eps)
    y = kernum.advance_pp_wrap(space.r, space.v, space.m, accs3, 0.5, G, eps)