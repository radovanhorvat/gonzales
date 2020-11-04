import numpy as np

from simulator.space import Space
from kernels.brute_force import calculate_accs_pp_wrap
from kernels.octree import calc_accs_octree
from kernels.numeric import calc_com_old, calc_com_wrap

if __name__ == '__main__':
    def vel_func(pos_vec):
        return np.array((0, 0, 0))


    def mass_func(pos_vec):
        return 1.0

    n = 100000
    cube_length = 1.0
    G = 1.0
    eps = 1.0e-3
    theta = 0.75

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    x = calc_com_old(space.r, space.m)
    y = calc_com_wrap(space.r, space.m)
    assert np.allclose(x, y, atol=1e-15, rtol=1e-10)

    #accs2 = calculate_accs_pp_wrap(space.r, space.m, G, eps)
    #accs3 = calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)
