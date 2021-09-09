import numpy as np
import functools
import timeit

from nbody.simulator.space import Space
import nbody.lib.octree as oct


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

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
    t = timeit.Timer(functools.partial(oct.calc_accs_octree, cube_length, 0., 0., 0., space.r, space.m, G, eps, theta))
    print(t.timeit(1))
