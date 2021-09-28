import numpy as np

from gonzales.simulator.space import Space
from gonzales.simulator.utils import calculate_relative_error
import gonzales.lib.brute_force as bf
import gonzales.lib.octree as oct


def vel_func(pos_vec):
    return np.array((0, 0, 0))


def mass_func(pos_vec):
    return 1.0


def test_brute_force_basic():
    # test correctness of Cython kernel
    G = 1.0
    eps = 0
    space = Space()
    # 1. zero acceleration cases
    # 1.1. single particle
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    accs = bf.calculate_accs_pp(space.r, space.m, G, eps)
    np.testing.assert_equal(accs, np.array([[0., 0., 0.]]))
    # 1.2. two particles at big distance
    space.add_particle(np.array((1.0e10, 1.0e15, 1.0e15)), np.array((0., 0., 0.)), 1.0)
    accs = bf.calculate_accs_pp(space.r, space.m, G, eps)
    np.testing.assert_almost_equal(accs, np.array([[0., 0., 0.], [0., 0., 0.]]))
    # 1.3. set G = 0
    accs = bf.calculate_accs_pp(space.r, space.m, 0., eps)
    np.testing.assert_equal(accs, np.array([[0., 0., 0.], [0., 0., 0.]]))
    # 1.4. two zero-mass particles
    space.clear_particles()
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 0.)
    space.add_particle(np.array((1., 0., 0.)), np.array((0., 0., 0.)), 0.)
    accs = bf.calculate_accs_pp(space.r, space.m, G, eps)
    np.testing.assert_equal(accs, np.array([[0., 0., 0.], [0., 0., 0.]]))

    # 2. two particles on x-axis at distance 1.0
    space.clear_particles()
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    space.add_particle(np.array((1., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    accs = bf.calculate_accs_pp(space.r, space.m, G, eps)
    np.testing.assert_almost_equal(accs, np.array([[1., 0., 0.], [-1., 0., 0.]]))
    # 3. two particles of unequal masses on x-axis at distance 1.0
    space.clear_particles()
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    space.add_particle(np.array((1., 0., 0.)), np.array((0., 0., 0.)), 0.5)
    accs = bf.calculate_accs_pp(space.r, space.m, G, eps)
    np.testing.assert_almost_equal(accs, np.array([[0.5, 0., 0.], [-1, 0., 0.]]))



def test_barnes_hut_theta_zero_C():
    # test Barnes-Hut C kernel relative to Cython brute force kernel, for theta = 0
    G = 1.0
    eps = 1.0e-3
    theta = 0.
    particle_nums = [2, 10, 100, 1000, 5000, 10000]    
    for num in particle_nums:
        cube_length = int(np.sqrt(num))
        space = Space()
        space.add_cuboid(num, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        accs1 = bf.calculate_accs_pp(space.r, space.m, G, eps)
        accs2 = oct.calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)
        np.testing.assert_almost_equal(accs1, accs2)



def test_barnes_hut_theta_non_zero_C():
    # test Barnes-Hut C kernel relative to Cython brute force kernel, for theta > 0
    G = 1.0
    eps = 1.0e-3
    theta = 0.5
    particle_nums = [2, 10, 100, 1000, 5000, 10000]
    for num in particle_nums:
        cube_length = int(np.sqrt(num))
        space = Space()
        space.add_cuboid(num, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        accs2 = bf.calculate_accs_pp(space.r, space.m, G, eps)
        accs3 = oct.calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)
        err, std_err = calculate_relative_error(accs3, accs2)
        assert err < 0.02 and std_err < 0.02
