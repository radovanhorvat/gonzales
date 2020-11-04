import numpy as np

from simulator.space import Space
from simulator.utils import calculate_relative_error
from kernels.brute_force import calculate_accs_pp_wrap, calc_accs_pp_cpp_wrap
from kernels.octree import calc_accs_octree


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
    accs = calculate_accs_pp_wrap(space.r, space.m, G, eps)
    assert np.allclose(accs, np.array((0., 0., 0.)))
    # 1.2. two particles at big distance
    space.add_particle(np.array((1.0e10, 1.0e10, 1.0e10)), np.array((0., 0., 0.)), 1.0)
    accs = calculate_accs_pp_wrap(space.r, space.m, G, eps)
    assert np.allclose(accs, np.array((0., 0., 0.)))
    # 1.3. set G = 0
    accs = calculate_accs_pp_wrap(space.r, space.m, 0., eps)
    assert np.allclose(accs, np.array((0., 0., 0.)))
    # 1.4. two zero-mass particles
    space.clear_particles()
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 0.)
    space.add_particle(np.array((1., 0., 0.)), np.array((0., 0., 0.)), 0.)
    accs = calculate_accs_pp_wrap(space.r, space.m, G, eps)
    assert np.allclose(accs, np.array([[0., 0., 0.], [0., 0., 0.]]))

    # 2. two particles on x-axis at distance 1.0
    space.clear_particles()
    space.add_particle(np.array((0., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    space.add_particle(np.array((1., 0., 0.)), np.array((0., 0., 0.)), 1.0)
    accs = calculate_accs_pp_wrap(space.r, space.m, G, eps)
    assert np.allclose(accs, np.array([[1., 0., 0.], [-1., 0., 0.]]))


def test_brute_force_cpp():
    # test brute force C++ kernel relative to brute force Cython kernel
    G = 1.0
    eps = 1.0e-3
    particle_nums = [2, 10, 100, 1000, 5000, 10000]
    for num in particle_nums:
        cube_length = int(np.sqrt(num))
        space = Space()
        space.add_cuboid(num, np.array((0, 0, 0)), cube_length, cube_length, cube_length, vel_func, mass_func)
        accs1 = calculate_accs_pp_wrap(space.r, space.m, G, eps)
        accs2 = calc_accs_pp_cpp_wrap(space.r, space.m, G, eps)
        assert np.allclose(accs1, accs2)


def test_barnes_hut_theta_zero():
    # test Barnes-Hut C++ kernel relative to Cython brute force kernel, for theta = 0
    G = 1.0
    eps = 1.0e-3
    theta = 0.
    particle_nums = [2, 10, 100, 1000, 5000, 10000]
    for num in particle_nums:
        cube_length = int(np.sqrt(num))
        space = Space()
        space.add_cuboid(num, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        accs2 = calculate_accs_pp_wrap(space.r, space.m, G, eps)
        accs3 = calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)
        assert np.allclose(accs2, accs3)


def test_barnes_hut_theta_non_zero():
    # test Barnes-Hut C++ kernel relative to Cython brute force kernel, for theta > 0
    G = 1.0
    eps = 1.0e-3
    theta = 0.5
    particle_nums = [2, 10, 100, 1000, 5000, 10000]
    for num in particle_nums:
        cube_length = int(np.sqrt(num))
        space = Space()
        space.add_cuboid(num, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        accs2 = calculate_accs_pp_wrap(space.r, space.m, G, eps)
        accs3 = calc_accs_octree(cube_length, 0., 0., 0., space.r, space.m, G, eps, theta)
        err, std_err = calculate_relative_error(accs3, accs2)
        assert err < 0.02 and std_err < 0.02
