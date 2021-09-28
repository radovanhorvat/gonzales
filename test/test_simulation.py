import os
import h5py
import shutil
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

from gonzales.simulator.space import Space
import gonzales.simulator.simulation as sim


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

@contextmanager
def make_temp_result_dir(dir_name):
    dir_path = os.path.dirname(os.path.join(os.path.abspath(__file__)))
    temp_dir = os.path.join(dir_path, dir_name)
    os.mkdir(temp_dir)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def _are_results_equal(sim1, sim2, n_steps):
    with h5py.File(sim1.output_filepath, 'r') as pp, h5py.File(sim2.output_filepath, 'r') as bh:
        pos_data_pp = pp['results/position']
        pos_data_bh = bh['results/position']
        vel_data_pp = pp['results/velocity']
        vel_data_bh = bh['results/velocity']
        for k in range(n_steps + 1):
            np.testing.assert_equal(pos_data_bh[k],  pos_data_pp[k])
            np.testing.assert_equal(vel_data_bh[k], vel_data_pp[k])


def _are_results_close(sim1, sim2, n_steps):
    with h5py.File(sim1.output_filepath, 'r') as pp, h5py.File(sim2.output_filepath, 'r') as bh:
        pos_data_pp = pp['results/position']
        pos_data_bh = bh['results/position']
        vel_data_pp = pp['results/velocity']
        vel_data_bh = bh['results/velocity']
        for k in range(n_steps + 1):
            assert _are_vecs_close(pos_data_bh[k], pos_data_pp[k])
            assert _are_vecs_close(vel_data_bh[k], vel_data_pp[k])


def _are_vecs_close(vec, vec_ref, rtol=1e-2, atol=1e-15):
    """

    :param vec: estimated vec
    :param vec_ref: vec to which we compare (correct one)
    :param rtol: relative tolerance
    :param atol: absolute tolerance
    :return: bool
    """
    x = np.mean(np.linalg.norm(vec - vec_ref, axis=1))
    y = np.mean(np.linalg.norm(vec_ref, axis=1) * rtol + atol)
    return x <= y


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def vel_func(pos_vec):
    return np.array((0, 0, 0))


def mass_func(pos_vec):
    return 1.0


def test_sim_barnes_hut_theta_zero():
    # test Barnes-Hut simulation relative to Cython brute force simulation, for theta = 0
    with make_temp_result_dir('temp_results') as res_dir:
        pp_file = os.path.join(res_dir, 'test_pp.hdf5')
        bh_file = os.path.join(res_dir, 'test_bh.hdf5')
        G = 1.0
        eps = 1.0e-3
        theta = 0.
        n_steps = 100
        step_size = 0.01
        n = 1000
        cube_length = int(np.sqrt(n))
        space = Space()
        space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        space1 = deepcopy(space)
        sim_pp = sim.PPSimulation(space, pp_file, G, eps)
        sim_bh = sim.BHSimulation(space1, bh_file, G, eps, cube_length, np.array((0., 0., 0.)), theta)
        sim_pp.run(n_steps, step_size)
        sim_bh.run(n_steps, step_size)
        # result comparsion
        _are_results_equal(sim_pp, sim_bh, n_steps)


def test_sim_barnes_hut_theta_non_zero():
    # test Barnes-Hut simulation relative to Cython brute force simulation, for theta = 0.5
    with make_temp_result_dir('temp_results') as res_dir:
        pp_file = os.path.join(res_dir, 'test_pp.hdf5')
        bh_file = os.path.join(res_dir, 'test_bh.hdf5')
        G = 1.0
        eps = 1.0e-3
        theta = 0.5
        n_steps = 100
        step_size = 0.01
        n = 1000
        cube_length = int(np.sqrt(n))
        space = Space()
        space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        space1 = deepcopy(space)
        sim_pp = sim.PPSimulation(space, pp_file, G, eps)
        sim_bh = sim.BHSimulation(space1, bh_file, G, eps, 10 * cube_length, np.array((0., 0., 0.)), theta)
        sim_pp.run(n_steps, step_size)
        sim_bh.run(n_steps, step_size)
        # result comparsion
        _are_results_close(sim_pp, sim_bh, n_steps)


def test_energy_conservation_barnes_hut():
    # test Barnes-Hut energy conservation
    with make_temp_result_dir('temp_results') as res_dir:
        bh_file = os.path.join(res_dir, 'test_bh.hdf5')
        n = 1000
        G = 1.0
        eps = 1.0e-3
        theta = 0.75
        n_steps = 1000
        step_size = 0.001
        cube_length = int(np.sqrt(n))
        space = Space()
        space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
        sim_bh = sim.BHSimulation(space, bh_file, G, eps, 10 * cube_length, np.array((0., 0., 0.)), theta)
        sim_bh.add_result('energy', n_steps)
        sim_bh.run(n_steps, step_size)
        with h5py.File(sim_bh.output_filepath, 'r') as bh:
            e_in = bh['results']['energy'][0]
            e_fin = bh['results']['energy'][1]
            assert np.abs(e_in - e_fin) / np.abs(e_in) <= 0.01
