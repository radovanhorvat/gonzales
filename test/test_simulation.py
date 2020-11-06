import os
import h5py
import shutil
import numpy as np
from copy import deepcopy

from simulator.space import Space
import simulator.simulation as sim


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _make_results_dir():
    dir_path = os.path.dirname(os.path.join(os.path.abspath(__file__)))
    temp_dir = os.path.join(dir_path, 'temp_results')
    os.mkdir(temp_dir)
    return temp_dir


def _compare_results(sim1, sim2, n_steps):
    with h5py.File(sim1.output_filepath, 'r') as pp, h5py.File(sim2.output_filepath, 'r') as bh:
        pos_data_pp = pp['results/positions']
        pos_data_bh = bh['results/positions']
        vel_data_pp = pp['results/velocities']
        vel_data_bh = bh['results/velocities']
        for k in range(n_steps + 1):
            print("step: ", k)
            np.testing.assert_equal(pos_data_bh[k],  pos_data_pp[k])
            np.testing.assert_equal(vel_data_bh[k], vel_data_pp[k])


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

def vel_func(pos_vec):
    return np.array((0, 0, 0))


def mass_func(pos_vec):
    return 1.0


def test_sim_barnes_hut_theta_zero():
    # test Barnes-Hut C++ simulation relative to Cython brute force simulation, for theta = 0
    res_dir = _make_results_dir()
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
    _compare_results(sim_pp, sim_bh, n_steps)
    shutil.rmtree(res_dir)
