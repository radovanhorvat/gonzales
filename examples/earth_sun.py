import os
import sys
import h5py
import numpy as np

from simulator.space import Space
from simulator.simulation import PPSimulation, ResultReader, BHSimulation

from ui.qt_test import run_viewer


if __name__ == '__main__':

    G = 6.67408e-11
    eps = 0.
    n_steps = 365
    step_size = 86400

    space = Space()
    space.add_particle(np.zeros(3), np.zeros(3), 1.989e30)
    space.add_particle(np.array([1.496e11, 0., 0.]), np.array([0., 29780, 0.]), 5.972e24)

    ofp = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output', 'earth_sun.hdf5'))
    sim = PPSimulation(space, ofp, G, eps)
    #sim = BHSimulation(space, ofp, G, eps, 2e11, np.array((0., 0., 0.)), 0.)
    sim.add_result('energy')
    sim.add_result('angular_momentum')
    sim.run(n_steps, step_size)

    run_viewer(ofp, body_sizes=[30, 10])
