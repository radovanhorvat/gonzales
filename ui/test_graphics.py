import os
import sys
import h5py
import numpy as np

from simulator.space import Space
from simulator.simulation import PPSimulation, BHSimulation, ResultReader

from ui.qt_test import run_viewer


if __name__ == '__main__':

    def vel_func(pos_vec):
        return np.array((0., 0., 0.))

    # def mass_func(pos_vec):
    #     return 1.0
    def mass_func(pos_vec):
        r = np.linalg.norm(pos_vec)
        return 4. / 3 * r**3 * np.pi * 1.0e-4

    # space = Space()
    # space.add_cuboid(10000, np.array((0, 0, 0)), 1, 1, 1, vel_func, mass_func)
    # space.add_cylinder(10000, np.array((5, 0, 0)), 1, 0.1, vel_func, mass_func)
    # space.add_sphere(10000, np.array((0, -5, 0)), 1, vel_func, mass_func)
    #
    # p = Points3DPlot(space)
    # p.show()

    n = 1000
    cube_length = np.sqrt(n)
    G = 1.0
    eps = 1.0e-3
    theta = 1.0
    n_steps = 100
    step_size = 0.001

    space = Space()
    #space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)
    space.add_sphere(n, np.array((0., 0., 0.)), 1., vel_func, mass_func)
    #space.add_sphere(n, np.array((-2., 0., 0.)), .5, vel_func, mass_func)

    ofp = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output', 'test_01.hdf5'))
    #s1 = PPSimulation(space, ofp, G, eps)
    s1 = BHSimulation(space, ofp, G, eps, 100000., np.array((0., 0., 0.)), theta)
    s1.add_result('velocity', (n, 3), res_frequency=50)
    s1.add_result('energy', (1,))
    s1.add_result('angular_momentum', (3,))
    s1.run(n_steps, step_size)

    #r = ResultReader(s1.output_filepath)
    #print(r.get_dataset_names())
    # x = r.get_result('velocity', 10)
    # print(x)
    #r.close()

    #anim = Points3DAnimation(ofp)
    #anim.animate()
    run_viewer(ofp)
