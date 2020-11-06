import os
import logging
import time
import h5py
import numpy as np

import kernels.brute_force as kernbf
import kernels.octree as kernoct
import kernels.numeric as kernum
from simulator.utils import ProgressBar
from simulator.space import Space


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


class SimulationBase:
    def __init__(self, space, output_filepath, G, eps):
        """
        Base class for all simulation types.

        :param space: instance of Space
        :param G: gravitational constant
        :param eps: gravitational softening
        """
        self.space = space
        self.output_filepath = output_filepath
        self.G = G
        self.eps = eps
        self._pb = ProgressBar(1, 30)
        self._kernel = None

    @staticmethod
    def set_metadata(hdf5_obj, **kwargs):
        """
            Sets metadata attributes for hdf5 object (can be a group or a dataset)
        """
        for k, v in kwargs.items():
            hdf5_obj[k] = v

    def set_initial_data(self, hdf5_obj, n_steps, step_size):
        """
            Creates basic hdf5 structure and sets some initial data.
        """
        info_grp = hdf5_obj.create_group('info')
        self.set_metadata(info_grp, number_of_steps=n_steps, time_step_size=step_size, G=self.G, epsilon=self.eps,
                          start_time=time.time(), number_of_particles=len(self.space), simulation_type=self.type)
        results_grp = hdf5_obj.create_group('results')
        pos_data = results_grp.create_dataset('positions', (n_steps + 1, *self.space.r.shape))
        vel_data = results_grp.create_dataset('velocities', (n_steps + 1, *self.space.v.shape))
        pos_data[0, :] = self.space.r
        vel_data[0, :] = self.space.v

    def set_kernel(self, kernel_func):
        """
            Sets the kernel function which will be used to calculate accelerations for each simulation
            step.

        :param kernel_func: function used to calculate accelerations
        """
        self._kernel = kernel_func

    def calc_accs(self):
        raise NotImplementedError

    def run(self, n_steps, step_size):
        # reset Progress bar
        self._pb.reset(n_steps)
        logging.info('Start simulation - type={}, N_particles={}, N_steps={}'.format(self.type, len(self.space),
                                                                                     n_steps))
        with h5py.File(self.output_filepath, 'w') as res_f:
            # set initial hdf5 data
            self.set_initial_data(res_f, n_steps, step_size)
            # calculate initial accelerations
            accs = self.calc_accs()
            # integration
            for i in range(1, n_steps + 1):
                kernum.advance_r_wrap(self.space.r, self.space.v, accs, step_size)
                new_accs = self.calc_accs()
                kernum.advance_v_wrap(self.space.v, accs, new_accs, step_size)
                accs = new_accs
                # update hdf5 data
                res_f['results/positions'][i, :] = self.space.r
                res_f['results/velocities'][i, :] = self.space.v
                self._pb.update()
            res_f['info']['end_time'] = time.time()
        logging.info('End simulation')


class PPSimulation(SimulationBase):
    """
        Simulation class for brute-force.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'Brute force'
        self.set_kernel(kernbf.calculate_accs_pp_wrap)

    def __repr__(self):
        return '<{}[{}] N={}, type={}>'.format(type(self).__name__, id(self), len(self.space), self.type)

    def calc_accs(self):
        return self._kernel(self.space.r, self.space.m, self.G, self.eps)


class BHSimulation(SimulationBase):
    """
        Simulation class for Barnes-Hut.
    """
    def __init__(self, space, output_filepath, G, eps, root_width, root_center, theta):
        super().__init__(space, output_filepath, G, eps)
        self.type = 'Barnes-Hut'
        self.root_width = root_width
        self.root_center = root_center
        self.theta = theta
        self.set_kernel(kernoct.calc_accs_octree_wrap)

    def __repr__(self):
        return '<{}[{}] N={}, type={}>'.format(type(self).__name__, id(self), len(self.space), self.type)

    def calc_accs(self):
        return self._kernel(self.root_width, *self.root_center, self.space.r, self.space.m, self.G, self.eps,
                            self.theta)


if __name__ == '__main__':
    def vel_func(pos_vec):
        return np.array((0., 0., 0.))


    def mass_func(pos_vec):
        return 1.0

    n = 100
    cube_length = 1.0
    G = 1.0
    eps = 1.0e-3
    theta = 0.
    n_steps = 10
    step_size = 0.01

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    ofp_pp = os.path.normpath(r'D:\Python_Projects\results\test_pp.hdf5')
    ofp_bh = os.path.normpath(r'D:\Python_Projects\results\test_bh.hdf5')

    sim_pp = PPSimulation(space, ofp_pp, G, eps)
    from copy import deepcopy
    space2 = deepcopy(space)
    sim_bh = BHSimulation(space2, ofp_bh, G, eps, 1.0, np.array((0., 0., 0.)), theta)

    sim_pp.run(n_steps, step_size)
    sim_bh.run(n_steps, step_size)

    # result comparsion
    with h5py.File(sim_pp.output_filepath, 'r') as pp, h5py.File(sim_bh.output_filepath, 'r') as bh:
        pos_data_pp = pp['results/positions']
        pos_data_bh = bh['results/positions']
        vel_data_pp = pp['results/velocities']
        vel_data_bh = bh['results/velocities']
        for k in range(n_steps + 1):
            print("step: ", k)
            np.testing.assert_equal(pos_data_bh[k],  pos_data_pp[k])
            np.testing.assert_equal(vel_data_bh[k], vel_data_pp[k])
