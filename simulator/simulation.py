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


# ------------------------------------------------------
# Simulation classes
# ------------------------------------------------------

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
        self._pb = ProgressBar(1, 40)
        self._kernel = None
        self._results = {'position': (1, self.space.r.shape), 'velocity': (1, self.space.v.shape)}
        self._result_writer_map = {'position': self._write_position,
                                   'velocity': self._write_velocity,
                                   'energy': self._write_total_energy,
                                   'angular_momentum': self._write_angular_momentum}

    def add_result(self, res_name, res_shape, res_frequency=1):
        """
        Adds a result which shall be written to the hdf5 file during the simulation. The shape
        of the result needs to be provided, as does the frequency of the output. The default res_frequency
        1 means the result will be written to the file at each simulation step.

        :param res_name: string, name of result
        :param res_shape: tuple, shape of result
        :param res_frequency: int, result will be written every res_frequency steps
        :return:
        """
        assert res_name in self._result_writer_map, 'Invalid result'
        self._results[res_name] = (res_frequency, res_shape)

    def _write_position(self, hdf5_fobj, step_num):
        hdf5_fobj['results/position'][step_num, :] = self.space.r

    def _write_velocity(self, hdf5_fobj, step_num):
        hdf5_fobj['results/velocity'][step_num, :] = self.space.v

    def _write_total_energy(self, hdf5_fobj, step_num):
        hdf5_fobj['results/energy'][step_num] = kernum.calc_te_wrap(self.space.r, self.space.v, self.space.m, self.G,
                                                                      self.eps)

    def _write_angular_momentum(self, hdf5_fobj, step_num):
        hdf5_fobj['results/angular_momentum'][step_num] = kernum.calc_ang_mom_wrap(
            self.space.r, self.space.v, self.space.m, self.G)

    def _write_results(self, hdf5_fobj, step_num):
        for res_name, res_data in self._results.items():
            res_freq, res_shape = res_data
            if step_num % res_freq == 0:
                self._result_writer_map[res_name](hdf5_fobj, int(step_num / res_freq))

    @staticmethod
    def set_metadata(hdf5_obj, **kwargs):
        """
        Sets metadata attributes for hdf5 object (can be a group or a dataset)
        """
        for k, v in kwargs.items():
            hdf5_obj[k] = v

    def create_datasets(self, hdf5_fobj, n_steps, step_size):
        """
        Creates hdf5 datasets.
        """
        info_grp = hdf5_fobj.create_group('info')
        self.set_metadata(info_grp, number_of_steps=n_steps, time_step_size=step_size, G=self.G, epsilon=self.eps,
                          start_time=time.time(), number_of_particles=len(self.space), simulation_type=self.type)
        results_grp = hdf5_fobj.create_group('results')
        for res_name, res_desc in self._results.items():
            res_freq, res_shape = res_desc
            n_rows = int(n_steps / res_freq)
            results_grp.create_dataset(res_name, (n_rows + 1, *res_shape))

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
        with h5py.File(self.output_filepath, 'w') as res_f:
            # set initial hdf5 data
            logging.info('Simulation - type={}, N_particles={}, N_steps={}'.format(self.type, len(self.space),
                                                                                   n_steps))
            logging.info('Creating datasets')
            self.create_datasets(res_f, n_steps, step_size)
            logging.info('Writing initial data')
            self._write_results(res_f, 0)
            # calculate initial accelerations
            logging.info('Calculating initial accelerations')
            accs = self.calc_accs()
            logging.info('Start simulation')
            # integration
            for i in range(1, n_steps + 1):
                kernum.advance_r_wrap(self.space.r, self.space.v, accs, step_size)
                new_accs = self.calc_accs()
                kernum.advance_v_wrap(self.space.v, accs, new_accs, step_size)
                accs = new_accs
                # update hdf5 data
                self._write_results(res_f, i)
                self._pb.update()
            res_f['info']['end_time'] = time.time()
            res_f['info']['total_time'] = res_f['info']['end_time'][()] - res_f['info']['start_time'][()]
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

    n = 1000
    cube_length = np.sqrt(n)
    G = 1.0
    eps = 1.0e-3
    theta = 0.75
    n_steps = 1000
    step_size = 0.001

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    ofp_bh = os.path.normpath(r'D:\Python_Projects\results\test_bh.hdf5')
    sim_bh = BHSimulation(space, ofp_bh, G, eps, 10 * cube_length, np.array((0., 0., 0.)), theta)
    sim_bh.add_result('energy', (1,), n_steps)
    sim_bh.run(n_steps, step_size)

    with h5py.File(sim_bh.output_filepath, 'r') as bh:
        init = bh['results']['energy'][0]
        fin = bh['results']['energy'][1]
        print(init, fin)
        print(np.abs(init - fin) / np.abs(init))
