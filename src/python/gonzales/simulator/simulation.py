import os
import logging
import time
import h5py

import gonzales.lib.brute_force as bf
import gonzales.lib.octree as oct
import gonzales.lib.physics as phy
from gonzales.simulator.utils import ProgressBar

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


# ------------------------------------------------------
# Result reader
# ------------------------------------------------------

class ResultReader:
    """
    Reader for hdf5 results. Should be closed after using.
    """
    def __init__(self, hdf5_filepath):
        """

        :param hdf5_filepath: path to hdf5 file
        """
        if not h5py.is_hdf5(hdf5_filepath):
            raise ValueError('Not an hdf5 file: {}'.format(hdf5_filepath))
        self._fobj = h5py.File(hdf5_filepath, 'r')

    def get_info(self):
        """
        Returns simulation info, which is dict-like iterable.
        """
        return self._fobj['info']

    def get_result(self, result_name, step_num):
        """
        Get a specific result for a given step.

        :param result_name: str, name of result, must be supported by simulation
        :param step_num: step number
        """
        return self._fobj['results'][result_name][step_num]

    def get_result_frequency(self, result_name):
        """
        Get result frequency.

        :param result_name: str, name of result, must be supported by simulation
        
        Returns result frequency
        """        
        return self._fobj['results'][result_name].attrs['frequency']

    def get_result_num_steps(self, result_name):
        """

        :param result_name: str, name of result, must be supported by simulation
        
        Returns number of steps for which result is calculated
        """
        return self._fobj['results'][result_name].shape[0]

    def get_result_names(self):
        return list(self._fobj['results'].keys())

    def close(self):
        """
        Closes the hdf5 file. This should be called when the reader is not needed anymore.
        """
        if bool(self._fobj):
            self._fobj.close()


# ------------------------------------------------------
# Simulation classes
# ------------------------------------------------------

class ResultDesc:
    def __init__(self, shape, writer_func, frequency):
        self.shape = shape
        self.writer_func = writer_func
        self.frequency = frequency


class SimulationBase:
    def __init__(self, space, output_filepath, G, eps):
        """
        Base class for all simulation types.

        :param space: instance of Space
        :param output_filepath: output filepath
        :param G: gravitational constant
        :param eps: gravitational softening
        """
        self.space = space        
        self.output_filepath = output_filepath
        self._ensure_output_dir()
        self.G = G
        self.eps = eps
        self._pb = ProgressBar(1, 40)
        self._kernel = None
        self._result_descs = {'position': ResultDesc(self.space.r.shape, self._write_position, 1),
                              'velocity': ResultDesc(self.space.v.shape, self._write_velocity, 1),
                              'energy': ResultDesc((1,), self._write_total_energy, 0),
                              'angular_momentum': ResultDesc((3,), self._write_angular_momentum, 0)}

    def _ensure_output_dir(self):
        head, tail = os.path.split(self.output_filepath)
        if not head:
            return
        if not os.path.isdir(head):
            try:
                os.mkdir(head)
            except:
                raise OSError('Error creating directory: {}'.format(head))

    def add_result(self, res_name, res_frequency=1):
        """
        Adds a result which shall be written to the hdf5 file during the simulation. The shape
        of the result needs to be provided, as does the frequency of the output. The default res_frequency
        1 means the result will be written to the file at each simulation step.

        :param res_name: string, name of result
        :param res_frequency: int, result will be written every res_frequency steps. If 0, result will not
            be written at all.
        :return:
        """
        assert res_name in self._result_descs, 'Invalid result'
        self._result_descs[res_name].frequency = res_frequency

    def _write_position(self, hdf5_fobj, step_num):
        hdf5_fobj['results/position'][step_num, :] = self.space.r

    def _write_velocity(self, hdf5_fobj, step_num):
        hdf5_fobj['results/velocity'][step_num, :] = self.space.v

    def _write_total_energy(self, hdf5_fobj, step_num):
        hdf5_fobj['results/energy'][step_num] = phy.calc_te(self.space.r, self.space.v, self.space.m, self.G, self.eps)

    def _write_angular_momentum(self, hdf5_fobj, step_num):
        hdf5_fobj['results/angular_momentum'][step_num] = phy.calc_ang_mom(
            self.space.r, self.space.v, self.space.m)

    def _write_results(self, hdf5_fobj, step_num):
        for res_name, res_data in self._result_descs.items():
            res_freq, res_shape = res_data.frequency, res_data.shape
            if res_freq == 0:
                continue
            if step_num % res_freq == 0:
                res_data.writer_func(hdf5_fobj, int(step_num / res_freq))

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
        for res_name, res_data in self._result_descs.items():
            res_freq, res_shape = res_data.frequency, res_data.shape
            if res_freq == 0:
                continue
            n_rows = int(n_steps / res_freq)
            results_grp.create_dataset(res_name, (n_rows + 1, *res_shape))
            results_grp[res_name].attrs['frequency'] = res_freq

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
        """
        Runs the simulation.

        :param n_steps: number of steps
        :param step_size: step size
        """
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
                phy.advance_r(self.space.r, self.space.v, accs, step_size)
                new_accs = self.calc_accs()
                phy.advance_v(self.space.v, accs, new_accs, step_size)
                accs = new_accs
                # update hdf5 data
                self._write_results(res_f, i)
                self._pb.update()
            res_f['info']['end_time'] = time.time()
            res_f['info']['total_time'] = res_f['info']['end_time'][()] - res_f['info']['start_time'][()]
        logging.info('End simulation')


class PPSimulation(SimulationBase):
    """
    Simulation class for brute-force simulations.
    """
    def __init__(self, *args, **kwargs):
        """
        :param space: instance of Space
        :param output_filepath: output filepath
        :param G: gravitational constant
        :param eps: gravitational softening
        """
        super().__init__(*args, **kwargs)
        self.type = 'Brute force'
        self.set_kernel(bf.calculate_accs_pp)

    def __repr__(self):
        return '<{}[{}] N={}, type={}>'.format(type(self).__name__, id(self), len(self.space), self.type)

    def calc_accs(self):
        """
        Calculates the acceleration vector, using the kernel which was set.
        """
        return self._kernel(self.space.r, self.space.m, self.G, self.eps)


class BHSimulation(SimulationBase):
    """
    Simulation class for Barnes-Hut simulations.
    """
    def __init__(self, space, output_filepath, G, eps, root_width, root_center, theta):
        """
        :param space: instance of Space
        :param output_filepath: output filepath
        :param G: gravitational constant
        :param eps: gravitational softening
        :param root_width: octree root node width
        :param root_center: octree root node center
        :param theta: threshold parameter
        """
        super().__init__(space, output_filepath, G, eps)
        self.type = 'Barnes-Hut'
        self.root_width = root_width
        self.root_center = root_center
        self.theta = theta
        self.set_kernel(oct.calc_accs_octree)

    def __repr__(self):
        return '<{}[{}] N={}, type={}>'.format(type(self).__name__, id(self), len(self.space), self.type)

    def calc_accs(self):
        """
        Calculates the acceleration vector, using the kernel which was set.
        """
        return self._kernel(self.root_width, *self.root_center, self.space.r, self.space.m, self.G, self.eps,
                            self.theta)
