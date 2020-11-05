import os
import time
import h5py
import numpy as np

import kernels.brute_force as kernbf
from simulator.utils import ProgressBar
from simulator.space import Space


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

    def set_kernel(self, kernel_func):
        """
            Sets the kernel function which will be used to calculate accelerations for each simulation
            step.

        :param kernel_func: function used to calculate accelerations
        """
        self._kernel = kernel_func

    def run(self, n_steps, step_size):
        """
            Runs the simulation for given number of steps and step size. Saves the results to the specified
            output_filepath.

        :param n_steps: number of simulation steps
        :param step_size: simulation step size
        :return:
        """
        raise NotImplementedError


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

    def run(self, n_steps, step_size):
        # reset Progress bar
        self._pb.reset(n_steps)
        with h5py.File(self.output_filepath, 'w') as f:
            # set initial hdf5 data
            info_grp = f.create_group('simulation_info')
            output_grp = f.create_group('simulation_output')
            self.set_metadata(info_grp, number_of_steps=n_steps, time_step_size=step_size, G=self.G, epsilon=self.eps,
                              start_time=time.time(), number_of_particles=len(self.space), simulation_type=self.type)
            pos_data = output_grp.create_dataset('positions', (n_steps + 1, *self.space.r.shape))
            vel_data = output_grp.create_dataset('velocities', (n_steps + 1, *self.space.v.shape))
            pos_data[0, :] = self.space.r
            vel_data[0, :] = self.space.v
            # integration
            for i in range(1, n_steps + 1):
                self._pb.update()
            info_grp.attrs['end_time'] = time.time()


if __name__ == '__main__':
    def vel_func(pos_vec):
        return np.array((1, 2, 3))


    def mass_func(pos_vec):
        return 1.0

    n = 100
    cube_length = 1.0
    G = 1.0
    eps = 1.0e-3
    theta = 0.75

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    ofp = os.path.normpath(r'D:\Python_Projects\results\test02.hdf5')
    sim = PPSimulation(space, ofp, 1.0, 1.0e-3)
    sim.run(100, 0.001)
