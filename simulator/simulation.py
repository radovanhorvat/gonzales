import os


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

    @staticmethod
    def set_metadata(hdf5_obj, **kwargs):
        """
            Sets metadata attributes for hdf5 object (can be a group or a dataset)
        """
        for k, v in kwargs.items():
            hdf5_obj[k] = v

    def run(self, n_steps, step_size):
        raise NotImplementedError


class PPSimulation(SimulationBase):
    """
        Simulation class for brute-force.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'Brute force'

    def __repr__(self):
        return '<{}[{}] N={}, type={}>'.format(type(self).__name__, id(self), len(self.space), self.type)

    def run(self, n_steps, step_size):
        pass


if __name__ == '__main__':
    s = PPSimulation('a', 'b', G=1.0, eps=1.0e-3)
    print(s)
