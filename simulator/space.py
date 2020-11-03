import numpy as np

from simulator.utils import timing


class Space:
    """
    Class used to represent a 3D space which contains particles.
    """
    def __init__(self):
        self.r = np.empty(shape=(0, 3))
        self.v = np.empty(shape=(0, 3))
        self.m = np.array([])

    def add_particle(self, r, v, m):
        """
        Add a single particle with the specified position, velocity and mass.
        :param r: numpy array, position vector of the particle
        :param v: numpy array, velocity vector of the particle
        :param m: float, particle mass
        """
        self.r = np.vstack((self.r, r))
        self.v = np.vstack((self.v, v))
        self.m = np.append(self.m, m)

    @timing
    def add_cuboid(self, n, center, l_x, l_y, l_z, v_func, m_func):
        """
        Generates uniform random particle distribution within a cuboid volume
        with given center and side lengths.

        :param n: number of particles to generate
        :param center: center of cuboid volume
        :param l_x: length of cuboid in the x-direction
        :param l_y: length of cuboid in the y-direction
        :param l_z: length of cuboid in the z-direction
        :param v_func: function of position vector
        :param m_func: function of position vector
        """
        r = np.random.uniform(
            np.array([-l_x, -l_y, -l_z]) * 0.5, np.array([l_x, l_y, l_z]) * 0.5,
            (n, 3))
        v = np.apply_along_axis(v_func, 1, r)
        self.r = np.vstack((self.r, r + center))
        self.v = np.vstack((self.v, v))
        self.m = np.append(self.m, np.apply_along_axis(m_func, 1, self.r))

    def clear_particles(self):
        """
        Removes all particles from the space.
        """
        self.r = np.empty(shape=(0, 3))
        self.v = np.empty(shape=(0, 3))
        self.m = np.array([])

    def __bool__(self):
        return np.any(self.m)

    def __len__(self):
        return len(self.m)
