import h5py
import numpy as np

from gonzales.simulator.utils import to_cartesian


class Space:
    """
    Class used to represent a 3D space which contains particles.
    """
    def __init__(self, r=None, v=None, m=None):
        """
        Initialize the space.

        :param r: N x 3 numpy array or None, position vectors of the particles
        :param v: N x 3 numpy array or None, velocity vectors of the particles
        :param m: N x 1 numpy array or None, mass vector
        """
        self.r = np.empty(shape=(0, 3)) if r is None else r
        self.v = np.empty(shape=(0, 3)) if v is None else v
        self.m = np.array([]) if m is None else m

    def to_hdf5(self, filepath):
        """
        Saves configuration to hdf5 file. Datasets are called "r", "v" and "m".
        """
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('r', self.r.shape)
            f.create_dataset('v', self.v.shape)
            f.create_dataset('m', self.m.shape)
            f['r'][:] = self.r
            f['v'][:] = self.v
            f['m'][:] = self.m

    def from_hdf5(self, filepath):
        """
        Reads data from hdf5 file.
        """
        with h5py.File(filepath, 'r') as f:
            self.add_particles(f['r'][:], f['v'][:], f['m'][:])

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

    def add_particles(self, r, v, m):
        """
        Adds N particles, specified by position and velocity matrices, and by mass vector.

        :param r: N x 3 numpy array, position vectors of the particles
        :param v: N x 3 numpy array, velocity vectors of the particles
        :param m: N x 1 numpy array, mass vector
        """
        self.r = np.vstack((self.r, r))
        self.v = np.vstack((self.v, v))
        self.m = np.append(self.m, m)

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
        self.m = np.append(self.m, np.apply_along_axis(m_func, 1, r))

    def add_sphere(self, n, center, radius, v_func, m_func):
        """
        Generates uniform random particle distribution within a spherical volume
        with a given center and radius.

        :param n: number of particles to generate
        :param center: center of spherical volume
        :param radius: radius of spherical volume
        :param v_func: function of r, theta, phi
        :param m_func: function of r, theta, phi
        """
        l = np.random.uniform(0, 1, n)
        u = np.random.uniform(-1, 1, n)
        r = radius * l ** (1/3)
        theta = np.arccos(-u)
        phi = np.random.uniform(0, 2 * np.pi, (n,))
        r_spherical = np.column_stack((r, theta, phi))
        v_spherical = np.apply_along_axis(v_func, 1, r_spherical)
        r_cartesian = to_cartesian(r_spherical, 'spherical')
        v_cartesian = to_cartesian(v_spherical, 'spherical')
        self.r = np.vstack((self.r, r_cartesian + center))
        self.v = np.vstack((self.v, v_cartesian))
        self.m = np.append(
            self.m, np.apply_along_axis(m_func, 1, r_spherical))

    def add_cylinder(self, n, center, radius, l_z, v_func, m_func):
        """
        Generates uniform random particle distribution within a cylindrical
        volume with a given center, radius and height.

        :param n: number of particles to generate
        :param center: center of cylindrical volume
        :param radius: radius of cylindrical volume
        :param l_z: height of cylindrical volume
        :param v_func: function of position vector in cylindrical coordinates
        :param m_func: function of position vector in cylindrical coordinates
        """
        r = np.random.uniform(0, radius, n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.uniform(-0.5 * l_z, 0.5 * l_z, n)
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        r_cylindrical = np.column_stack((r, theta, z))
        v_cylindrical = np.apply_along_axis(v_func, 1, r_cylindrical)
        r_cartesian = np.column_stack((x, y, z))
        v_cartesian = to_cartesian(v_cylindrical, 'cylindrical')
        self.r = np.vstack((self.r, r_cartesian + center))
        self.v = np.vstack((self.v, v_cartesian))
        self.m = np.append(
            self.m, np.apply_along_axis(m_func, 1, r_cylindrical))

    def add_plummer(self, n, center):
        m = np.ones(n) * 1. / n
        self.m = np.append(self.m, m)
        r = 1. / np.sqrt(np.random.rand(n)**(-2. / 3.) - 1)
        phi = np.random.uniform(0, 2 * np.pi, n)
        theta = np.arccos(np.random.uniform(-1, 1, n))
        R = np.ones((n, 3))
        R[:, 0] = np.multiply(np.multiply(r, np.sin(theta)), np.cos(phi))
        R[:, 1] = np.multiply(np.multiply(r, np.sin(theta)), np.sin(phi))
        R[:, 2] = np.multiply(r, np.cos(theta))
        self.r = np.vstack((self.r, R + center))
        x = np.zeros(n)
        y = np.ones(n) * 0.1
        while all(y > x**2 * (1. - x**2)**3.5):
            x = np.random.uniform(0, 1, n)
            y = np.random.uniform(0, 0.1, n)
        phi = np.random.uniform(0, 2 * np.pi, n)
        theta = np.arccos(np.random.uniform(-1, 1, n))
        vel = np.multiply(x, np.sqrt(2) * (1 + np.multiply(r, r)) ** -0.25)
        V = np.ones((n, 3))
        V[:, 0] = np.multiply(np.multiply(vel, np.sin(theta)), np.cos(phi))
        V[:, 1] = np.multiply(np.multiply(vel, np.sin(theta)), np.sin(phi))
        V[:, 2] = np.multiply(vel, np.cos(theta))
        self.v = np.vstack((self.v, V))

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
