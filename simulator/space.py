import numpy as np

from simulator.utils import timing, to_cartesian


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

    #@timing
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

    #@timing
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

    #@timing
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
