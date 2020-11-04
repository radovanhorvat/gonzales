import time
import numpy as np
from functools import wraps


# --------------------------------------
# General utility functions
# --------------------------------------

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        t = end - start
        print('Function name: {0}, Elapsed time: {1}'.format(f.__name__, t))
        return result
    return wrapper


# --------------------------------------
# Numerical utility functions
# --------------------------------------

def calculate_relative_error(accs, accs_ref):
    error_vec = np.linalg.norm(accs - accs_ref, axis=1) / np.linalg.norm(accs_ref, axis=1)
    return np.mean(error_vec), np.std(error_vec)


def to_cartesian(r, coord_sys):
    """
    Transforms an array of vectors from a coord_sys to the cartesian coordinate
    system.
    :param r: n x 3 matrix of position or any other vectors in spherical
    coordinates
    :param coord_sys: 'spherical' or 'cylindrical'
    :return: n x 3 matrix of position vectors in cartesian coordinates
    """
    if coord_sys == 'cylindrical':
        r, phi, z = r[:, 0], r[:, 1], r[:, 2]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.column_stack((x, y, z))
    elif coord_sys == 'spherical':
        r, theta, phi = r[:, 0], r[:, 1], r[:, 2]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.column_stack((x, y, z))
    else:
        raise ValueError("'coord_sys' must be 'cylindrical' or 'spherical'.")
