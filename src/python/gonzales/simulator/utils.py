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


# --------------------------------------
# Progress bar
# --------------------------------------

class ProgressBarStyle:
    def __init__(self, fill=u'\u2588', empty='.', left='[', right=']', text='Progress: '):
        """

        :param fill: symbol for completed percentage
        :param empty: symbol for remaining percentage
        :param left: left bound symbol
        :param right: right bound symbol
        :param text: text which is displayed before the progress bar
        """
        self.fill = fill
        self.empty = empty
        self.left = left
        self.right = right
        self.text = text


class ProgressBar:
    def __init__(self, iterations, length, style=ProgressBarStyle()):
        """

        :param iterations: total number of iterations
        :param length: length of progress bar
        :param style: instance of ProgressBarStyle, determines visual properties
        """
        self._iterations = iterations
        self._length = length
        self._style = style
        self._current_iter = 0

    def set_style(self, style):
        """
        Sets the style.

        :param style: instance of ProgressBarStyle
        """
        self._style = style

    def reset(self, iterations=None):
        """
        Resets the progress bar - sets the current iteration to 0, and, optionally, changes the total
        number of iterations.

        :param iterations: if given, resets the total number of iterations
        """
        self._current_iter = 0
        if iterations:
            self._iterations = iterations

    def update(self):
        """
        Updates the progress bar. An update consists of incrementing the current iteration, and
        displaying the progress bar. Also, it is checked if the current iteration is within specified bounds.
        """
        self._current_iter += 1
        assert self._current_iter <= self._iterations
        self._show()

    def _show(self):
        progress = self._current_iter / self._iterations
        filled_length = int(progress * self._length)
        empty_length = self._length - filled_length
        progress_percentage = str(int(progress * 100)) + ' %'
        print('\r{}{}{}{}{} {}'.format(self._style.text, self._style.left, self._style.fill * filled_length,
                                       self._style.empty * empty_length, self._style.right,
                                       progress_percentage), end='')
        if self._current_iter == self._iterations:
            print()
