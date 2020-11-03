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
