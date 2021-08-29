import os
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages

CYTHON_SRC_PATH = os.path.join('src', 'cython')
C_SRC_PATH = os.path.join('src', 'c')


def get_compile_args():
    if sys.platform == 'linux':
        return ['-fopenmp', '-O3', '-ffast-math', '-march=native']
    elif sys.platform == 'win32':
        return ['/openmp']
    return []


def get_link_args():
    if sys.platform == 'linux':
        return ['-fopenmp']
    elif sys.platform == 'win32':
        return []
    return []
    

compile_args = get_compile_args()
link_args = get_link_args()


extensions = [Extension("brute_force", sources=[os.path.join(CYTHON_SRC_PATH, 'brute_force.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[CYTHON_SRC_PATH, np.get_include()]),
              Extension("octree_c", sources=[os.path.join(CYTHON_SRC_PATH, 'octree_c.pyx'),
                                             os.path.join(C_SRC_PATH, 'data_structs.c'),
                                             os.path.join(C_SRC_PATH, 'octnode.c'),
                                             os.path.join(C_SRC_PATH, 'brute_force.c')],
                        include_dirs=[CYTHON_SRC_PATH, C_SRC_PATH, np.get_include()],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        language="c"),

              Extension("numeric", sources=[os.path.join(CYTHON_SRC_PATH, 'numeric.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[C_SRC_PATH, np.get_include()]),
              ]

setup(
    name='N-body simulator',
    version='0.1',
    packages=find_packages(os.path.join('src', 'python')),
    description='N-body simulator',
    author='Radovan Horvat',
    author_email='radovan.horvat@gmail.com',
    ext_modules=cythonize(extensions),
)
