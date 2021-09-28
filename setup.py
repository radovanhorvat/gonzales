import os
import sys
import numpy as np
from pathlib import Path
from Cython.Build import cythonize
from setuptools import setup, Extension, find_packages


CYTHON_SRC_PATH = os.path.join('src', 'cython')
C_SRC_PATH = os.path.join('src', 'c')
PYTHON_SRC_PATH = os.path.join('src', 'python')

long_description = (Path(__file__).parent / "README.md").read_text()


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


extensions = [Extension("gonzales.lib.brute_force", sources=[os.path.join(CYTHON_SRC_PATH, 'brute_force.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[CYTHON_SRC_PATH, np.get_include()]),
              Extension("gonzales.lib.octree", sources=[os.path.join(CYTHON_SRC_PATH, 'octree.pyx'),
                                                        os.path.join(C_SRC_PATH, 'data_structs.c'),
                                                        os.path.join(C_SRC_PATH, 'octnode.c'),
                                                        os.path.join(C_SRC_PATH, 'brute_force.c')],
                        include_dirs=[CYTHON_SRC_PATH, C_SRC_PATH, np.get_include()],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        language="c"),

              Extension("gonzales.lib.physics", sources=[os.path.join(CYTHON_SRC_PATH, 'physics.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[C_SRC_PATH, np.get_include()]),
              ]


setup(
    name='gonzales',
    version='0.1.1',
    packages=find_packages(PYTHON_SRC_PATH),
    package_dir={'': PYTHON_SRC_PATH},
    description='N-body simulator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Radovan Horvat',
    author_email='radovan.horvat@gmail.com',
    url='https://github.com/radovanhorvat/gonzales',
    license='MIT',
    install_requires=['numpy>=1.21.0', 'Cython>=0.29.23', 'h5py>=3.3.0', 'psutil>=5.8.0', 'vispy>=0.7.1',
                      'PyQt5>=5.15.4'],
    ext_modules=cythonize(extensions),
)
