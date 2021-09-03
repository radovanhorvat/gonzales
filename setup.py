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


extensions = [Extension("nbody.kernels.brute_force", sources=[os.path.join(CYTHON_SRC_PATH, 'brute_force.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[CYTHON_SRC_PATH, np.get_include()]),
              Extension("nbody.kernels.octree_c", sources=[os.path.join(CYTHON_SRC_PATH, 'octree_c.pyx'),
                                                           os.path.join(C_SRC_PATH, 'data_structs.c'),
                                                           os.path.join(C_SRC_PATH, 'octnode.c'),
                                                           os.path.join(C_SRC_PATH, 'brute_force.c')],
                        include_dirs=[CYTHON_SRC_PATH, C_SRC_PATH, np.get_include()],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        language="c"),

              Extension("nbody.kernels.numeric", sources=[os.path.join(CYTHON_SRC_PATH, 'numeric.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=[C_SRC_PATH, np.get_include()]),
              ]


# class build_ext_first(setuptools.command.install.install):
#     def run(self):
#         self.run_command('build_ext --build-lib=src/python/nbody/kernels')
#         return setuptools.command.install.install.run(self)


# class build_ext(_build_ext):
#     'to install numpy'
#     def initialize_options(self):
#         super().initialize_options()
#         self.build_lib = os.path.join('src', 'python', 'nbody', 'kernels')


setup(
    name='nbodytest',
    version='0.60',
    packages=find_packages(os.path.join('src', 'python')),
    package_dir={'': os.path.join('src', 'python')},
    #ext_package='nbody.kernels',
    #package_data={'': [os.path.join('src', 'cython', '*.pyx')]},
    #cmdclass={'build_ext' : build_ext},
    description='N-body simulator',
    author='Radovan Horvat',
    author_email='radovan.horvat@gmail.com',
    install_requires=['numpy', 'Cython', 'h5py', 'psutil', 'vispy', 'PyQt5'],
    ext_modules=cythonize(extensions),
)
