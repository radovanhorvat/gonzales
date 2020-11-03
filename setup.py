import os
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

# pip install -r requirements.txt
# pip install -e .
# python setup.py build_ext --build-lib=kernels --> this should be added to setup

extensions = [Extension('brute_force',
                        sources=[os.path.join('kernels', 'brute_force.pyx')],
                        extra_compile_args=['/openmp'],
                        extra_link_args=['/openmp'],
                        include_dirs=['kernels', np.get_include()]),
              Extension("octree", sources=[os.path.join('kernels', 'octree.pyx'),
                                           os.path.join('kernels', 'cpp_src', 'octnode.cpp'),
                                           os.path.join('kernels', 'cpp_src', 'octree.cpp')],
                        include_dirs=['kernels', os.path.join('kernels', 'cpp_src'), np.get_include()],
                        language="c++"),
              ]


setup(
    name='N-body simulator',
    version='0.1',
    packages=['simulator', 'test'],
    ext_modules=cythonize(extensions),
)