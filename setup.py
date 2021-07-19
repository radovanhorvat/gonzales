import os
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

# python3 -m venv env
# source env/bin/activate (deactivate with: deactivate)
# pip install -r requirements.txt
# pip install -e .
# python setup.py build_ext --build-lib=kernels --> this should be added to setup

def get_compile_args():
    if sys.platform == 'linux':
        return ['-fopenmp', '-O3', '-ffast-math', '-march=native']
    elif sys.platform == 'win32':
        return ['/openmp', '/fp:fast']
    return []


def get_link_args():
    if sys.platform == 'linux':
        return ['-fopenmp']
    elif sys.platform == 'win32':
        return []
    return []
    

compile_args = get_compile_args()
link_args = get_link_args()


extensions = [Extension("brute_force", sources=[os.path.join('kernels', 'brute_force.pyx'),
                                                os.path.join('kernels', 'cpp_src', 'brute_force.cpp')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=['kernels', os.path.join('kernels', 'cpp_src'), np.get_include()],
                        language="c++"),
              Extension("octree", sources=[os.path.join('kernels', 'octree.pyx'),
                                           os.path.join('kernels', 'cpp_src', 'octnode.cpp'),
                                           os.path.join('kernels', 'cpp_src', 'octree.cpp')],
                        include_dirs=['kernels', os.path.join('kernels', 'cpp_src'), np.get_include()],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        language="c++"),
              Extension("octree_c", sources=[os.path.join('kernels', 'octree_c.pyx'),
                                             os.path.join('kernels', 'c_src', 'data_structs.c'),
                                             os.path.join('kernels', 'c_src', 'octnode.c'),
                                             os.path.join('kernels', 'c_src', 'brute_force.c')],
                        include_dirs=['kernels', os.path.join('kernels', 'c_src'), np.get_include()],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        language="c"),

              Extension("numeric", sources=[os.path.join('kernels', 'numeric.pyx')],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args,
                        include_dirs=['kernels', np.get_include()]),
              ]


setup(
    name='N-body simulator',
    version='0.1',
    packages=['simulator', 'test'],
    ext_modules=cythonize(extensions),
)