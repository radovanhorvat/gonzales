# nbody

A 3D N-body simulator. 

## Project description

This project is a N-body simulator, which includes a user interface for overview and animation of simulation results.
The low level code is written in Cython and C, and the rest was developed with Python. The GUI is written using PyQt5.

Two main simulation types are available:

- Brute-force simulation
- Barnes-Hut simulation

 For both cases, the simulator is collisionless, with a gravitational softening parameter which can be specified.
 Simulation results are written into HDF5 format.


## Requirements

- Python 3, Pip
- a C/C++ compiler

## Installation

- clone the repository
- in the repo root folder, create a virtual environment using `python -m venv venv`
- activate the virtual environment with `source env/bin/activate` (Linux) or `venv\Scripts\activate.bat` (Windows)
- install requirements using pip: `pip install -r requirements.txt`
- build the extension modules (simulator kernels) with: `python setup.py build_ext --build-lib=src/python/nbody/kernels`

Now, assuming everything was successful, you can do the following in order to verify everything is working properly:

- navigate to directory `src/python`
- run tests with `python -m pytest`
- run benchmarks with `python -m nbody.benchmark.benchmark`
- run one of the examples, like the Solar system simulation: `python -m nbody.examples.solar_system`

Have fun!

## Demo

to be updated

