# gonzales

A 3D N-body simulator. 

## Project description

This project is a parallel N-body simulator, which includes a user interface for overview and animation of simulation results.
The low level code is written in Cython and C, and the rest was developed with Python. Parallelization was achieved with OpenMP.
The GUI is written using PyQt5.

Two main simulation types are available:

- Brute-force simulation
- Barnes-Hut simulation

 For both cases, the simulator is collisionless, with a gravitational softening parameter which can be specified.
 Simulation results are written into HDF5 format.

## Documentation

Documentation can be found [here](https://gonzales.readthedocs.io/en/latest/).

## Requirements

- Python 3, Pip
- a C compiler

## Installation

- install from [PyPi](https://pypi.org/project/nbody-solver/) with `pip install gonzales`, preferably using 
a virtual environment. For more details, see the [installation docs](https://gonzales.readthedocs.io/en/latest/install.html)

After installing, you can do the following in order to verify everything is working correctly:

- run one of the examples, like the Solar system simulation: `python -m gonzales.examples.solar_system`
- run the default performance benchmark suite with `import gonzales; gonzales.run_default_benchmark()`
- run the quick start snippet below

## Quick start snippet

```
import gonzales as gnz
import numpy as np

# number of particles, gravitational constant and softening length
N = 10000
G, eps = 0.01, 1e-2

# initial conditions and masses
r = np.random.uniform(-1, 1, (N, 3))
v = np.zeros((N, 3))
m = np.ones(N)

# output file name
fn = 'results.hdf5'

# particle container
s = gnz.Space(r, v, m)

# Barnes-Hut simulation
sim = gnz.BHSimulation(s, fn, G, eps, 1000, np.zeros(3), 0.75)
sim.run(100, 0.01)

# Viewer for animation and results
gnz.run_viewer(fn)
```