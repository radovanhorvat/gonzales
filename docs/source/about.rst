About
=====

`nbody-solver` is a package for simulating collisionless 3D particle systems under Newtonian gravitational
interaction. It is intented to be a lightweight and easy to use, yet comparatively powerful tool, which uses
the hdf5_ file format to store simulation data. Also, it includes a graphical user interface which is capable of
animating a simulation and efficiently displaying simulation info and results in tabular form to the user.

Two simulation types are supported: PP-simulations (name derived from Particle-Particle methods, or brute-force
methods), and BH-simulations_ (name derived from Barnes-Hut tree code methods) - the former having a computational
complexity of :math:`O(n^2)`, and the latter :math:`O(n\,log\,n)`. For both simulations, numerical
integration of the equations of motion is done using the Leapfrog_ integration method.

Functionally, the program consists of three main packages:

- ``nbody.simulator`` - provides interfaces for generating initial conditions and creating and running simulation instances
- ``nbody.lib``- consists of functions used to calculate gravitational interactions and values of physical quantities
- ``nbody.ui``- defines the graphical user interface

Detailed documentation of these packages can be found in the :ref:`modules-ref-label` documentation section.

A typical workflow can be described as follows:

- create initial conditions for the simulation using ``nbody.simulator.space.Space``
- create a simulation instance using ``nbody.simulator.simulation.PPSimulation`` or ``nbody.simulator.simulation.BHSimulation``
- run the simulation
- view generated results using ``nbody.ui.viewer.run_viewer``

More details can be found in the :ref:`quick-start-ref-label` documentation section.


.. _hdf5: https://en.wikipedia.org/wiki/Hierarchical_Data_Format

.. _Leapfrog: https://en.wikipedia.org/wiki/Leapfrog_integration

.. _BH-simulations: https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
