.. _quick-start-ref-label:

Quick start
===========

.. rubric:: Initialization

Let us set up a minimal brute force simulation. First, we need numpy:

``import numpy as np``

Let's create initial conditions for 1000 particles:

``>>> r = np.random.uniform(-1, 1, (1000, 3))``

``>>> v = np.zeros((1000, 3)``

Next, add a mass vector:

``>>> m = np.ones((1000)``

Then, let's import the ``Space`` class and initialize it with the data:

``>>> from nbody.simulator.space import Space``

``>>> s = Space(r, v, m)``

.. rubric:: Simulation

Now, we're ready to use the ``PPSimulation`` class:

``>>> from nbody.simulator.simulation import PPSimulation``

Let's first set an output filepath, where the results will be stored:

``>>> import os``

``>>> fp = os.path.join('output', 'test.hdf5')``

Now we can initialize the simulation with a value of ``G = 1`` and ``eps = 0``,
where ``G`` and ``eps`` represent the gravitational constant and the softening parameter, respectively:

``>>> sim = PPSimulation(s, fp, 1., 0.)``

The default results, which are written for every step, are positions and velocities. We can add additional
results with an arbitrary writing frequency like so:

``>>> sim.add_result('energy', 10)``

That means the total energy will be written every 10 steps. Now, Let's run the simulation with `100` steps of size `0.01`:

``>>> sim.run(100, 0.01)``

Some neat output should appear, along with a progress bar:

::

   2021-09-06 20:30:59,397 INFO Simulation - type=Brute force, N_particles=1000, N_steps=100
   2021-09-06 20:30:59,398 INFO Creating datasets
   2021-09-06 20:30:59,402 INFO Writing initial data
   2021-09-06 20:30:59,403 INFO Calculating initial accelerations
   2021-09-06 20:30:59,405 INFO Start simulation
   Progress: [████████████████████████████████████████] 100 %
   2021-09-06 20:30:59,640 INFO End simulation

.. rubric:: Postprocessing

Great. If you came this far, the output was written to the specified hdf5 file. Let's first
see how to browse the output with our ``ResultReader``:

``>>> from nbody.simulator.simulation import ResultReader``

``>>> r = ResultReader(fp)``

Let's see which results we have stored by default (we can add additional ones also):

``>>> r.get_result_names()``

``['energy', 'position', 'velocity']``

OK, let's inspect some specific result for a specific step number:

``>>> r.get_result('velocity', 35)``

Boring? Well, let's demonstrate the viewer:

``>>> from nbody.ui.viewer import run_viewer``

``>>> run_viewer(fp)``

As you can see, the Viewer can animate the simulation, read the results for each step and display some basic simulation info.

.. rubric:: Something different

The animation above probably wasn't very interesting, so let's generate something that looks a bit better,
and also demonstrate the capabilities of the ``BHSimulation`` class. First, let's clear all the existing particles:

``>>> s.clear_particles()``

Now, let's add a uniform cuboid of side length ``1``, centered at the origin, and containing ``100000`` particles:

``>>> center = np.zeros(3)``

``>>> s.add_cuboid(100000, center, 1., 1., 1., lambda pos_vec: np.zeros(3), lambda pos_vec: 1e-4)``

The last two arguments are velocity and mass distribution functions, for a simple case of zero initial velocity
and uniform mass. Let's instantiate and run a Barnes-Hut simulation, with parameter ``theta = 0.75``. We let
the root size be ``10000``, so no particles can exceed its bounds during the 100 steps.

``>>> sim = BHSimulation(s, fp, 1., 1.e-3, 10000, center, 0.75)``

``>>> sim.run(100, 0.01)``

``>>> run_viewer(fp)``


This should yield a more interesting animation. Also, on my humble i5 processor, this took a bit under 10 seconds.

To wrap up - have fun!