import os
import numpy as np

from gonzales.simulator.space import Space
from gonzales.simulator.simulation import PPSimulation

from gonzales.ui.viewer import run_viewer


if __name__ == '__main__':

    # Data taken from: https://ssd.jpl.nasa.gov/horizons.cgi, date 20-07-2021
    # Units used: Solar mass / AU / day
    
    G_old = 6.67408e-11
    M_solar = 1.989e30
    M_mercury = 3.285e23
    M_venus = 4.867e24
    M_earth = 5.972e24
    M_mars = 6.39e23
    M_jupiter = 1.898e27
    M_saturn = 5.683e26
    M_uranus = 8.681e25
    M_neptune = 1.024e26

    AU = 1.495978707e11
    # Gravitational constant must be converted
    G = G_old * (1. / AU)**3 / ((1. / M_solar) * (1. / 86400)**2)

    # No close encounters, so softening parameter is zero
    eps = 0.
    # Simulated time is 10 years
    n_steps = 365 * 10
    # Step size is one day
    step_size = 1

    space = Space()
    # Sun
    space.add_particle(np.zeros(3), np.zeros(3), 1)
    # Mercury
    space.add_particle(np.array([2.185348993935225e-1, 2.273633321330415e-1, -1.466652015897708e-3]),
                       np.array([-2.584933975624156e-2, 2.067679899283321e-2, 4.060795109164917e-3]), M_mercury / M_solar)
    # Venus
    space.add_particle(np.array([-7.073117862716856e-1, -1.329903946314239e-1, 3.899034851507831e-2]),
                       np.array([3.608302825050088e-3, -1.996897793395259e-2, -4.822730163473640e-4]), M_venus / M_solar)
    # Earth
    space.add_particle(np.array([4.498031285897389e-1, -9.112652994636136e-1, 4.076525631862657e-5]),
                       np.array([1.514203345104269e-2, 7.555238763182754e-3, 2.188032562853510e-7]), M_earth / M_solar)
    # Mars
    space.add_particle(np.array([-1.552370154286541, 6.019885575267463e-1, 5.069526134016548e-2]),
                       np.array([-4.536866693514593e-3, -1.185244632129359e-2, -1.370992590183020e-4]), M_mars / M_solar)
    # Jupiter
    space.add_particle(np.array([4.076159263821152,-2.952369556329466, -7.893452587677123e-2]),
                       np.array([4.340575155056248e-3, 6.474220433145525e-3, -1.239750763580018e-4]), M_jupiter / M_solar)
    # Saturn
    space.add_particle(np.array([6.322978029998636, -7.685224692129240, -1.180247483666776e-1]),
                       np.array([4.002206194869687e-3, 3.537638556162297e-3, -2.204327999164067e-4]), M_saturn / M_solar)
    # Uranus
    space.add_particle(np.array([1.484037656777784e1, 1.302427327404588e1, -1.439132671601537e-1]),
                       np.array([-2.619185972108102e-3, 2.779162944689609e-3, 4.419686279276467e-5]), M_uranus / M_solar)
    # Neptune
    space.add_particle(np.array([2.956026299315944e1, -4.610612365323029, -5.863606760383681e-1]),
                       np.array([4.674200510936484e-4, 3.127475540110565e-3, -7.500779722081749e-5]), M_neptune / M_solar)


    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'output', 'earth_sun.hdf5'))
    
    # Use a Brute-force simulation here, since we only have a few bodies
    sim = PPSimulation(space, output_file, G, eps)
    # Add energy and angular momentum as additional results which will be written
    sim.add_result('energy')
    sim.add_result('angular_momentum')
    
    # Run the simulation
    sim.run(n_steps, step_size)

    # Run the GUI to view animation and results. Body sizes and body colors are defined for each particle
    # in definition order. 
    run_viewer(output_file, body_sizes=[30, 15, 15, 15, 15, 25, 20, 20, 20],
                            colors=['yellow', 'brown', 'green', 'blue', 'red', 'brown', 'brown', 'blue', 'blue'])
