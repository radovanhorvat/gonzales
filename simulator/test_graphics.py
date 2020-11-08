import os
import sys
import h5py
import numpy as np

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

from simulator.space import Space
import simulator.simulation as sim


class ViewWidget(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts['distance'] = 10
        self.setGeometry(50, 50, 1600, 900)
        self.setWindowTitle('3D space')
        self.points = gl.GLScatterPlotItem()
        self.addItem(self.points)
        self.show()


class Points3DPlot(object):
    def __init__(self, space):
        self.app = QtGui.QApplication(sys.argv)
        self.space = space
        self.view_widget = ViewWidget()
        self.view_widget.points.setData(pos=self.space.r, size=3)

    def show(self):
        self.app.exec_()


class Points3DAnimation:
    def __init__(self, filename):
        self.app = QtGui.QApplication(sys.argv)
        self._filename = filename
        self._fobj = h5py.File(filename, 'r')
        self.view_widget = ViewWidget()
        self._timer = QtCore.QTimer()
        self._cnt = 0
        self.view_widget.points.setData(pos=self._fobj['results']['positions'][self._cnt], size=3)

    def _update(self):
        self._cnt += 1
        if self._cnt >= self._fobj['info']['number_of_steps'][()]:
            return
        pos_data = self._fobj['results']['positions'][self._cnt]
        self.view_widget.points.setData(pos=pos_data, size=3)

    def animate(self):
        self._timer.timeout.connect(self._update)
        self._timer.start(50)
        self.app.exec_()


if __name__ == '__main__':

    def vel_func(pos_vec):
        return np.array((0., 0., 0.))

    def mass_func(pos_vec):
        return 1.0e3

    # space = Space()
    # space.add_cuboid(10000, np.array((0, 0, 0)), 1, 1, 1, vel_func, mass_func)
    # space.add_cylinder(10000, np.array((5, 0, 0)), 1, 0.1, vel_func, mass_func)
    # space.add_sphere(10000, np.array((0, -5, 0)), 1, vel_func, mass_func)
    #
    # p = Points3DPlot(space)
    # p.show()

    n = 1000
    cube_length = np.sqrt(n)
    G = 1.0
    eps = 1.0e-5
    theta = 0.75
    n_steps = 1000
    step_size = 0.001

    space = Space()
    space.add_cuboid(n, np.array((0., 0., 0.)), cube_length, cube_length, cube_length, vel_func, mass_func)

    ofp = os.path.normpath(r'D:\Python_Projects\results\test_bh.hdf5')
    #s1 = sim.PPSimulation(space, ofp, G, eps)
    s1 = sim.BHSimulation(space, ofp, G, eps, cube_length, np.array((0., 0., 0.)), theta)
    #s1.add_result('energies', (1,), 1)
    s1.run(n_steps, step_size)

    anim = Points3DAnimation(ofp)
    anim.animate()
