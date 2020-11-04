import sys
import numpy as np

import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

from simulator.space import Space


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


if __name__ == '__main__':

    def vel_func(pos_vec):
        return np.array((0, 0, 0))

    def mass_func(pos_vec):
        return 1.0e-3

    space = Space()
    space.add_cuboid(10000, np.array((0, 0, 0)), 1, 1, 1, vel_func, mass_func)
    space.add_cylinder(10000, np.array((5, 0, 0)), 1, 0.1, vel_func, mass_func)
    space.add_sphere(10000, np.array((0, -5, 0)), 1, vel_func, mass_func)

    p = Points3DPlot(space)
    p.show()
