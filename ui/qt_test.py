import h5py

from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QAction, QFileDialog, QApplication, QVBoxLayout)
from PyQt5.QtGui import QIcon
from pyqtgraph.Qt import QtGui, QtCore

import sys
from pathlib import Path

import pyqtgraph.opengl as gl


class ViewWidget(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.opts['distance'] = 10
        #self.setGeometry(50, 50, 1600, 900)
        self.setWindowTitle('3D space')
        self.points = gl.GLScatterPlotItem()
        self.addItem(self.points)
        self.show()


class FormWidget(QWidget):

    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.view_widget = ViewWidget()

        play_button = QPushButton('Play', self)
        play_button.clicked.connect(self.parent().on_play)

        self.layout.addWidget(play_button)
        self.layout.addWidget(self.view_widget)
        self.info_label = QLabel()
        self.params_label = QLabel()

        info_groupbox = QGroupBox("Simulation info")
        info_box = QVBoxLayout()
        info_box.addWidget(self.info_label)
        info_box.addWidget(self.params_label)
        info_groupbox.setLayout(info_box)
        info_groupbox.setMaximumHeight(80)
        self.layout.addWidget(info_groupbox)

        self.info_label.setMaximumHeight(40)
        self.params_label.setMaximumHeight(40)
        self.setLayout(self.layout)


class NBodyViewer(QMainWindow):

    def __init__(self, filename=''):
        super().__init__()
        self._filename = filename
        self._fobj = None
        self._timer = QtCore.QTimer()
        self._cnt = 0
        self._color = (.5, .3, .1, .7)
        self.init_ui()
        if filename:
            self._set_data_from_file(filename)

    def _refresh_status_bar(self):
        self.statusBar().showMessage('Loaded file: {}'.format(self._filename))

    def init_ui(self):
        self.cw = FormWidget(self)
        self.setCentralWidget(self.cw)
        self.statusBar()
        self._refresh_status_bar()

        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')

        play = QAction('Play', self)
        play.setStatusTip('Run animation')

        openFile.triggered.connect(self.on_file_open)
        play.triggered.connect(self.on_play)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(play)

        self.setGeometry(50, 50, 1600, 900)
        self.setWindowTitle('N-body visualization')
        self.show()

    def _update_anim(self):
        self._cnt += 1
        if self._cnt >= self._fobj['info']['number_of_steps'][()]:
            return
        pos_data = self._fobj['results']['position'][self._cnt]
        self.cw.view_widget.points.setData(pos=pos_data, size=3, color=self._color)
        self.cw.params_label.setText('Step: {}'.format(self._cnt))

    def on_play(self):
        if not self._filename:
            self.statusBar().showMessage('No file loaded')
            return
        self._cnt = 0
        self._timer.timeout.connect(self._update_anim)
        self._timer.start(50)

    def on_file_open(self):
        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        if not fname:
            return
        self._set_data_from_file(fname[0])

    def _set_data_from_file(self, filename):
        self._filename = filename
        self._fobj = h5py.File(self._filename, 'r')
        self.cw.view_widget.points.setData(pos=self._fobj['results']['position'][self._cnt], size=3,
                                           color=self._color)
        self._refresh_status_bar()
        info_str = 'N: {}, G: {}, eps: {}, type: {}'.format(
            str(self._fobj['info']['number_of_particles'][()]),
            str(self._fobj['info']['G'][()]),
            str(self._fobj['info']['epsilon'][()]),
            str(self._fobj['info']['simulation_type'][()])
        )
        self.cw.info_label.setText(info_str)


def run_viewer(filename=''):
    app = QApplication(sys.argv)
    nbv = NBodyViewer(filename)
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_viewer()
