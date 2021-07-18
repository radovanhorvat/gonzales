import h5py
import numpy as np

from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QAction, QFileDialog, QApplication, QVBoxLayout, QSlider, QTabWidget, QTableWidget,
                             QTableWidgetItem, QComboBox, QHBoxLayout)
from PyQt5.QtGui import QIcon
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtCore import Qt

import sys
from pathlib import Path

from simulator.simulation import ResultReader

import vispy
import vispy.scene
from vispy.scene import visuals

 
class TableView(QTableWidget):        
    def set_data(self, res_data, res_headers=None):        
        n, m = res_data.shape
        self.setRowCount(n)
        self.setColumnCount(m)
        for i in range(n):            
            for j in range(m):
                newitem = QTableWidgetItem(str(res_data[i][j]))
                #newitem.setFlags(QtCore.Qt.ItemIsEnabled)
                self.setItem(i, j, newitem)
        if res_headers:
            self.setHorizontalHeaderLabels(res_headers)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()


class SliderLabelWidget(QWidget):
    def __init__(self, parent):
        super(SliderLabelWidget, self).__init__(parent)
        hbox = QHBoxLayout()

        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setRange(0, 100)
        self.sld.setFocusPolicy(Qt.NoFocus)
        self.sld.setPageStep(5)

        self.label = QLabel('0', self)

        hbox.addWidget(self.sld)
        hbox.addSpacing(15)
        hbox.addWidget(self.label)

        self.setLayout(hbox)
   


class MainWidget(QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Animation")
        self.tabs.addTab(self.tab2, "Results")

        self.tab1.layout = QVBoxLayout(self)
        self.tab2.layout = QVBoxLayout(self)

        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        self.scatter = visuals.Markers()
        view.add(self.scatter)
        self.view_widget = canvas.native

        play_button = QPushButton('Play', self)
        play_button.clicked.connect(self.parent().on_play)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.parent().on_pause)
        
        # tab1
        self.tab1.layout.addWidget(play_button)
        self.tab1.layout.addWidget(self.pause_button)
        self.tab1.layout.addWidget(self.view_widget)
        self.info_label = QLabel()
        self.params_label = QLabel()

        info_groupbox = QGroupBox("Simulation info")
        info_box = QVBoxLayout()
        info_box.addWidget(self.info_label)
        info_box.addWidget(self.params_label)
        info_groupbox.setLayout(info_box)
        info_groupbox.setMaximumHeight(80)
        self.tab1.layout.addWidget(info_groupbox)
        self.tab1.setLayout(self.tab1.layout)

        self.info_label.setMaximumHeight(40)
        self.params_label.setMaximumHeight(40)

        # tab2
        self.combo = QComboBox(self)
        self.table = TableView()
        self.slider = SliderLabelWidget(self)
        self.combo.activated.connect(self.parent().on_combo_activated)
        self.slider.sld.valueChanged.connect(self.parent().on_slider_changed)

        self.tab2.layout.addWidget(self.combo)
        self.tab2.layout.addWidget(self.slider)
        self.tab2.layout.addWidget(self.table)
        self.tab2.setLayout(self.tab2.layout)

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)    


class NBodyViewer(QMainWindow):
    _RES_HEADER_MAP = {'position': ['r_x', 'r_y', 'r_z'], 'velocity': ['v_x', 'v_y', 'v_z'], 'energy': ['E']}

    def __init__(self, filename=''):
        super().__init__()
        self._filename = None
        self._reader = None
        self._timer = QtCore.QTimer()
        self._cnt = 0
        self._num_steps = 0
        self._color = (.9, .9, .1, .7)        
        self.init_ui()
        if filename:
            self._set_data_from_file(filename)
        self._is_playing = False

    def _refresh_status_bar(self):
        self.statusBar().showMessage('Loaded file: {}'.format(self._filename))

    def init_ui(self):
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self.statusBar()
        self._refresh_status_bar()

        openFile = QAction('Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')

        play = QAction('Play', self)
        play.setStatusTip('Run animation')

        quit = QAction('Quit', self)
        quit.setStatusTip('Close viewer')

        openFile.triggered.connect(self.on_file_open)
        play.triggered.connect(self.on_play)
        quit.triggered.connect(self.close)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(play)
        fileMenu.addAction(quit)

        self.setGeometry(50, 50, 1600, 900)
        self.setWindowTitle('N-body visualization')
        self.show()

    def _update_view(self):
        self._cnt += 1
        if self._cnt > self._num_steps:
            return
        pos_data = self._reader.get_result('position', self._cnt)
        self.main_widget.scatter.set_data(pos_data, edge_color=None, face_color=self._color, size=4)
        self.main_widget.params_label.setText('Step: {}'.format(self._cnt))

    def on_play(self):
        if not self._filename:
            self.statusBar().showMessage('No file loaded')
            return
        self._cnt = 0
        self._timer.timeout.connect(self._update_view)
        self._timer.start(50)
        self._is_playing = True

    def on_pause(self):
        if self._is_playing:
            self._is_playing = False
            self.main_widget.pause_button.setText('Resume')
            self._timer.stop()
            return
        self._timer.start(50)
        self._is_playing = True
        self.main_widget.pause_button.setText('Pause')

    def on_file_open(self):
        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        if not fname:
            return
        self._cnt = 0
        self._set_data_from_file(fname[0])

    def on_combo_activated(self):
        res_name = str(self.main_widget.combo.currentText())
        res_data = self._reader.get_result(res_name, 0)
        self.main_widget.table.set_data(res_data, self._RES_HEADER_MAP[res_name])
        self.main_widget.slider.sld.setRange(0, self._reader.get_result_num_steps(res_name) - 1)

    def on_slider_changed(self, value):
        self.main_widget.slider.label.setText(str(value))
        res_name = str(self.main_widget.combo.currentText())
        res_data = self._reader.get_result(res_name, value)
        self.main_widget.table.set_data(res_data)

    def _set_data_from_file(self, filename):
        self._filename = filename
        self._reader = ResultReader(filename)
        pos_data = self._reader.get_result('position', self._cnt)
        self.main_widget.scatter.set_data(pos_data, edge_color=None, face_color=self._color, size=4)
        self._refresh_status_bar()
        info_str = 'N: {}, G: {}, eps: {}, type: {}'.format(
            str(self._reader.get_info()['number_of_particles'][()]),
            str(self._reader.get_info()['G'][()]),
            str(self._reader.get_info()['epsilon'][()]),
            str(self._reader.get_info()['simulation_type'][()])
        )
        self._num_steps = self._reader.get_info()['number_of_steps'][()]
        self.main_widget.info_label.setText(info_str)
        for res_name in self._reader.get_result_names():
            self.main_widget.combo.addItem(res_name)
        res_name = str(self.main_widget.combo.currentText())
        res_data = self._reader.get_result(res_name, 0)
        self.main_widget.table.set_data(res_data, self._RES_HEADER_MAP[res_name])
        self.main_widget.slider.sld.setRange(0, self._reader.get_result_num_steps(res_name) - 1)

    def closeEvent(self, event):
        self._timer.stop()
        if self._reader:
            self._reader.close()
        super(QMainWindow, self).closeEvent(event)


def run_viewer(filename=''):
    app = QApplication(sys.argv)
    nbv = NBodyViewer(filename)
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_viewer()
