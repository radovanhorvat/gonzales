from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QLabel, QGroupBox,
                             QAction, QFileDialog, QApplication, QVBoxLayout, QSlider, QTabWidget,
                             QTableWidgetItem, QComboBox, QHBoxLayout, QTableView, QTableWidget)
from PyQt5.QtCore import Qt, QAbstractTableModel, QTimer, QModelIndex

import sys
from pathlib import Path

from gonzales.simulator.simulation import ResultReader

import vispy
import vispy.scene
from vispy.scene import visuals


class TableModel(QAbstractTableModel):
    _header_label_map = {'position': ['r_x', 'r_y', 'r_z'],
                         'velocity': ['v_x', 'v_y', 'v_z'],
                         'angular_momentum': ['L_x', 'L_y', 'L_z'],
                         'energy': ['E']}

    def __init__(self, data, res_name):
        super(TableModel, self).__init__()
        self._data = data
        self._res_name = res_name

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if len(self._data.shape) > 1:
                return str(self._data[index.row()][index.column()])
            return str(self._data[index.column()])

    def rowCount(self, parent=QModelIndex()):
        if len(self._data.shape) > 1:
            return self._data.shape[0]
        return 1

    def columnCount(self, parent=QModelIndex()):
        if len(self._data.shape) > 1:
            return self._data.shape[1]
        return self._data.shape[0]

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        labels = self._header_label_map.get(self._res_name)
        if labels and role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return labels[section]            
        return QAbstractTableModel.headerData(self, section, orientation, role)


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
 

class WidgetWithLabel(QWidget):
    def __init__(self, parent, widget_inst, label_text):
        super(WidgetWithLabel, self).__init__(parent)
        hbox = QHBoxLayout()

        self.wgt = widget_inst
        self.label = QLabel(label_text, self)

        hbox.addWidget(self.label)
        hbox.addWidget(self.wgt)
        hbox.setAlignment(Qt.AlignLeft)
        
        self.setLayout(hbox)


class MainWidget(QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent)
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.tabs.addTab(self.tab1, "Animation")
        self.tabs.addTab(self.tab2, "Results")
        self.tabs.addTab(self.tab3, "Info")

        tab1_layout = QVBoxLayout()
        tab2_layout = QVBoxLayout()
        tab3_layout = QVBoxLayout()

        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        #view.camera.scale_factor = 1e11
        axis = vispy.scene.visuals.XYZAxis(parent=view.scene)
        self.scatter = visuals.Markers()
        view.add(self.scatter)
        self.view_widget = canvas.native

        play_button = QPushButton('Play', self)
        play_button.clicked.connect(self.parent().on_play)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.parent().on_pause)
        
        # tab1
        tab1_layout.addWidget(play_button)
        tab1_layout.addWidget(self.pause_button)
        tab1_layout.addWidget(self.view_widget)
        self.step_label = QLabel()
        self.filename_label = QLabel()

        info_groupbox = QGroupBox("Simulation info")
        info_box = QVBoxLayout()
        info_box.addWidget(self.filename_label)
        info_box.addWidget(self.step_label)
        info_groupbox.setLayout(info_box)
        tab1_layout.addWidget(info_groupbox)
        self.tab1.setLayout(tab1_layout)

        # tab2
        self.combo = WidgetWithLabel(self, QComboBox(self), 'Result: ')


        self.table = QTableView()
        self.slider = WidgetWithLabel(self, SliderLabelWidget(self), 'Step: ')
        self.combo.wgt.activated.connect(self.parent().on_combo_activated)
        self.slider.wgt.sld.valueChanged.connect(self.parent().on_slider_changed)

        tab2_layout.addWidget(self.combo)
        tab2_layout.addWidget(self.slider)
        tab2_layout.addWidget(self.table)
        self.tab2.setLayout(tab2_layout)

        #tab3
        self.info_table = QTableWidget()
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.horizontalHeader().setVisible(False)
        tab3_layout.addWidget(self.info_table)
        self.tab3.setLayout(tab3_layout)

        layout.addWidget(self.tabs)
        self.setLayout(layout)


class NBodyViewer(QMainWindow):
    def __init__(self, filename='', body_sizes=4, colors=(.9, .9, .1, .7)):
        super().__init__()
        self._filename = None
        self._reader = None
        self._timer = QTimer()
        self._cnt = 0
        self._num_steps = 0
        self._color = colors        
        self._sizes = body_sizes

        self.init_ui()
        if filename:
            self._set_data_from_file(filename)
        self._is_playing = False

    def _refresh_status_bar(self):
        self.main_widget.filename_label.setText('Loaded file: {}'.format(self._filename))

    def init_ui(self):
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self.statusbar = self.statusBar()
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
        self.main_widget.scatter.set_data(pos_data, edge_color=None, face_color=self._color, size=self._sizes)
        self.main_widget.step_label.setText('Step: {}'.format(self._cnt))

    def _populate_info_table(self):
        n = len(self._reader.get_info().keys())
        self.main_widget.info_table.setRowCount(n)
        self.main_widget.info_table.setColumnCount(2)
        for i, kname in enumerate(self._reader.get_info().keys()):
            val = self._reader.get_info()[kname][()]
            self.main_widget.info_table.setItem(i, 0, QTableWidgetItem(str(kname)))
            if kname in ['start_time', 'end_time']:
                val = datetime.fromtimestamp(val).strftime("%m/%d/%Y, %H:%M:%S")
            self.main_widget.info_table.setItem(i, 1, QTableWidgetItem(str(val)))
        self.main_widget.info_table.resizeColumnsToContents()

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
        if not any(fname):
            return
        self._cnt = 0
        if self._reader:
            self._reader.close()
        self._sizes = 4
        self._set_data_from_file(fname[0])

    def on_combo_activated(self):
        res_name = str(self.main_widget.combo.wgt.currentText())
        res_data = self._reader.get_result(res_name, 0)
        self.main_widget.table.setModel(TableModel(res_data, res_name))
        self.main_widget.slider.wgt.sld.setRange(0, self._reader.get_result_num_steps(res_name) - 1)
        self.main_widget.slider.wgt.sld.setValue(0)

    def on_slider_changed(self, value):
        res_name = str(self.main_widget.combo.wgt.currentText())
        res_data = self._reader.get_result(res_name, value)
        res_freq = self._reader.get_result_frequency(res_name)
        self.main_widget.slider.wgt.label.setText(str(res_freq * value))
        self.main_widget.slider.wgt.sld.setValue(value)
        self.main_widget.table.setModel(TableModel(res_data, res_name))

    def _set_data_from_file(self, filename):
        self._filename = filename
        self._reader = ResultReader(filename)
        pos_data = self._reader.get_result('position', self._cnt)
        self.main_widget.scatter.set_data(pos_data, edge_color=None, face_color=self._color, size=self._sizes)
        self._refresh_status_bar()
        self._num_steps = self._reader.get_info()['number_of_steps'][()]
        self._populate_info_table()
        self.main_widget.combo.wgt.clear()
        for res_name in self._reader.get_result_names():
            self.main_widget.combo.wgt.addItem(res_name)
        res_name = str(self.main_widget.combo.wgt.currentText())
        res_data = self._reader.get_result(res_name, 0)
        self.main_widget.table.setModel(TableModel(res_data, res_name))
        self.main_widget.slider.wgt.sld.setRange(0, self._reader.get_result_num_steps(res_name) - 1)

    def closeEvent(self, event):
        self._timer.stop()
        if self._reader:
            self._reader.close()
        super(QMainWindow, self).closeEvent(event)


def run_viewer(filename='', body_sizes=4, colors=(.9, .9, .1, .7)):
    """
    Runs the viewer with given filename.

    :param filename: filepath of hdf5 file
    :param body_sizes: int or tuple of ints, respective body size(s) which will be drawn.
    :param colors: double or tuple of doubles, respective body color(s) which will be drawn.
    """
    app = QApplication(sys.argv)
    nbv = NBodyViewer(filename, body_sizes, colors)
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_viewer()
