import sys, os, matplotlib
matplotlib.use("TkAgg")

from PyQt5.QtWidgets import QApplication, QMainWindow, \
                            QShortcut, QCheckBox
from PyQt5 import QtCore, uic
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import qdarkstyle

from dafy.projects.xrv.core.xrv_run_control import RunApp
from dafy.projects.xrv.core.gui_operations import GuiOperations
from dafy.core.util.path import xrv_path, user_config_path

class MyMainWindow(QMainWindow, GuiOperations):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        self.init_ui()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(xrv_path/'icons'/'xrv.png')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(24, 24))

    def init_ui(self):
        #load ui script
        uic.loadUi(str(xrv_path / 'ui' / 'XRV_GUI.ui'),self)
        self.setWindowTitle('Data analysis factory: XRV data analasis')
        self.use_q_mapping = self.radioButton_q.isChecked()
        self.app_ctr = RunApp(self.use_q_mapping, self.spinBox_order,self.comboBox_loader.currentText())
        self.current_image_no = 0
        self.current_scan_number = None
        self.bkg_intensity = 0
        self.bkg_clip_image = None
        self.stop = False
        self.widget_terminal.update_name_space('xrv',self.app_ctr)
        self.widget_terminal.update_name_space('main_gui',self)
        self.single_point_strain_ax_handle = None
        self.single_point_size_ax_handle = None

        #pre-set the data path and data name
        self.lineEdit_data_file_name.setText('temp_data_xrv.xlsx')
        self.lineEdit_data_file_path.setText(self.app_ctr.data_path)
        setattr(self.app_ctr,'data_path',os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text()))

        #signal-slot setting
        self.open.clicked.connect(self.load_file)
        self.launch.clicked.connect(self.launch_file)
        self.stopBtn.clicked.connect(self.stop_func)
        self.saveas.clicked.connect(self.save_file_as)
        self.save.clicked.connect(self.save_file)
        self.plot.clicked.connect(self.plot_figure)
        self.runstepwise.clicked.connect(self.plot_)
        self.nextShort = QShortcut(QtGui.QKeySequence("Right"), self)
        self.nextShort.activated.connect(self.plot_)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        for each in self.groupBox_2.findChildren(QCheckBox):
            each.released.connect(self.update_poly_order)
        self.pushButton_remove_current_point.clicked.connect(self.remove_data_point)
        self.pushButton_remove_single_point.clicked.connect(self.remove_current_data_point)
        self.pushButton_recenter.clicked.connect(self.recenter)
        #self.doubleSpinBox_ss_factor.valueChanged.connect(self.update_ss_factor)
        self.comboBox_p2.activated.connect(self.select_source_for_plot_p2)
        self.actionOpenConfig.triggered.connect(self.load_file)
        self.actionSaveConfig.triggered.connect(self.save_file)
        self.actionRun.triggered.connect(self.plot_)
        self.actionStop.triggered.connect(self.stop_func)
        self.actionSaveData.triggered.connect(self.save_data)
        self.p2_data_source = self.comboBox_p2.currentText()
        setattr(self.app_ctr,'p2_data_source',self.comboBox_p2.currentText())
        self.timer_save_data = QtCore.QTimer(self)
        self.timer_save_data.timeout.connect(self.save_data)
        #apply offset
        self.pushButton_apply_cut_width_offset.clicked.connect(self.apply_cut_width_offset)
        self.pushButton_apply_center_offset.clicked.connect(self.apply_center_offset)
        
def main():
    import shutil
    #copy config file
    shutil.copytree(xrv_path / 'config', user_config_path, dirs_exist_ok=True)
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())    

if __name__ == "__main__":
    main()
    