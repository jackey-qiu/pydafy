import sys, os, logging
import pyqtgraph as pg
import matplotlib
matplotlib.use("Qt5Agg")
from pyqtgraph.Qt import QtGui
import qdarkstyle
from PyQt5 import uic, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QShortcut, QCheckBox, QRadioButton
from dafy.core.util.path import *
from dafy.projects.ctr.core.ctr_run_control import RunApp
from dafy.projects.ctr.core.gui_operations import GuiOperations
from dafy.core.util.DebugFunctions import QTextEditLogger

# pg.setConfigOption('background', (50,50,100))
# pg.setConfigOption('foreground', 'k')

class MyMainWindow(QMainWindow, GuiOperations):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        self.init_ui()
        #self.horizontalSlider.valueChanged.connect(self.change_peak_width)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(ctr_path/'icons'/'ctr.png')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(24, 24))

    def init_ui(self):
        uic.loadUi(str(ctr_path / 'ui' / 'CTR_bkg_pyqtgraph_new.ui'),self)
        self.widget_config.init_pars(data_type = self.comboBox_beamline.currentText())
        self.setWindowTitle('Data analysis factory: CTR data analasis')
        self.widget_terminal.update_name_space('main_gui',self)
        
        #set redirection of error message to embeted text browser widget
        logTextBox = QTextEditLogger(self.textBrowser_error_msg)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.WARNING)

        self.app_ctr=RunApp(beamline = self.comboBox_beamline.currentText())
        self.lineEdit.setText(os.path.join(self.app_ctr.data_path,'default.ini'))
        self.ref_data = None
        self.ref_fit_pars_current_point = {}
        self.current_image_no = 0
        self.current_scan_number = None
        self.bkg_intensity = 0
        self.bkg_clip_image = None
        self.image_log_scale = False
        self.run_mode = False
        self.image_set_up = False
        self.tag_reprocess = False
        self.roi_pos = None
        self.roi_size = None

        self.stop = False
        self.open.clicked.connect(self.load_file)
        self.launch.clicked.connect(self.launch_file)
        self.spinBox_peak_width.valueChanged.connect(self.change_peak_width)
        self.stopBtn.clicked.connect(self.stop_func)
        self.saveas.clicked.connect(self.save_file_as)
        self.save.clicked.connect(self.save_file)
        self.plot.clicked.connect(self.plot_figure)
        self.runstepwise.clicked.connect(self.set_tag_process_type)
        self.runstepwise.clicked.connect(self.plot_)
        self.pushButton_filePath.clicked.connect(self.locate_data_folder)
        self.pushButton_load_ref_data.clicked.connect(self.load_ref_data)
        self.lineEdit_data_file_name.setText('temp_data_ctr.xlsx')
        self.lineEdit_data_file_path.setText(self.app_ctr.data_path)
        self.actionOpenConfig.triggered.connect(self.load_file)
        self.actionSaveConfig.triggered.connect(self.save_file)
        self.actionRun.triggered.connect(self.set_tag_process_type)
        self.actionRun.triggered.connect(self.plot_)
        self.actionStop.triggered.connect(self.stop_func)
        self.actionSaveData.triggered.connect(self.save_data)
        self.pushButton_save_rod_data.clicked.connect(self.start_dL_BL_editor_dialog)
        setattr(self.app_ctr,'data_path',os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text()))
        for each in self.groupBox_2.findChildren(QCheckBox):
            each.released.connect(self.update_poly_order)
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            each.toggled.connect(self.update_cost_func)
        self.pushButton_remove_current_point.clicked.connect(self.remove_data_point)
        self.pushButton_left.clicked.connect(self.move_roi_left)
        self.pushButton_right.clicked.connect(self.move_roi_right)
        self.pushButton_up.clicked.connect(self.move_roi_up)
        self.pushButton_down.clicked.connect(self.move_roi_down)
        self.pushButton_set_roi.clicked.connect(self.set_roi)
        self.pushButton_go.clicked.connect(self.reprocess_previous_frame)
        self.comboBox_beamline.currentTextChanged.connect(self.change_config_layout)
        self.pushButton_track_peak.clicked.connect(self.track_peak)
        self.pushButton_set_peak.clicked.connect(self.set_peak)

        self.leftShort = QShortcut(QtGui.QKeySequence("Ctrl+Left"), self)
        self.leftShort.activated.connect(self.move_roi_left)
        self.rightShort = QShortcut(QtGui.QKeySequence("Ctrl+Right"), self)
        self.rightShort.activated.connect(self.move_roi_right)
        self.upShort = QShortcut(QtGui.QKeySequence("Ctrl+Up"), self)
        self.upShort.activated.connect(self.move_roi_up)
        self.downShort = QShortcut(QtGui.QKeySequence("Ctrl+Down"), self)
        self.downShort.activated.connect(self.move_roi_down)
        self.switch_roi_adjustment_type_short = QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self.switch_roi_adjustment_type_short.activated.connect(self.switch_roi_adjustment_type)

        self.nextShort = QShortcut(QtGui.QKeySequence("Right"), self)
        self.nextShort.activated.connect(self.plot_)
        self.deleteShort = QShortcut(QtGui.QKeySequence("Down"), self)
        self.deleteShort.activated.connect(self.remove_data_point)

        # self.checkBox_use_log_scale.stateChanged.connect(self.set_log_image)
        self.radioButton_fixed_percent.clicked.connect(self.update_image)
        self.radioButton_fixed_between.clicked.connect(self.update_image)
        self.radioButton_automatic_set_hist.clicked.connect(self.update_image)
        self.lineEdit_scale_factor.returnPressed.connect(self.update_image)
        self.lineEdit_tailing_factor.returnPressed.connect(self.update_image)
        self.lineEdit_left.returnPressed.connect(self.update_image)
        self.lineEdit_right.returnPressed.connect(self.update_image)

        self.radioButton_traditional.toggled.connect(self.update_ss_factor)
        self.radioButton_vincent.toggled.connect(self.update_ss_factor)
        self.doubleSpinBox_ss_factor.valueChanged.connect(self.update_ss_factor)

        self.comboBox_p3.activated.connect(self.select_source_for_plot_p3)
        self.comboBox_p4.activated.connect(self.select_source_for_plot_p4)
        self.p3_data_source = self.comboBox_p3.currentText()
        self.p4_data_source = self.comboBox_p4.currentText()
        setattr(self.app_ctr,'p3_data_source',self.comboBox_p3.currentText())
        setattr(self.app_ctr,'p4_data_source',self.comboBox_p4.currentText())
        self.timer_save_data = QtCore.QTimer(self)

def main():
    import shutil
    #copy config file
    shutil.copytree(ctr_path / 'config', user_config_path, dirs_exist_ok=True)
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())    

if __name__ == "__main__":
    main()