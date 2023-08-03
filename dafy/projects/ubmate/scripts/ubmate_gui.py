import sys,os,configparser
from os import walk
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvas
import qdarkstyle
from PyQt5 import uic, QtCore
from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import imageio, cv2
from dafy.core.util.path import *
#api for surface unit cell generator
#diffcalc api funcs setup
sys.path.append(os.path.join(DaFy_path,'projects','orcalc'))
sys.path.append(os.path.join(ubmate_path,'core'))
import diffcalc.util
diffcalc.util.DEBUG = True
from diffcalc import settings
from diffcalc.hkl.you.geometry import SixCircle
from diffcalc.hardware import DummyHardwareAdapter
settings.hardware = DummyHardwareAdapter(('mu', 'delta', 'gam', 'eta', 'chi', 'phi'))
settings.geometry = SixCircle()  # @UndefinedVariable
import diffcalc.dc.dcyou as dc
from diffcalc.ub import ub
from diffcalc import hardware
from diffcalc.hkl.you import hkl as hkl_dc

from dafy.projects.ubmate.core.gui_operations import GuiOperations
from dafy.projects.ubmate.core.viewer_operations import ViewerOperations
from dafy.projects.ubmate.core.rspace_operations import RspaceOperations
from dafy.projects.ubmate.core.instrument_operations import InstrumentOperations

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MyMainWindow(QMainWindow, GuiOperations, ViewerOperations, RspaceOperations, InstrumentOperations):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        self.init_ui()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(ubmate_path/'icons'/'ubmate.png')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(24, 24))

    def init_ui(self):
        uic.loadUi(str(ubmate_path / 'ui' / 'ubmate.ui'),self)
        self.widget_terminal.update_name_space('main_gui',self)
        self.setWindowTitle('Data analysis factory: UBMate')
        self.widget_config.init_pars()
        #UB settings from diffcalc module
        self.ub = ub
        # self.ub.newub('current_ub')
        self.UB_INDEX = 0
        self.hkl = hkl_dc
        self.settings = settings
        self.hardware = hardware
        self.dc = dc
        self.cons = []

        #config parser
        self.structure_container = {}
        self.load_default_structure_file()
        self.lineEdit_save_folder.setText(os.path.join(DaFy_path,'cif'))
        self.comboBox_substrate_surfuc.currentIndexChanged.connect(self.update_file_names)
        self.pushButton_use_selected.clicked.connect(self.init_pandas_model_in_parameter)
        self.pushButton_load_more.clicked.connect(self.open_structure_files)
        self.config = configparser.RawConfigParser()
        self.config.optionxform = str # make entries in config file case sensitive
        self.base_structures = {}
        self.structures = []
        self.HKLs_dict = {}
        self.peaks_in_zoomin_viewer = {}
        self.trajactory_pos = []
        self.ax_simulation =None
        self.pushButton_draw.clicked.connect(lambda:self.show_structure(widget_name = 'widget_glview'))
        # self.pushButton_load.clicked.connect(self.load_config_file)
        self.pushButton_extract_in_viewer.clicked.connect(self.extract_peaks_in_zoom_viewer)
        # self.pushButton_update.clicked.connect(self.update_config_file)
        self.pushButton_launch.clicked.connect(self.launch_config_file_new)
        self.comboBox_names.currentIndexChanged.connect(self.update_HKs_list)
        self.comboBox_names.currentIndexChanged.connect(self.update_Bragg_peak_list)
        self.comboBox_bragg_peak.currentIndexChanged.connect(self.select_rod_based_on_Bragg_peak)
        self.pushButton_panup.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,-0.5))
        self.pushButton_pandown.clicked.connect(lambda:self.widget_glview_zoomin.pan(0,0,0.5))
        self.pushButton_plot_XRD_profiles.clicked.connect(self.draw_ctrs)
        # self.comboBox_working_substrate.currentIndexChanged.connect(self.fill_matrix)
        # self.pushButton_convert_abc.clicked.connect(self.cal_xyz)
        # self.pushButton_convert_xyz.clicked.connect(self.cal_abc)
        self.pushButton_convert_hkl.clicked.connect(self.cal_qxqyqz)
        self.pushButton_convert_qs.clicked.connect(self.cal_hkl)
        self.pushButton_extract.clicked.connect(self.extract_rod_from_a_sym_list)
        self.pushButton_calculate_hkl_reference.clicked.connect(self.cal_hkl_in_reference)
        self.pushButton_lscan.clicked.connect(lambda:self.calc_angs_in_scan(scan_type = 'l'))
        self.pushButton_escan.clicked.connect(lambda:self.calc_angs_in_scan(scan_type = 'energy'))
        # self.pushButton_compute.clicked.connect(self.compute_angles)
        self.timer_spin = QtCore.QTimer(self)
        self.timer_spin.timeout.connect(self.spin)
        self.timer_spin_sample = QtCore.QTimer(self)
        self.timer_spin_sample.timeout.connect(self.rotate_sample_SN)
        self.timer_l_scan = QtCore.QTimer(self)
        self.timer_l_scan.timeout.connect(self.simulate_l_scan_live)
        # self.timer_spin_sample.timeout.connect(self.simulate_image)
        self.azimuth_angle = 0
        self.pushButton_azimuth0.clicked.connect(self.azimuth_0)
        self.pushButton_azimuth90.clicked.connect(self.azimuth_90)
        self.pushButton_azimuth180.clicked.connect(self.azimuth_180)
        self.pushButton_panup_2.clicked.connect(lambda:self.pan_view([0,0,-1]))
        self.pushButton_pandown_2.clicked.connect(lambda:self.pan_view([0,0,1]))
        self.pushButton_panleft.clicked.connect(lambda:self.pan_view([0,-1,0]))
        self.pushButton_panright.clicked.connect(lambda:self.pan_view([0,1,0]))
        self.pushButton_start_spin.clicked.connect(self.start_spin)
        self.pushButton_stop_spin.clicked.connect(self.stop_spin)
        self.pushButton_bragg_peaks.clicked.connect(self.simulate_image_Bragg_reflections)
        #real space viewer control
        self.pushButton_azimuth0_2.clicked.connect(self.azimuth_0_2)
        self.pushButton_azimuth90_2.clicked.connect(self.azimuth_90_2)
        #self.pushButton_azimuth180_2.clicked.connect(self.azimuth_180_2)
        self.pushButton_elevation90.clicked.connect(self.elevation_90)
        self.pushButton_panup_4.clicked.connect(lambda:self.pan_view([0,0,-1],'widget_real_space'))
        self.pushButton_pandown_4.clicked.connect(lambda:self.pan_view([0,0,1],'widget_real_space'))

        self.pushButton_rotate.clicked.connect(self.rotate_sample)
        # self.pushButton_rotate.clicked.connect(self.simulate_image)

        self.pushButton_spin.clicked.connect(self.spin_)
        self.pushButton_draw_real_space.clicked.connect(self.draw_real_space)
        self.pushButton_simulate.clicked.connect(self.simulate_image)
        self.pushButton_send_detector.clicked.connect(self.send_detector)
        #ub matrix control
        self.pushButton_show_constraint_editor.clicked.connect(self.show_or_hide_constraints)
        self.pushButton_apply_constraints.clicked.connect(self.set_cons)
        self.comboBox_predefined_cons.currentTextChanged.connect(self.set_predefined_cons)
        self.pushButton_clear_cons.clicked.connect(self.clear_all_cons)
        self.pushButton_calc_angs.clicked.connect(self.calc_angs)
        self.pushButton_calc_hkl.clicked.connect(self.calc_hkl_dc)
        self.pushButton_to_HKL.clicked.connect(self.send_hkl_angs)
        self.comboBox_UB.currentTextChanged.connect(self.display_UB)
        self.pushButton_update_ub.clicked.connect(self.set_UB_matrix)
        self.pushButton_cal_ub.clicked.connect(self.add_refs)
        #surface unit cell generator
        self.pushButton_generate.clicked.connect(self.generate_surface_unitcell_info)
        self.pushButton_save_files.clicked.connect(self.save_structure_files)
        self.comboBox_TM_type.currentIndexChanged.connect(self.update_TM)
        #display diffractometer geometry figure
        
        image = imageio.imread(str(ubmate_path / 'icons' / '4s_2d_diffractometer.png'))
        #image = imageio.imread()
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, dsize=(int(745/1.2), int(553/1.2)), interpolation=cv2.INTER_CUBIC)
        self.im = QImage(image,image.shape[1],image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        # self.im = QImage(image,200,200, 200 * 3, QImage.Format_RGB888)
        # self.im.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.label_diffractometer.setPixmap(QPixmap(self.im))

        # self.pushButton_draw.clicked.connect(self.prepare_peaks_for_render)
        ##set style for matplotlib figures
        plt.style.use('ggplot')
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rc('font',size = 10)
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.grid'] = True
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['axes.facecolor']='0.7'
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['mathtext.default']='regular'
        #style.use('ggplot','regular')

def main():
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())    

if __name__ == "__main__":
    main()