import sys,os,qdarkstyle
import matplotlib.pyplot as plt
import matplotlib
from pyqtgraph.Qt import QtGui
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from dafy.core.util.path import viewer_path
from dafy.core.util.cv_tool import cvAnalysis
from dafy.core.util.PlotSetup import plot_tafel_from_formatted_cv_info
from dafy.projects.viewer.core.data_preprocessing import DataPreprocessing
from dafy.projects.viewer.core.graph_operations import GraphOperations
from dafy.projects.viewer.core.gui_operations import GuiOperations
from dafy.projects.viewer.core.viewer_io_control import ViewerIOControl

os.environ["QT_MAC_WANTS_LAYER"] = "1"

class MyMainWindow(QMainWindow, DataPreprocessing, GraphOperations, GuiOperations, ViewerIOControl):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)

        #setting plotting style
        matplotlib.rc('xtick', labelsize=10)
        matplotlib.rc('ytick', labelsize=10)
        plt.rcParams.update({'axes.labelsize': 10})
        plt.rc('font',size = 10)
        plt.rcParams['axes.linewidth'] = 1.5
        # plt.rcParams['axes.grid'] = True
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams["errorbar.capsize"] = 5
        # plt.rcParams['axes.facecolor']='0.7'
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['mathtext.default']='regular'
        plt.rcParams['savefig.dpi'] = 300

        #init meta info
        self.data_to_save = {}
        self.image_range_info = {}
        self.cv_tool = cvAnalysis()
        self.potential_offset = 0.055
        self.data = None
        self.data_summary = {}
        self.data_range = None
        self.pot_range = None
        self.potential = []
        self.show_frame = True
        self.plot_lib = {}
        self.charge_info = {}
        self.grain_size_info_all_scans = {}
        self.strain_info_all_scans = {}#key is scan_no, each_item is {(pot1,pot2):{"vertical":(abs_value,value_change),"horizontal":(abs_value,value_change)},"pH":pH value}}
        self.pot_ranges = {}
        self.cv_info = {}
        self.tick_label_settings = {}
        self.plot_tafel = plot_tafel_from_formatted_cv_info
        self.GUI_metaparameter_channels  = ['lineEdit_data_file',
                                            'lineEdit_resistance',
                                            'checkBox_time_scan',
                                            'checkBox_use',
                                            'checkBox_mask',
                                            'checkBox_max',
                                            'lineEdit_x',
                                            'lineEdit_y',
                                            'scan_numbers_append',
                                            'lineEdit_fmt',
                                            'lineEdit_potential_range', 
                                            'lineEdit_pot_range', 
                                            'lineEdit_scan_rate',
                                            'lineEdit_data_range',
                                            'lineEdit_colors_bar',
                                            'checkBox_use_external_cv',
                                            'checkBox_use_internal_cv',
                                            'checkBox_plot_slope',
                                            'checkBox_use_external_slope',
                                            'lineEdit_pot_offset',
                                            'lineEdit_cv_folder',
                                            'lineEdit_slope_file',
                                            'lineEdit_reference_potential',
                                            'checkBox_show_marker',
                                            'checkBox_merge',
                                            'lineEdit_input_values',
                                            'lineEdit_input_name',
                                            'lineEdit_OER_j',
                                            'lineEdit_OER_E',
                                            'lineEdit_hwspace',
                                            'checkBox_panel1',
                                            'checkBox_panel2',
                                            'checkBox_panel3',
                                            'checkBox_panel4',
                                            'lineEdit_partial_set_p1',
                                            'lineEdit_partial_set_p2',
                                            'lineEdit_partial_set_p3',
                                            'lineEdit_partial_set_p4',
                                            'checkBox_marker',
                                            'comboBox_link_container'
                                            ]

        #init ui
        self.init_ui()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(viewer_path/'icons'/'viewer.png')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(30, 30))
        self.setWindowTitle('Data analysis factory: CTR data modeling')

    def init_ui(self):
        uic.loadUi(os.path.join(viewer_path,'ui','data_viewer_xrv_gui.ui'),self)
        self.lineEdit_data_range.hide()
        self.widget_terminal.update_name_space('main_gui',self)
        self.addToolBar(self.mplwidget.navi_toolbar)
        #connect signal and slot
        self.actionLoadData.triggered.connect(self.load_file)
        self.actionPlotData.triggered.connect(self.plot_figure_xrv)
        self.actionPlotData.triggered.connect(self.print_data_summary)
        self.actionPlotRate.triggered.connect(self.plot_data_summary_xrv)
        self.actionShowHide.triggered.connect(self.show_or_hide)
        self.pushButton_cal_charge.clicked.connect(self.plot_figure_xrv)
        self.pushButton_cal_charge.clicked.connect(self.print_data_summary)
        self.PushButton_append_scans.clicked.connect(self.append_scans_xrv)
        self.checkBox_time_scan.clicked.connect(self.set_plot_channels)
        self.checkBox_mask.clicked.connect(self.append_scans_xrv)
        self.actionLoadConfig.triggered.connect(self.load_config)
        self.actionSaveConfig.triggered.connect(self.save_config)
        self.pushButton_update.clicked.connect(self.update_plot_range)
        self.pushButton_update.clicked.connect(self.append_scans_xrv)
        self.pushButton_update_info.clicked.connect(self.make_plot_lib)
        self.pushButton_append_rows.clicked.connect(lambda:self.update_pandas_model_cv_setting())
        self.pushButton_apply.clicked.connect(self.update_pot_offset)
        self.pushButton_tweak.clicked.connect(self.tweak_one_channel)
        self.pushButton_load_cv_config.clicked.connect(self.load_cv_config_file)
        self.pushButton_update_cv_config.clicked.connect(self.update_cv_config_file)
        self.pushButton_plot_cv.clicked.connect(self.plot_cv_data)
        self.pushButton_cal_charge_2.clicked.connect(self.calculate_charge_2)
        self.pushButton_plot_reaction_order.clicked.connect(self.plot_reaction_order_and_tafel)
        self.pushButton_get_pars.clicked.connect(self.project_cv_settings)

        #self.pushButton_save_data.clicked.connect(self.save_data_method)
        #self.pushButton_save_xrv_data.clicked.connect(self.save_xrv_data)
        #self.pushButton_plot_datasummary.clicked.connect(self.plot_data_summary_xrv)

        self.init_pandas_model_ax_format()
        self.init_pandas_model_cv_setting()
        self.pushButton_bkg_fit.clicked.connect(self.perform_bkg_fitting)
        self.pushButton_extract_cv.clicked.connect(self.extract_cv_data)
        self.pushButton_project.clicked.connect(self.project_to_master)
        self.pushButton_add_link.clicked.connect(self.add_one_link)
        self.pushButton_remove_selected.clicked.connect(self.remove_one_item)
        self.pushButton_remove_all.clicked.connect(lambda:self.comboBox_link_container.clear())

def main():    
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()