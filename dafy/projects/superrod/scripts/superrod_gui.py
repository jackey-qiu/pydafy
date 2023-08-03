import sys,os
import logging
root = logging.getLogger()
root.setLevel(logging.DEBUG)
import qdarkstyle
import pyqtgraph as pg
import matplotlib
matplotlib.use("TkAgg")
os.environ["QT_MAC_WANTS_LAYER"] = "1"
matplotlib.rc('image', cmap='prism')
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QAbstractItemView, QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqtgraph.Qt import QtGui
from dafy.projects.superrod.widgets import syntax_pars
from dafy.core.util.path import superrod_path
from dafy.core.util.DebugFunctions import QTextEditLogger
from dafy.core.EnginePool.diffev import fit_model_NLLS
from dafy.projects.superrod.core.models import model
from dafy.projects.superrod.core.models import solvergui
from dafy.projects.superrod.core.superrod_run_control import ScanPar, RunFit, RunBatch, RunControl
from dafy.projects.superrod.core.superrod_io_control import SuperRodIoControl
from dafy.projects.superrod.core.graph_operations import GraphOperations
from dafy.projects.superrod.core.view_operations import ViewOperations
from dafy.projects.superrod.core.data_preprocessing import DataProcessing
from dafy.projects.superrod.core.gui_operations import GuiOperations

class MyMainWindow(QMainWindow, RunControl, GuiOperations, SuperRodIoControl, GraphOperations, ViewOperations, DataProcessing):
    """
    GUI class for this app
    ....
    Attributes (selected)
    -----------
    <<widgets>>
    tableWidget_data: QTableWidget holding a list of datasets
    tableWidget_data_view: QTableWidget displaying each dataset
    widget_solver:pyqtgraph.parameter_tree_widget where you define
                  intrinsic parameters for undertaking DE optimization
    tableWidget_pars: QTableWidget displaying fit parameters
    widget_data: pyqtgraph.GraphicsLayoutWidget showing figures of
                 each ctr profile (data, fit, ideal and errorbars)
    widget_fom: pyqtgraph.GraphicsLayoutWidget showing evolution of
                figure of merit during fit
    widget_pars:pyqtgraph.GraphicsLayoutWidget showing best fit of
                each parameter at current generation and the search
                range in bar chart at this moment. longer bar means
                more aggressive searching during fit. If the bars 
                converge to one narrow line, fit quality cannot improved
                anymore. That means the fit is finished.
    widget_edp: GLViewWidget showing the 3d molecular structure of the
                current best fit model.
    widget_msv_top: GLViewWidget showing the top view of 3d molecular
                structure of the current best fit model. Only one sorbate
                and one layer of surface atoms are shown for clarity.
    plainTextEdit_script: QCodeEditor widget showing the model script
    widget_terminal:TerminalWidget, where you can do whatever you can in
                a normal python terminal. Three variables are loaded in 
                the namespace of the terminal:
                1) win: GUI main frame
                2) model: model that bridget script_module, pars and Fit engine
                you can explore the variables defined in your model script
                using model.script_module.vars (if vars is defined in script)
    <<others>>
    run_fit: Run_Fit instance to be launched to start/stop a fit. Refer to
             Run_Fit.solver to learn more about implementation of multi-processing
             programe method.
    model: model instance to coordinate script name space, dataset instance and par
           instance
    f_ideal: a list holding the structure factor values for unrelaxed structure
    data_profile: list of handles to plot ctr profiles including data and fit reuslts

    Methods (selected)
    -----------
    """
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        # self.setupUi(self)
        #pyqtgraph preference setting
        pg.setConfigOptions(imageAxisOrder='row-major', background = (50,50,100))
        pg.mkQApp()
        self.init_ui()
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(str(superrod_path/'icons'/'superrod.png')), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.setIconSize(QtCore.QSize(30, 30))
        self.setWindowTitle('Data analysis factory: CTR data modeling')

    def init_ui(self):
        #load GUI ui file made by qt designer
        uic.loadUi(str(superrod_path / 'ui' / 'superrod_gui.ui'),self)
        self.widget_terminal.update_name_space("win",self)
        #set redirection of error message to embeted text browser widget
        self.logTextBox = QTextEditLogger(self.textBrowser_error_msg)
        # You can format what is printed to text box
        self.logTextBox.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(self.logTextBox)

        # self.comboBox_all_motif.insertItems(0, sorbate_tool.ALL_MOTIF_COLLECTION)
        #self.stop = False
        self.show_checkBox_list = []
        self.domain_tag = 1
        #structure factor for ideal structure
        self.f_ideal=[]
        self.data_profiles = []
        self.model = model.Model()
        self.nlls_fit = fit_model_NLLS(self.model)
        
        self.timer_nlls = QTimer(self)
        self.timer_nlls.timeout.connect(self.update_status_nlls)
        #init run_fit
        self.run_fit = RunFit(solvergui.SolverController(self.model))
        self.fit_thread = QtCore.QThread()
        self.run_fit.moveToThread(self.fit_thread)#move run_fit to a different thread
        #signal-slot connection
        self.run_fit.updateplot.connect(self.update_par_during_fit)
        self.run_fit.updateplot.connect(self.update_status)
        self.run_fit.fitended.connect(self.stop_model_slot)
        self.fit_thread.started.connect(self.run_fit.run)

        #init run_batch
        self.run_batch = RunBatch(solvergui.SolverController(self.model))
        self.batch_thread = QtCore.QThread()
        self.run_batch.moveToThread(self.batch_thread)
        #signal-slot connection
        self.run_batch.updateplot.connect(self.update_par_during_fit)
        self.run_batch.updateplot.connect(self.update_status_batch)
        self.run_batch.fitended.connect(self.stop_model_batch)
        # self.run_batch.fitended.connect(lambda:self.timer_update_structure.stop())
        self.batch_thread.started.connect(self.run_batch.run)
        # self.batch_thread.started.connect(lambda:self.timer_update_structure.start(2000))

        self.scan_par = ScanPar(self.model)
        self.scan_par_thread = QtCore.QThread()
        self.scan_par.moveToThread(self.scan_par_thread)
        self.scan_par_thread.started.connect(self.scan_par.run)
        self.pushButton_scan.clicked.connect(self.start_scan_par_thread)
        self.pushButton_stop_scan.clicked.connect(self.stop_scan_par_thread)
        self.timer_scan_par = QTimer(self)
        self.timer_scan_par.timeout.connect(self.update_structure_during_scan_par)
        #tool bar buttons to operate model
        self.actionNew.triggered.connect(self.init_new_model)
        self.actionOpen.triggered.connect(self.open_model)
        self.actionSaveas.triggered.connect(self.save_model_as)
        self.actionSave.triggered.connect(self.save_model)
        self.actionSimulate.triggered.connect(lambda:self.simulate_model(compile = True))
        # self.actionCompile.triggered.connect(lambda:self.simulate_model(compile = False))
        self.actionRun.triggered.connect(self.run_model)
        self.actionStop.triggered.connect(self.stop_model)
        self.actionCalculate.triggered.connect(self.calculate_error_bars)
        self.actionRun_batch_script.triggered.connect(self.run_model_batch)
        self.actionStopBatch.triggered.connect(self.terminate_model_batch)

        #menu items
        self.actionOpen_model.triggered.connect(self.open_model)
        self.actionSave_model.triggered.connect(self.save_model_as)
        self.actionSimulate_2.triggered.connect(lambda:self.simulate_model(compile = True))
        self.actionStart_fit.triggered.connect(self.run_model)
        self.actionNLLS.triggered.connect(self.start_nlls)
        self.actionStop_fit.triggered.connect(self.stop_model)
        self.actionSave_table.triggered.connect(self.save_par)
        self.actionSave_script.triggered.connect(self.save_script)
        self.actionSave_data.triggered.connect(self.save_data)
        self.actionData.changed.connect(self.toggle_data_panel)
        self.actionPlot.changed.connect(self.toggle_plot_panel)
        self.actionScript.changed.connect(self.toggle_script_panel)

        self.pushButton_generate_script.clicked.connect(self.generate_script_dialog)

        #pushbuttons for model file navigator 
        self.pushButton_load_files.clicked.connect(self.load_rod_files)
        self.pushButton_clear_selected_files.clicked.connect(self.remove_selected_rod_files)
        self.listWidget_rod_files.itemDoubleClicked.connect(self.open_model_selected_in_listWidget)
        self.pushButton_open_selected_rod_file.clicked.connect(self.open_model_selected_in_listWidget)
        self.pushButton_hook_to_batch.clicked.connect(self.hook_to_batch)
        self.pushButton_purge_from_batch.clicked.connect(self.purge_from_batch)
        self.actionpreviousModel.triggered.connect(self.load_previous_rod_file_in_batch)
        self.actionnextModel.triggered.connect(self.load_next_rod_file_in_batch)

        #pushbuttons for data handeling
        self.pushButton_load_data.clicked.connect(self.load_data_ctr)
        self.pushButton_append_data.clicked.connect(self.append_data)
        self.pushButton_delete_data.clicked.connect(self.delete_data)
        self.pushButton_save_data.clicked.connect(self.save_data)
        self.pushButton_update_mask.clicked.connect(self.update_mask_info_in_data)
        self.pushButton_use_all.clicked.connect(self.use_all_data)
        self.pushButton_use_none.clicked.connect(self.use_none_data)
        self.pushButton_use_selected.clicked.connect(self.use_selected_data)
        self.pushButton_invert_use.clicked.connect(self.invert_use_data)
        self.pushButton_dummy_data.clicked.connect(self.generate_dummy_data_dialog)

        #pushbuttons for structure view
        self.pushButton_azimuth_0.clicked.connect(self.azimuth_0)
        self.pushButton_azimuth_90.clicked.connect(self.azimuth_90)
        self.pushButton_elevation_0.clicked.connect(self.elevation_0)
        self.pushButton_elevation_90.clicked.connect(self.elevation_90)
        self.pushButton_parallel.clicked.connect(self.parallel_projection)
        self.pushButton_projective.clicked.connect(self.projective_projection)
        self.pushButton_pan.clicked.connect(self.pan_msv_view)
        self.pushButton_start_spin.clicked.connect(self.start_spin)
        self.pushButton_stop_spin.clicked.connect(self.stop_spin)
        self.pushButton_xyz.clicked.connect(self.save_structure_file)

        #spinBox to save the domain_tag
        self.spinBox_domain.valueChanged.connect(self.update_domain_index)

        #pushbutton to load/save script
        self.pushButton_load_script.clicked.connect(self.load_script)
        self.pushButton_save_script.clicked.connect(self.save_script)
        # self.pushButton_modify_script.clicked.connect(self.modify_script)

        #pushbutton to load/save parameter file
        self.pushButton_load_table.clicked.connect(self.load_par)
        self.pushButton_save_table.clicked.connect(self.save_par)
        self.pushButton_remove_rows.clicked.connect(self.remove_selected_rows)
        self.pushButton_add_one_row.clicked.connect(self.append_one_row)
        self.pushButton_add_par_set.clicked.connect(lambda:self.append_par_set(par_selected=None))
        self.pushButton_add_all_pars.clicked.connect(self.append_all_par_sets)
        self.pushButton_fit_all.clicked.connect(self.fit_all)
        self.pushButton_fit_none.clicked.connect(self.fit_none)
        self.pushButton_fit_selected.clicked.connect(self.fit_selected)
        self.pushButton_fit_next_5.clicked.connect(self.fit_next_5)
        self.pushButton_invert_fit.clicked.connect(self.invert_fit)
        self.pushButton_update_pars.clicked.connect(self.update_model_parameter)
        self.horizontalSlider_par.valueChanged.connect(self.play_with_one_par)
        self.pushButton_scan.clicked.connect(self.scan_one_par)

        #pushButton to operate plots
        self.pushButton_update_plot.clicked.connect(lambda:self.update_structure_view(compile = True))
        self.pushButton_update_plot.clicked.connect(lambda:self.update_plot_data_view_upon_simulation(q_correction = False))
        self.pushButton_update_plot.clicked.connect(self.update_par_bar_during_fit)
        self.pushButton_update_plot.clicked.connect(self.update_electron_density_profile)
        self.pushButton_previous_screen.clicked.connect(self.show_plots_on_previous_screen)
        self.pushButton_next_screen.clicked.connect(self.show_plots_on_next_screen)
        #q correction widgets
        self.groupBox_q_correction.hide()
        self.pushButton_show.clicked.connect(lambda:self.update_plot_data_view_upon_simulation(q_correction = True))
        self.pushButton_append.clicked.connect(self.append_L_scale)
        self.pushButton_reset.clicked.connect(self.reset_L_scale)
        self.fit_q_correction = False
        self.apply_q_correction = False
        self.pushButton_fit.clicked.connect(self.fit_q)
        self.pushButton_apply.clicked.connect(self.update_q)
        self.pushButton_q_correction.clicked.connect(lambda:self.groupBox_q_correction.show())
        self.pushButton_hide.clicked.connect(lambda:self.groupBox_q_correction.hide())
        #select dataset in the viewer
        self.comboBox_dataset.activated.connect(self.update_data_view)

        #GAF viewer
        self.pushButton_generate_GAF.clicked.connect(self.generate_gaf_plot)

        #syntax highlight for script
        self.plainTextEdit_script.setStyleSheet("""QPlainTextEdit{
                                font-family:'Consolas';
                                font-size:14pt;
                                color: #ccc;
                                background-color: #2b2b2b;}""")
        self.plainTextEdit_script.setTabStopWidth(self.plainTextEdit_script.fontMetrics().width(' ')*4)

        #table view for parameters set to selecting row basis
        self.tableWidget_pars.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget_data.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.timer_update_structure = QtCore.QTimer(self)
        self.timer_update_structure.timeout.connect(self.pushButton_update_plot.click)
        self.timer_spin_msv = QtCore.QTimer(self)
        self.timer_spin_msv.timeout.connect(self.spin_msv)
        self.azimuth_angle = 0
        self.setup_plot()
        self._load_par()

        #widgets for plotting figures
        self.widget_fig.parent = self
        self.pushButton_extract_data.clicked.connect(lambda:self.widget_fig.extract_data_all())
        self.pushButton_reset_plot.clicked.connect(lambda:self.widget_fig.reset())
        self.pushButton_init_pars.clicked.connect(lambda:self.widget_fig.init_pandas_model())
        self.pushButton_plot_figures.clicked.connect(lambda:self.widget_fig.create_plots())
        self.pushButton_clear_plot.clicked.connect(lambda:self.widget_fig.clear_plot())

        #widgets for model result evaluation
        self.pushButton_cov.clicked.connect(self.generate_covarience_matrix)
        self.pushButton_sensitivity.clicked.connect(self.screen_parameters)

        #set logging
        self.pushButton_clear_log.clicked.connect(self.clear_log)
        self.comboBox_sel_log_type.activated.connect(self.select_log_level)
        self.pushButton_save_log.clicked.connect(self.save_log_info)
        #help tree widget
        # self.treeWidget.itemDoubleClicked.connect(self.open_help_doc)

def main():    
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    #get dpi info: dots per '''IINFOh
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()
    myWin = MyMainWindow()
    myWin.dpi = dpi
    hightlight = syntax_pars.PythonHighlighter(myWin.plainTextEdit_script.document())
    myWin.plainTextEdit_script.show()
    myWin.plainTextEdit_script.setPlainText(myWin.plainTextEdit_script.toPlainText())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()