import sys
from PyQt5 import uic
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow,QApplication, QLabel
sys.path.append(str(Path(__file__).parent.parent.parent))
import dafy.projects.archiver.core.db_operations as db
import dafy.projects.archiver.core.gui_operations as gui
from dafy.bin.dafy_launcher import dispatcher
import logging
logging.basicConfig(filemode='w', filename= 'app.log', level=logging.INFO,\
                    format='%(levelname)s : %(name)s : %(message)s : %(asctime)s : %(lineno)d')
logger = logging.getLogger('')
f_handler = logging.FileHandler('app2.log',mode = 'w')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter('%(levelname)s : %(name)s : %(message)s : %(asctime)s : %(lineno)d'))
logger.addHandler(f_handler)

class MyMainWindow(QMainWindow):
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)
        self.db_operations = db
        logger.info('App is started successfully!')
        logging.info('Log once more!')

    def init_gui(self, ui):
        logger.info('Start init gui widgets!')
        self.ui = ui
        uic.loadUi(ui, self)
        gui.populate_config_template_files(self)
        self.statusbar = self.statusBar()
        self.statusLabel = QLabel(f"Welcome to ccg lib management system!")
        self.statusbar.addPermanentWidget(self.statusLabel)
        self.widget_terminal.update_name_space('main_gui',self)
        self.actionDatabaseCloud.triggered.connect(lambda:db.start_mongo_client_cloud(self))
        self.actionLogout.triggered.connect(lambda:db.logout(self))
        self.actionRegistration.triggered.connect(lambda:db.register_new_user(self))
        self.pushButton_load.clicked.connect(lambda:db.load_project(self))
        self.pushButton_new_project.clicked.connect(lambda:db.new_project_dialog(self))
        self.pushButton_update_project_info.clicked.connect(lambda:db.update_project_info(self))
        self.pushButton_new_scan.clicked.connect(lambda:db.add_scan_info(self))
        self.pushButton_update_scan_info.clicked.connect(lambda:db.update_scan_info(self))
        self.pushButton_delete_scan.clicked.connect(lambda:db.delete_one_scan(self))
        self.pushButton_new_sample.clicked.connect(lambda:db.add_sample_info(self))
        self.pushButton_update_sample_info.clicked.connect(lambda:db.update_sample_info(self))
        self.pushButton_delete_sample.clicked.connect(lambda:db.delete_one_sample(self))
        self.comboBox_sample_ids.activated.connect(lambda:db.extract_sample_info(self))
        self.comboBox_sample_ids.activated.connect(lambda:db.init_pandas_model_from_db(self))
        self.pushButton_extract_config.clicked.connect(lambda:gui.extract_config_template(self))
        self.pushButton_search.clicked.connect(lambda:gui.parse_query_conditions(self))
        self.pushButton_select_all.clicked.connect(lambda:gui.select_all(self))
        self.pushButton_select_none.clicked.connect(lambda:gui.select_none(self))
        self.pushButton_load_data_locally.clicked.connect(lambda:gui.load_csv_file(self))
        self.pushButton_save_local.clicked.connect(lambda:gui.save_csv_file(self))
        self.pushButton_plot.clicked.connect(lambda:gui.plot_processed_data(self))
        self.pushButton_save_cloud.clicked.connect(lambda:db.save_processed_data_to_cloud(self))
        self.pushButton_load_from_cloud.clicked.connect(lambda:db.load_processed_data_from_cloud(self))
        logger.info('Finish setting gui widgets!')

#@click.command()
#@click.option('--ui', default='beamtime_manager.ui',help="main gui ui file generated from Qt Desinger, possible ui files are :")
#@click.option('--ss', default ='darkstyle', help='style sheet file *.qss, possible qss files include: Takezo.qss')
#@click.option('--tm', default = 'False', help='show terminal widget (--tm True) or not (--tm False)')
def main():
    ss= 'darkstyle'
    tm = True
    ui_file = str(Path(__file__).parent.parent/ "ui" / 'beamtime_manager.ui')
    QApplication.setStyle("windows")
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.init_gui(ui_file)
    if tm=='False':
        myWin.widget_terminal.hide()
        myWin.label_3.hide()
    elif tm=='True':
        pass
    myWin.setWindowTitle('Beamtime management system')
    if ss == 'darkstyle':
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    else:
        style_sheet_path = str(Path(__file__).parent.parent/ "resources" / "stylesheets" / ss)
        File = open(style_sheet_path,'r')
        with File:
            qss = File.read()
            app.setStyleSheet(qss)    
    myWin.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
    