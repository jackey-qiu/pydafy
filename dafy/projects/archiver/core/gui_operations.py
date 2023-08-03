from functools import partial
import numpy as np
import PyQt5
import re
import pyqtgraph as pg
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
from pathlib import Path
from os import listdir
from .db_operations import logical_query, init_pandas_model_from_db
from ..config import config
from ..core.util import PandasModel
from .util import error_pop_up        

def populate_config_template_files(self):
    root = Path(__file__).parent.parent/ "resources" / "templates"
    init_files = [each[0:-4] for each in listdir(str(root)) if each.endswith('ini')]
    self.comboBox_templates.clear()
    self.comboBox_templates.addItems(init_files)

def extract_config_template(self):
    self.widget_analysis_config.init_pars(self.comboBox_templates.currentText()+'.ini')

def parse_query_conditions(self):
    logic = self.comboBox_logic.currentText()
    left_field = {self.comboBox_search_field.currentText():self.lineEdit_search_item1.text()}
    right_field = {self.comboBox_search_field2.currentText():self.lineEdit_search_item2.text()}
    fields = [each for each in config.display_fields if each!='select']
    targets = logical_query(self,'scan_info',logic,left_field,right_field,fields)
    init_pandas_model_from_db(self,targets)

def select_all(self):
    self.pandas_model_scan_info._data['select'] = True

def select_none(self):
    self.pandas_model_scan_info._data['select'] = False

def load_csv_file(self, header = 0, sep = ','):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","csv Files (*.csv);;text Files (*.txt)", options=options)
    if fileName:
        data = pd.read_csv(fileName, sep = ',', header = 0)
        data.insert(0, column = 'select', value = False)
        self.pandas_model_processed_data_info = PandasModel(data = data, tableviewer = self.tableView_processed_data, main_gui = self, rgb_bkg=(25,35,45),rgb_fg=(200,200,200))
        self.tableView_processed_data.setModel(self.pandas_model_processed_data_info)
        self.tableView_processed_data.resizeColumnsToContents()
        self.tableView_processed_data.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        self.tableView_processed_data.horizontalHeader().setStretchLastSection(True)

def save_csv_file(self):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "","csv Files (*.csv);;text Files (*.txt)", options=options)
    if fileName:
        columns = self.lineEdit_channels.text()
        try:
            if columns!='':
                columns = columns.replace(' ','').rsplit(',')
                self.pandas_model_processed_data_info._data.to_csv(fileName, index = False, columns = columns)
            else:
                self.pandas_model_processed_data_info._data.to_csv(fileName, index = False)
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Success to update processed data to cloud!')
        except Exception as e:
            error_pop_up('Fail to save csv file due to:'+str(e))

def plot_processed_data(self):
    conditions, x, y = self.lineEdit_conditions.text(),self.lineEdit_x.text(), self.lineEdit_y.text()
    pattern = re.compile(r'(\w+)([<,>,=]=?)(\d+\.?\d*)([&,|])?')
    groups = pattern.findall(conditions.replace(' ',''))
    shared_head = '(self.pandas_model_processed_data_info._data["{}"]{}{}) {}'
    final_str = ''
    for group in groups:
        final_str=final_str+shared_head.format(*group)
    complete_str = 'self.pandas_model_processed_data_info._data[{}].index'.format(final_str)
    index = eval(complete_str)
    x_value = np.array(self.pandas_model_processed_data_info._data[x][index])
    y_value = np.array(self.pandas_model_processed_data_info._data[y][index])
    self.widget_plot.clear()
    self.plot = self.widget_plot.addPlot(title=conditions)
    #make the line glow
    alphas = np.linspace(25, 25, 10, dtype=int)
    lws = np.linspace(1, 8, 10)
    for alpha, lw in zip(alphas, lws):
        pen = pg.mkPen(color='{}{:02x}'.format('#00ff41', alpha),
                               width=lw,
                               connect="finite")
        self.plot.plot(x_value, y_value, pen = pen, clear=(lw==lws[0]))
    self.plot.setLabel('left', "Y Axis", units='A')
    self.plot.setLabel('bottom', "Y Axis", units='s')
    self.plot.setLogMode(x=self.checkBox_x.isChecked(), y=self.checkBox_y.isChecked())


