import copy
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QBrush, QColor
from PyQt5.QtWidgets import QCheckBox,QTableWidgetItem
from dafy.projects.superrod.widgets import syntax_pars
from dafy.projects.superrod.widgets.custome_widgets import ScriptGeneraterDialog, DummydataGeneraterDialog
from dafy.projects.superrod.core.models.structure_tools.sxrd_dafy import AtomGroup
from dafy.projects.superrod.core.models.utils import UserVars
import logging

class GuiOperations(object):

    def clear_log(self):
        self.textBrowser_error_msg.clear()
        self.logTextBox.records = 0

    def select_log_level(self):
        log = self.comboBox_sel_log_type.currentText()
        self.logTextBox.set_level(log)
        logging.root.critical('New logging level ({}) started from here.'.format(log))

    def open_help_doc(self):
        print('Double clicked signal received!')
        # return self.treeWidget.currentItem()

    def generate_script_dialog(self):
        dlg = ScriptGeneraterDialog(self)
        hightlight = syntax_pars.PythonHighlighter(dlg.plainTextEdit_script.document())
        dlg.plainTextEdit_script.show()
        dlg.plainTextEdit_script.setPlainText(dlg.plainTextEdit_script.toPlainText())
        dlg.exec()

    def generate_dummy_data_dialog(self):
        dlg = DummydataGeneraterDialog(self)
        dlg.exec()

    def toggle_data_panel(self):
        """data panel on the left side of GUI main frame"""
        self.tabWidget_data.setVisible(self.actionData.isChecked())

    def toggle_plot_panel(self):
        """plot panel on the top right side of main GUI frame"""
        self.tabWidget.setVisible(self.actionPlot.isChecked())

    def toggle_script_panel(self):
        """script panel on the bottom right side of main GUI frame"""
        self.tabWidget_2.setVisible(self.actionScript.isChecked())

    def update_domain_index(self):
        """update domain index, triggering the associated structure to show"""
        self.domain_tag = int(self.spinBox_domain.text())
        if self.model.compiled:
            self.widget_edp.items = []
            # self.widget_msv_top.items = []
            self.init_structure_view()
        else:
            pass

    def update_data_check_attr(self):
        """update the checkable attr of each dataset: use, show, showerror"""
        re_simulate = False
        for i in range(len(self.model.data)):
            #model.data: masked data
            self.model.data[i].show = self.tableWidget_data.cellWidget(i,1).isChecked()
            self.model.data[i].use_error = self.tableWidget_data.cellWidget(i,3).isChecked()
            #model.data_original: unmasked data for model saving later
            self.model.data_original[i].show = self.tableWidget_data.cellWidget(i,1).isChecked()
            self.model.data_original[i].use_error = self.tableWidget_data.cellWidget(i,3).isChecked()
            if self.model.data[i].use!=self.tableWidget_data.cellWidget(i,2).isChecked():
                re_simulate = True
                self.model.data[i].use = self.tableWidget_data.cellWidget(i,2).isChecked()
                self.model.data_original[i].use = self.tableWidget_data.cellWidget(i,2).isChecked()
        if re_simulate:
            self.simulate_model()

    def calc_f_ideal(self):
        self.f_ideal = []
        for i in range(len(self.model.data)):
            each = self.model.data[i]
            if each.x[0]>1000:#indicate energy column
                self.f_ideal.append(self.model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.extra_data['Y'])**2)
            else:
                self.f_ideal.append(self.model.script_module.sample.calc_f_ideal(each.extra_data['h'], each.extra_data['k'], each.x)**2)

    def update_combo_box_list_par_set(self):
        """atomgroup and uservars instances defined in script will be colleced and displayed in this combo box"""
        attrs = self.model.script_module.__dir__()
        attr_wanted = [each for each in attrs if type(getattr(self.model.script_module, each)) in [AtomGroup, UserVars]]
        num_items = self.comboBox_register_par_set.count()
        for i in range(num_items):
            self.comboBox_register_par_set.removeItem(0)
        self.comboBox_register_par_set.insertItems(0,attr_wanted)

    def update_table_widget_data(self):
        self.tableWidget_data.clear()
        self.tableWidget_data.setRowCount(len(self.model.data))
        self.tableWidget_data.setColumnCount(5)
        self.tableWidget_data.setHorizontalHeaderLabels(['DataID','logY','Use','Errors','fmt'])
        for i in range(len(self.model.data)):
            current_data = self.model.data[i]
            name = current_data.name
            for j in range(5):
                if j == 0:
                    qtablewidget = QTableWidgetItem(name)
                    self.tableWidget_data.setItem(i,j,qtablewidget)
                elif j == 4:
                    qtablewidget = QTableWidgetItem('sym:6bw;l:3r')
                    self.tableWidget_data.setItem(i,j,qtablewidget)
                else:
                    #note j=1 to j=3 corresponds to data.show, data.use, data.use_error
                    #data.show is not used for judging showing or not(all datasets are shown)
                    #It is instead used to specify the scale of Y(log or not)
                    check = getattr(current_data, ['show', 'use', 'use_error'][j-1])
                    check_box = QCheckBox()
                    check_box.setChecked(check)
                    #check_box.stateChanged.connect(self.update_plot_data_view)
                    self.tableWidget_data.setCellWidget(i,j,check_box)
        
        # self.tableWidget_data.resizeColumnsToContents()
        # self.tableWidget_data.resizeRowsToContents()

    def use_all_data(self):
        """fit all datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            self.tableWidget_data.cellWidget(i,2).setChecked(True)
        self.simulate_model()

    def use_none_data(self):
        """fit none of those datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def use_selected_data(self):
        """fit those that have been selected
        """
        selected_row_index = [each.row() for each in self.tableWidget_data.selectionModel().selectedRows()]
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            if i in selected_row_index:
                self.tableWidget_data.cellWidget(i,2).setChecked(True)
            else:
                self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def invert_use_data(self):
        """invert the selection of to-be-fit datasets
        """
        num_rows_table = self.tableWidget_data.rowCount()
        for i in range(num_rows_table):
            checkstate = self.tableWidget_data.cellWidget(i,2).checkState()
            if checkstate == 0:
                self.tableWidget_data.cellWidget(i,2).setChecked(True)
            else:
                self.tableWidget_data.cellWidget(i,2).setChecked(False)
        self.simulate_model()

    def update_combo_box_dataset(self):
        new_items = [each.name for each in self.model.data]
        self.comboBox_dataset.clear()
        self.comboBox_dataset.addItems(new_items)
        self.comboBox_dataset2.clear()
        self.comboBox_dataset2.addItems(new_items)

    #used in q correction
    def return_L_I(self):
        dataset_name = self.comboBox_dataset2.currentText()
        dataset = None
        for each in self.model.data_original:
            if each.name == dataset_name:
                dataset = each
                break
            else:
                pass
        return dataset.x, dataset.y

    def update_data_view(self):
        """update the data view widget to show data values as table"""
        dataset_name = self.comboBox_dataset.currentText()
        dataset = None
        for each in self.model.data_original:
            if each.name == dataset_name:
                dataset = each
                break
            else:
                pass
        column_labels_main = ['x','y','error','mask']
        extra_labels = ['h', 'k', 'dL', 'LB']
        all_labels = ['x','y','error','h','k','dL','LB','mask']
        self.tableWidget_data_view.setRowCount(len(dataset.x))
        self.tableWidget_data_view.setColumnCount(len(all_labels))
        self.tableWidget_data_view.setHorizontalHeaderLabels(all_labels)
        for i in range(len(dataset.x)):
            for j in range(len(all_labels)):
                if all_labels[j] in column_labels_main:
                    item_ = getattr(dataset,all_labels[j])[i]
                    if all_labels[j] == 'mask':
                        qtablewidget = QTableWidgetItem(str(item_))
                    else:
                        qtablewidget = QTableWidgetItem(str(round(item_,4)))
                elif all_labels[j] in extra_labels:
                    qtablewidget = QTableWidgetItem(str(dataset.get_extra_data(all_labels[j])[i]))
                else:
                    qtablewidget = QTableWidgetItem('True')
                self.tableWidget_data_view.setItem(i,j,qtablewidget)

    def update_mask_info_in_data(self):
        """if the mask value is False, the associated data point wont be shown and wont be fitted as well"""
        dataset_name = self.comboBox_dataset.currentText()
        dataset = None
        for each in self.model.data_original:
            if each.name == dataset_name:
                dataset = each
                break
            else:
                pass
        for i in range(len(dataset.x)):
            dataset.mask[i] = (self.tableWidget_data_view.item(i,7).text() == 'True')
        self.model.data = copy.deepcopy(self.model.data_original)
        [each.apply_mask() for each in self.model.data]
        #updae the data infomation
        self.model.data.concatenate_all_ctr_datasets()
        self.simulate_model()

    def init_mask_info_in_data_upon_loading_model(self):
        """apply mask values to each dataset"""
        self.model.data = copy.deepcopy(self.model.data_original)
        [each.apply_mask() for each in self.model.data]
        self.simulate_model()

    #not implemented!
    def change_plot_style(self):
        if self.background_color == 'w':
            self.widget_data.getViewBox().setBackgroundColor('k')
            self.widget_edp.getViewBox().setBackgroundColor('k')
            # self.widget_msv_top.getViewBox().setBackgroundColor('k')
            self.background_color = 'k'
        else:
            self.widget_data.getViewBox().setBackgroundColor('w')
            self.widget_edp.getViewBox().setBackgroundColor('w')
            # self.widget_msv_top.getViewBox().setBackgroundColor('w')
            self.background_color = 'w'

    def remove_selected_rows(self):
        # Delete the selected mytable lines
        self._deleteRows(self.tableWidget_pars.selectionModel().selectedRows(), self.tableWidget_pars)
        self.update_model_parameter()

    # DeleteRows function
    def _deleteRows(self, rows, table):
            # Get all row index
            indexes = []
            for row in rows:
                indexes.append(row.row())

            # Reverse sort rows indexes
            indexes = sorted(indexes, reverse=True)

            # Delete rows
            for rowidx in indexes:
                table.removeRow(rowidx)

    def append_one_row(self):
        rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(rows) == 0:
            row_index = self.tableWidget_pars.rowCount()
        else:
            row_index = rows[-1].row()
        self.tableWidget_pars.insertRow(row_index+1)
        for i in range(7):
            if i==2:
                check_box = QCheckBox()
                check_box.setChecked(False)
                self.tableWidget_pars.setCellWidget(row_index+1,2,check_box)
            else:
                qtablewidget = QTableWidgetItem('')
                if i == 0:
                    qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                elif i == 1:
                    qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                self.tableWidget_pars.setItem(row_index+1,i,qtablewidget)
        self.update_model_parameter()

    def append_one_row_at_the_end(self):
        row_index = self.tableWidget_pars.rowCount()
        self.tableWidget_pars.insertRow(row_index)
        for i in range(7):
            if i==2:
                check_box = QCheckBox()
                check_box.setChecked(False)
                self.tableWidget_pars.setCellWidget(row_index,2,check_box)
            else:
                qtablewidget = QTableWidgetItem('')
                if i == 0:
                    qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                elif i == 1:
                    qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                self.tableWidget_pars.setItem(row_index,i,qtablewidget)
        self.update_model_parameter()

    def update_model_parameter(self):
        """After you made changes in the par table, this func is executed to update the par values in model"""
        self.model.parameters.data = []
        vertical_label = []
        label_tag=1
        for i in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.item(i,0)==None:
                items = ['',0,False,0,0,'-','']
                vertical_label.append('')
            elif self.tableWidget_pars.item(i,0).text()=='':
                items = ['',0,False,0,0,'-','']
                vertical_label.append('')
            else:
                items = [self.tableWidget_pars.item(i,0).text(),float(self.tableWidget_pars.item(i,1).text()),self.tableWidget_pars.cellWidget(i,2).isChecked(),\
                         float(self.tableWidget_pars.item(i,3).text()), float(self.tableWidget_pars.item(i,4).text()), self.tableWidget_pars.item(i,5).text()]
                if self.tableWidget_pars.item(i,6)==None:
                    items.append('')
                else:
                    items.append(self.tableWidget_pars.item(i,6).text())
                self.model.parameters.data.append(items)
                if self.tableWidget_pars.cellWidget(i,2).isChecked():
                    vertical_label.append(str(label_tag))
                    label_tag += 1
                else:
                    vertical_label.append('')
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_label)

    def fit_all(self):
        """fit all fit parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i,2).setChecked(True)
            except:
                pass
        self.update_model_parameter()

    def fit_next_5(self):
        """fit next 5 parameters starting from first selected row
        """
        num_rows_table = 5
        rows = self.tableWidget_pars.selectionModel().selectedRows()
        starting_row = 0
        if len(rows)!=0:
            starting_row = rows[0].row()

        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i+starting_row,2).setChecked(True)
            except:
                pass
        self.update_model_parameter()

    def fit_none(self):
        """fit none of parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                self.tableWidget_pars.cellWidget(i,2).setChecked(False)
            except:
                pass
        self.update_model_parameter()

    def fit_selected(self):
        """fit selected parameters
        """
        selected_row_index = [each.row() for each in self.tableWidget_pars.selectionModel().selectedRows()]
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            if i in selected_row_index:
                try:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(True)
                except:
                    pass
            else:
                try:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(False)
                except:
                    pass
        self.update_model_parameter()

    def invert_fit(self):
        """invert the selection of fit parameters
        """
        num_rows_table = self.tableWidget_pars.rowCount()
        for i in range(num_rows_table):
            try:
                checkstate = self.tableWidget_pars.cellWidget(i,2).checkState()
                if checkstate == 0:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(True)
                else:
                    self.tableWidget_pars.cellWidget(i,2).setChecked(False)
            except:
                pass
        self.update_model_parameter()

    def update_par_upon_load(self):
        """upon loading model, the par table widget content will be updated with this func"""
        vertical_labels = []
        lines = self.model.parameters.data
        how_many_pars = len(lines)
        self.tableWidget_pars.clear()
        self.tableWidget_pars.setRowCount(how_many_pars)
        self.tableWidget_pars.setColumnCount(7)
        self.tableWidget_pars.setHorizontalHeaderLabels(['Parameter','Value','Fit','Min','Max','Error','Link'])
        for i in range(len(lines)):
            items = lines[i]
            #j = 0
            if items[0] == '':
                vertical_labels.append('')
                # j += 1
            else:
                #add items to table view
                if len(vertical_labels)==0:
                    if items[2]:
                        vertical_labels.append('1')
                    else:
                        vertical_labels.append('')
                else:
                    #if vertical_labels[-1] != '':
                    if items[2]:#ture or false
                        if '1' not in vertical_labels:
                            vertical_labels.append('1')
                        else:
                            jj=0
                            while vertical_labels[-1-jj]=='':
                                jj = jj + 1
                            vertical_labels.append('{}'.format(int(vertical_labels[-1-jj])+1))
                    else:
                        vertical_labels.append('')
                for j,item in enumerate(items):
                    if j == 2:
                        check_box = QCheckBox()
                        check_box.setChecked(item==True)
                        self.tableWidget_pars.setCellWidget(i,2,check_box)
                    else:
                        if j == 1:
                            qtablewidget = QTableWidgetItem(str(round(item,10)))
                        else:
                            qtablewidget = QTableWidgetItem(str(item))
                        if j == 0:
                            qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                        elif j == 1:
                            qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                        self.tableWidget_pars.setItem(i,j,qtablewidget)
                    #j += 1
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        """
        header = self.tableWidget_pars.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
        """
        self.tableWidget_pars.setShowGrid(True)
        self.tableWidget_pars.setVerticalHeaderLabels(vertical_labels)

    def update_par_upon_change(self):
        """will be executed before simulation"""
        self.model.parameters.data = []
        for each_row in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.item(each_row,0)==None:
                items = ['',0,False,0,0,'-','']
            elif self.tableWidget_pars.item(each_row,0).text()=='':
                items = ['',0,False,0,0,'-','']
            else:
                items = [self.tableWidget_pars.item(each_row,0).text()] + [float(self.tableWidget_pars.item(each_row,i).text()) for i in [1,3,4]] + [self.tableWidget_pars.item(each_row,5).text()]
                if self.tableWidget_pars.item(each_row,6)!=None:
                    items.append(self.tableWidget_pars.item(each_row,6).text())
                else:
                    items.append('')
                items.insert(2, self.tableWidget_pars.cellWidget(each_row,2).isChecked())
            self.model.parameters.data.append(items)

    @QtCore.pyqtSlot(str,object,bool)
    def update_par_during_fit(self,string,model,save_tag):
        """slot func to update par table widgets during fit"""
        for i in range(len(model.parameters.data)):
            if model.parameters.data[i][0] !='':
                item_temp = self.tableWidget_pars.item(i,1)
                item_temp.setText(str(round(model.parameters.data[i][1],8)))
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()
        self.tableWidget_pars.setShowGrid(False)

    @QtCore.pyqtSlot(str,object,bool)
    def update_status(self,string,model,save_tag):
        """slot func to update status info displaying fit status"""
        self.statusbar.clearMessage()
        self.statusbar.showMessage(string)
        self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,round(self.run_fit.solver.optimizer.best_fom,5)))
        if save_tag:
            try:
                self.auto_save_model()
                logging.root.info('Auto save model file successfully!')
            except:
                logging.root.error('Fail to save model!')

    @QtCore.pyqtSlot(str,object,bool)
    def update_status_batch(self,string,model,save_tag):
        """slot func to update status info displaying fit status"""
        self.statusbar.clearMessage()
        self.statusbar.showMessage(string)
        self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,round(self.run_batch.solver.optimizer.best_fom,5)))
        if save_tag:
            self.auto_save_model()