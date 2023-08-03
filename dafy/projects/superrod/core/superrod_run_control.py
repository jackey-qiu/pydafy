import os
import time
import copy
import logging
import numpy as np
import pandas as pd
from threading import Thread
from PyQt5.QtWidgets import QCheckBox,QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QFont, QBrush, QColor
from dafy.projects.superrod.core.models import model

from PyQt5 import QtCore

class ScanPar(QtCore.QObject):
    def __init__(self,model):
        super(ScanPar, self).__init__()
        self.model = model
        self.running = False
        self.sign = 1
        self.row = 0
        self.steps = 10

    def run(self):
        self.running = True
        while True:
            # print('Running!')
            if self.running:
                self.rock_par()
                time.sleep(1.5)
            else:
                break

    def rock_par(self):
        par_set = self.model.parameters.data[self.row]
        par_min, par_max = par_set[-4], par_set[-3]
        #steps = int(self.spinBox_steps.value())
        steps = self.steps
        old_value = self.model.parameters.get_value(self.row, 1)
        new_value = (par_max - par_min)/steps*self.sign + old_value
        if (new_value>par_max) or (new_value<par_min):
            self.model.parameters.set_value(self.row, 1, new_value - 2*self.sign*(par_max - par_min)/steps)
            self.sign = -self.sign
        else:
            self.model.parameters.set_value(self.row, 1, new_value)

    def stop(self):
        self.running = False

class RunFit(QtCore.QObject):
    """
    RunFit class to interface GUI to operate fit-engine, which is ran on a different thread
    ...
    Attributes
    ----------
    updateplot : pyqtSignal be emitted to be received by main GUI thread during fit
    solver: api for model fit using differential evolution algorithm

    Methods
    ----------
    run: start the fit
    stop: stop the fit

    """
    updateplot = QtCore.pyqtSignal(str,object,bool)
    fitended = QtCore.pyqtSignal(str)
    def __init__(self,solver):
        super(RunFit, self).__init__()
        self.solver = solver
        self.running = False

    def run(self):
        self.running = True
        self.solver.optimizer.stop = False
        self.solver.StartFit(self.updateplot,self.fitended)

    def stop(self):
        self.running = False
        self.solver.optimizer.stop = True

    def is_running(self):
        return self.running

class RunBatch(QtCore.QObject):
    """
    RunFit class to interface GUI to operate fit-engine, which is ran on a different thread
    ...
    Attributes
    ----------
    updateplot : pyqtSignal be emitted to be received by main GUI thread during fit
    solver: api for model fit using differential evolution algorithm
    multiple_files_hooker: 
        True if you want to run sequentially the rod files in the listWidget
        False if you want to rolling the fit on one rod files (used in fitting many RAXR spectrum)

    Methods
    ----------
    run: start the fit
    stop: stop the fit

    """
    updateplot = QtCore.pyqtSignal(str,object,bool)
    fitended = QtCore.pyqtSignal(str)
    def __init__(self,solver):
        super(RunBatch, self).__init__()
        self.solver = solver
        self.running = False
        self.multiple_files_hooker = False

    def run(self):
        self.running = True
        self.solver.optimizer.stop = False
        self.solver.StartFit(self.updateplot,self.fitended)

    def set_hooker(self,hooker):
        self.multiple_files_hooker = hooker

    def stop(self):
        self.running = False
        self.solver.optimizer.stop = True

class RunControl(object):
    stopNLLS = QtCore.pyqtSignal()
    def generate_covarience_matrix(self):
        fom_level = float(self.lineEdit_error.text())
        if len(self.run_fit.solver.optimizer.par_evals)==0:
            return
        condition = (self.run_fit.solver.optimizer.fom_evals.array()+1)<(self.run_fit.solver.model.fom+1)*(1+fom_level)
        target_matrix = self.run_fit.solver.optimizer.par_evals[condition]
        df = pd.DataFrame(target_matrix)
        corr = df.corr()
        corr.index += 1
        corr = corr.rename(columns = lambda x:str(int(x)+1))
        self.covariance_matrix = corr
        #cmap: coolwarm, plasma, hsv
        self.textEdit_cov.setHtml(corr.style.background_gradient(cmap='coolwarm').set_precision(3).render())

    #calculate the sensitivity of each fit parameter
    #sensitivity: how much percentage increase of a parameter has to be applied to achived ~10% increase in fom?
    # the increase rate of fom divided by that for par give rise to the numerical representative of sensitivity
    #In the end, all sensitivity values are normalized to have the max value equal to 1 for better comparison.
    def screen_parameters(self):
        index_fit_pars = [i for i in range(len(self.model.parameters.data)) if self.model.parameters.data[i][2]]
        #par_names = ['{}.'.format(i) for i in range(1,len(index_fit_pars)+1)]
        #print(par_names)
        epoch_list = [0]*len(index_fit_pars)
        fom_diff_list = [0]*len(index_fit_pars)
        #each epoch, increase value by 2%
        epoch_step = float(self.lineEdit_step.text())
        max_epoch = int(self.lineEdit_epoch.text())
        for i in index_fit_pars:
            par = self.model.parameters.get_value(i, 0)
            print('Screen par {} now!'.format(par))
            current_value = self.model.parameters.get_value(i, 1)
            current_fom = self.model.fom
            current_vec = copy.deepcopy(self.run_fit.solver.optimizer.best_vec)
            epoch = 0
            while epoch<max_epoch:
                epoch = epoch + 1
                #self.model.parameters.set_value(i, 1, current_value*(1+epoch_step*epoch))
                #self.model.simulate()
                current_vec[index_fit_pars.index(i)] = current_value+abs(current_value)*epoch_step*epoch
                print(epoch, current_value, abs(current_value)*epoch_step*epoch)
                fom = self.run_fit.solver.optimizer.calc_fom(current_vec)
                #offset off 1 is used just in case the best fom is very close to 0
                if (fom+1)>(current_fom+1)*(1+0.1):
                    epoch_list[index_fit_pars.index(i)] = epoch*epoch_step
                    fom_diff_list[index_fit_pars.index(i)] = (fom - current_fom)/current_fom
                    #set the original value back
                    self.model.parameters.set_value(i, 1, current_value)
                    #print(epoch_list)
                    print('Break')
                    break
                if epoch == max_epoch:
                    fom_diff_list[index_fit_pars.index(i)] = (fom - current_fom)/current_fom
                    print(fom, current_fom)
                    epoch_list[index_fit_pars.index(i)] = epoch*epoch_step
                    self.model.parameters.set_value(i, 1, current_value)

        sensitivity = np.array(fom_diff_list)/np.array(epoch_list)
        self.plot_bar_chart(sensitivity/max(sensitivity))

    def start_nlls(self):
        self.stopNLLS.connect(self.stop_nlls)
        self.thread_nlls_fit = Thread(target = self.nlls_fit.fit_model, args = (self.stopNLLS,))
        self.thread_nlls_fit.start()
        self.timer_nlls.start(50)

    @QtCore.pyqtSlot()
    def stop_nlls(self):
        self.timer_nlls.stop()

    def update_status_nlls(self):
        if not self.nlls_fit.running:
            self.timer_nlls.stop()
            self.statusbar.showMessage('Finish running model based on NLLS: fom = {} at trial_{}'.format(self.nlls_fit.fom,self.nlls_fit.run_num))
            self.update_error_bars_from_nlls()
        else:
            self.statusbar.showMessage('Running model based on NLLS: fom = {} at trial_{}'.format(self.nlls_fit.fom,self.nlls_fit.run_num))

    def update_error_bars_from_nlls(self):
        errors = self.nlls_fit.perr
        accum_fit_par = -1
        for i in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.cellWidget(i,2).isChecked():
                accum_fit_par = accum_fit_par+1
                #we only change the error but the values are maintained from DE fit results
                self.tableWidget_pars.item(i,5).setText(str(round(errors[accum_fit_par],9)))

    def hook_to_batch(self):
        self.run_batch.set_hooker(True)
        self.run_batch.rod_files = []
        for i in range(self.listWidget_rod_files.count()):
            self.run_batch.rod_files.append(os.path.join(self.lineEdit_folder_of_rod_files.text(),self.listWidget_rod_files.item(i).text()))
        self.open_model_with_path(self.run_batch.rod_files[0])
        self.listWidget_rod_files.setCurrentRow(0)

    def purge_from_batch(self):
        self.run_batch.set_hooker(False)
        self.run_batch.rod_files = []

    def init_new_model(self):
        reply = QMessageBox.question(self, 'Message', 'Would you like to save the current model first?', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.save_model()
        try:
            self.model = model.Model()
            self.run_fit.solver.model = self.model
            self.tableWidget_data.setRowCount(0)
            self.tableWidget_pars.setRowCount(0)
            self.plainTextEdit_script.setPlainText('')
            self.comboBox_dataset.clear()
            self.tableWidget_data_view.setRowCount(0)
            # self.update_plot_data_view()
            self._load_par()
        except Exception:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to init a new model!')
            logging.getLogger().exception('Fatal error encountered during model initiation!')
            self.tabWidget_data.setCurrentIndex(4)

    def append_all_par_sets(self):
        """append fit parameters for all parset listed in the combo box, handy tool to save manual adding them in par table"""
        if "table_container" in self.model.script_module.__dir__():
            if len(self.model.script_module.table_container)!=0:
                table = self.model.script_module.table_container[::-1]
                rows = self.tableWidget_pars.selectionModel().selectedRows()
                if len(rows) == 0:
                    row_index = self.tableWidget_pars.rowCount()
                else:
                    row_index = rows[-1].row()
                for ii in range(len(table)):
                    self.tableWidget_pars.insertRow(row_index)
                    for i in range(6):
                        if i==2:
                            check_box = QCheckBox()
                            check_box.setChecked(eval(table[ii][i]))
                            self.tableWidget_pars.setCellWidget(row_index,2,check_box)
                        else:
                            if i == 0:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                                qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                            elif i in [1]:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                                qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                            elif i ==5:
                                qtablewidget = QTableWidgetItem('(0,0)')
                            else:
                                qtablewidget = QTableWidgetItem(table[ii][i])
                            self.tableWidget_pars.setItem(row_index,i,qtablewidget)
                self.append_one_row_at_the_end()
        else:
            par_all = [self.comboBox_register_par_set.itemText(i) for i in range(self.comboBox_register_par_set.count())]
            for par in par_all:
                self.append_par_set(par)
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()

    def append_par_set(self, par_selected = None):
        #boundary mapping for quick setting the bounds of fit pars
        bounds_map = {"setR":[0.8,1.8],"setScale":[0,1],("setdx","sorbate"):[-0.5,0.5],\
                     ("setdy","sorbate"):[-0.5,0.5],("setdz","sorbate"):[-0.1,1],("setoc","sorbate"):[0.5,3],\
                     ("setdx","surface"):[-0.1,0.1],("setdy","surface"):[-0.1,0.1],("setdz","surface"):[-0.1,0.1],\
                     ("setoc","surface"):[0.6,1],"setDelta":[-20,60],"setGamma":[0,180],"setBeta":[0,0.1]}
        def _get_bounds(attr_head,attr_item):
            for key in bounds_map:
                if type(key)==str:
                    if key in attr_item:
                        return bounds_map[key]
                else:
                    if (key[0] in attr_item) and (key[1] in attr_head):
                        return bounds_map[key]
            return []
        if par_selected==None:
            par_selected = self.comboBox_register_par_set.currentText()
        else:
            pass
        attrs = eval("self.model.script_module.{}.__dir__()".format(par_selected))
        attrs_wanted = [each for each in attrs if each.startswith("set")][::-1]

        rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(rows) == 0:
            row_index = self.tableWidget_pars.rowCount()
        else:
            row_index = rows[-1].row()
        for ii in range(len(attrs_wanted)):
            self.tableWidget_pars.insertRow(row_index)
            # current_value = eval("self.model.script_module."+par_selected+'.'+attrs_wanted[ii].replace('set','get')+"()")
            attr_temp = list(attrs_wanted[ii])
            attr_temp[0] = 'g'#set replaced by get this way
            current_value = eval("self.model.script_module."+par_selected+'.'+''.join(attr_temp)+"()")
            bounds_temp = _get_bounds(par_selected,attrs_wanted[ii])
            #update the bounds if the current value is out of the bound range
            if len(bounds_temp)==2:
                if current_value<bounds_temp[0]:
                    bounds_temp[0] = current_value
                if current_value>bounds_temp[1]:
                    bounds_temp[1] = current_value
            for i in range(6):
                if i==2:
                    check_box = QCheckBox()
                    check_box.setChecked(True)
                    self.tableWidget_pars.setCellWidget(row_index,2,check_box)
                else:
                    if i == 0:
                        qtablewidget = QTableWidgetItem(".".join([par_selected,attrs_wanted[ii]]))
                        qtablewidget.setFont(QFont('Times',10,QFont.Bold))
                    elif i in [1]:
                        qtablewidget = QTableWidgetItem(str(round(current_value,4)))
                        qtablewidget.setForeground(QBrush(QColor(255,0,255)))
                    elif i ==5:
                        qtablewidget = QTableWidgetItem('(0,0)')
                    elif i ==3:
                        #left boundary of fit parameter
                        if len(bounds_temp)!=0:
                            qtablewidget = QTableWidgetItem(str(round(bounds_temp[0],4)))
                        else:
                            qtablewidget = QTableWidgetItem(str(round(current_value*0.5,4)))
                    elif i ==4:
                        #right boundary of fit parameter
                        if len(bounds_temp)!=0:
                            qtablewidget = QTableWidgetItem(str(round(bounds_temp[1],4)))
                        else:
                            qtablewidget = QTableWidgetItem(str(round(current_value*1.5,4)))

                    self.tableWidget_pars.setItem(row_index,i,qtablewidget)
        self.append_one_row_at_the_end()
        self.tableWidget_pars.resizeColumnsToContents()
        self.tableWidget_pars.resizeRowsToContents()

    def simulate_model(self, compile = True):
        """
        simulate the model
        script will be updated and compiled to make name spaces in script_module
        """
        self.update_data_check_attr()
        self.update_plot_dimension()
        self.update_par_upon_change()
        self.model.script = (self.plainTextEdit_script.toPlainText())
        self.widget_solver.update_parameter_in_solver(self)
        self.tableWidget_pars.setShowGrid(True)
        try:
            self.model.simulate(compile = compile)
            self.update_structure_view(compile = compile)
            try:
                self.calc_f_ideal()
            except:
                # self.calc_f_ideal()
                pass
            self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,self.model.fom))
            self.update_plot_data_view_upon_simulation()
            self.update_electron_density_profile()
            if hasattr(self.model.script_module,'model_type'):
                if self.model.script_module.model_type=='ctr':
                    self.init_structure_view()
                else:
                    pass
            else:
                self.init_structure_view()
            self.statusbar.clearMessage()
            self.update_combo_box_list_par_set()
            # self.textBrowser_error_msg.clear()
            self.spinBox_domain.setMaximum(len(self.model.script_module.sample.domain)-1)
            self.statusbar.showMessage("Model is simulated successfully!")
            logging.root.info("Model is simulated successfully!")
        except model.ModelError as e:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to simulate model!')
            logging.root.exception('Fatal error encountered during model simulation!')
            self.tabWidget_data.setCurrentIndex(7)
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)

    #execution when you move the slide bar to change only one parameter
    def simulate_model_light(self):
        try:
            self.model.simulate()
            self.label_2.setText('FOM {}:{}'.format(self.model.fom_func.__name__,self.model.fom))
            self.update_plot_data_view_upon_simulation()
            self.update_structure_view(compile = False)
            self.update_electron_density_profile()
            self.statusbar.showMessage("Model is simulated now!")
        except model.ModelError as e:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to simulate model!')
            logging.getLogger().exception('Fatal error encountered during model simulation!')
            self.tabWidget_data.setCurrentIndex(4)
            _ = QMessageBox.question(self, 'Runtime error message', str(e), QMessageBox.Ok)

    def play_with_one_par(self):
        selected_rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(selected_rows)>0:
            #only get the first selected item
            par_set = self.model.parameters.data[selected_rows[0].row()]
            par_min, par_max = par_set[-4], par_set[-3]
            value = (par_max - par_min)*self.horizontalSlider_par.value()/100 + par_min
            self.model.parameters.set_value(selected_rows[0].row(), 1, value)
            self.lineEdit_scan_par.setText('{}:{}'.format(par_set[0],value))
            self.simulate_model_light()
        else:
            print('Doing nothing!')
            pass

    def scan_one_par(self):
        selected_rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(selected_rows)>0:
            par_set = self.model.parameters.data[selected_rows[0].row()]
            par_min, par_max = par_set[-4], par_set[-3]
            steps = int(self.spinBox_steps.value())
            for i in range(steps+1):
                value = (par_max - par_min)/steps*i + par_min
                self.model.parameters.set_value(selected_rows[0].row(), 1, value)
                self.horizontalSlider_par.setValue(int(i/steps*100))
                self.lineEdit_scan_par.setText('{}:{}'.format(par_set[0],value))
                self.simulate_model_light()

    def rock_one_par(self, sign):
        selected_rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(selected_rows)>0:
            par_set = self.model.parameters.data[selected_rows[0].row()]
            par_min, par_max = par_set[-4], par_set[-3]
            steps = int(self.spinBox_steps.value())
            old_value = self.model.parameters.get_value(selected_rows[0].row(), 1)
            new_value = (par_max - par_min)/steps*sign + old_value
            if (new_value>par_max) or (new_value<par_min):
                self.model.parameters.set_value(selected_rows[0].row(), 1, new_value - 2*sign*(par_max - par_min)/steps)
                self.simulate_model_light()
                return -sign
            else:
                self.model.parameters.set_value(selected_rows[0].row(), 1, new_value)
                self.simulate_model_light()
                return sign

    def update_structure_during_scan_par(self):
        self.simulate_model_light()

    def start_scan_par_thread(self):
        selected_rows = self.tableWidget_pars.selectionModel().selectedRows()
        if len(selected_rows)>0:
            self.scan_par.row = selected_rows[0].row()
            self.scan_par.steps = int(self.spinBox_steps.value())
            self.scan_par_thread.start()
            self.timer_scan_par.start(1000)
        else:
            pass

    def stop_scan_par_thread(self):
        self.scan_par.stop()
        self.scan_par_thread.terminate()
        self.timer_scan_par.stop()

    def run_model(self):
        """start the model fit looping"""
        #button will be clicked every 2 second to update figures
        try:
            # self.stop_model()
            self.simulate_model()
            self.statusbar.showMessage("Initializing model running ...")
            self.timer_update_structure.start(2000)
            self.widget_solver.update_parameter_in_solver(self)
            self.fit_thread.start()
        except:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to launch a model fit!')
            logging.getLogger().exception('Fatal error encountered during init model fitting!')
            self.tabWidget_data.setCurrentIndex(5)

    def stop_model(self):
        if self.run_fit.is_running():
            self.run_fit.stop()
            self.fit_thread.terminate()
            self.timer_update_structure.stop()
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model run is stopped manually!")
        else:
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Model run is already stopped. No action!")   
        
    @QtCore.pyqtSlot(str)
    def stop_model_slot(self,message):
        self.stop_model()
        logging.root.warning(message)
        self.tabWidget_data.setCurrentIndex(7)

    def _stop_model(self):
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.timer_update_structure.stop()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted!")

    def run_model_batch(self):
        """start the model fit looping in a batch mode
        To speed up the structure and plots are not to be updated!
        """
        try:
            #self._stop_model()
            self.simulate_model()
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Initializing model running ...")
            self.widget_solver.update_parameter_in_solver_batch(self)
            self.statusbar.clearMessage()
            self.statusbar.showMessage("Parameters in solver are updated!")
            self.batch_thread.start()
            self.timer_update_structure.start(5000)
        except Exception:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to batch run a model!')
            logging.getLogger().exception('Fatal error encountered during batch run a model!')
            self.tabWidget_data.setCurrentIndex(4)

    def stop_model_batch(self):
        #now need to distinguish how the thread is stopped
        #stop after the finishing all generations
        finish_normal = not self.run_batch.solver.optimizer.stop
        self.update_par_upon_change()
        # self.run_batch.solver.model.simulate()#update the error bar info in the model
        #stop the batch run
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.timer_update_structure.stop()
        #save model first before rolling to next one
        self.calculate_error_bars()
        self.save_model()
        self.auto_save_model()
        #if the run is terminated by user then stop here, otherwise continue on
        if not finish_normal:
            return
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted to work on next task!")
        if self.run_batch.multiple_files_hooker:
            if self.rolling_to_next_rod_file():
                self.run_model_batch()
            else:
                pass
        else:
            if self.update_fit_setup_for_batch_run():
                self.run_model_batch()
            else:
                pass

    def terminate_model_batch(self):
        self.run_batch.stop()
        self.batch_thread.terminate()
        self.statusbar.clearMessage()
        self.statusbar.showMessage("Batch model run is aborted now!")

    def rolling_to_next_rod_file(self):
        which = self.run_batch.rod_files.index(self.rod_file)
        if which == self.run_batch.rod_files.__len__()-1:
            return False
        else:
            self.open_model_with_path(self.run_batch.rod_files[which+1])
            self.listWidget_rod_files.setCurrentRow(which+1)
            return True

    def update_fit_setup_for_batch_run(self):
        """
        Update the fit parameters and the fit dataset for next batch job!
        
        Returns:
            [bool] -- move to the end of datasets or not?
        """
        first_checked_data_item, first_checked_par_item = None, None
        for i in range(self.tableWidget_data.rowCount()):
            if self.tableWidget_data.cellWidget(i,2).checkState()!=0:
                first_checked_data_item = i
                break
        for i in range(self.tableWidget_pars.rowCount()):
            if self.tableWidget_pars.cellWidget(i,2)!=None:
                if self.tableWidget_pars.cellWidget(i,2).checkState()!=0:
                    first_checked_par_item = i
                    break
        self.use_none_data()
        self.fit_none()
        try:
            [self.tableWidget_pars.cellWidget(i+6+first_checked_par_item,2).setChecked(True) for i in range(5)]
            self.tableWidget_data.cellWidget(1+first_checked_data_item,2).setChecked(True)
            self.update_model_parameter()
            return True
        except:
            return False
