import copy
import numpy as np
from dafy.core.util.UtilityFunctions import fit_q_correction
from pyqtgraph.Qt import QtGui


class DataProcessing(object):
    def append_L_scale(self):
        if self.lineEdit_L_container.text()=='':
            self.lineEdit_L_container.setText(self.lineEdit_L.text())
            self.lineEdit_scale_container.setText(self.lineEdit_scale.text())
        else:
            self.lineEdit_L_container.setText(self.lineEdit_L_container.text()+',{}'.format(self.lineEdit_L.text()))
            self.lineEdit_scale_container.setText(self.lineEdit_scale_container.text()+',{}'.format(self.lineEdit_scale.text()))

    def reset_L_scale(self):
        self.lineEdit_L_container.setText('')
        self.lineEdit_scale_container.setText('')

    def fit_q(self):
        L_list = eval('[{}]'.format(self.lineEdit_L_container.text()))
        scale_list = eval('[{}]'.format(self.lineEdit_scale_container.text()))
        if len(L_list)==0 or len(scale_list)==0 or len(L_list)!=len(scale_list):
            return
        lam = self.model.script_module.wal
        R_tth = float(self.lineEdit_r_tth.text())
        unitcell = self.model.script_module.unitcell
        delta,c_off,scale = fit_q_correction(lam,R_tth,L_list,scale_list, unitcell.c*np.sin(unitcell.beta))
        self.q_correction_factor = {'L_bragg':L_list[0],'delta':delta, 'c_off':c_off, 'scale':scale}
        self.fit_q_correction = True
        self.update_plot_data_view_upon_simulation(q_correction = True)
        self.fit_q_correction = False

    def update_q(self):
        reply = QtGui.QMessageBox.question(self, 'Message',
        "Are you sure to update the data with q correction results?", QtGui.QMessageBox.Yes | 
        QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            #print("YES")
            self.apply_q_correction = True
            self.fit_q()
            self.apply_q_correction = False
        else:
            return

    def apply_q_correction_results(self, index, LL):
        self.model.data_original[index].x = LL
        self.model.data = copy.deepcopy(self.model.data_original)
        [each.apply_mask() for each in self.model.data]
        self.model.data.concatenate_all_ctr_datasets()
        self.simulate_model()