import os
from os import walk
import numpy as np
import pandas as pd
import configparser
import PyQt5
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore
from dafy.core.util.DebugFunctions import error_pop_up
from dafy.core.util.path import DaFy_path
from dafy.projects.ubmate.core.rsp_pkg import reciprocal_space as rsp

class GuiOperations(object):
    def __init__(self):
        pass

    def extract_rod_from_a_sym_list(self):
        self.comboBox_names.setCurrentText(self.comboBox_working_substrate.currentText())
        self.comboBox_bragg_peak.setCurrentText(self.comboBox_sym_HKL.currentText())
        self.extract_peaks_in_zoom_viewer(symHKLs = [self.comboBox_sym_HKL.itemText(i) for i in range(self.comboBox_sym_HKL.count())])

    def update_TM(self):
        map_ = {'unity':'[[1,0,0],[0,1,0],[0,0,1]]',
                'fcc':'[[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]',
                'hex':'[[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]]'}
        if self.comboBox_TM_type.currentText() in map_:
            self.lineEdit_TM.setText(map_[self.comboBox_TM_type.currentText()])
        else:
            print('The matrix transformation for type {} is not specified!'.format(self.comboBox_TM_type.currentText()))

    def update_file_names(self):
        termination_text = '{}{}{}'.format(*eval(self.lineEdit_hkl.text()))
        self.lineEdit_bulk.setText('{}_{}_bulk.str'.format(self.comboBox_substrate_surfuc.currentText().rsplit('.')[0],termination_text))
        self.lineEdit_surface.setText('{}_{}_surface.str'.format(self.comboBox_substrate_surfuc.currentText().rsplit('.')[0],termination_text))

    def save_structure_files(self):
        # self.surf._generate_structure_files(os.path.join(self.lineEdit_save_folder.text(),self.lineEdit_bulk.text()),os.path.join(self.lineEdit_save_folder.text(),self.lineEdit_surface.text()))
        try:
            self.surf._generate_structure_files(os.path.join(self.lineEdit_save_folder.text(),self.lineEdit_bulk.text()),os.path.join(self.lineEdit_save_folder.text(),self.lineEdit_surface.text()))
            error_pop_up('Success in saving structure files in {}'.format(self.lineEdit_save_folder.text()),'Information')
        except:
            error_pop_up('Failure to save structure files')

    def show_or_hide_constraints(self):
        if self.frame_constraints.isHidden():
            self.frame_constraints.show()
        else:
            self.frame_constraints.hide()

    def fill_matrix(self):
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        RecTM = structure.lattice.RecTM.flatten()
        RealTM = structure.lattice.RealTM.flatten()
        for i in range(1,10):
            exec(f'self.lineEdit_reaTM_{i}.setText(str(round(RealTM[i-1],3)))')
            exec(f'self.lineEdit_recTM_{i}.setText(str(round(RecTM[i-1],3)))')

    def update_HKs_list(self):
        name = self.comboBox_names.currentText()
        self.comboBox_HKs.clear()
        self.comboBox_HKs.addItems(list(map(str,self.HKLs_dict[name])))

    def update_Bragg_peak_list(self):
        name = self.comboBox_names.currentText()
        structure = None
        for each in self.structures:
            if each.name == name:
                structure = each
                break
        peaks = self.peaks_dict[name]
        peaks_unique = []
        for i, peak in enumerate(peaks):
            qxqyqz, _, intensity,_,_ = peak
            HKL = [int(round(each_, 0)) for each_ in structure.lattice.HKL(qxqyqz)]
            peaks_unique.append(HKL+[intensity])
        peaks_unique = np.array(peaks_unique)
        peaks_unique = peaks_unique[peaks_unique[:,-1].argsort()[::-1]]
        HKLs_unique = [str(list(map(int,each[:3]))) for each in peaks_unique]
        self.comboBox_bragg_peak.clear()
        self.comboBox_bragg_peak.addItems(HKLs_unique)

    def select_rod_based_on_Bragg_peak(self):
        name = self.comboBox_names.currentText()
        structure = None
        for each in self.structures:
            if each.name == name:
                structure = each
                break
        hkl_selected = list(eval(self.comboBox_bragg_peak.currentText()))
        qxqy_reference = np.array(structure.lattice.q(hkl_selected)[0:2])
        for each in self.HKLs_dict[name]:
            qxqy = np.array(structure.lattice.q(each)[0:2])
            if np.sum(np.abs(qxqy - qxqy_reference))<0.05:
                self.comboBox_HKs.setCurrentText(str(each))
                break

    def load_config_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Config Files (*.cfg);;All Files (*.*)", options=options)
        if fileName:
            self.lineEdit_config_path.setText(fileName)
            with open(fileName,'r') as f:
                self.plainTextEdit_config.setPlainText(''.join(f.readlines()))
    
    def load_default_structure_file(self):
        #load all structure files (*.cif or *.str) in script_path/cif or its subfolder, deeper folder will be ignored
        temp_dict = {}
        _, dirs_, filenames =  next(walk(os.path.join(DaFy_path,'resources','cif')), (None, None, []))
        for file in filenames:
            if file.endswith('.str') or file.endswith('.cif'):
                temp_dict[file] = os.path.join(DaFy_path, 'resources', 'cif',file)
        for dir_ in dirs_:
            _, _, filenames =  next(walk(os.path.join(DaFy_path, 'resources','cif', dir_)), (None, None, []))
            for file in filenames:
                if file.endswith('.str') or file.endswith('.cif'):
                    temp_dict[file] = os.path.join(DaFy_path, 'resources','cif',dir_,file)
        self.structure_container.update(temp_dict)
        self.update_list_widget()

    def open_structure_files(self):
        dir_ = QFileDialog.getExistingDirectory(None, 'Select project folder:', 'C:\\', QFileDialog.ShowDirsOnly)
        filenames = next(walk(dir_), (None, None, []))[2]
        temp_dict = {}
        for file in filenames:
            if file.endswith('.str') or file.endswith('.cif'):
                temp_dict[file] = os.path.join(dir_,file)
        self.structure_container.update(temp_dict)
        self.update_list_widget()

    def update_list_widget(self):
        self.listWidget_base_structures.addItems(list(self.structure_container.keys()))
        self.comboBox_substrate_surfuc.clear()
        self.comboBox_substrate_surfuc.addItems(list(self.structure_container.keys()))

    def update_config_file(self):
        with open(self.lineEdit_config_path.text(),'w') as f:
            f.write(self.plainTextEdit_config.toPlainText())
            error_pop_up('The config file is overwritten!','Information')

    def init_pandas_model_in_parameter(self):
        selected_items = [each.text() for each in self.listWidget_base_structures.selectedItems()]
        total = len(selected_items)
        data_ = {}
        data_['use'] = [True]*total
        data_['ID'] = selected_items
        data_['SN_vec'] = ['[0,0,1]']*total
        data_['x_vec'] = ['[1,0,0]']*total
        data_['x_offset'] = ['0']*total
        data_['reference'] = ['True'] + ['False']*(total-1)
        data_['plot_peaks'] = ['True']*total
        data_['plot_rods'] = ['True']*total
        data_['plot_grids'] = ['True']*total
        data_['plot_unitcell'] = ['True']*total
        if total==1:
            data_['color'] = ['[0.8,0.8,0]']*total
        elif total == 2:
            data_['color'] = ['[0.8,0.8,0]','[0.,0.8,0]']
        elif total == 3:
             data_['color'] = ['[0.8,0.8,0]','[0.,0.8,0]','[0.,0.8,0.8]']
        else:
            data_['color'] = ['[0.8,0.8,0]']*total
        self.pandas_model_in_parameter = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_used_structures, main_gui = self)
        self.tableView_used_structures.setModel(self.pandas_model_in_parameter)
        self.tableView_used_structures.resizeColumnsToContents()
        self.tableView_used_structures.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def launch_config_file(self):
        self.config = configparser.RawConfigParser()
        self.config.optionxform = str # make entries in config file case sensitive
        self.config.read(self.lineEdit_config_path.text())
        self.common_offset_angle = float(self.config.get('Plot', 'common_offset_angle'))
        self.plot_axes = int(self.config.get('Plot', 'plot_axes'))
        self.energy_keV = float(self.config.get('Plot', 'energy_keV'))
        self.lineEdit_eng.setText(str(self.energy_keV))
        self.lineEdit_eng_ref1.setText(str(self.energy_keV))
        self.lineEdit_eng_ref2.setText(str(self.energy_keV))
        self.k0 = rsp.get_k0(self.energy_keV)
        self.load_base_structures_in_config()
        self.load_structures_in_config()
        self.update_draw_limits()
        self.prepare_objects_for_render()
        self.UB_INDEX += 1
        self.ub.newub('current_ub_{}'.format(self.UB_INDEX))
        bs = self.structures[0].base_structure
        self.ub.setlat('Triclinic',bs.a,bs.b,bs.c,bs.alpha,bs.beta,bs.gamma)
        self.hardware.settings.hardware.energy = self.energy_keV
        self.dc.setu([[1,0,0],[0,1,0],[0,0,1]])
        self.set_cons()

    def launch_config_file_new(self):
        #self.config = ConfigParser.RawConfigParser()
        #self.config.optionxform = str # make entries in config file case sensitive
        #self.config.read(self.lineEdit_config_path.text())
        self.common_offset_angle = self.widget_config.par.names['Plot'].names['common_offset_angle'].value() 
        self.plot_axes = int(self.widget_config.par.names['Plot'].names['plot_axes'].value())
        self.energy_keV = self.widget_config.par.names['Plot'].names['energy_keV'].value() 
        self.lineEdit_eng.setText(str(self.energy_keV))
        self.lineEdit_eng_ref1.setText(str(self.energy_keV))
        self.lineEdit_eng_ref2.setText(str(self.energy_keV))
        self.k0 = rsp.get_k0(self.energy_keV)
        #self.load_base_structures()
        self.load_structures()
        self.update_draw_limits_new()
        self.prepare_objects_for_render()
        self.UB_INDEX += 1
        self.ub.newub('current_ub_{}'.format(self.UB_INDEX))
        lat_temp = self.structures[0].lattice
        self.ub.setlat('Triclinic',lat_temp.a,lat_temp.b,lat_temp.c,np.rad2deg(lat_temp.alpha),np.rad2deg(lat_temp.beta),np.rad2deg(lat_temp.gamma))
        self.hardware.settings.hardware.energy = self.energy_keV
        self.dc.setu([[1,0,0],[0,1,0],[0,0,1]])
        self.set_cons()

    def load_structures(self):
        self.base_structures = {}
        self.structures = []
        ids = self.pandas_model_in_parameter._data[self.pandas_model_in_parameter._data['use']]['ID'].tolist()
        for id in ids:
            file_path = self.structure_container[id]
            if id.endswith('.cif'):
                self.base_structures[id] = Base_Structure.from_cif(id, file_path)
                self.structures.append(self._extract_structure(id))
            elif id.endswith('.str'):
                with open(file_path) as f:
                    lines = f.readlines()
                    for line in lines:
                        #comment lines heading with # are ignore
                        if not line.startswith('#'):
                            toks = line.strip().split(',')
                            a = float(toks[0])
                            b = float(toks[1])
                            c = float(toks[2])
                            alpha = float(toks[3])
                            beta = float(toks[4])
                            gamma = float(toks[5])
                            basis = []
                            for i in range(6, len(toks)):
                                toks2 = toks[i].split(';')
                                basis.append([toks2[0], float(toks2[1]), float(toks2[2]), float(toks2[3])])
                            self.base_structures[id] = Base_Structure(id,a,b,c,alpha,beta,gamma,basis)
                self.structures.append(self._extract_structure(id))
            else:
                error_pop_up('File not in right format. It should be either cif or str file.')
        names = ids
        self.comboBox_names.clear()
        self.comboBox_working_substrate.clear()
        self.comboBox_reference_substrate.clear()
        self.comboBox_substrate.clear()
        # self.comboBox_substrate_surfuc.clear()
        # self.comboBox_names.addItems(names)
        self.comboBox_working_substrate.addItems(names)
        self.comboBox_reference_substrate.addItems(names)
        self.comboBox_substrate.addItems(names)
        # self.comboBox_substrate_surfuc.addItems(names)
        # put reference structure at first position in list
        for i in range(len(self.structures)):
            if(self.structures[i].is_reference_coordinate_system):
                self.structures[0], self.structures[i] = self.structures[i], self.structures[0]
                break

    def _extract_structure(self, id):
        data = self.pandas_model_in_parameter._data
        data = data[data['ID']==id]
        HKL_normal = eval(data['SN_vec'].iloc[0])
        HKL_para_x = eval(data['x_vec'].iloc[0])
        offset_angle = float(data['x_offset'].iloc[0]) + self.common_offset_angle
        is_reference_coordinate_system = eval(data['reference'].iloc[0])
        plot_peaks =  eval(data['plot_peaks'].iloc[0])
        plot_rods = eval(data['plot_rods'].iloc[0])
        plot_grid = eval(data['plot_grids'].iloc[0])
        plot_unitcell = eval(data['plot_unitcell'].iloc[0])
        #print(type(data['color'].iloc[0]),data['color'].iloc[0])
        color = eval(data['color'].iloc[0])
        return Structure(self.base_structures[id], HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, id, self.energy_keV)

    #old config loader
    def load_base_structures_in_config(self):
        # read from settings file
        self.base_structures = {}
        base_structures_ = self.config.items('BaseStructures')
        for base_structure in base_structures_:
            toks = base_structure[1].split(',')
            if(len(toks) == 2):
                id = toks[0]
                self.base_structures[id] = Base_Structure.from_cif(id, os.path.join(DaFy_path,'core', 'util', 'cif',toks[1]))
            else:
                id = toks[0]
                a = float(toks[1])
                b = float(toks[2])
                c = float(toks[3])
                alpha = float(toks[4])
                beta = float(toks[5])
                gamma = float(toks[6])
                basis = []
                for i in range(7, len(toks)):
                    toks2 = toks[i].split(';')
                    basis.append([toks2[0], float(toks2[1]), float(toks2[2]), float(toks2[3])])
                self.base_structures[id] = Base_Structure(id,a,b,c,alpha,beta,gamma,basis)

    #old structure config loader
    def load_structures_in_config(self):
        self.structures = []
        names = []
        structures_ = self.config.items('Structures')
        for structure_ in structures_:
            name = structure_[0]
            names.append(name)
            toks = structure_[1].split(',')
            id = toks[0]
            HKL_normal = toks[1].split(';')
            HKL_normal = [float(HKL_normal[0]), float(HKL_normal[1]), float(HKL_normal[2])]
            HKL_para_x = toks[2].split(';')
            HKL_para_x = [float(HKL_para_x[0]), float(HKL_para_x[1]), float(HKL_para_x[2])]
            offset_angle = float(toks[3]) + self.common_offset_angle
            is_reference_coordinate_system = int(toks[4])
            plot_peaks = int(toks[5])
            plot_rods = int(toks[6])
            plot_grid = int(toks[7])
            plot_unitcell = int(toks[8])
            color = toks[9].split(';')
            color = (float(color[0]), float(color[1]), float(color[2]))
            self.structures.append(Structure(self.base_structures[id], HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name, self.energy_keV))

        self.comboBox_names.clear()
        self.comboBox_working_substrate.clear()
        self.comboBox_reference_substrate.clear()
        self.comboBox_substrate.clear()
        self.comboBox_working_substrate.clear()
        # self.comboBox_names.addItems(names)
        self.comboBox_working_substrate.addItems(names)
        self.comboBox_reference_substrate.addItems(names)
        self.comboBox_substrate.addItems(names)
        self.comboBox_working_substrate.addItems(names)
        # put reference structure at first position in list
        for i in range(len(self.structures)):
            if(self.structures[i].is_reference_coordinate_system):
                self.structures[0], self.structures[i] = self.structures[i], self.structures[0]
                break

class Base_Structure():
    def __init__(self, id, a=1, b=1, c=1, alpha=90, beta=90, gamma=90, basis=[], filename=None, create_from_cif=False):
        self.id = id
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.basis = basis
        self.filename = filename
        self.create_from_cif = create_from_cif

    @staticmethod
    def from_cif(id, filename):
        return Base_Structure(id, filename=filename, create_from_cif=True)

class Structure():
    def __init__(self, base_structure, HKL_normal, HKL_para_x, offset_angle, is_reference_coordinate_system, plot_peaks, plot_rods, plot_grid, plot_unitcell, color, name, energy_keV):
        self.HKL_normal = HKL_normal
        self.HKL_para_x = HKL_para_x
        self.offset_angle = offset_angle
        self.is_reference_coordinate_system = is_reference_coordinate_system
        self.plot_peaks = plot_peaks
        self.plot_rods = plot_rods
        self.plot_grid = plot_grid
        self.plot_unitcell = plot_unitcell
        self.base_structure = base_structure
        self.color = color
        self.name = name
        self.energy_keV = energy_keV
        if(base_structure.create_from_cif):
            self.lattice = rsp.lattice.from_cif(base_structure.filename, self.HKL_normal, self.HKL_para_x, offset_angle, self.energy_keV)
        else:
            a = base_structure.a
            b = base_structure.b
            c = base_structure.c
            alpha = base_structure.alpha
            beta = base_structure.beta
            gamma = base_structure.gamma
            basis = base_structure.basis
            self.lattice = rsp.lattice(a, b, c, alpha, beta, gamma, basis, HKL_normal, HKL_para_x, offset_angle, self.energy_keV)

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, tableviewer, main_gui, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.tableviewer = tableviewer
        self.main_gui = main_gui

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role):
        if index.isValid():
            if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole]:
                return str(self._data.iloc[index.row(), index.column()])
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 0:
                # return QtGui.QColor('DeepSkyBlue')
                return QtGui.QColor('green')
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 1:
                return QtGui.QColor('dark')
                # return QtGui.QColor('lightGreen')
            if role == QtCore.Qt.ForegroundRole and index.row()%2 == 1:
                return QtGui.QColor('white')
            if role == QtCore.Qt.CheckStateRole and index.column()==0:
                if self._data.iloc[index.row(),index.column()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            if value == QtCore.Qt.Checked:
                self._data.iloc[index.row(),index.column()] = True
            else:
                self._data.iloc[index.row(),index.column()] = False
        else:
            if str(value)!='':
                self._data.iloc[index.row(),index.column()] = str(value)
        #if self._data.columns.tolist()[index.column()] in ['select','archive_data','user_label','read_level']:
        #    self.main_gui.update_meta_info_paper(paper_id = self._data['paper_id'][index.row()])
        self.dataChanged.emit(index, index)
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()
        self.tableviewer.resizeColumnsToContents() 
        return True

    def update_view(self):
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[rowcol]         
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[rowcol]         
        return None

    def flags(self, index):
        if not index.isValid():
           return QtCore.Qt.NoItemFlags
        else:
            if index.column()==0:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            else:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        self.layoutAboutToBeChanged.emit()
        self._data = self._data.sort_values(self._data.columns.tolist()[Ncol],
                                        ascending=order == QtCore.Qt.AscendingOrder, ignore_index = True)
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()