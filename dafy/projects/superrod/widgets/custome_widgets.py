import os
import numpy as np
import pandas as pd
import PyQt5
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QDialog
from pyqtgraph.Qt import QtGui
from dafy.core.util.path import superrod_path, DaFy_path, superrod_batch_path
from dafy.projects.superrod.core.models.structure_tools import sorbate_tool_beta
from dafy.projects.superrod.core.models.structure_tools.sxrd_dafy import lattice
from dafy.core.util.UtilityFunctions import  replace_block, apply_modification_of_code_block as script_block_modifier

superrod_standard_script_path = superrod_path / 'scripts' / 'standard_scripts'

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
                return QtGui.QColor('DeepSkyBlue')
                # return QtGui.QColor('green')
            if role == QtCore.Qt.BackgroundRole and index.row()%2 == 1:
                return QtGui.QColor('aqua')
                # return QtGui.QColor('lightGreen')
            if role == QtCore.Qt.ForegroundRole and index.row()%2 == 1:
                return QtGui.QColor('black')
            '''
            if role == QtCore.Qt.CheckStateRole and index.column()==0:
                if self._data.iloc[index.row(),index.column()]:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
            '''
        return None

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        '''
        if role == QtCore.Qt.CheckStateRole and index.column() == 0:
            if value == QtCore.Qt.Checked:
                self._data.iloc[index.row(),index.column()] = True
            else:
                self._data.iloc[index.row(),index.column()] = False
        else:
        '''
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
            return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable)

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        self.layoutAboutToBeChanged.emit()
        self._data = self._data.sort_values(self._data.columns.tolist()[Ncol],
                                        ascending=order == QtCore.Qt.AscendingOrder, ignore_index = True)
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

class DummydataGeneraterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # Load the dialog's GUI
        uic.loadUi(str(superrod_path / 'ui' / 'generate_dummy_data_gui.ui'), self)
        self.setWindowTitle('Dummy data Generator in an easy way')

        self.pushButton_extract_ctr.clicked.connect(self.extract_ctr_set_table)
        self.pushButton_extract_raxs.clicked.connect(self.extract_raxs_set_table)
        self.pushButton_generate.clicked.connect(self.generate_dummy_data)

    def extract_ctr_set_table(self):
        rows = self.spinBox_ctr.value()
        data_ = {'h':[0]*rows, 'k':[0]*rows, 'l_min':[0]*rows, 'l_max':[5]*rows, 'delta_l': [0.1]*rows, 'Bragg_Ls':[str([2,4,6])]*rows}
        self.pandas_model_ctr = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_ctr, main_gui = self.parent)
        self.tableView_ctr.setModel(self.pandas_model_ctr)
        self.tableView_ctr.resizeColumnsToContents()
        self.tableView_ctr.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def extract_raxs_set_table(self):
        rows = self.spinBox_raxs.value()
        data_ = {'h':[0]*rows, 'k':[0]*rows, 'l':[0.3]*rows, 'E_min':[float(self.lineEdit_E_min.text())]*rows, 'E_max':[float(self.lineEdit_E_max.text())]*rows, 'delta_E': [1]*rows, 'E0':[float(self.lineEdit_E0.text())]*rows}
        self.pandas_model_raxs = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_raxs, main_gui = self.parent)
        self.tableView_raxs.setModel(self.pandas_model_raxs)
        self.tableView_raxs.resizeColumnsToContents()
        self.tableView_raxs.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    @staticmethod
    def _generate_dummy_data(data,file_path):
        np.savetxt(file_path,data)

    def generate_dummy_data(self):
        from functools import partial
        data_ctr = self._extract_hkl_all()
        data_raxs = self._extract_hkl_all_raxs()
        if len(data_raxs)!=0:
            kwargs = {'ctr':data_ctr,'raxs':data_raxs,'func':partial(self._generate_dummy_data, file_path = self.lineEdit_data_path.text())}
        else:
            kwargs = {'ctr':data_ctr,'func':partial(self._generate_dummy_data, file_path = self.lineEdit_data_path.text())}
        self.parent.model.script_module.Sim(self.parent.model.script_module.data, kwargs = kwargs)

    def _extract_hkl_one_rod(self, h, k, l_min, l_max, Bragg_ls, delta_l):
        ls = np.linspace(l_min, l_max, int((l_max-l_min)/delta_l))
        for each in Bragg_ls:
            ls = ls[abs(ls - each)>delta_l]
        hs = np.array([h]*len(ls))
        ks = np.array([k]*len(ls))
        return np.array([hs,ks,ls]).T

    def _extract_hkl_one_raxs(self, h, k, l, E_min, E_max, delta_E, E0):
        Es_left = list(np.linspace(E_min, E0-20, int((E0-20-E_min)/10)))
        Es_edge = list(np.linspace(E0-20, E0+20, int(40/delta_E)))
        Es_right = list(np.linspace(E0+20, E_max, int((E_max-E0-20)/10)))
        Es = Es_left + Es_edge + Es_right
        #ls = [l]*len(Es)
        hs = [h]*len(Es)
        ks = [k]*len(Es)
        ls = [l]*len(Es)
        #dls = [2]*len(Es)
        #Bls = [2]*len(Es)
        #fs_ = self.parent.model.script_module.sample.calc_f_all_RAXS(np.array(hs), np.array(ks), np.array(ys), np.array(Es))
        #fs = abs(fs_*fs_)
        #errs = fs*0.005
        return np.array([hs,ks,ls, Es]).T

    def _extract_hkl_all(self):
        rows = len(self.pandas_model_ctr._data.index)
        data_temp = np.zeros((1,3))[0:0]
        for i in range(rows):
            h, k, l_min, l_max, delta_l, Bragg_ls = self.pandas_model_ctr._data.iloc[i,:].tolist()
            Bragg_ls = eval(Bragg_ls)
            result = self._extract_hkl_one_rod(int(h),int(k),float(l_min),float(l_max),Bragg_ls, float(delta_l))
            data_temp = np.vstack((data_temp,result))
        return data_temp

    def _extract_hkl_all_raxs(self):
        rows = len(self.pandas_model_raxs._data.index)
        data_temp = np.zeros((1,4))[0:0]
        for i in range(rows):
            h, k, l, E_min, E_max, delta_E, E0 = self.pandas_model_raxs._data.iloc[i,:].tolist()
            result = self._extract_hkl_one_raxs(int(h),int(k),float(l),float(E_min),float(E_max),float(delta_E), float(E0))
            data_temp = np.vstack((data_temp,result))
        return data_temp

class ScriptGeneraterDialog(QDialog):
    def __init__(self, parent=None):
        # print(sorbate_tool_beta.STRUCTURE_MOTIFS)
        super().__init__(parent)
        self.parent = parent
        
        # Load the dialog's GUI
        uic.loadUi(str(superrod_path / 'ui' / "ctr_model_creator.ui"), self)
        self.setWindowTitle('Script Generator in an easy way')
        self.plainTextEdit_script.setStyleSheet("""QPlainTextEdit{
                        font-family:'Consolas';
                        font-size:14pt;
                        color: #ccc;
                        background-color: #2b2b2b;}""")
        self.plainTextEdit_script.setTabStopWidth(self.plainTextEdit_script.fontMetrics().width(' ')*4)
        #set combobox text items
        self.comboBox_motif_types.addItems(list(sorbate_tool_beta.STRUCTURE_MOTIFS.keys()))

        self.comboBox_predefined_symmetry.addItems(list(sorbate_tool_beta.SURFACE_SYMS.keys()))
        self.comboBox_predefined_symmetry.currentTextChanged.connect(self.reset_sym_info)
        self.pushButton_add_symmetry.clicked.connect(self.append_sym_info)
        self.pushButton_add_all.clicked.connect(self.append_all_sym)

        self.pushButton_extract_surface.clicked.connect(self.extract_surface_slabs)
        self.pushButton_generate_script_surface.clicked.connect(self.generate_script_surface_slabs_and_surface_atm_group)

        self.comboBox_predefined_subMotifs.clear()
        self.comboBox_predefined_subMotifs.addItems(sorbate_tool_beta.STRUCTURE_MOTIFS[self.comboBox_motif_types.currentText()])
        self.comboBox_motif_types.currentTextChanged.connect(self.reset_combo_motif)
        self.pushButton_make_setting_table.clicked.connect(self.setup_sorbate_setting_table)
        self.pushButton_apply_setting.clicked.connect(self.apply_settings_for_one_sorbate)
        self.pushButton_generate_script_sorbate.clicked.connect(self.generate_script_snippet_sorbate)

        self.pushButton_generate_full_script.clicked.connect(self.generate_full_script)
        self.pushButton_transfer_script.clicked.connect(self.transfer_script)

        self.pushButton_draw_structure.clicked.connect(self.show_3d_structure)
        self.pushButton_pan.clicked.connect(self.pan_view)

        self.pushButton_hematite.clicked.connect(self.load_hematite)
        self.pushButton_mica.clicked.connect(self.load_mica)
        self.pushButton_cu.clicked.connect(self.load_cu)

        self.script_lines_sorbate = {}
        self.script_lines_update_sorbate = {'update_sorbate':[]}
        self.script_container = {}
        self.lineEdit_bulk.setText(str(superrod_batch_path / 'Cu100' / 'Cu100_bulk.str'))
        self.lineEdit_folder_suface.setText(str(superrod_batch_path / 'Cu100'))
        self.lineEdit_template_script.setText(str(superrod_standard_script_path /'template_script.py'))

    def load_hematite(self):
        self.lineEdit_bulk.setText(str(superrod_batch_path / 'hematite_rcut'/'bulk.str'))
        self.lineEdit_folder_suface.setText(str(superrod_batch_path / 'hematite_rcut'))
        self.lineEdit_files_surface.setText('half_layer2.str')
        self.lineEdit_lattice.setText(str([5.038,5.434,7.3707,90,90,90]))
        self.lineEdit_surface_offset.setText(str({'delta1':0.,'delta2':0.1391}))
        self.lineEdit_template_script.setText(str(superrod_standard_script_path / 'template_script.py'))
        self.comboBox_T_factor.setCurrentText('B')

    def load_cu(self):
        self.lineEdit_bulk.setText(str(superrod_batch_path / 'Cu100' / 'Cu100_bulk.str'))
        self.lineEdit_folder_suface.setText(str(superrod_batch_path / 'Cu100'))
        self.lineEdit_files_surface.setText('Cu100_surface_1.str')
        self.lineEdit_lattice.setText(str([3.615,3.615,3.615,90,90,90]))
        self.lineEdit_surface_offset.setText(str({'delta1':0.,'delta2':0.}))
        self.lineEdit_template_script.setText(str(superrod_standard_script_path / 'template_script.py'))
        self.comboBox_T_factor.setCurrentText('u')

    def load_mica(self):
        self.lineEdit_bulk.setText(str(superrod_batch_path / 'Muscovite001' / 'muscovite_001_bulk_u_corrected_new.str'))
        self.lineEdit_folder_suface.setText(str(superrod_batch_path / 'Muscovite001'))
        self.lineEdit_files_surface.setText('muscovite_001_surface_AlSi_u_corrected_new_1.str')
        self.lineEdit_lattice.setText(str([5.1988,9.0266,20.04156,90,95.782,90]))
        self.lineEdit_surface_offset.setText(str({'delta1':0.,'delta2':0.}))
        self.lineEdit_template_script.setText(str(superrod_standard_script_path / 'template_script.py'))
        self.comboBox_T_factor.setCurrentText('u')

    def show_3d_structure(self):
        self.lattice = lattice(*eval(self.lineEdit_lattice.text()))
        self.widget_structure.T = self.lattice.RealTM
        self.widget_structure.T_INV = self.lattice.RealTMInv
        self.widget_structure.show_bond_length = True
        self.widget_structure.clear()
        self.widget_structure.opts['distance'] = 2000
        self.widget_structure.opts['fov'] = 1
        self.widget_structure.abc = np.array(eval(self.lineEdit_lattice.text())[0:3])
        xyz_init = self.pandas_model_slab._data[self.pandas_model_slab._data['show']=='1'][['el','id','x','y','z']]
        translation_offsets = [np.array([0,0,0]),np.array([1,0,0]),np.array([-1,0,0]),np.array([0,1,0]),np.array([0,-1,0]),np.array([1,-1,0]),np.array([-1,1,0]),np.array([1,1,0]),np.array([-1,-1,0])]
        xyz_ = pd.DataFrame({'el':[],'id':[],'x':[],'y':[],'z':[]})
        xyz = []
        for i in range(len(xyz_init.index)):
            el, id, x, y, z = xyz_init.iloc[i].tolist()
            for tt in translation_offsets:
                i, j, k = tt
                tag = '_'
                if i!=0:
                    tag=tag+'{}x'.format(i)
                if j!=0:
                    tag=tag+'{}y'.format(j)
                if k!=0:
                    tag=tag+'{}z'.format(k)
                if tag == '_':
                    tag = ''
                _x, _y, _z = x+i, y+j, z+k
                _id = id+tag
                xyz_.loc[len(xyz_.index)] = [el, _id, _x, _y, _z]
                x_c, y_c, z_c = np.dot(self.widget_structure.T,np.array([_x,_y,_z]))
                xyz.append([el,_id, x_c, y_c, z_c])
        #xyz = list(zip(xyz_['el'].tolist(),xyz_['id'].tolist(),xyz_['x']*self.widget_structure.abc[0].tolist(),xyz_['y']*self.widget_structure.abc[1].tolist(),xyz_['z']*self.widget_structure.abc[2].tolist()))
        self.widget_structure.show_structure(xyz, show_id = True)

    def pan_view(self):
        value = int(self.spinBox_pan_pixel.text())
        self.widget_structure.pan(value*int(self.checkBox_x.isChecked()),value*int(self.checkBox_y.isChecked()),value*int(self.checkBox_z.isChecked()))

    def reset_sym_info(self):
        self.lineEdit_2d_rotation_matrix.setText(str(sorbate_tool_beta.SURFACE_SYMS[self.comboBox_predefined_symmetry.currentText()][0:2]))
        self.lineEdit_translation_offset.setText(str(sorbate_tool_beta.SURFACE_SYMS[self.comboBox_predefined_symmetry.currentText()][2]))

    def append_sym_info(self):
        text = self.textEdit_symmetries.toPlainText()
        if text == '':
            self.textEdit_symmetries.setPlainText(f"model.SymTrans({self.lineEdit_2d_rotation_matrix.text()},t={self.lineEdit_translation_offset.text()})")
        else:
            self.textEdit_symmetries.setPlainText(text+f"\nmodel.SymTrans({self.lineEdit_2d_rotation_matrix.text()},t={self.lineEdit_translation_offset.text()})")

    def append_all_sym(self):
        sym_symbols = [self.comboBox_predefined_symmetry.itemText(i) for i in range(self.comboBox_predefined_symmetry.count())]
        for each in sym_symbols:
            mt = str(sorbate_tool_beta.SURFACE_SYMS[each][0:2])
            tt = str(sorbate_tool_beta.SURFACE_SYMS[each][2])
            text = self.textEdit_symmetries.toPlainText()
            if text=='':
                self.textEdit_symmetries.setPlainText(f"model.SymTrans({mt},t={tt})")
            else:
                self.textEdit_symmetries.setPlainText(text+f"\nmodel.SymTrans({mt},t={tt})")

    def reset_combo_motif(self):
        self.comboBox_predefined_subMotifs.clear()
        self.comboBox_predefined_subMotifs.addItems(sorbate_tool_beta.STRUCTURE_MOTIFS[self.comboBox_motif_types.currentText()])

    def setup_sorbate_setting_table(self):
        module = getattr(sorbate_tool_beta,self.comboBox_motif_types.currentText())
        data_ = {}
        #if self.checkBox_use_predefined.isChecked():
        data_['sorbate'] = [str(self.spinBox_sorbate_index.value())]
        data_['motif'] = [self.comboBox_predefined_subMotifs.currentText()]
        data_.update(module.get_par_dict(self.comboBox_predefined_subMotifs.currentText()))
        '''
        else:
            data_['sorbate'] = [str(self.spinBox_sorbate_index.value())]
            data_['xyzu_oc_m'] = str([0.5, 0.5, 1.5, 0.1, 1, 1])
            data_['els'] = str(['O','C','C','O'])
            data_['flat_down_index'] = str([2])
            data_['anchor_index_list'] = str([1, None, 1, 2])
            data_['lat_pars'] = str([3.615, 3.615, 3.615, 90, 90, 90])
            data_['structure_pars_dict'] = str({'r':1.5, 'delta':0})
            data_['binding_mode'] = 'OS'
            data_['structure_index'] = str(self.spinBox_sorbate_index.value())
        '''
        self.pandas_model = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_sorbate_setting, main_gui = self.parent)
        self.tableView_sorbate_setting.setModel(self.pandas_model)
        self.tableView_sorbate_setting.resizeColumnsToContents()
        self.tableView_sorbate_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def apply_settings_for_one_sorbate(self):
        module = getattr(sorbate_tool_beta,self.comboBox_motif_types.currentText())
        # module = eval(f"sorbate_tool_beta.{self.comboBox_motif_types.currentText()}")
        '''
        if self.checkBox_use_predefined.isChecked():
            results = module.generate_script_from_setting_table(use_predefined_motif = True, predefined_motif = self.pandas_model._data.iloc[0]['motif'], structure_index = self.pandas_model._data.iloc[0]['sorbate'])
            self.script_lines_sorbate[self.pandas_model._data.iloc[0]['sorbate']]=results[0]
            self.script_lines_update_sorbate['update_sorbate'].append(results[1])
        else:
        '''
        kwargs = {each:self.pandas_model._data.iloc[0][each] for each in self.pandas_model._data.columns}
        results = module.generate_script_from_setting_table(use_predefined_motif = False, structure_index = self.pandas_model._data.iloc[0]['sorbate'], kwargs = kwargs)
        self.script_lines_sorbate[self.pandas_model._data.iloc[0]['sorbate']]=results[0]
        self.script_lines_update_sorbate['update_sorbate'].append(results[1])
        self.reset_sorbate_set()

    def reset_sorbate_set(self):
        self.lineEdit_sorbate_set.setText(','.join(['sorbate_{}'.format(each) for each in self.script_lines_sorbate.keys()]))

    def generate_script_snippet_sorbate(self):
        keys = sorted(list(self.script_lines_sorbate.keys()))
        scripts = '\n\n'.join([self.script_lines_sorbate[each] for each in keys])
        syms = ''
        for each in keys:
            syms= syms+ "sorbate_syms_{}=[{}]\n".format(each,','.join(self.textEdit_symmetries.toPlainText().rsplit('\n')))
        self.script_container['sorbateproperties'] = scripts+'\n'
        self.script_container['sorbatesym'] = syms
        self.script_container['update_sorbate'] = '\n'.join(list(set(self.script_lines_update_sorbate['update_sorbate'])))+'\n'
        if 'slabnumber' not in self.script_container:
            self.script_container['slabnumber'] = {'num_sorbate_slabs':str(len(keys))}
        else:
            self.script_container['slabnumber'].update({'num_sorbate_slabs':str(len(keys))})

        self.plainTextEdit_script.setPlainText(scripts+syms)

    def extract_surface_slabs(self):
        files = [os.path.join(self.lineEdit_folder_suface.text(),each) for each in self.lineEdit_files_surface.text().rsplit()]
        def _make_df(file, slab_index):
            df = pd.read_csv(file, comment = '#', names = ['id','el','x','y','z','u','occ','m'])
            df['slab'] = slab_index
            df['show'] = str(0)
            df['sym_matrix']=str([1,0,0,0,1,0,0,0,1])
            df['gp_tag'] = 'NaN'
            return df
        dfs = []
        for i in range(len(files)):
            dfs.append(_make_df(files[i],i+1))
        df = pd.concat(dfs, ignore_index = True)
        df.sort_values(by = ['slab','z','id'], ascending = False, inplace = True, ignore_index=True)
        self.pandas_model_slab = PandasModel(data = pd.DataFrame(df), tableviewer = self.tableView_surface_slabs, main_gui = self.parent)
        self.tableView_surface_slabs.setModel(self.pandas_model_slab)
        self.tableView_surface_slabs.resizeColumnsToContents()
        self.tableView_surface_slabs.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def generate_script_surface_slabs(self):
        scripts = []
        files = [os.path.join(self.lineEdit_folder_suface.text(),each) for each in self.lineEdit_files_surface.text().rsplit()]
        for i in range(len(files)):
            scripts.append("surface_{} = model.Slab(T_factor = '{}')".format(i+1,self.comboBox_T_factor.currentText()))
            scripts.append("tool_box.add_atom_in_slab(surface_{}, '{}')".format(i+1, files[i]))
        # scripts.append('\n')
        self.script_container['surfaceslab'] = '\n'.join(scripts) + '\n'
        #self.script_container['slabnumber'] = {'num_surface_slabs':str(len(files))}
        if 'slabnumber' not in self.script_container:
            self.script_container['slabnumber'] = {'num_surface_slabs':str(len(files))}
        else:
            self.script_container['slabnumber'].update({'num_surface_slabs':str(len(files))})
        return self.script_container['surfaceslab']

    def generate_script_surface_atm_group(self):
        scripts = []
        atm_gps_df = self.pandas_model_slab._data[self.pandas_model_slab._data['gp_tag']!='NaN']
        gp_tags = sorted(list(set(atm_gps_df['gp_tag'].tolist())))
        for each in gp_tags:
            scripts.append("{} = model.AtomGroup(instance_name = '{}')".format(each, each))
            temp = atm_gps_df[atm_gps_df['gp_tag']==each]
            for i in range(temp.shape[0]):
                scripts.append("{}.add_atom({},'{}',matrix={})".format(each,'surface_'+str(temp.iloc[i]['slab']),temp.iloc[i]['id'],str(temp.iloc[i]['sym_matrix'])))
        # scripts.append('\n')
        self.script_container['atmgroup'] = '\n'.join(scripts)
        return self.script_container['atmgroup']

    def generate_script_surface_slabs_and_surface_atm_group(self):
        self.plainTextEdit_script.setPlainText(self.generate_script_surface_slabs()+'\n\n'+self.generate_script_surface_atm_group())

    def generate_script_bulk(self):
        self.script_container['sample'] = {'surface_parms':self.lineEdit_surface_offset.text()}
        if os.path.isfile(self.lineEdit_bulk.text()):
            self.script_container['bulk'] = "bulk = model.Slab(T_factor = '{}')\ntool_box.add_atom_in_slab(bulk,'{}')\n".format(self.comboBox_T_factor.currentText(),self.lineEdit_bulk.text())

    def generate_script_raxs(self):
        self.script_container['raxs']  = "RAXS_EL = '{}'\nRAXS_FIT_MODE = '{}'\nNUMBER_SPECTRA = {}\nE0 = {}\nF1F2_FILE = '{}'\n".format(self.lineEdit_res_el.text(),
                                                                                                               self.comboBox_mode.currentText(),
                                                                                                               str(self.spinBox_num_raxs.value()),
                                                                                                               self.lineEdit_e0.text(),
                                                                                                               self.lineEdit_f1f2.text())


    def generate_full_script(self):
        with open(self.lineEdit_template_script.text(),'r') as f:
            lines = f.readlines()
            if len(lines)== 0:
                print('There are 0 lines in the file!')
                return
            #bulk file
            self.generate_script_bulk()
            #raxs setting
            self.generate_script_raxs()
            #lattice parameters
            self.script_container['unitcell']={'lat_pars':eval(self.lineEdit_lattice.text())}
            #energy
            self.script_container['wavelength']={'wal':round(12.398/eval(self.lineEdit_E.text()),4)}

            #surface slabs
            self.generate_script_surface_slabs()
            #surface atm groups
            self.generate_script_surface_atm_group()
            #sorbate and symmetry
            self.generate_script_snippet_sorbate()
            ##Now let us modify the script
            for key in self.script_container:
                if type(self.script_container[key])==type({}):
                    lines = script_block_modifier(lines, key, list(self.script_container[key].keys()), list(self.script_container[key].values()))
                else:
                    lines = replace_block(lines, key, self.script_container[key])
            self.plainTextEdit_script.setPlainText(''.join(lines))

    def transfer_script(self):
        self.parent.plainTextEdit_script.setPlainText(self.plainTextEdit_script.toPlainText())