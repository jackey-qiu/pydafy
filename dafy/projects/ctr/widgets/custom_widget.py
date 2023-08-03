import os
import PyQt5
import pandas as pd
from PyQt5 import uic, QtCore
from pyqtgraph.Qt import QtGui
from PyQt5.QtWidgets import QDialog
from dafy.core.util.path import ctr_path
from dafy.core.util.DebugFunctions import error_pop_up

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
            if role == QtCore.Qt.CheckStateRole and (index.column() in [0, 6]):
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
        elif role == QtCore.Qt.CheckStateRole and index.column() == 6:
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
            if index.column() == 0:
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsUserCheckable)
            elif index.column() == 6:
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

class DataEditorDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.parent = parent
        uic.loadUi(str(ctr_path / 'ui' / "save_ctr_data_dialog.ui"), self)
        self.lineEdit_file_path.setText(os.path.join(self.parent.lineEdit_data_file_path.text(),'ctr_data_temp.csv'))
        self.pushButton_close.clicked.connect(lambda:self.close())
        self.pushButton_save.clicked.connect(self.save_data)
        self.display_table()

    def _generate_scan_info(self):
        info_list = list(set(zip(self.parent.app_ctr.data['scan_no'],self.parent.app_ctr.data['H'], self.parent.app_ctr.data['K'])))
        scan_no, H, K, dL, BL, save, escan = [],[],[],[],[],[], []
        for each in info_list:
            scan_no.append(each[0])
            H.append(each[1])
            K.append(each[2])
            dL.append(0)
            BL.append(0)
            save.append(True)
            escan.append(False)
        result = pd.DataFrame({'save': save,'scan_no':scan_no, 'H': H, 'K': K, 'dL': dL, 'BL': BL, 'escan': escan})
        return result

    def save_data(self):
        self.parent.app_ctr.save_rod_data(self.lineEdit_file_path.text(), self.pandas_model._data)
        error_pop_up('Successful to save rod data!','Information')

    def display_table(self):
        self.pandas_model = PandasModel(data = pd.DataFrame(self._generate_scan_info()), tableviewer = self.tableView_editor, main_gui = self)
        self.tableView_editor.setModel(self.pandas_model)
        self.tableView_editor.resizeColumnsToContents()
        self.tableView_editor.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        self.parent.pandas_model_ = self.pandas_model