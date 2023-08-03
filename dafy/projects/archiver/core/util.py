from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QMenu
import base64
import bcrypt

def encrypt_password(password, encode = 'utf-8'):
    password = password.encode(encode)
    salt = bcrypt.gensalt(10)
    return bcrypt.hashpw(password, salt)

def confirm_password(password,encrypted_password, encode = 'utf-8'):
    return bcrypt.checkpw(str(password).encode(encode), encrypted_password)

def error_pop_up(msg_text = 'error', window_title = ['Error','Information','Warning'][0]):
    msg = QMessageBox()
    if window_title == 'Error':
        msg.setIcon(QMessageBox.Critical)
    elif window_title == 'Warning':
        msg.setIcon(QMessageBox.Warning)
    else:
        msg.setIcon(QMessageBox.Information)

    msg.setText(msg_text)
    # msg.setInformativeText('More information')
    msg.setWindowTitle(window_title)
    msg.exec_()

def extract_vars_from_config(config_file, section_var):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)
    kwarg = {}
    for each in config.items(section_var):
        try:
            kwarg[each[0]] = eval(each[1])
        except:
            kwarg[each[0]] = each[1]
    return kwarg

def image_to_64base_string(image_path):
    with open(image_path, "rb") as img_file:
         my_string = base64.b64encode(img_file.read())
    return my_string

def image_string_to_qimage(my_string, img_format = 'PNG'):
    QByteArr = QtCore.QByteArray.fromBase64(my_string)
    QImage = QtGui.QImage()
    QImage.loadFromData(QByteArr, img_format)
    return QImage

class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, tableviewer, main_gui, parent=None, rgb_bkg = None, rgb_fg = None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
        self.tableviewer = tableviewer
        self.main_gui = main_gui
        self.rgb_bkg = rgb_bkg
        self.rgb_fg = rgb_fg

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role):
        if index.isValid():
            if role in [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole] and index.column()!=0:
                return str(self._data.iloc[index.row(), index.column()])
            #if role == QtCore.Qt.BackgroundRole and index.row()%2 == 0:
                # return QtGui.QColor('green')
                # return QtGui.QColor('DeepSkyBlue')
                #return QtGui.QColor('Blue')
            # if role == QtCore.Qt.BackgroundRole and index.row()%2 == 1:
            if role == QtCore.Qt.BackgroundRole:
                if index.column()==0:
                    if self._data.iloc[index.row(), index.column()]:
                        return QtGui.QColor('red')
                    else:
                        if self.rgb_bkg!=None:
                            return QtGui.QColor(*self.rgb_bkg)
                        else:
                            return QtGui.QColor('white')                        
                else:
                    if self.rgb_bkg!=None:
                        return QtGui.QColor(*self.rgb_bkg)
                    else:
                        return QtGui.QColor('white')
                # return QtGui.QColor('aqua')
                # return QtGui.QColor('lightGreen')
            # if role == QtCore.Qt.ForegroundRole and index.row()%2 == 1:
            if role == QtCore.Qt.ForegroundRole:
                if self.rgb_fg!=None:
                    return QtGui.QColor(*self.rgb_fg)
                else:
                    return QtGui.QColor('black')
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
        # if self._data.columns.tolist()[index.column()] in ['select','archive_date','user_label','read_level']:
            # self.update_meta_info_paper(paper_id = self._data['paper_id'][index.row()])
        self.dataChanged.emit(index, index)
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()
        self.tableviewer.resizeColumnsToContents() 
        # self.tableviewer.horizontalHeader().setStretchLastSection(True)
        return True
    
    def update_view(self):
        self.layoutAboutToBeChanged.emit()
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

    def headerData(self, rowcol, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            tag = self._data.columns[rowcol]         
            return tag
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
                return (QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)

    def sort(self, Ncol, order):
        """Sort table by given column number."""
        self.layoutAboutToBeChanged.emit()
        self._data = self._data.sort_values(self._data.columns.tolist()[Ncol],
                                        ascending=order == QtCore.Qt.AscendingOrder, ignore_index = True)
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(self.rowCount(0), self.columnCount(0)))
        self.layoutChanged.emit()

    def get_dict_from_data(self, ref_key = 'scan_id', selected_columns = None):
        if selected_columns==None:
            columns = [each for each in list(self._data.columns) if each!=ref_key]
        else:
            columns = selected_columns
        vals = list(set(self._data[ref_key]))
        return_result = []
        for val in vals:
            one = {ref_key:val}
            index = self._data[self._data[ref_key]==val].index
            for col in columns:
                one[col] = list(self._data[col][index])
            return_result.append(one)
        return columns, vals, return_result
