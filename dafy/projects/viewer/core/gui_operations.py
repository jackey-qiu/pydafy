import os
import pandas as pd
import numpy as np
import PyQt5
from PyQt5.QtWidgets import QFileDialog
from dafy.core.util.DebugFunctions import error_pop_up
from dafy.core.util.UtilityFunctions import PandasModel
from dafy.core.util.path import user_data_path

class GuiOperations(object):
    def add_one_link(self):
        link_1 = self.comboBox_link_1.currentText()
        link_2 = self.comboBox_link_2.currentText()
        link_group = f"{link_1}+{link_2}"
        current_list = [self.comboBox_link_container.itemText(i) for i in range(self.comboBox_link_container.count())]
        if link_group in current_list:
            return
        if len(current_list)==4:
            error_pop_up('You can add 4 group links at most. Remove some item and continue to add this new item.')
        self.comboBox_link_container.clear()
        self.comboBox_link_container.addItems(current_list + [link_group])

    def remove_one_item(self):
        current_list = [self.comboBox_link_container.itemText(i) for i in range(self.comboBox_link_container.count())]
        current_item = self.comboBox_link_container.currentText()
        new_list = [each for each in current_list if each!=current_item]
        self.comboBox_link_container.clear()
        self.comboBox_link_container.addItems(new_list)

    def tweak_one_channel(self):
        scan = int(self.comboBox_scans_2.currentText())
        channel = self.comboBox_tweak_channel.currentText()
        tweak_value = self.doubleSpinBox_offset.value()
        if channel == 'image_no':
            tweak_value = int(tweak_value)
        setattr(self, f'{channel}_offset_{scan}',tweak_value)
        self.plot_figure_xrv()

    def init_pandas_model_cv_setting(self):
        data_ = {}
        rows = self.spinBox_cv_rows.value()
        data_['use'] = [False] * rows
        data_['scan'] = [''] * rows
        data_['cv_name'] = [''] * rows
        data_['cycle'] = ['0'] * rows
        data_['scaling'] = ['30'] * rows
        data_['smooth_len'] = ['15'] * rows
        data_['smooth_order'] = ['1'] * rows
        data_['color'] = ['r'] * rows
        data_['pH'] = ['13'] * rows
        data_['extract_func'] = ['extract_cv_file_fouad'] * rows
        self.pandas_model_cv_setting = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
        self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
        self.tableView_cv_setting.resizeColumnsToContents()
        self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def update_pandas_model_cv_setting(self, reset = False, data = {}):
        if not reset:
            rows = self.spinBox_cv_rows.value() - self.pandas_model_cv_setting._data.shape[0]
            data_ = self.pandas_model_cv_setting._data.to_dict()
            data_new = {}
            if rows<=0:
                return
            else:
                for each in data_:
                    data_new[each] = [data_[each][self.pandas_model_cv_setting._data.shape[0]-1]]*rows
                self.pandas_model_cv_setting._data = pd.concat([self.pandas_model_cv_setting._data,pd.DataFrame(data_new, index = np.arange(self.pandas_model_cv_setting._data.shape[0],self.pandas_model_cv_setting._data.shape[0]+rows))])
                self.pandas_model_cv_setting = PandasModel(data = self.pandas_model_cv_setting._data, tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
                self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
                self.tableView_cv_setting.resizeColumnsToContents()
                self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)
        else:
            self.pandas_model_cv_setting._data = pd.DataFrame(data)
            self.pandas_model_cv_setting = PandasModel(data = self.pandas_model_cv_setting._data, tableviewer = self.tableView_cv_setting, main_gui = self, check_columns = [0])
            self.tableView_cv_setting.setModel(self.pandas_model_cv_setting)
            self.tableView_cv_setting.resizeColumnsToContents()
            self.tableView_cv_setting.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def project_cv_settings(self):
        self.lineEdit_cv_config_path.setText(str(user_data_path / 'temp.ini'))
        num_items = self.pandas_model_cv_setting._data.shape[0]
        pot_padding = eval(self.pandas_model_in_ax_format._data.iloc[0,4])
        pot_min = eval(self.pandas_model_in_ax_format._data.iloc[0,3])[0]-pot_padding
        pot_max = eval(self.pandas_model_in_ax_format._data.iloc[0,3])[-1]+pot_padding
        pot_bounds = f'[{round(pot_min,3)},{round(pot_max,3)}]'

        current_padding = eval(self.pandas_model_in_ax_format._data.iloc[1,4])
        current_min = eval(self.pandas_model_in_ax_format._data.iloc[1,3])[0]-current_padding
        current_max = eval(self.pandas_model_in_ax_format._data.iloc[1,3])[-1]+current_padding
        current_bounds = f'[{current_min},{current_max}]'

        key_value_map = {'Data_Info':
                                    {'sequence_id':str(list(map(eval,self.pandas_model_cv_setting._data['scan'].to_list()))),
                                     'selected_scan':str(list(map(eval,self.pandas_model_cv_setting._data['scan'].to_list()))),
                                     'cv_folder':self.lineEdit_cv_folder.text(),
                                     'path':str(self.pandas_model_cv_setting._data['cv_name'].to_list()),
                                     'ph':str(list(map(eval, self.pandas_model_cv_setting._data['pH'].to_list())))},
                          'General_Format_Settings':
                                     {'fmt':"['-']*{}".format(num_items),
                                      'color':str(self.pandas_model_cv_setting._data['color'].to_list()),
                                      'index_header_pos_offset_cv':str([0]*2),
                                      'index_header_pos_offset_tafel':str([0]*2),
                                      'index_header_pos_offset_order':str([0]*2),
                                      'tafel_show_tick_label_x_y':str([True]*num_items),
                                      'order_show_tick_label_x_y':str([True]*num_items),
                                      },
                          'Axis_Format_Settings':
                                      {'cv_bounds_pot':pot_bounds+'+'+'+'.join(self.pandas_model_in_ax_format._data.iloc[0,3:].to_list()),
                                       'cv_bounds_current':current_bounds+'+'+'+'.join(self.pandas_model_in_ax_format._data.iloc[1,3:].to_list()),
                                       'cv_show_tick_label_x':str([False]*(num_items-1) + [True]),
                                       'cv_show_tick_label_y':str([True]*num_items)
                                      },
                          'Data_Analysis_settings':
                                      {'cv_scale_factor':str([int(each) for each in self.pandas_model_cv_setting._data.iloc[:,4].to_list()]),
                                       'scale_factor_text_pos':str([(1.35,2.0)]*num_items),
                                       'cv_spike_cut':str([0.002]*num_items),
                                       'scan_rate':str([float(self.lineEdit_scan_rate.text())]*num_items),
                                       'resistance':str([50]*num_items),
                                       'which_cycle':str([eval(each)[0] for each in self.pandas_model_cv_setting._data.iloc[:,3].to_list()]),
                                       'method':str(self.pandas_model_cv_setting._data.iloc[:,-1].to_list()),
                                       'pot_range':str([[1.2,1.6]]*num_items),
                                       'pot_starts_tafel':str([1.6]*num_items),
                                       'pot_ends_tafel':str([1.72]*num_items),
                                      }
                        }

        for each_section in key_value_map:
            for each_item in key_value_map[each_section]:
                print(each_section, each_item)
                self.widget_par_tree.set_field(each_section, each_item, key_value_map[each_section][each_item])        

    def show_or_hide(self):
        self.frame.setVisible(not self.show_frame)
        self.show_frame = not self.show_frame


    #this update the data range for the selected scan
    #it will take effect after you replot the figures
    def update_plot_range(self):
        scan = int(self.comboBox_scans.currentText())
        l,r = int(self.lineEdit_img_range_left.text()),int(self.lineEdit_img_range_right.text())
        self.image_range_info[scan] = [l,r]
        all_info=[]
        for each in self.image_range_info:
            all_info.append('{}:{}'.format(each,self.image_range_info[each]))
        self.plainTextEdit_img_range.setPlainText('\n'.join(all_info))

    #open data file
    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName))



