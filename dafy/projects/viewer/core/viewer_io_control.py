import os
import numpy as np
import pandas as pd
import pickle, zipfile
from PyQt5.QtWidgets import QFileDialog
from dafy.core.util.path import user_data_path
from dafy.core.util.DebugFunctions import error_pop_up

class ViewerIOControl(object):
    
    def print_data_summary(self):
        try:
            self.print_data_summary_()
        except:
            if len(self.cv_info)==0:
                print('You should provide CV file information to continue on!')

    def print_data_summary_(self):
        header = ["scan", "pH", "pot_lf", "pot_rt", "q_skin", "q_film", "q_cv", "hor_size","d_hor_size","hor_size_err","ver_size","d_ver_size","ver_size_err","hor_strain","d_hor_strain","hor_strain_err","ver_strain","d_ver_strain","ver_strain_err","d_bulk_vol","d_bulk_vol_err","skin_vol_fraction","skin_vol_fraction_err","d_skin_avg", "d_skin_avg_err",'OER_E', 'OER_j']
        output_data = []
        for scan in self.scans:
            for pot_range in self.pot_ranges[scan]:
                which = self.pot_ranges[scan].index(pot_range)
                pot_range_value = 1
                if pot_range[0] != pot_range[1]:
                    pot_range_value = abs(pot_range[0]-pot_range[1])
                #scan = each_scan
                ph = self.grain_size_info_all_scans[scan][pot_range]['pH']
                charges = [round(self.charge_info[scan][pot_range][each],2) for each in ['skin_charge','film_charge','total_charge']]
                size_hor = [round(each,2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["horizontal"])]
                size_hor_error = [round(self.data_summary[scan]['grain_size_ip'][which*2+1]*pot_range_value,4)]
                size_ver = [round(each, 2) for each in list(self.grain_size_info_all_scans[scan][pot_range]["vertical"])]
                size_ver_error = [round(self.data_summary[scan]['grain_size_oop'][which*2+1]*pot_range_value,4)]
                strain_hor = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["horizontal"])]
                strain_hor_error = [round(self.data_summary[scan]['strain_ip'][which*2+1]*pot_range_value,4)]
                strain_ver = [round(each,4) for each in list(self.strain_info_all_scans[scan][pot_range]["vertical"])]
                strain_ver_error = [round(self.data_summary[scan]['strain_oop'][which*2+1]*pot_range_value,4)]
                d_bulk_vol = [-(2*abs(strain_hor[1])+abs(strain_ver[1]))]
                d_bulk_vol_error = [round((4*strain_hor_error[0]**2 + strain_ver_error[0]**2)**0.5,4)]
                skin_vol_fraction = [round(abs(size_ver[1]/size_ver[0] + 2*size_hor[1]/size_hor[0])*100,3)]
                skin_vol_fraction_error = [round(((size_ver_error[0]/size_ver[0])**2 + 4 * (size_hor_error[0]/size_hor[0])**2)**0.5*100,4)]
                d_skin_avg = [round(abs(size_ver[1] + 2* size_ver[0]/size_hor[0] * size_hor[1]), 4)] #refer to ACS catalysis paper https://doi.org/10.1021/acscatal.1c05169 SI (Section 2)
                d_skin_avg_error = [round((size_ver_error[0]**2 + 4 * (size_ver[0]/size_hor[0])**2*size_hor_error[0]**2)**0.5,4)]
                try:
                    idx_OER_E = sorted(list(np.argpartition(abs(self.cv_info[scan]['potential']-float(self.lineEdit_OER_E.text())), 18)[0:18]))[0]
                    idx_OER_j = sorted(list(np.argpartition(abs(self.cv_info[scan]['current_density']-float(self.lineEdit_OER_j.text())), 18)[0:18]))[0]
                except:
                    idx_OER_E = sorted(list(np.argpartition(abs(self.data_to_plot[scan]['potential']-float(self.lineEdit_OER_E.text())), 18)[0:18]))[0]
                    idx_OER_j = sorted(list(np.argpartition(abs(self.data_to_plot[scan]['current']*8-float(self.lineEdit_OER_j.text())), 18)[0:18]))[0]
                OER_E = [round(self.cv_info[scan]['potential'][idx_OER_j],4)]
                OER_j = [round(self.cv_info[scan]['current_density'][idx_OER_E],4)]
                data_temp = [scan, ph] +[round(each,3) for each in list(pot_range)]+ charges + size_hor + size_hor_error + size_ver + size_ver_error + strain_hor + strain_hor_error + strain_ver + strain_ver_error + d_bulk_vol + d_bulk_vol_error + skin_vol_fraction + skin_vol_fraction_error + d_skin_avg + d_skin_avg_error + OER_E + OER_j
                output_data.append(data_temp)
        self.summary_data_df = pd.DataFrame(np.array(output_data),columns = header)
        self.widget_terminal.update_name_space('charge_info',self.charge_info)
        self.widget_terminal.update_name_space('size_info',self.grain_size_info_all_scans)
        self.widget_terminal.update_name_space('strain_info',self.strain_info_all_scans)
        self.widget_terminal.update_name_space('main_win',self)
        self.widget_terminal.update_name_space('cv_info', self.cv_info)
        self.widget_terminal.update_name_space('summary_data', self.summary_data_df)

        def _tag_p(text):
            return '<p>{}</p>'.format(text)
        output_text = []
        output_text.append("*********Notes*********")
        output_text.append("*scan: scan number")
        output_text.append("*pot_lf (V_RHE): left boundary of potential range considered ")
        output_text.append("*pot_rt (V_RHE): right boundary of potential range considered ")
        output_text.append("*q_skin(mc/m2): charge calculated based on skin layer thickness")
        output_text.append("*q_film(mc/m2): charge calculated assuming all Co2+ in the film material has been oxidized to Co3+")
        output_text.append("*q_cv(mc/m2): charge calculated from electrochemistry data (CV data)")
        output_text.append("*(d)_hor/ver_size(nm): horizontal/vertical size or the associated change with a d_ prefix")
        output_text.append("*hor/ver_size_err(nm): error for horizontal/vertical size")
        output_text.append("*(d)_hor/ver_strain(%): horizontal/vertical strain or the associated change with a d_ prefix")
        output_text.append("*hor/ver_strain_err(%): error for horizontal/vertical strain")
        output_text.append("*d_bulk_vol (%): The change of bulk vol wrt the total film volume: 2*d_hor_strain + d_ver_strain")
        output_text.append("*d_bulk_vol_err: error of d_bulk_vol")
        output_text.append("*skin_vol_fraction (%): The skin volume fraction wrt the total film volume")
        output_text.append("*skin_vol_fraction_err: error of skin_vol_fraction")
        output_text.append("*d_skin_avg (nm): The average thickness of the skin layer normalized to surface area of the crystal")
        output_text.append("*d_skin_avg_err: the error of d_skin_avg")
        output_text.append(f"*OER_E: The OER potential at j(mA/cm2) = {float(self.lineEdit_OER_j.text())}")
        output_text.append(f"*OER_j: The OER current at E (RHE/V) = {float(self.lineEdit_OER_E.text())}")
        self.output_text = output_text
        for i in range(len(output_text)):
            output_text[i] = _tag_p(output_text[i])

        self.plainTextEdit_summary.setHtml(self.summary_data_df.to_html(index = False)+''.join(output_text))
        #print("\n".join(output_text))
        #self.plainTextEdit_summary.setPlainText("\n".join(output_text))
    #save the data according to the specified data ranges
    def save_data_method(self):
        # print(self.data_to_save.keys())
        if len(self.data_to_save)==0:
            error_pop_up('No data prepared to be saved!','Error')
        else:
            for each in self.data_to_save:
                # print(self.data_to_save[each])
                self.data_to_save[each].to_csv(os.path.join(self.lineEdit_data_file_path.text(), self.lineEdit_data_file_name.text()+'_{}.csv'.format(each)),header = False, sep =' ',index=False)


    #save a segment of data to be formated for loading in superrod
    def save_xrv_data(self):
        key_map_lib = {
                       #'peak_intensity':1,
                       'strain_oop':2,
                       'strain_ip':3,
                       'grain_size_ip':4,
                       'grain_size_oop':5
                       }
        scan = self.scans
        ph = self.phs
        data_range = self.data_range
        data = self.data_to_plot
        for i in range(len(scan)):
            scan_ = scan[i]
            ph_ = ph[i]
            data_range_ = data_range[i]
            data_ = data[scan_]
            temp_data = {'potential':[],
                         'scan_no':[],
                         'items':[],
                         'Y':[],
                         #'I':[],
                         #'eI':[],
                         'e1':[],
                         'e2':[]}
            for key in key_map_lib:
                temp_data['potential'] = temp_data['potential'] + list(data_['potential'][data_range_[0]:])
                #temp_data['eI'] = temp_data['eI'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['Y'] = temp_data['Y'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e1'] = temp_data['e1'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['e2'] = temp_data['e2'] + [0]*len(data_['potential'][data_range_[0]:])
                temp_data['items'] = temp_data['items'] + [key_map_lib[key]]*len(data_['potential'][data_range_[0]:])
                temp_data['scan_no'] = temp_data['scan_no'] + [scan_]*len(data_['potential'][data_range_[0]:])
                #temp_data['I'] = temp_data['I'] + list(data_[key][data_range_[0]:])
            df = pd.DataFrame(temp_data)
            df.to_csv(self.lineEdit_data_file_path.text().replace('.csv','_{}.csv'.format(scan_)),\
                      header = False, sep =' ',columns = list(temp_data.keys()), index=False)

    #open the cv config file for cv analysis only
    def load_cv_config_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CV config Files (*.ini);;zip Files (*.zip)", options=options)
        if fileName:
            self.lineEdit_cv_config_path.setText(fileName)
            self.widget_par_tree.update_parameter(fileName)
            '''
            with open(fileName,'r') as f:
                lines = f.readlines()
                self.plainTextEdit_cv_config.setPlainText(''.join(lines))
            '''

    #update the config file after edition in the plainText block
    def update_cv_config_file(self):
        # with open(self.lineEdit_cv_config_path.text(),'w') as f:
            # f.write(self.plainTextEdit_cv_config.toPlainText())
        self.widget_par_tree.save_parameter(self.lineEdit_cv_config_path.text())
        if self.lineEdit_cv_config_path.text().endswith('.ini'):
            missed_items = self.cv_tool._extract_parameter_from_config(self.lineEdit_cv_config_path.text())
        elif self.lineEdit_cv_config_path.text().endswith('.zip'): 
            missed_items = self.cv_tool._extract_parameter_from_config(self.lineEdit_cv_config_path.text().replace('.zip','.ini'))
        if len(missed_items)==0:
            self.cv_tool.extract_cv_info()
            error_pop_up('The config file is overwritten!','Information')
        else:
            error_pop_up(f'The config file is overwritten, but the config file has the following items missed:{missed_items}!','Error')

    #load config file
    def load_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","zip Files (*.zip);;config Files (*.ini)", options=options)
        if fileName.endswith('.ini'):
            with open(fileName,'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.rstrip().rsplit(':')
                    if len(items)>2:
                        channel,value = items[0], ':'.join(items[1:])
                    else:
                        channel,value = items
                    if value=='True':
                        getattr(self,channel).setChecked(True)
                    elif value=='False':
                        getattr(self,channel).setChecked(False)
                    else:
                        try:
                            if channel == "textEdit_plot_lib":
                                getattr(self,channel).setText(value.replace(";","\n"))
                            else:
                                getattr(self,channel).setText(value)
                        except:
                            if channel == 'plainTextEdit_img_range':
                                getattr(self,channel).setPlainText(value.replace(";","\n"))
                                if value=="":
                                    pass
                                else:
                                    self.image_range_info = {}
                                    items = value.rsplit(';')
                                    for each_item in items:
                                        a,b = each_item.rstrip().rsplit(":")
                                        self.image_range_info[int(a)] = eval(b)
                            elif channel == 'plainTextEdit_tick_label_settings':
                                getattr(self,channel).setPlainText(value.replace(";","\n"))
                            elif channel == 'tableView_ax_format':
                                value=eval(value)
                                cols = self.pandas_model_in_ax_format._data.columns.tolist()
                                data_shape = self.pandas_model_in_ax_format._data.shape
                                for i in range(len(cols)):
                                    #print(i,cols[i], value['use'])
                                    for j in value[cols[i]]:
                                        if j<data_shape[0] and i<data_shape[1]:
                                            self.pandas_model_in_ax_format._data.iloc[j,i] = value[cols[i]][j]
                                #self.pandas_model_in_ax_format._data = pd.DataFrame(eval(value))
                            elif channel == 'tableView_cv_setting':
                                self.update_pandas_model_cv_setting(reset = True, data = eval(value))
                            elif channel.startswith('comboBox'):
                                getattr(self, channel).clear()
                                getattr(self, channel).addItems(eval(value))

        elif fileName.endswith('.zip'):
            self.load_config_raw(zipfile.ZipFile(fileName,'r'))

        self._load_file()
        self.append_scans_xrv()
        self.update_pot_offset()
        self.make_plot_lib()

    #zipfile is of format zipfile.ZipFile('','r')
    #this step will be done first before lauching the plot func
    def _save_temp_cv_excel_file(self, zipfile):
        root_folder = str(user_data_path)
        #pandas DataFrame
        cv_data_list = pickle.loads(zipfile.read('cv_data_raw'))
        cv_data_names = pickle.loads(zipfile.read('cv_data_names'))
        try:
            xrv_data = pickle.loads(zipfile.read('xrv_data'))
            if type(xrv_data)==pd.DataFrame:
                print('The xrv data was saved directly in pd.DataFrame format')
                xrv_data.to_excel(os.path.join(root_folder,zipfile.read('xrv_data_file_name').decode()), index=False)
            else:
                print('The xrv data was saved in dict format, convert it to dataframe first')
                pd.DataFrame.from_dict(xrv_data).to_excel(os.path.join(root_folder,zipfile.read('xrv_data_file_name').decode()), index=False)
        except:
            print('fail to pickle load due to pandas version dismatch, you should manually copy the exel file to the targeted folder')
        for i in range(len(cv_data_list)):
            #str format already
            cv_data = cv_data_list[i]
            _, cv_name = os.path.split(cv_data_names[i])
            with open(os.path.join(root_folder,cv_name), 'w') as f:
                f.write(cv_data)     

    def load_config_raw(self, zipfile):
        print('save temp cv and xrv excel file...')
        self._save_temp_cv_excel_file(zipfile)
        print('files being saved and set meta parameters now ...')
        #lineedit or checkbox channels
        #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
        #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
        #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
        #            'checkBox_show_marker','checkBox_merge']
        more_channels = ['plainTextEdit_img_range', 'tableView_ax_format', 'tableView_cv_setting']
        
        for channel in self.GUI_metaparameter_channels:
            if channel.startswith('lineEdit'):
                if channel == 'lineEdit_cv_folder':
                    getattr(self, channel).setText(str(user_data_path))
                elif channel == 'lineEdit_data_file':
                    filename = zipfile.read('xrv_data_file_name').decode()
                    getattr(self, channel).setText(os.path.join(user_data_path / filename))
                else:
                    try:
                        getattr(self, channel).setText(zipfile.read(channel).decode())
                    except:
                        pass
            elif channel.startswith('checkBox'):
                try:
                    getattr(self, channel).setChecked(eval(zipfile.read(channel).decode()))
                except:
                    pass
            elif channel.startswith('comboBox'):
                try:
                    getattr(self, channel).clear()
                    getattr(self, channel).addItems(eval(zipfile.read(channel).decode()))
                except:
                    pass
            else:#scan_numbers_append
                getattr(self, channel).setText(zipfile.read(channel).decode())
        for channel in more_channels:
            value = zipfile.read(channel).decode()
            if channel == 'plainTextEdit_img_range':
                getattr(self,channel).setPlainText(value.replace(";","\n"))
                if value=="":
                    pass
                else:
                    self.image_range_info = {}
                    items = value.rsplit(';')
                    for each_item in items:
                        a,b = each_item.rstrip().rsplit(":")
                        self.image_range_info[int(a)] = eval(b)
            elif channel == 'tableView_ax_format':
                value=eval(value)
                cols = self.pandas_model_in_ax_format._data.columns.tolist()
                data_shape = self.pandas_model_in_ax_format._data.shape
                for i in range(len(cols)):
                    #print(i,cols[i], value['use'])
                    for j in value[cols[i]]:
                        if j<data_shape[0] and i<data_shape[1]:
                            self.pandas_model_in_ax_format._data.iloc[j,i] = value[cols[i]][j]
                #self.pandas_model_in_ax_format._data = pd.DataFrame(eval(value))
            elif channel == 'tableView_cv_setting':
                self.update_pandas_model_cv_setting(reset = True, data = eval(value))
        zipfile.close()
        print('everything for setup is finished now!')

    #save config file
    def save_config(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","zip File (*.zip);;config file (*.ini)", options=options)
        if fileName.endswith('.zip'):
            self.save_config_raw(fileName)
            return
        with open(fileName,'w') as f:
            #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
            #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
            #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
            #            'checkBox_show_marker','checkBox_merge']
            for channel in self.GUI_metaparameter_channels:
                try:#checkBox case
                    f.write(channel+':'+str(getattr(self,channel).isChecked())+'\n')
                except:
                    try:#lineEdit case
                        f.write(channel+':'+getattr(self,channel).text()+'\n')
                    except:#comboBox case
                        f.write(channel+':'+str([getattr(self, channel).itemText(i) for i in range(getattr(self, channel).count())])+'\n')

            f.write("plainTextEdit_img_range:"+self.plainTextEdit_img_range.toPlainText().replace("\n",";")+'\n')

            #f.write("textEdit_plot_lib:"+self.textEdit_plot_lib.toPlainText().replace("\n",";")+'\n')
            if hasattr(self,'plainTextEdit_tick_label_settings'):
                f.write("plainTextEdit_tick_label_settings:"+self.plainTextEdit_tick_label_settings.toPlainText().replace("\n",";")+'\n')
            if hasattr(self, 'tableView_ax_format'):
                f.write("tableView_ax_format:"+str(self.pandas_model_in_ax_format._data.to_dict())+'\n')
            if hasattr(self,'tableView_cv_setting'):
                f.write("tableView_cv_setting:"+str(self.pandas_model_cv_setting._data.to_dict())+'\n')

    #save all meta-parameters and data files (xrv data and cv data) into a zip file
    def save_config_raw(self, filename):
        #options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        #filename, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","config file (*.zip)", options=options)
        if filename:
            try:
                savefile = zipfile.ZipFile(filename, 'w')
            except Exception as e:
                raise IOError(str(e), filename)
            #channels = ['lineEdit_data_file','checkBox_time_scan','checkBox_use','checkBox_mask','checkBox_max','lineEdit_x','lineEdit_y','scan_numbers_append','lineEdit_fmt',\
            #            'lineEdit_potential_range', 'lineEdit_data_range','lineEdit_colors_bar','checkBox_use_external_cv','checkBox_use_internal_cv',\
            #            'checkBox_plot_slope','checkBox_use_external_slope','lineEdit_pot_offset','lineEdit_cv_folder','lineEdit_slope_file','lineEdit_reference_potential',\
            #            'checkBox_show_marker','checkBox_merge']
            for each in self.GUI_metaparameter_channels:
                if each.startswith('checkBox'):
                    savefile.writestr(each, str(getattr(self,each).isChecked()))
                elif each.startswith('comboBox'):
                    savefile.writestr(each, str([getattr(self, each).itemText(i) for i in range(getattr(self, each).count())]))
                else:
                    savefile.writestr(each, getattr(self,each).text())
            savefile.writestr("plainTextEdit_img_range", self.plainTextEdit_img_range.toPlainText().replace("\n",";"))
            # savefile.writestr("plainTextEdit_tick_label_settings", self.plainTextEdit_tick_label_settings.toPlainText().replace("\n",";"))
            savefile.writestr("tableView_ax_format", str(self.pandas_model_in_ax_format._data.to_dict()))
            savefile.writestr("tableView_cv_setting",str(self.pandas_model_cv_setting._data.to_dict()))
            data_tmp = self.data.copy(deep = True)
            data_tmp['potential'] += data_tmp['iR']
            data_tmp['potential_cal'] += data_tmp['iR']
            data_tmp = data_tmp[[each for each in data_tmp.columns if not each.startswith('Unnamed')]]
            savefile.writestr('xrv_data', pickle.dumps(data_tmp.to_dict()))
            savefile.writestr('xrv_data_file_name', os.path.split(self.lineEdit_data_file.text())[1])
            savefile.writestr('cv_data_raw', pickle.dumps([open(self.plot_lib[each][0],'r').read() for each in self.plot_lib]))
            savefile.writestr('cv_data_names', pickle.dumps([os.path.split(self.plot_lib[each][0])[1] for each in self.plot_lib]))
            savefile.close()

    #fill the info in the summary text block
    #and init the self.data attribute by reading data from excel file
    def _load_file(self):
        fileName = self.lineEdit_data_file.text()
        self.lineEdit_data_file.setText(fileName)
        self.data = pd.read_excel(fileName)
        #do ir correction now
        self._ir_correction(Rs = eval(self.lineEdit_resistance.text()))
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        scans = list(set(list(self.data['scan_no'])))
        self.scans_all = scans
        scans.sort()
        scan_numbers = 'scan_nos\n'+str(scans)+'\n'
        # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
        self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
        phs = 'pHs\n'+str(self.phs_all)+'\n'
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))

    #load excel data file
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;All Files (*.csv)", options=options)
        if fileName:
            self.lineEdit_data_file.setText(fileName)
            self.data = pd.read_excel(fileName)
        col_labels = 'col_labels\n'+str(list(self.data.columns))+'\n'
        scans = list(set(list(self.data['scan_no'])))
        self.scans_all = scans
        self.comboBox_scans.clear()
        self.comboBox_scans.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_2.clear()
        self.comboBox_scans_2.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_3.clear()
        self.comboBox_scans_3.addItems([str(each) for each in sorted(scans)])
        self.image_range_info = {}
        self.plainTextEdit_img_range.setPlainText("")
        scans.sort()
        scan_numbers = 'scan_nos\n'+str(scans)+'\n'
        # print(list(self.data[self.data['scan_no']==scans[0]]['phs'])[0])
        self.phs_all = [list(self.data[self.data['scan_no']==scan]['phs'])[0] for scan in scans]
        phs = 'pHs\n'+str(self.phs_all)+'\n'
        self.textEdit_summary_data.setText('\n'.join([col_labels,scan_numbers,phs]))            