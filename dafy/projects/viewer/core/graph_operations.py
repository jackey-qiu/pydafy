import numpy as np
import pandas as pd
from scipy import stats
import PyQt5
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from dafy.core.util.UtilityFunctions import PandasModel
from dafy.core.util.charge_calculation import calculate_charge
from dafy.core.util.PlotSetup import RHE

class GraphOperations(object):
    def project_to_master(self):
        scan_no = int(self.comboBox_scans_3.currentText())
        index_lf, index_rg = list(map(int,list(self.widget.region.getRegion())))
        pot = self.widget.p3_handle.getData()[1][index_lf:index_rg]
        current_bkg = self.widget.p1_bkg_handle.getData()[1][index_lf:index_rg]
        cv_scale_factor= self.plot_lib[scan_no][2]
        getattr(self,'plot_axis_scan{}'.format(scan_no))[0].plot(pot, current_bkg * cv_scale_factor, '--k')
        self.mplwidget.fig.tight_layout()
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

    def init_pandas_model_ax_format(self):
        data_ = {}
        data_['use'] = [True]*6 + [False] * 15
        data_['type'] = ['master']*6 + ['bar']*15
        data_['channel'] = ['potential','current','strain_ip','strain_oop','grain_size_ip','grain_size_oop']
        data_['channel'] = data_['channel'] + ['strain_ip','strain_oop','grain_size_ip','grain_size_oop'] + \
                           ['dV_bulk', 'dV_skin', '<dskin>','OER_E', 'OER_j', 'OER_j/<dskin>','pH', 'q_cv','q_film','input','TOF']
        data_['tick_locs'] = ['[0,1,2,3]']*(6+15)
        data_['padding'] = ['0.1']*(6+15)
        data_['#minor_tick'] = ['4']*(6+15)
        data_['fmt_str'] = ["{: 4.2f}"]*(6+15)
        data_['func'] = ['set_xlim'] + ['set_ylim']*(5+15)
        self.pandas_model_in_ax_format = PandasModel(data = pd.DataFrame(data_), tableviewer = self.tableView_ax_format, main_gui = self, check_columns = [0])
        self.tableView_ax_format.setModel(self.pandas_model_in_ax_format)
        self.tableView_ax_format.resizeColumnsToContents()
        self.tableView_ax_format.setSelectionBehavior(PyQt5.QtWidgets.QAbstractItemView.SelectRows)

    def extract_tick_label_settings(self):
        if hasattr(self,'plainTextEdit_tick_label_settings'):
            strings = self.plainTextEdit_tick_label_settings.toPlainText()
            lines = strings.rsplit('\n')
            for each_line in lines:
                if each_line.startswith('#'):
                    pass
                else:
                    items = each_line.rstrip().rsplit('+')
                    key,item,locator,padding,tick_num,fmt,func = items
                    locator = eval(locator)
                    if key in self.tick_label_settings:
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
                    else:
                        self.tick_label_settings[key] = {}
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
        elif hasattr(self, 'tableView_ax_format'):
            cols = self.pandas_model_in_ax_format._data.shape[0]
            for i in range(cols):
                if self.pandas_model_in_ax_format._data.iloc[i,0]:
                    key,item,locator,padding,tick_num,fmt,func = self.pandas_model_in_ax_format._data.iloc[i,1:].tolist()
                    locator = eval(locator)
                    if key in self.tick_label_settings:
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}
                    else:
                        self.tick_label_settings[key] = {}
                        self.tick_label_settings[key][item] = {'locator':locator,'padding':float(padding),'tick_num':int(tick_num),'fmt':fmt,'func':func}

    def plot_pot_step_current_from_external(self,ax,scan_no, plot_channel = 'potential'):
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        results = func(file_name, which_cycle, use_all = True)
        pot,current = results
        #pot_filtered, current_filtered = pot, current
        #pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        #pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*cv_scale_factor, smooth_length, smooth_order)

        #ax.plot(pot_filtered,current_filtered*8,label='',color = color)
        if plot_channel=='current':
            ax.plot(range(len(current)),current*8,label='',color = color)
        elif plot_channel == 'potential':
            ax.plot(range(len(current)),RHE(pot,pH=ph),label='',color = color)

    def plot_cv_from_external(self,ax,scan_no,marker_pos):
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        pot, current = [], []
        if type(which_cycle)==list:
            for each in which_cycle:
                _pot, _current = func(file_name, each)
                pot = pot + list(_pot)
                current = current + list(_current)
            pot, current = np.array(pot), np.array(current)
        elif type(which_cycle)==int:
            results = func(file_name, which_cycle)
            pot,current = results
        else:
            print('unsupported cycle index:', which_cycle)
        print('IR correction to the potential with:-->')
        Rs = eval(self.lineEdit_resistance.text())
        #get scan numbers
        text = self.scan_numbers_append.text()
        scans = list(set([int(each) for each in text.rstrip().rsplit(',')]))
        scans.sort()
        assert len(Rs)>=len(scans), 'Shape of Rs and scans mismatch!'
        print(f'R = {Rs[scans.index(scan_no)]}')
        pot = pot - abs(current)*0.001*Rs[scans.index(scan_no)]
        pot_filtered, current_filtered = pot, current
        pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*cv_scale_factor, smooth_length, smooth_order)
        '''
        if scan_no == 23999:
            self.ax_test = ax
            self.pot_test = pot_filtered
            self.current_test = current_filtered*8
        '''
        ax.plot(pot_filtered,current_filtered*8,label='',color = color)
        ax.plot(RHE(pot,pH=ph),current*8,ls=':',label='',color = color)
        #get the position to show the scaling text on the plot
        # current_temp = current_filtered[np.argmin(np.abs(pot_filtered[0:int(len(pot_filtered)/2)]-1.1))]*8*cv_scale_factor
        current_temp = 0
        ax.text(1.1,current_temp+1.5,'x{}'.format(cv_scale_factor),color=color)
        #store the cv data
        self.cv_info[scan_no] = {'current_density':current*8,'potential':RHE(pot,pH = ph),'pH':ph, 'color':color}
        if self.checkBox_show_marker.isChecked():
            for each in marker_pos:
                ax.plot([each,each],[-100,100],':k')
        #ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
        print('scan{} based on cv'.format(scan_no))
        #scan rate in V/s
        scan_rate = float(self.lineEdit_scan_rate.text())
        #potential range in V_RHE
        pot_ranges = self.pot_ranges[scan_no]
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {}
        for pot_range in pot_ranges:
            try:
                charge_cv, output = self.cv_tool.calculate_pseudocap_charge_stand_alone(pot_filtered, current_filtered/cv_scale_factor*8, scan_rate = scan_rate, pot_range = pot_range)
            except:
                charge_cv = 0
            if pot_range not in self.charge_info[scan_no]:
                self.charge_info[scan_no][pot_range] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
            else:
                self.charge_info[scan_no][pot_range]['total_charge'] = charge_cv

        return min(current_filtered*8),max(current*8)

    def plot_cv_from_external_original(self,ax,scan_no,marker_pos):
        file_name,which_cycle,cv_spike_cut,cv_scale_factor, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.{}'.format(func_name))
        results = func(file_name, which_cycle)
        if len(results) == 3:
            t, pot,current = results
            t_filtered, pot_filtered, current_filtered = t, pot, current
        elif len(results) == 2:
            pot,current = results
            pot_filtered, current_filtered = pot, current
        for ii in range(4):
            filter_index = np.where(abs(np.diff(current_filtered*8))<cv_spike_cut)[0]
            filter_index = filter_index+1#index offset by 1
            # t_filtered = t_filtered[(filter_index,)]
            pot_filtered = pot_filtered[(filter_index,)]
            current_filtered = current_filtered[(filter_index,)]
        pot_filtered = RHE(pot_filtered,pH=ph)
        ax.plot(pot_filtered,current_filtered*8*cv_scale_factor,label='',color = color)
        ax.plot(RHE(pot,pH=ph),current*8,label='',color = color)
        #get the position to show the scaling text on the plot
        # current_temp = current_filtered[np.argmin(np.abs(pot_filtered[0:int(len(pot_filtered)/2)]-1.1))]*8*cv_scale_factor
        current_temp = 0
        ax.text(1.1,current_temp+1.5,'x{}'.format(cv_scale_factor),color=color)
        #store the cv data
        self.cv_info[scan_no] = {'current_density':current*8,'potential':RHE(pot,pH = ph),'pH':ph, 'color':color}

        for each in marker_pos:
            ax.plot([each,each],[-100,100],':k')
        #ax.set_ylim([min(current_filtered*8*cv_scale_factor),max(current*8)])
        print('scan{} based on cv'.format(scan_no))
        # print([self.pot_range1, self.pot_range2, self.pot_range3])
        # print(self.get_integrated_charge(pot_filtered, current_filtered, t_filtered, plot = False))
        #scan rate in V/s
        scan_rate = float(self.lineEdit_scan_rate.text())
        #potential range in V_RHE
        """
        pot_range = eval('[{}]'.format(self.lineEdit_pot_range.text().rstrip()))
        charge_cv = calculate_charge(t, pot, current, which_cycle=0, ph=ph, cv_spike_cut=cv_spike_cut, cv_scale_factor=cv_scale_factor, scan_rate = scan_rate, pot_range = pot_range)
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
        else:
            self.charge_info[scan_no]['total_charge'] = charge_cv
        # pot_ranges = [self.pot_range1, self.pot_range2, self.pot_range3]
        """
        pot_ranges = self.pot_ranges[scan_no]
        if scan_no not in self.charge_info:
            self.charge_info[scan_no] = {}
        for pot_range in pot_ranges:
            charge_cv = calculate_charge(t, pot, current, which_cycle=0, ph=ph, cv_spike_cut=cv_spike_cut, cv_scale_factor=cv_scale_factor, scan_rate = scan_rate, pot_range = pot_range)
            if pot_range not in self.charge_info[scan_no]:
                self.charge_info[scan_no][pot_range] = {'skin_charge':0,'film_charge':0,'total_charge':charge_cv}
            else:
                self.charge_info[scan_no][pot_range]['total_charge'] = charge_cv

        # print(self.charge_info)

        return min(current_filtered*8*cv_scale_factor),max(current*8)            

    def set_plot_channels(self):
        time_scan = self.checkBox_time_scan.isChecked()
        if time_scan:
            self.lineEdit_x.setText('image_no')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop')
        else:
            self.lineEdit_x.setText('potential')
            self.lineEdit_y.setText('current,strain_ip,strain_oop,grain_size_ip,grain_size_oop')        

    #plot bar chart using linear fit results
    def plot_data_summary_xrv_from_external_file(self):
        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            #label mapping
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\partial\mid\Delta\varepsilon_\parallel\mid\slash\partial E$ (%/V)',
                        'strain_oop':r'$\partial\mid\Delta\varepsilon_\perp\mid\slash\partial E$ (%/V)',
                        'grain_size_oop':r'$\partial\mid\Delta d_\perp\mid\slash\partial E$ (nm/V)',
                        'grain_size_ip':r'$\partial\mid\Delta d_\parallel\mid\slash\partial E$ (nm/V)',
                        'peak_intensity':r'Intensity / a.u.'}
            #get color tags
            colors_bar = self.lineEdit_colors_bar.text().rsplit(',')
            if len(colors_bar) == 1:
                colors_bar = colors_bar*len(self.scans)
            else:
                if len(colors_bar) > len(self.scans):
                    colors_bar = colors_bar[0:len(self.scans)]
                elif len(colors_bar) < len(self.scans):
                    colors_bar = colors_bar + [colors_bar[-1]]*(len(self.scans)-len(colors_bar))
            plot_y_labels = [each for each in list(self.data_summary[self.scans[0]].keys()) if each in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']]
            #TODO this has to be changed to set the y_max automatically in different cases.
            lim_y_temp = {'strain_ip':-0.18,'strain_oop':-0.5,'grain_size_ip':-1.2,'grain_size_oop':-1.6}
            for each in plot_y_labels:
                for i in range(int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)):#each value come with an error value
                    # plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)],self.data_summary[each_scan][each][-1]] for each_scan in self.scans])
                    plot_data_y = np.array([[self.data_summary[each_scan][each][i*2],self.data_summary[each_scan][each][i*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    '''
                    plot_data_y_charge = []
                    for each_scan in self.scans:
                        key_ = list(self.charge_info[each_scan].keys())[-1]
                        plot_data_y_charge.append(self.charge_info[each_scan][key_]['total_charge']/abs(key_[0]-key_[1]))
                    '''
                    labels = ['pH {}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    count_pH13 = 1
                    for j,each_label in enumerate(labels):
                        if each_label == 'pH 13':
                            labels[j] = '{} ({})'.format(each_label,count_pH13)
                            count_pH13 += 1
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), int(len(self.data_summary[self.scans[0]]['strain_ip'])/2), i+1+int(len(self.data_summary[self.scans[0]]['strain_ip'])/2)*plot_y_labels.index(each))
                    # ax_temp_2 = ax_temp.twinx()
                    if i==0 and each == plot_y_labels[0]:
                        #ax_temp.legend(labels)
                        for ii in range(len(plot_data_x)):
                            if labels[ii] in ['pH 13 (1)', 'pH 8', 'pH 7', 'pH 10']:
                                label = labels[ii]
                                if label == 'pH 13 (1)':
                                    label = 'pH 13'
                                ax_temp.bar(plot_data_x[ii],-plot_data_y[ii,0],0.5, yerr = plot_data_y[ii,-1], color = colors_bar[ii], label = label)
                            else:
                                ax_temp.bar(plot_data_x[ii],-plot_data_y[ii,0],0.5, yerr = plot_data_y[ii,-1], color = colors_bar[ii])
                        ax_temp.legend(loc = 2,ncol = 1)
                    else:
                        ax_temp.bar(plot_data_x,-plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                    # ax_temp_2.plot(plot_data_x,-np.array(plot_data_y_charge),'k:*')
                    self._format_axis(ax_temp)
                    # self._format_axis(ax_temp_2)
                    if 'bar' in self.tick_label_settings:
                        if (each in self.tick_label_settings['bar']) and self.checkBox_use.isChecked():
                            self._format_ax_tick_labels(ax = ax_temp,
                                    fun_set_bounds = self.tick_label_settings['bar'][each]['func'],
                                    bounds = [0,abs(lim_y_temp[each])], #[lim_y_temp[each],0],
                                    bound_padding = self.tick_label_settings['bar'][each]['padding'],
                                    major_tick_location =self.tick_label_settings['bar'][each]['locator'],
                                    show_major_tick_label = i==0, #show major tick label for the first scan
                                    num_of_minor_tick_marks=self.tick_label_settings['bar'][each]['tick_num'],
                                    fmt_str = self.tick_label_settings['bar'][each]['fmt'])
                    if i == 0:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=10)
                        ax_temp.set_ylim([0,abs(lim_y_temp[each])])

                    else:
                        ax_temp.set_ylim([0,abs(lim_y_temp[each])])
                    # if each == plot_y_labels[0]:
                        # ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=13)
                    if each != plot_y_labels[-1]:
                        ax_temp.set_xticklabels([])
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                    if i!=0:
                        ax_temp.set_yticklabels([])

                    # ax_temp.set_xticklabels(plot_data_x,labels)
            self.mplwidget2.fig.subplots_adjust(wspace = 0.04,hspace=0.04)
            self.mplwidget2.canvas.draw()
        else:
            pass            

    #bar chart based on slope values calculated directely from the xrv master figure
    def plot_data_summary_xrv(self):
        if self.checkBox_use_external_slope.isChecked():
            self.make_data_summary_from_external_file()
            self.plot_data_summary_xrv_from_external_file()
            print("new_data summary is built!")
            return
        #here you should update the self.data_summary info
        self.plot_figure_xrv()
        self.print_data_summary()
        #plain text to be displayed in the data summary tab
        plain_text = []
        #set it manually to True when you want to plot TOF and j/V on the same plot
        plot_dual_y_axis_TOF_j = False

        if self.data_summary!={}:
            self.mplwidget2.fig.clear()
            y_label_map = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\Delta\varepsilon_\parallel$  (%/V)',
                        'strain_oop':r'$\Delta\varepsilon_\perp$  (%/V)',
                        'grain_size_oop':r'$\Delta d_\perp$  (nm/V)',
                        'grain_size_ip':r'$\Delta d_\parallel$  (nm/V)',
                        'peak_intensity':r'Intensity / a.u.',
                        '<dskin>': r'$<d_{skin}>$ / nm',
                        'dV_bulk':r'($\Delta V / V$) / %',
                        'dV_skin':r'($\Delta V_{skin} / V$) / %',
                        'OER_E': r'$\eta (1 mAcm^{-2}) / V$',
                        'TOF': r'$TOF \ / \ s^{-1}$',
                        'OER_j':r'$j (1.65 V) / mAcm^{-2})$',
                        'q_cv':r'$Q_0\hspace{1} /\hspace{1} mC{\bullet}cm^{-2}$',
                        'q_film': r'$Q_0\hspace{1} /\hspace{1} mC{\bullet}cm^{-2}$',
                        'OER_j/<dskin>':r'$log(j/V_{skin})\hspace{1}/\hspace{1}mA{\bullet}nm^{-3}$'
                        }

            y_label_map_abs = {'potential':'E / V$_{RHE}$',
                        'current':r'j / mAcm$^{-2}$',
                        'strain_ip':r'$\varepsilon_\parallel$  (%)',
                        'strain_oop':r'$\varepsilon_\perp$  (%)',
                        'grain_size_oop':r'$d_\perp$  (nm)',
                        'grain_size_ip':r'$d_\parallel$  (nm)',
                        'peak_intensity':r'Intensity / a.u.'}

            colors_bar = self.lineEdit_colors_bar.text().rsplit(',')
            if len(colors_bar) == len(self.scans):
                pass
            else:
                if len(colors_bar) > len(self.scans):
                    colors_bar = colors_bar[0:len(self.scans)]
                elif len(colors_bar) < len(self.scans):
                    colors_bar = colors_bar + [colors_bar[-1]]*(len(self.scans)-len(colors_bar))
            plot_y_labels = [each for each in list(self.data_summary[self.scans[0]].keys()) if each in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']]

            lim_y_temp = {'strain_ip':[],'strain_oop':[],'grain_size_ip':[],'grain_size_oop':[]}
            for each_pot in self.pot_range:
                #if pot_range = [1,1] for eg, the bar value is actually the associated absolute value at pot = 1
                #if pot_range = [1,1.5] for eg, the bar value is the value difference between 1 and 1.5 V
                use_absolute_value = each_pot[0] == each_pot[1]
                #force using the absolute value
                # use_absolute_value = True
                for each in lim_y_temp.keys():
                    for each_scan in self.scans:
                        if use_absolute_value:
                            lim_y_temp[each].append(self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
                        else:
                            lim_y_temp[each].append(self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2])
            for each in lim_y_temp:
                lim_y_temp[each] = [min(lim_y_temp[each]),max(lim_y_temp[each])]
            for each in lim_y_temp:
                offset = (lim_y_temp[each][1]-lim_y_temp[each][0])*0.1
                lim_y_temp[each] = [lim_y_temp[each][0]-offset,lim_y_temp[each][1]+offset]
            if use_absolute_value:
               y_label_map = y_label_map_abs
            gs_left = plt.GridSpec(len(plot_y_labels), len(self.pot_range)+1,hspace=0.02,wspace=0.2)
            hwspace = eval(self.lineEdit_hwspace.text())
            gs_right = plt.GridSpec(max([2,self.comboBox_link_container.count()]), len(self.pot_range)+1,hspace=hwspace[0],wspace=hwspace[1])
            #print(self.data_summary)
            def _extract_setting(channel):
                data = self.pandas_model_in_ax_format._data
                return data[(data['type'] == 'bar') & (data['channel'] == channel)].to_dict(orient = 'records')[0]

            for each_pot in self.pot_range:
                output_data = []
                #use_absolute_value = each_pot[0] == each_pot[1]
                # use_absolute_value = True
                for each in plot_y_labels:
                    plot_data_y = np.array([[self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2],self.data_summary[each_scan][each][self.pot_range.index(each_pot)*2+1]] for each_scan in self.scans])
                    plot_data_x = np.arange(len(plot_data_y))
                    labels = ['{}'.format(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans]
                    # ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range)+1, self.pot_range.index(each_pot)+1+(len(self.pot_range)+1)*plot_y_labels.index(each))
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(gs_left[plot_y_labels.index(each), self.pot_range.index(each_pot)])
                    self._format_axis(ax_temp)
                    settings = _extract_setting(each)
                    if settings['use'] and self.checkBox_use.isChecked():
                        self._format_ax_tick_labels(ax = ax_temp,
                                                    fun_set_bounds = settings['func'],#'set_xlim',
                                                    bounds = [0,1],#will be replaced
                                                    bound_padding = float(settings['padding']),
                                                    major_tick_location = eval(settings['tick_locs']), #x_locator
                                                    show_major_tick_label = True, #show major tick label for the first scan
                                                    num_of_minor_tick_marks=int(settings['#minor_tick']), #4
                                                    fmt_str = settings['fmt_str'])#'{:3.1f}'
                    if use_absolute_value:
                        ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(plot_data_y[:,0])
                    else:
                        ax_temp.bar(plot_data_x,plot_data_y[:,0],0.5, yerr = plot_data_y[:,-1], color = colors_bar)
                        ax_temp.plot(plot_data_x,plot_data_y[:,0], '*:',color='0.1')
                        output_data.append(plot_data_y[:,0])
                    if each_pot == self.pot_range[0]:
                        ax_temp.set_ylabel(y_label_map[each],fontsize=10)
                        # ax_temp.set_ylim([lim_y_temp[each],0])
                        # ax_temp.set_ylim(lim_y_temp[each])
                    else:
                        pass
                        # ax_temp.set_ylim(lim_y_temp[each])
                    if each == plot_y_labels[0]:
                        if use_absolute_value:
                            ax_temp.set_title('E = {:4.2f} V'.format(each_pot[0]), fontsize=10)
                        else:
                            ax_temp.set_title('E range:{:4.2f}-->{:4.2f} V'.format(*each_pot), fontsize=10)
                    if each != plot_y_labels[-1]:
                        #ax_temp.set_xticklabels([])
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                    else:
                        ax_temp.set_xticks(plot_data_x)
                        ax_temp.set_xticklabels(labels,fontsize=10)
                        ax_temp.set_xlabel('pH')
                    if each_pot!=self.pot_range[0]:
                        ax_temp.set_yticklabels([])
                def _extract_data(channel, which_pot_range):
                    data_len = self.summary_data_df.shape[0]+1
                    name_map = {'dV_bulk':lambda:self.summary_data_df['d_bulk_vol'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'dV_skin':lambda:self.summary_data_df['skin_vol_fraction'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                '<dskin>':lambda:self.summary_data_df['d_skin_avg'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_E':lambda:self.summary_data_df['OER_E'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_j':lambda:self.summary_data_df['OER_j'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_j/<dskin>':lambda:(self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'TOF':lambda:(0.3437*self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'pH':lambda:self.summary_data_df['pH'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'q_film':lambda:self.summary_data_df['q_film'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'q_cv':lambda:self.summary_data_df['q_cv'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'input':lambda:eval(self.lineEdit_input_values.text())}
                    name_map_error = {'dV_bulk':lambda:self.summary_data_df['d_bulk_vol_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'dV_skin':lambda:self.summary_data_df['skin_vol_fraction_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                '<dskin>':lambda:self.summary_data_df['d_skin_avg_err'].to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'OER_E':lambda:[None]*self.summary_data_df.shape[0],
                                'OER_j':lambda:[None]*self.summary_data_df.shape[0],
                                'OER_j/<dskin>':lambda:(self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']**2*self.summary_data_df['d_skin_avg_err']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'TOF':lambda:(0.3437*self.summary_data_df['OER_j']/self.summary_data_df['d_skin_avg']**2*self.summary_data_df['d_skin_avg_err']).to_list()[which_pot_range:data_len:len(self.pot_range)],
                                'pH':lambda:[None]*self.summary_data_df.shape[0],
                                'q_film':lambda:[None]*self.summary_data_df.shape[0],
                                'q_cv':lambda:[None]*self.summary_data_df.shape[0],
                                'input':lambda:[None]*self.summary_data_df.shape[0]}
                    if which_pot_range>len(self.pot_range)-1:
                        which_pot_range = 0
                    if self.checkBox_error.isChecked():
                        return name_map[channel](), name_map_error[channel]()
                    else:
                        return name_map[channel](), [None]*self.summary_data_df.shape[0]
                    #return [self.data_summary[each_][channel][which_pot_range*2] for each_ in self.scans]

                def _get_xy_for_linear_fit(panel_index, x, y):
                    tag = getattr(self,f'lineEdit_partial_set_p{panel_index+1}').text()
                    x_, y_ = [], []
                    if tag == '[*]': #use all
                        return x, y
                    else:
                        if tag.startswith('-'):
                            tag = [each for each in range(len(x)) if each not in eval(tag[1:])]
                        else:
                            tag = eval(tag)
                        if type(tag)!=list:
                            return x, y
                        else:
                            for each in tag:
                                x_.append(x[each])
                                y_.append(y[each])
                            return x_, y_

                for i in range(self.comboBox_link_container.count()):
                    channels = self.comboBox_link_container.itemText(i).rsplit('+')
                    x, x_error = _extract_data(channels[0],self.spinBox_pot_range_idx.value())
                    y, y_error = _extract_data(channels[1],self.spinBox_pot_range_idx.value())
                    x_, y_ = _get_xy_for_linear_fit(i, x, y)
                    # ax_temp = self.mplwidget2.canvas.figure.add_subplot(len(plot_y_labels), len(self.pot_range)+1, 2+(len(self.pot_range)+1)*i)
                    ax_temp = self.mplwidget2.canvas.figure.add_subplot(gs_right[i,len(self.pot_range)])
                    if plot_dual_y_axis_TOF_j:
                        if 'TOF' == channels[1]:
                            ax_temp_right = ax_temp.twinx()
                            self._format_axis_customized(ax_temp, right=False, left=True, labelright=False, labelleft=True)
                            self._format_axis_customized(ax_temp_right, left=False, right=True, labelleft=False, labelright=True)
                        else:
                            self._format_axis(ax_temp)
                    else:
                        self._format_axis(ax_temp)
                    for channel in channels:
                        settings = _extract_setting(channel)
                        if settings['use'] and self.checkBox_use.isChecked():
                            self._format_ax_tick_labels(ax = ax_temp,
                                                        fun_set_bounds = settings['func'],#'set_xlim',
                                                        bounds = [0,1],#will be replaced
                                                        bound_padding = float(settings['padding']),
                                                        major_tick_location = eval(settings['tick_locs']), #x_locator
                                                        show_major_tick_label = True, #show major tick label for the first scan
                                                        num_of_minor_tick_marks=int(settings['#minor_tick']), #4
                                                        fmt_str = settings['fmt_str'])#'{:3.1f}'
                    y_axis_range_l, y_axis_range_r = ax_temp.get_ylim()
                    y_axis_span = abs(y_axis_range_r - y_axis_range_l)
                    x_axis_range_l, x_axis_range_r = ax_temp.get_xlim()
                    x_axis_span = abs(x_axis_range_r - x_axis_range_l)
                    HA = 'center' if channels[0]!='pH' else 'left'
                    VA = 'baseline'
                    if channels[0]!='pH':
                        x_axis_span = 0
                    else:
                        x_axis_span = x_axis_span/5
                        y_axis_span = y_axis_span*0
                    if channels[0] in y_label_map:
                        ax_temp.set_xlabel(y_label_map[channels[0]])
                    else:
                        ax_temp.set_xlabel(channels[0])
                    if channels[1] in y_label_map:
                        ax_temp.set_ylabel(y_label_map[channels[1]])
                    else:
                        ax_temp.set_ylabel(channels[1])
                    if channels[0]=='input':
                        ax_temp.set_xlabel(self.lineEdit_input_name.text())
                    if channels[1]=='input':
                        ax_temp.set_ylabel(self.lineEdit_input_name.text())
                    if 'OER_j/<dskin>' == channels[1]:
                        #-14 term is due to the unit of cm-2 transformed to nm-2, the scalling factor will be 10-14 becoming -14 after applying the log
                        slope_, intercept_, r_value_, *_ = stats.linregress(x_, np.log10(y_)-14)
                        print(f'R2={r_value_}, slope for log(OER_j/<dskin>) as y axis = {slope_}')
                        # ax_temp.set_ylabel('log({})'.format(channels[1]))
                        ax_temp.set_ylabel(y_label_map[channels[1]])
                        # [ax_temp.scatter(x[jj], np.log10(y[jj]), c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if y_error[jj] == None:
                                ax_temp.errorbar(x[jj], np.log10(y[jj])-14, xerr=x_error[jj], yerr =None, c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(x[jj], np.log10(y[jj])-14, xerr=x_error[jj], yerr = y_error[jj]/y[jj]/np.log(10), c=colors_bar[jj], marker = 's', ms = 4)
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj]+x_axis_span/20, np.log10(y[jj]+y_axis_span/20)-14, str(jj+1), ha=HA, va=VA, c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x, np.array(x)*slope_ + intercept_, '-k')
                    elif 'OER_j/<dskin>' == channels[0]:
                        slope_, intercept_, r_value_, *_ = stats.linregress(np.log10(x_)-14, y_)
                        # ax_temp.set_xlabel('log({})'.format(channels[0]))
                        ax_temp.set_xlabel(y_label_map[channels[0]])
                        # [ax_temp.scatter(np.log10(x[jj]), y[jj], c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if x_error[jj] == None:
                                ax_temp.errorbar(np.log10(x[jj])-14, y[jj], xerr=None, yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(np.log10(x[jj])-14, y[jj], xerr=x_error[jj]/x[jj]/np.log(10), yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(np.log10(x[jj]+x_axis_span/20), y[jj]+y_axis_span/20, str(jj+1), ha=HA, va=VA, c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(np.log10(x), np.log10(x)*slope_ + intercept_, '-k')
                    elif 'TOF' == channels[1]:
                        #scale_factor of 0.3437 has been applied to j/<d> to convert to TOF: TOF = sf * (j/<d>)
                        #note j in mA/cm2, d in nm
                        #read details from Wiegmann_2022 (https://doi.org/10.1021/acscatal.1c05169): last paragraph in section 4.2
                        #-14 term is due to the unit of cm-2 transformed to nm-2, the scalling factor will be 10-14 becoming -14 after applying the log
                        slope_, intercept_, r_value_, *_ = stats.linregress(x_, np.log10(y_))
                        print(f'R2={r_value_}, slope for log(TOF) as y axis = {slope_}')
                        # ax_temp.set_ylabel('log({})'.format(channels[1]))
                        ax_temp.set_ylabel(y_label_map[channels[1]])
                        if plot_dual_y_axis_TOF_j:
                            ax_temp_right.set_ylabel(r'$(j/V_{skin})\hspace{1}/\hspace{1}mA{\bullet}nm^{-3}$')
                        # [ax_temp.scatter(x[jj], np.log10(y[jj]), c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if y_error[jj] == None:
                                ax_temp.errorbar(x[jj], y[jj], xerr=x_error[jj], yerr =None, c=colors_bar[jj], marker = 's', ms = 4)
                                # ax_temp_right.errorbar(x[jj], y[jj]/0.3437, xerr=x_error[jj], yerr =None, c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(x[jj], y[jj], xerr=x_error[jj], yerr = y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                                # ax_temp_right.errorbar(x[jj], y[jj]/0.3437, xerr=x_error[jj], yerr = y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj]+x_axis_span/20, y[jj]+y_axis_span/20, str(jj+1), ha=HA, va=VA,c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x_, 10**(np.array(x_)*slope_ + intercept_), '-k')
                        ax_temp.set_yscale('log')
                        if plot_dual_y_axis_TOF_j:
                            ax_temp_right.set_yscale('log')
                            #with log scale, TOF data points are just shift down by log10(0.3437) compared to (j/V), therefore to only show one set of points, let us shift the y_lim accordingly
                            ax_temp_right.set_ylim(*np.array(ax_temp.get_ylim())/0.3437)
                    elif 'TOF' == channels[0]:
                        slope_, intercept_, r_value_, *_ = stats.linregress(np.log10(x_), y_)
                        # ax_temp.set_ylabel('log({})'.format(channels[1]))
                        ax_temp.set_xlabel(y_label_map[channels[0]])
                        # [ax_temp.scatter(x[jj], np.log10(y[jj]), c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        for jj in range(len(x)):
                            if x_error[jj] == None:
                                ax_temp.errorbar(x[jj], y[jj], xerr=None, yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                            else:
                                ax_temp.errorbar(x[jj], y[jj], xerr=x_error[jj], yerr=y_error[jj], c=colors_bar[jj], marker = 's', ms = 4)
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj]+x_axis_span/20, y[jj]+y_axis_span/20, str(jj+1), ha=HA, va=VA,c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x_, np.log10(x_)*slope_ + intercept_, '-k') 
                        ax_temp.set_xscale('log')
                    else:
                        slope_, intercept_, r_value_, *_ = stats.linregress(x_, y_)
                        # [ax_temp.scatter(x[jj], y[jj], c=colors_bar[jj], marker = '.') for jj in range(len(x))]
                        [ax_temp.errorbar(x[jj], y[jj], xerr = x_error[jj], yerr = y_error[jj], c=colors_bar[jj], marker = 's', ms = 4) for jj in range(len(x))]
                        if self.checkBox_marker.isChecked():
                            [ax_temp.text(x[jj]+x_axis_span/20, y[jj]+y_axis_span/20, str(jj+1), ha=HA, va=VA,c=colors_bar[jj], size = 'small') for jj in range(len(x))]
                        if getattr(self, f'checkBox_panel{i+1}').isChecked():
                            ax_temp.plot(x, np.array(x)*slope_ + intercept_, '-k')

                #print output data
                output_data = np.array(output_data).T
                output_data = np.append(output_data,np.array([int(self.phs[self.scans.index(each_scan)]) for each_scan in self.scans])[:,np.newaxis],axis=1)
                output_data = np.append(np.array([int(each_) for each_ in self.scans])[:,np.newaxis],output_data,axis = 1)
                # print('\n')
                # print(each_pot)
                plain_text.append(f'<p>\npot = {each_pot} V</p>')
                plain_text.append('<p>scan_no\tstrain_ip\tstrain_oop\tgrain_size_ip\tgrain_size_oop\tpH</p>')
                for each_row in output_data:
                    # print("{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:2.0f}".format(*each_row))
                    plain_text.append("<p>{:3.0f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t\t{:2.0f}</p>".format(*each_row))
            #self.mplwidget2.fig.subplots_adjust(hspace=0.5, wspace=0.2)
            self.mplwidget2.canvas.draw()
            # self.plainTextEdit_summary.setPlainText('\n'.join(plain_text))
            self.plainTextEdit_summary.setHtml('<h3>Table of complete information of pseudocapacitive charge and film structure (results extracted from master figure)</h3>'\
                                               +self.summary_data_df.to_html(index = False)+''.join(self.output_text)
                                               +'<br><h3>structural change normalized to potential (delta/V) (data used for plotting bar chart)</h3>'+''.join(plain_text))
        else:
            pass

    def plot_reaction_order_and_tafel(self,axs = []):
        if len(axs)== 0:
            self.widget_cv_view.canvas.figure.clear()
            ax_tafel = self.widget_cv_view.canvas.figure.add_subplot(1,2,1)
            ax_order = self.widget_cv_view.canvas.figure.add_subplot(1,2,2)
        else:
            assert len(axs) == 2, 'You need only two axis handle here!'
            ax_tafel, ax_order = axs
        #self.ax_tafel = ax_tafel
        #self.ax_order = ax_order
        if self.cv_tool.info['reaction_order_mode'] == 'constant_potential':
            constant_value = self.cv_tool.info['potential_reaction_order']
        elif self.cv_tool.info['reaction_order_mode'] == 'constant_current':
            constant_value = self.cv_tool.info['current_reaction_order']
        mode = self.cv_tool.info['reaction_order_mode']
        forward_cycle = True
        text_log_tafel = self.cv_tool.plot_tafel_with_reaction_order(ax_tafel, ax_order,constant_value = constant_value,mode = mode, forward_cycle = forward_cycle, use_marker = self.checkBox_use_marker.isChecked(), use_all = self.checkBox_use_all.isChecked())
        plainText = '\n'.join([text_log_tafel[each] for each in text_log_tafel])
        self.plainTextEdit_cv_summary.setPlainText(self.plainTextEdit_cv_summary.toPlainText() + '\n\n' + plainText)

        self._format_axis(ax_tafel)
        self._format_axis(ax_order)
        #set item tag e) and f) for tafel and order
        # ax_tafel.text(1.5, 4.4, 'e)',weight = 'bold', fontsize = 12)
        # ax_order.text(4.8, 1.81, 'f)',weight = 'bold', fontsize = 12)

        tafel_bounds_pot, tick_locs_tafel_pot, padding_tafel_pot, num_tick_marks_tafel_pot, fmt_tafel_pot, func_tafel_pot = self.cv_tool.info['tafel_bounds_pot'].rsplit('+')
        tafel_bounds_current, tick_locs_tafel_current, padding_tafel_current, num_tick_marks_tafel_current, fmt_tafel_current, func_tafel_current = self.cv_tool.info['tafel_bounds_current'].rsplit('+')
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = func_tafel_pot,
                bounds = eval(tafel_bounds_pot),
                bound_padding = float(padding_tafel_pot),
                major_tick_location =eval(tick_locs_tafel_pot),
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks= int(num_tick_marks_tafel_pot),
                fmt_str = fmt_tafel_pot)
        self._format_ax_tick_labels(ax = ax_tafel,
                fun_set_bounds = func_tafel_current,
                bounds = eval(tafel_bounds_current),
                bound_padding = float(padding_tafel_current),
                major_tick_location =eval(tick_locs_tafel_current),
                show_major_tick_label = True, #show major tick label for the first scan
                num_of_minor_tick_marks= int(num_tick_marks_tafel_current),
                fmt_str = fmt_tafel_current)

        order_bounds_ph, tick_locs_order_ph, padding_order_ph, num_tick_marks_order_ph, fmt_order_ph, func_order_ph = self.cv_tool.info['order_bounds_ph'].rsplit('+')
        order_bounds_y, tick_locs_order_y, padding_order_y, num_tick_marks_order_y, fmt_order_y, func_order_y = self.cv_tool.info['order_bounds_y'].rsplit('+')
        self._format_ax_tick_labels(ax = ax_order,
                                    fun_set_bounds = func_order_ph,
                                    bounds = eval(order_bounds_ph),
                                    bound_padding = eval(padding_order_ph),
                                    major_tick_location =eval(tick_locs_order_ph),
                                    show_major_tick_label = True, #show major tick label for the first scan
                                    num_of_minor_tick_marks= int(num_tick_marks_order_ph),
                                    fmt_str = fmt_order_ph)
        self._format_ax_tick_labels(ax = ax_order,
                                    fun_set_bounds = func_order_y,
                                    bounds = eval(order_bounds_y),
                                    bound_padding = eval(padding_order_y),
                                    major_tick_location =eval(tick_locs_order_y),
                                    show_major_tick_label = True, #show major tick label for the first scan
                                    num_of_minor_tick_marks= int(num_tick_marks_order_y),
                                    fmt_str = fmt_order_y)

        coord_top_left = [eval(tafel_bounds_pot)[0]-float(padding_tafel_pot),eval(tafel_bounds_current)[1]+float(padding_tafel_current)]
        offset = np.array(self.cv_tool.info['index_header_pos_offset_tafel'])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(15),list('abcdefghijklmno')))
        cvs_total_num = len([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())])
        ax_tafel.text(*coord_top_index_marker, '{})'.format(label_map[cvs_total_num]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))

        coord_top_left = [eval(order_bounds_ph)[0]-float(padding_order_ph),eval(order_bounds_y)[1]+float(padding_order_y)]
        offset = np.array(self.cv_tool.info['index_header_pos_offset_order'])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(15),list('abcdefghijklmno')))
        #cvs_total_num = len([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())])
        ax_order.text(*coord_top_index_marker, '{})'.format(label_map[cvs_total_num+1]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))
        for each in [ax_tafel,ax_order]:
            for tick in each.xaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))
            for tick in each.yaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))

        #move labels to right side of the plot

        '''
        ax_tafel.yaxis.set_label_position("right")
        ax_tafel.yaxis.tick_right()
        ax_order.yaxis.set_label_position("right")
        ax_order.yaxis.tick_right()
        '''
        self.widget_cv_view.canvas.draw()

    def plot_cv_data(self):
        self.widget_cv_view.canvas.figure.clear()
        '''
        if self.checkBox_default.isChecked():
            col_num = 2
            row_num = len(self.cv_tool.cv_info)
        else:
            col_num = max([2,self.spinBox_cols.value()])
            row_num = max([len(self.cv_tool.cv_info),self.spinBox_rows.value()])
        '''
        if self.checkBox_default.isChecked():
            col_num = 2
            row_num = max([3,len(self.cv_tool.cv_info)])
        else:
            col_num = max([2,int(self.widget_par_tree.par[('Figure_Layout_settings','total_columns')])])
            row_num = max([len(self.cv_tool.cv_info),int(self.widget_par_tree.par[('Figure_Layout_settings','total_rows')])])
        
        if not self.checkBox_use_all.isChecked():
            row_num = max([3,len(self.cv_tool.info['selected_scan'])])
        gs_left = plt.GridSpec(row_num,col_num,hspace=self.cv_tool.info['hspace'][0],wspace=self.cv_tool.info['wspace'][0])
        gs_right = plt.GridSpec(row_num,col_num, hspace=self.cv_tool.info['hspace'][1],wspace=self.cv_tool.info['wspace'][1])
        if self.checkBox_use_all.isChecked():
            # axs = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.cv_info), col_num, 1 + col_num*(i-1) ) for i in range(1,len(self.cv_tool.cv_info)+1)]
            axs = [self.widget_cv_view.canvas.figure.add_subplot(gs_left[i, 0]) for i in range(0,len(self.cv_tool.cv_info))]
            #self.cv_tool.plot_cv_files(axs = axs)
            self.cv_tool.plot_cv_files(axs = axs)
        else:
            # axs = [self.widget_cv_view.canvas.figure.add_subplot(len(self.cv_tool.info['selected_scan']), col_num, 1 + col_num*(i-1) ) for i in range(1,len(self.cv_tool.info['selected_scan'])+1)]
            axs = [self.widget_cv_view.canvas.figure.add_subplot(gs_left[i, 0]) for i in range(0,len(self.cv_tool.info['selected_scan']))]
            #self.cv_tool.plot_cv_files_selected_scans(axs = axs, scans = self.cv_tool.info['selected_scan'])
            self.cv_tool.plot_cv_files_selected_scans(axs = axs, scans = self.cv_tool.info['selected_scan'])
        labels = []
        for scan, each in zip([self.cv_tool.info['selected_scan'],self.cv_tool.info['sequence_id']][int(self.checkBox_use_all.isChecked())],axs):
            #index in the selected scan, if use all scans, then i=i_full
            i = axs.index(each)
            #index in the full sequence
            i_full = self.cv_tool.info['sequence_id'].index(scan)

            bounds_pot, tick_locs_pot, padding_pot, num_tick_marks_pot, fmt_pot, func_pot = self.cv_tool.info['cv_bounds_pot'].rsplit('+')
            bounds_current, tick_locs_current, padding_current, num_tick_marks_current, fmt_current, func_current = self.cv_tool.info['cv_bounds_current'].rsplit('+')
            show_tick_label_pot = self.cv_tool.info['cv_show_tick_label_x'][i_full]
            show_tick_label_current = self.cv_tool.info['cv_show_tick_label_y'][i_full]

            self._format_axis(each)
            self._format_ax_tick_labels(ax = each,
                                        fun_set_bounds = func_pot,
                                        bounds = eval(bounds_pot),
                                        bound_padding = float(padding_pot),
                                        major_tick_location = eval(tick_locs_pot),
                                        show_major_tick_label = show_tick_label_pot, #show major tick label for the first scan
                                        num_of_minor_tick_marks=int(num_tick_marks_pot),
                                        fmt_str = fmt_pot)
            self._format_ax_tick_labels(ax = each,
                                        fun_set_bounds = func_current,
                                        bounds = eval(bounds_current),
                                        bound_padding = float(padding_current),
                                        major_tick_location = eval(tick_locs_current),
                                        show_major_tick_label = show_tick_label_current, #show major tick label for the first scan
                                        num_of_minor_tick_marks=int(num_tick_marks_current),
                                        fmt_str = fmt_current)

            #set the index text marker for figure (eg. a), b) and so on ... )
            coord_top_left = np.array([eval(bounds_pot)[0]-float(padding_pot),eval(bounds_current)[1]+float(padding_current)])
            offset = np.array(self.cv_tool.info['index_header_pos_offset_cv'])
            coord_top_index_marker = coord_top_left+offset
            label_map = dict(zip(range(26),list('abcdefghijklmnopqrstuvwxyz')))
            each.text(*coord_top_index_marker, '{})'.format(label_map[i]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))
            #set pH label as title
            pH_text = 'pH {}'.format(self.cv_tool.info['ph'][i_full])
            which_pH13 = 0
            if self.cv_tool.info['ph'][i_full]==13:
                for each_scan in self.cv_tool.info['sequence_id']:
                    if self.cv_tool.info['ph'][self.cv_tool.info['sequence_id'].index(each_scan)]==13:
                        which_pH13 = which_pH13+1
                        if each_scan==scan:
                            pH_text = pH_text+'({})'.format(which_pH13)
                            break
            labels.append(pH_text)
            # ph_marker_pos = coord_top_left-[-0.1,eval(bounds_current)[1]*0.3]
            ph_marker_pos = coord_top_left-[-abs(eval(bounds_pot)[0]-eval(bounds_pot)[1])*0.2,eval(bounds_current)[1]*0.35]
            each.text(*ph_marker_pos, pH_text, fontsize = int(self.cv_tool.info['fontsize_text_marker']),color = self.cv_tool.info['color'][i_full])
            #set axis label
            if len(axs)<7:#show all y labels when the total num of ax is fewer than 7
                each.set_ylabel(r'j / mAcm$^{-2}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))
            else:#only show the y label for the middle panel
                if i== int((len(axs)-1)/2):
                    each.set_ylabel(r'j / mAcm$^{-2}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))

            if each == axs[-1]:
                each.set_xlabel(r'E / V$_{RHE}$',fontsize = int(self.cv_tool.info['fontsize_axis_label']))
            #now set the fontsize for tick marker
            for tick in each.xaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))
            for tick in each.yaxis.get_major_ticks():
                tick.label.set_fontsize(int(self.cv_tool.info['fontsize_tick_label']))

            #add scalling factor marker on each panel
            text_pos = (1,3)
            if 'scale_factor_text_pos' in self.cv_tool.info:
                if i_full < len(self.cv_tool.info['scale_factor_text_pos']):
                    text_pos = self.cv_tool.info['scale_factor_text_pos'][i_full]
                elif len(self.cv_tool.info['scale_factor_text_pos'])==0:
                    print('The length of text_pos doesnot match the lenght of scans, use default pos (1,3) instead!')
                else:
                    text_pos = self.cv_info.info['scale_factor_text_pos'][-1]
                    print('The length of text_pos doesnot match the lenght of scans, use the last item instead!')
            else:
                print('scale_factor_text_pos NOT existing in the config file, use default pos (1,3) instead!')
            each.text(*text_pos,'x{}'.format(self.cv_tool.info['cv_scale_factor'][i_full]),color=self.cv_tool.info['color'][i_full], fontsize = int(self.cv_tool.info['fontsize_text_marker']))
        tafel_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','tafel_row_range')])
        tafel_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','tafel_col_range')])
        tafel_row_lf = [tafel_row_range[0],0][int(self.checkBox_default.isChecked())]
        tafel_row_rt = [tafel_row_range[1],1][int(self.checkBox_default.isChecked())]
        tafel_col_lf = [tafel_col_range[0],1][int(self.checkBox_default.isChecked())]
        tafel_col_rt = [tafel_col_range[1],2][int(self.checkBox_default.isChecked())]

        order_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','rxn_order_row_range')])
        order_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','rxn_order_col_range')])
        order_row_lf = [order_row_range[0],1][int(self.checkBox_default.isChecked())]
        order_row_rt = [order_row_range[1],2][int(self.checkBox_default.isChecked())]
        order_col_lf = [order_col_range[0],1][int(self.checkBox_default.isChecked())]
        order_col_rt = [order_col_range[1],2][int(self.checkBox_default.isChecked())]

        charge_row_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','charge_row_range')])
        charge_col_range = eval(self.widget_par_tree.par[('Figure_Layout_settings','charge_col_range')])
        charge_row_lf = [charge_row_range[0],2][int(self.checkBox_default.isChecked())]
        charge_row_rt = [charge_row_range[1],3][int(self.checkBox_default.isChecked())]
        charge_col_lf = [charge_col_range[0],1][int(self.checkBox_default.isChecked())]
        charge_col_rt = [charge_col_range[1],2][int(self.checkBox_default.isChecked())]
        #charge_col = [int(self.lineEdit_col_charge.text()),1][int(self.checkBox_default.isChecked())]
        # axs_2 = [self.widget_cv_view.canvas.figure.add_subplot(gs_right[0:1,1]),self.widget_cv_view.canvas.figure.add_subplot(gs_right[1:3,1])]
        axs_2 = [self.widget_cv_view.canvas.figure.add_subplot(gs_right[tafel_row_lf:tafel_row_rt,tafel_col_lf:tafel_col_rt]),self.widget_cv_view.canvas.figure.add_subplot(gs_right[order_row_lf:order_row_rt,order_col_lf:order_col_rt])]

        self.plot_reaction_order_and_tafel(axs = axs_2)
        # ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[3:,1])
        ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[charge_row_lf:charge_row_rt,charge_col_lf:charge_col_rt])
        self._format_axis(ax_3)
        if self.checkBox_use_all.isChecked():
            bar_list = ax_3.bar(range(len(self.cv_tool.info['charge'])),self.cv_tool.info['charge'],0.5)
            bar_colors = self.cv_tool.info['color']
        else:
            index_ = [i for i in range(len(self.cv_tool.info['sequence_id'])) if self.cv_tool.info['sequence_id'][i] in self.cv_tool.info['selected_scan']]
            bar_list = ax_3.bar(range(len(self.cv_tool.info['selected_scan'])),[self.cv_tool.info['charge'][i] for i in index_],0.5)
            bar_colors = [self.cv_tool.info['color'][i] for i in index_]

        ax_3.set_ylabel(r'q / mCcm$^{-2}$')
        ax_3.set_xlabel(r'Measurement sequence')
        ax_3.set_ylim(0,max(self.cv_tool.info['charge'])*1.3)
        ax_3.set_xticks(range(len(bar_list)))
        ax_3.set_xticklabels(range(1,1+len(bar_list)))
        # ax_3.set_xticklabels(labels[0:len(self.cv_tool.info['charge'])])

        coord_top_left = np.array([len(bar_list)-1-0.1, max(self.cv_tool.info['charge'])*1.1])
        offset = np.array([0,0])
        coord_top_index_marker = coord_top_left+offset
        label_map = dict(zip(range(26),list('abcdefghijklmnopqrstuvwxyz')))
        ax_3.text(*coord_top_index_marker, '{})'.format(label_map[len(axs)+2]),weight = 'bold', fontsize = int(self.cv_tool.info['fontsize_index_header']))

        for i, bar_ in enumerate(bar_list):
            # bar_.set_color(self.cv_tool.info['color'][i])
            bar_.set_color(bar_colors[i])
        # try:
        #     self.plot_reaction_order_and_tafel(axs = axs_2)
        #     # ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[3:,1])
        #     ax_3 = self.widget_cv_view.canvas.figure.add_subplot(gs_right[charge_row_lf:charge_row_rt,charge_col_lf:charge_col_rt])
        #     bar_list = ax_3.bar(range(len(self.cv_tool.info['charge'])),self.cv_tool.info['charge'],0.5)
        #     ax_3.set_ylabel(r'q / mCcm$^{-2}$')
        #     #ax_3.set_title('Comparison of charge')
        #     ax_3.set_xticks(range(len(self.cv_tool.info['charge'])))
        #     #labels = ['HM1','HM2', 'HM3', 'PEEK1', 'PEEK2']
        #     ax_3.set_xticklabels(labels[0:len(self.cv_tool.info['charge'])])
        #     for i, bar_ in enumerate(bar_list):
        #         bar_.set_color(self.cv_tool.info['color'][i])
        # except:
        #     pass

        #ax_3.legend()
        #self.widget_cv_view.fig.subplots_adjust(wspace=0.31,hspace=0.15)
        self.widget_cv_view.canvas.figure.set_size_inches(self.cv_tool.info['figsize'])
        self.widget_cv_view.canvas.draw()

    def _setup_matplotlib_fig(self,plot_dim):
        self.mplwidget.fig.clear()
        # #[rows, columns]
        for scan in self.scans:
            setattr(self,'plot_axis_scan{}'.format(scan),[])
            j = self.scans.index(scan) + 1
            for i in range(plot_dim[0]):
                getattr(self,'plot_axis_scan{}'.format(scan)).append(self.mplwidget.canvas.figure.add_subplot(plot_dim[0], plot_dim[1],j+plot_dim[1]*i))
                self._format_axis(getattr(self,'plot_axis_scan{}'.format(scan))[-1])

    def _plot_one_panel_x_is_potential(self, scan, channel, channel_index, y_values, y_values_smooth, fmt, marker_index_container, slope_info):
        if self.checkBox_use_external_slope.isChecked():
            seperators = self.return_seperator_values(scan)
        else:
            seperators = list(set(marker_index_container))
        if channel!='current':
            #plot the channel values now
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_merge.isChecked():
                if scan!=self.scans[0]:
                    getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_show_smoothed_curve.isChecked():
                # x, y = self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]], y_values_smooth
                # z = np.linspace(0, 1, len(x))
                # line_segment = colorline(x, y, z, cmap=plt.get_cmap('binary'), linewidth=3)
                # getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].add_collection(line_segment)
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values_smooth,'-', color = '0.4')
            #plot the slope line segments
            cases = self._plot_slope_segment(scan = scan, channel = channel, channel_index = channel_index, y_values_smooth = y_values_smooth,
                                        slope_info = slope_info, seperators = seperators,  marker_index_container = marker_index_container)
            #store the calculated strain/size change
            if 'ip' in channel and len(cases)!=0:
                if 'grain' in channel:
                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
                elif 'strain' in channel:
                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'horizontal',cases)
            elif 'oop' in channel and len(cases)!=0:
                if 'grain' in channel:
                    self.set_grain_info_all_scan(self.grain_size_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
                elif 'strain' in channel:
                    self.set_grain_info_all_scan(self.strain_info_all_scans,scan,self.pot_ranges[scan],'vertical',cases)
        #now plot current channel
        else:
            #extract seperators for displaying vertical line segments
            _seperators = []
            if self.checkBox_use_external_slope.isChecked():
                _seperators = seperators[scan][channel]
            else:
                _seperators = [[self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][each_index]] for each_index in seperators]
            if self.checkBox_use_external_cv.isChecked():
                #plot cv profile from external files
                lim_y = self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],scan,_seperators)
                if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                    self.plot_cv_from_external(getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index],scan,_seperators)
                #overplot the internal cv data if you want
                #here y is already scaled by 8 considerring the current density to be shown
                if self.checkBox_use_internal_cv.isChecked():
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt, ls = '-', marker = None)
            else:
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]],y_values,fmt,ls = '-', marker = None)
                #show marker and vert line segments
                if self.checkBox_show_marker.isChecked():
                    [getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([each_seperator, each_seperator],[-100,100],'k:') for each_seperator in _seperators]

    def _plot_one_panel(self, scan, channel, channel_index, y_values, y_values_smooth, x_values, fmt, marker_index_container):
        current_channel = channel == 'current'
        if current_channel:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values,fmt, lw=2.5, marker = None, ls='-')
            if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(x_values,y_values,fmt, lw=2.5, marker = None, ls='-')
        else:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_merge.isChecked() and scan!=self.scans[0]:
                #if merged than plot the profile also at column 0 corresponding to self.scans[0]
                getattr(self,'plot_axis_scan{}'.format(self.scans[0]))[channel_index].plot(x_values,y_values,fmt,markersize = self.spinBox_marker_size.value())
            if self.checkBox_show_smoothed_curve.isChecked() and (channel!='potential'):#not show smooth line for grain size channels
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot(x_values,y_values_smooth, fmt, color = 'red', lw=2.5, marker = None, ls='-')
            if self.checkBox_show_marker.isChecked():#also display the bounds for specified pot_ranges
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([x_values[iii] for iii in marker_index_container],[y_values_smooth[iii] for iii in marker_index_container],'k*')
                for iii in marker_index_container:
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([x_values[iii]]*2,[-100,100],':k')

    def _plot_slope_segment(self, scan, channel, channel_index, y_values_smooth, slope_info, seperators, marker_index_container):
        if self.checkBox_plot_slope.isChecked() and self.checkBox_use_external_slope.isChecked():
            cases = []
            if slope_info[scan][channel]!=None:
                #one known point coords (the cross point): (p1, y1)
                #slopes are: a1 and a2
                #the other two points coords are: (p0,y0) and (p2, y2)
                p0,p1,p2,y1,a1,a2 = slope_info[scan][channel]
                y0 = a1*(p0-p1)+y1
                y2 = a2*(p2-p1)+y1
                cases = [self.calculate_size_strain_change(p0,p1,p2,y1,a1,a2,pot_range = each_pot) for each_pot in self.pot_ranges[scan]]
                #slope line segments
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([p0,p1,p2],np.array([y0,y1,y2])-self.data_to_plot[scan][channel+"_max"],'k--')
                #vertical line segments
                for pot in seperators[scan][channel]:
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([pot,pot],[-100,100],'k:')
        else:
            cases = [self.calculate_size_strain_change_from_plot_data(scan, channel, self.data_range[self.scans.index(scan)], each_pot) for each_pot in self.pot_range]
            #vertical line segments only
            if self.checkBox_show_marker.isChecked():
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][iii] for iii in marker_index_container],[y_values_smooth[iii] for iii in marker_index_container],'k*')
                for each_index in seperators:
                    pot = self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]][each_index]
                    getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].plot([pot,pot],[-100,100],'k:')
        return cases

    def _get_fmt_style(self, scan, channel):
        #fmt here is a list of one or two format strings (eg. -b;-r,-b;-r will destructure to ['-b','-r'] and ['-b', '-r'])
        #you may want to show lines only for strain channels, but only show symbols for grain size channels due to the large error bar
        #if two items: second one is for grain size channels, first is for the other channels
        #if only one item: all channel share the same fmt style
        try:
            fmt = self.lineEdit_fmt.text().rsplit(',')[self.scans.index(scan)].rsplit(";")
        except:
            fmt = ['b-']
        #extract the fmt tag
        if len(fmt)==2:
            fmt = fmt[int('size' in channel)]
        else:
            fmt = fmt[0]
        return fmt

    def _update_bounds_xy(self, scan, y_values, channel_index, x_min_value, x_max_value, y_min_values, y_max_values):
        temp_max, temp_min = max(y_values), min(y_values)
        temp_max_x, temp_min_x = max(list(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]])), min(list(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]]))
        if temp_max_x > x_max_value:
            x_max_value = temp_max_x
        if temp_min_x < x_min_value:
            x_min_value = temp_min_x
        if y_max_values[channel_index]<temp_max:
            y_max_values[channel_index] = temp_max
        if y_min_values[channel_index]>temp_min:
            y_min_values[channel_index] = temp_min
        return x_min_value, x_max_value, y_min_values, y_max_values

    def _set_xy_tick_labels(self, scan, channel, channel_index, channel_length):
        ##set x tick labels
        #the x tick lable only shown for the last panel, will be hidden for the others
        if channel_index!=(channel_length-1):
            ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index]
            ax.set_xticklabels([])
        else:#show x tick label for last panel, either potential or image_no
            ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index]
            x_label = [r'Time (s)','E / V$_{RHE}$'][self.plot_label_x[self.scans.index(scan)]=='potential']
            ax.set_xlabel(x_label, fontsize = 13)
        ##set y tick labels
        #the y tick label only shown for the first column panel
        if scan!=self.scans[0]:
            getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].set_yticklabels([])
        else:
            #according to relative scale
            y_label_map = {'potential':'E / V$_{RHE}$',
                            'current':r'j / mAcm$^{-2}$',
                            'strain_ip':r'$\Delta\varepsilon_\parallel$  (%)',
                            'strain_oop':r'$\Delta\varepsilon_\perp$  (%)',
                            'grain_size_oop':r'$\Delta d_\perp$ / nm',
                            'grain_size_ip':r'$\Delta d_\parallel$ / nm',
                            'peak_intensity':r'Intensity / a.u.'}
            #based on absolute values
            y_label_map_abs = {'potential':'E / V$_{RHE}$',
                                'current':r'j / mAcm$^{-2}$',
                                'strain_ip':r'$\varepsilon_\parallel$  (%)',
                                'strain_oop':r'$\varepsilon_\perp$  (%)',
                                'grain_size_oop':r'$ d_\perp$ / nm',
                                'grain_size_ip':r'$ d_\parallel$ / nm',
                                'peak_intensity':r'Intensity / a.u.'}
            if not self.checkBox_max.isChecked():
                y_label_map = y_label_map_abs
            if channel in y_label_map:
                getattr(self,'plot_axis_scan{}'.format(scan))[channel_index].set_ylabel(y_label_map[channel], fontsize = 13)

    def _do_text_label(self, scan, count_pH13, x_min_value, y_max_values):
        #overwrite max_y using the format setting
        if 'master' in self.tick_label_settings:
            if 'current' in self.tick_label_settings['master']:
                y_max_values[0] = self.tick_label_settings['master']['current']['locator'][-1] + float(self.tick_label_settings['master']['current']['padding'])

        #extract color
        try:#from cv settings
            _,_,_,_,_,color, _, _ = self.plot_lib[scan]
        except:#specified in gui
            color = self.comboBox_color.currentText()

        # pH labeling
        text = r'pH {}'.format(self.phs[self.scans.index(scan)])
        tag = ''
        if self.radioButton_pH.isChecked():
            if self.phs[self.scans.index(scan)]==13:
                text = r'pH {} ({})'.format(self.phs[self.scans.index(scan)],count_pH13)
                count_pH13 += 1
            else:
                text = r'pH {}'.format(self.phs[self.scans.index(scan)])
        # scan number label
        elif self.radioButton_scan.isChecked():
            text = f'scan{scan}'
        elif self.radioButton_custom.isChecked():
            labels = self.lineEdit_custom_label.text().rstrip().rsplit(',')
            index = self.scans.index(scan)
            if len(labels)>index:
                text = labels[index]
            else:
                print('The dimention of text label does not match the total number of scans!')
                text = f'{scan}'
        # without label
        else:
            text = ''
        #set label here
        text_obj = getattr(self,'plot_axis_scan{}'.format(scan))[0].text(x_min_value, y_max_values[0]*0.8,text,color = color,fontsize=11)
        setattr(self, f'text_obj_{scan}', text_obj)
        return count_pH13

    def _decorate_axis_tick_labels(self, scan, channel, channel_index, x_min_value, x_max_value, y_min_values, y_max_values):

        if self.plot_label_x[self.scans.index(scan)] == 'potential':
            if 'master' in self.tick_label_settings:
                if 'potential' in self.tick_label_settings['master']:
                    if self.checkBox_use.isChecked():
                        self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],
                                                    fun_set_bounds = self.tick_label_settings['master']['potential']['func'],#'set_xlim',
                                                    bounds = [x_min_value,x_max_value],#[0.4,2.1],#[0.95,1.95],
                                                    bound_padding = float(self.tick_label_settings['master']['potential']['padding']),
                                                    major_tick_location = self.tick_label_settings['master']['potential']['locator'], #x_locator
                                                    show_major_tick_label = (len(self.plot_labels_y)-1)==channel_index, #show major tick label for the first scan
                                                    num_of_minor_tick_marks=self.tick_label_settings['master']['potential']['tick_num'], #4
                                                    fmt_str = self.tick_label_settings['master']['potential']['fmt'])#'{:3.1f}'

        #y axis
        if 'master' in self.tick_label_settings:
            if channel in self.tick_label_settings['master']:
                if self.checkBox_use.isChecked():
                    self._format_ax_tick_labels(ax = getattr(self,'plot_axis_scan{}'.format(scan))[channel_index],
                                                fun_set_bounds = self.tick_label_settings['master'][channel]['func'],#'set_xlim',
                                                bounds = [y_min_values[channel_index],y_max_values[channel_index]],#[0.4,2.1],#[0.95,1.95],
                                                bound_padding = float(self.tick_label_settings['master'][channel]['padding']),
                                                major_tick_location = self.tick_label_settings['master'][channel]['locator'], #x_locator
                                                show_major_tick_label = self.scans.index(scan)==0, #show major tick label for the first scan
                                                num_of_minor_tick_marks=self.tick_label_settings['master'][channel]['tick_num'], #4
                                                fmt_str = self.tick_label_settings['master'][channel]['fmt'])#'{:3.1f}'

    #plot the master figure
    def plot_figure_xrv(self):
        #update state and reset meta data
        self.make_plot_lib()#external cv files
        self.reset_meta_data()#calculated values (eg strain and grain size) and tick label setting reset to empty {}
        self.extract_tick_label_settings()#extract the latest tick label setting

        #extract slope info if any
        slope_info_temp = None
        if self.checkBox_use_external_slope.isChecked():
            slope_info_temp = self.return_slope_values()

        #init plot settings, create figure axis, and init the bounds of x and y axis
        self._setup_matplotlib_fig(plot_dim = [len(self.plot_labels_y), len(self.scans)])
        #these are extreme values, these values will be updated
        y_max_values,y_min_values = [-100000000]*len(self.plot_labels_y),[100000000]*len(self.plot_labels_y) #multiple sets of ylim
        x_min_value, x_max_value = [1000000000,-10000000000] # one set of xlim

        #prepare ranges for viewing datasummary, which summarize the variance of structural pars in each specified range
        #this is a way to remove duplicate data points if there are multiple cycles
        self._prepare_data_range_and_pot_range()

        #the main loop starts from here
        for scan in self.scans:
            self.cal_potential_ranges(scan)
            #data_summary, summarizing values of structural changes, is used to plot bar char afterwards
            self.data_summary[scan] = {}
            if 'potential' in self.plot_labels_y and self.plot_label_x[self.scans.index(scan)] == 'potential':
                plot_labels_y = [each for each in self.plot_labels_y if each!='potential'] # remove potential in y lables if potential is set as x channel already
            else:
                plot_labels_y = self.plot_labels_y
            #plot each y channel from here
            for each in plot_labels_y:
                #each is the y channel string tag
                self.data_summary[scan][each] = []
                #i is the channel index
                i = plot_labels_y.index(each)
                #is this the current channel
                current_channel = each == 'current'
                #y vs image_no?
                x_is_frame_no = self.plot_label_x[self.scans.index(scan)] == 'image_no'
                #extract fmt style
                fmt = self._get_fmt_style(scan = scan, channel = each)
                #extract the channel values
                y, y_smooth_temp, std_val = self._extract_y_values(scan = scan, channel = each)
                #this marker container will contain the positions of potentials bounds according to the specified potential ranges
                marker_index_container = []
                for ii in range(len(self.pot_range)):
                    marker_index_container = self._cal_structural_change_rate(scan = scan, channel =each,
                                                                              y_values = y_smooth_temp,
                                                                              std_val = std_val,
                                                                              data_range = self.data_range[self.scans.index(scan)],
                                                                              pot_range = self.pot_range[ii],
                                                                              marker_index_container = marker_index_container)
                # print(marker_index_container)
                # if the plot channel is versus time (image_no)
                if x_is_frame_no:
                    #x offset
                    offset_ = 0
                    if hasattr(self,f'image_no_offset_{scan}'):
                        offset_ = getattr(self, f'image_no_offset_{scan}')
                    #here two situations, plot current density or plot other channels
                    #current density: you can either use internal data points or extract values from external files
                    #NOTE: the sampling rate of internal data is way smaller than that of external files, so we don't want to use external file for plotting current
                    self._plot_one_panel(scan = scan, channel = each, channel_index = i, y_values = y,
                                         y_values_smooth = y_smooth_temp, x_values = np.arange(len(y))+offset_,
                                         fmt = fmt, marker_index_container = marker_index_container)
                #if the plot channel is versus potential
                else:
                    self._plot_one_panel_x_is_potential(scan = scan, channel = each, channel_index = i, y_values = y,
                                                        y_values_smooth = y_smooth_temp, fmt = fmt, marker_index_container = marker_index_container,
                                                        slope_info = slope_info_temp)
                #update the x and y bounds
                x_min_value, x_max_value, y_min_values, y_max_values  = self._update_bounds_xy(scan = scan, y_values = y, channel_index = i,
                                                                                              x_min_value = x_min_value, x_max_value = x_max_value,
                                                                                              y_min_values = y_min_values, y_max_values = y_max_values)
                #set xy tick labels
                self._set_xy_tick_labels(scan = scan, channel = each, channel_index = i, channel_length = len(plot_labels_y))
            #now calculate the pseudocapacitative charge values
            self._cal_pseudcap_charge(scan = scan)
        count_pH13_temp = 1#count the times of dataset for pH 13
        #text labeling on the master figure
        for scan in self.scans:
            # label display only on the first row
            count_pH13_temp = self._do_text_label(scan = scan, count_pH13 = count_pH13_temp, x_min_value = x_min_value, y_max_values = y_max_values)
            for each in self.plot_labels_y:
                i = self.plot_labels_y.index(each)
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_xlim(x_min_value, x_max_value)
                getattr(self,'plot_axis_scan{}'.format(scan))[i].set_ylim(*[y_min_values[i],y_max_values[i]])
                ####The following lines are customized for axis formating (tick locations, padding, ax bounds)
                #decorate axis tick labels
                self._decorate_axis_tick_labels(scan = scan, channel = each, channel_index = i, x_min_value = x_min_value, x_max_value = x_max_value,
                                                y_min_values = y_min_values, y_max_values = y_max_values)
        self.mplwidget.fig.tight_layout()
        self.mplwidget.fig.subplots_adjust(wspace=0.04,hspace=0.04)
        self.mplwidget.canvas.draw()

    #format the axis tick so that the tick facing inside, showing both major and minor tick marks
    #The tick marks on both sides (y tick marks on left and right side and x tick marks on top and bottom side)
    def _format_axis(self,ax):
        major_length = 4
        minor_length = 2
        if hasattr(ax,'__len__'):
            for each in ax:
                each.tick_params(which = 'major', axis="x", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
                each.tick_params(which = 'major', axis="y", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
                each.tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
                each.tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
                each.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        else:
            ax.tick_params(which = 'major', axis="x", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
            ax.tick_params(which = 'major', axis="y", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
            ax.tick_params(which = 'major', bottom=True, top=True, left=True, right=True)
            ax.tick_params(which = 'minor', bottom=True, top=True, left=True, right=True)
            ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

    def _format_axis_customized(self,ax, **settings):
        major_length = 4
        minor_length = 2
        bottom, top, left, right=True, True, True, True
        labelbottom, labeltop,labelleft,labelright=True, False, True, False
        if 'bottom' in settings:
            bottom = settings['bottom']
        if 'top' in settings:
            top = settings['top']
        if 'left' in settings:
            left = settings['left']
        if 'right' in settings:
            right = settings['right']
        if 'labelbottom' in settings:
            labelbottom = settings['labelbottom']
        if 'labeltop' in settings:
            labeltop = settings['labeltop']
        if 'labelleft' in settings:
            labelleft = settings['labelleft']
        if 'labelright' in settings:
            labelright = settings['labelright']

        if hasattr(ax,'__len__'):
            for each in ax:
                each.tick_params(which = 'major', axis="x", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
                each.tick_params(which = 'major', axis="y", length = major_length, direction="in")
                each.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
                each.tick_params(which = 'major', bottom=bottom, top=top, left=left, right=right)
                each.tick_params(which = 'minor', bottom=bottom, top=top, left=left, right=right)
                each.tick_params(labelbottom=labelbottom, labeltop=labeltop, labelleft=labelleft, labelright=labelright)
        else:
            ax.tick_params(which = 'major', axis="x", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="x", length = minor_length,direction="in")
            ax.tick_params(which = 'major', axis="y", length = major_length,direction="in")
            ax.tick_params(which = 'minor', axis="y", length = minor_length,direction="in")
            ax.tick_params(which = 'major', bottom=bottom, top=top, left=left, right=right)
            ax.tick_params(which = 'minor', bottom=bottom, top=top, left=left, right=right)
            ax.tick_params(labelbottom=labelbottom, labeltop=labeltop, labelleft=labelleft, labelright=labelright)

    def _format_ax_tick_labels(self,ax,fun_set_bounds = 'set_ylim', bounds = [0,1], bound_padding = 0, major_tick_location = [], show_major_tick_label = True, num_of_minor_tick_marks=5, fmt_str = '{: 4.2f}'):
        mapping = {'set_ylim':'yaxis','set_xlim':'xaxis'}
        which_axis = mapping[fun_set_bounds]
        #redefine the bounds using major tick locations
        major_tick_values = [float(each) for each in major_tick_location]
        bounds = [min(major_tick_values), max(major_tick_values)]
        bounds_after_add_padding = bounds[0]-bound_padding, bounds[1]+bound_padding
        major_tick_labels = []
        for each in major_tick_location:
            if show_major_tick_label:
                major_tick_labels.append(fmt_str.format(each))
            else:
                major_tick_labels.append('')
        minor_tick_location = []
        minor_tick_labels = []
        for i in range(len(major_tick_location)-1):
            start = major_tick_location[i]
            end = major_tick_location[i+1]
            tick_spacing = (end-start)/(num_of_minor_tick_marks+1)
            for j in range(num_of_minor_tick_marks):
                minor_tick_location.append(start + tick_spacing*(j+1))
                minor_tick_labels.append('')#not showing minor tick labels
            #before starting point
            if i==0:
                count = 1
                while True:
                    if (start-count*abs(tick_spacing))<bounds_after_add_padding[0]:
                        break
                    else:
                        minor_tick_location.append(start - abs(tick_spacing)*count)
                        minor_tick_labels.append('')#not showing minor tick labels
                        count = count+1
            #after the last point
            elif i == (len(major_tick_location)-2):
                count = 1
                while True:
                    if (end+count*abs(tick_spacing))>bounds_after_add_padding[1]:
                        break
                    else:
                        minor_tick_location.append(end + abs(tick_spacing)*count)
                        minor_tick_labels.append('')#not showing minor tick labels
                        count = count+1

        #set limits
        getattr(ax,fun_set_bounds)(*bounds_after_add_padding)
        #set major tick and tick labels
        getattr(ax, which_axis).set_major_formatter(FixedFormatter(major_tick_labels))
        getattr(ax, which_axis).set_major_locator(FixedLocator(major_tick_location))
        #set minor tick and tick lables (not showing the lable)
        getattr(ax, which_axis).set_minor_formatter(FixedFormatter(minor_tick_labels))
        getattr(ax, which_axis).set_minor_locator(FixedLocator(minor_tick_location))