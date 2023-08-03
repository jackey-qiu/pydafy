import os 
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from dafy.core.util.DebugFunctions import error_pop_up
from dafy.core.util.PlotSetup import RHE

class DataPreprocessing(object):
    def perform_bkg_fitting(self):
        order = 0
        if self.checkBox_order1.isChecked():
            order+=1
        if self.checkBox_order2.isChecked():
            order+=2
        if self.checkBox_order3.isChecked():
            order+=3
        if self.checkBox_order4.isChecked():
            order+=4
        fct = 'atq'
        if self.radioButton_stq.isChecked():
            fct = 'stq'
        if self.radioButton_sh.isChecked():
            fct = 'sh'
        if self.radioButton_ah.isChecked():
            fct = 'ah'
        s = self.doubleSpinBox_ss_factor.value()
        scan_rate = float(self.lineEdit_scan_rate.text())
        charge = self.widget.perform_bkg_fitting(order, s, fct, scan_rate)
        self.lineEdit_charge.setText(f'{charge} mC/cm2')

    def _init_meta_data_for_scans(self, scans):
        for each in scans:
            setattr(self, f'strain_ip_offset_{each}',0)
            setattr(self, f'strain_oop_offset_{each}',0)
            setattr(self, f'grain_size_ip_offset_{each}',0)
            setattr(self, f'grain_size_oop_offset_{each}',0)
            setattr(self, f'potential_offset_{each}',0)
            setattr(self, f'current_offset_{each}',0)
            setattr(self, f'image_no_offset_{each}',0)    

    def calculate_charge_2(self):
        output = self.cv_tool.calc_charge_all()
        self.plainTextEdit_cv_summary.setPlainText(output)

    def set_grain_info_all_scan(self,grain_object, scan, pot_ranges, direction, cases):
        pot_ranges = [tuple(each) for each in pot_ranges]
        if scan not in grain_object:
            grain_object[scan] = {}
        if pot_ranges[0] not in grain_object[scan]:
            for each_pot, case in zip(pot_ranges,cases):
                grain_object[scan][each_pot] = {direction:case,'pH':self.phs[self.scans.index(scan)]}
        else:
            for each_pot, case in zip(pot_ranges,cases):
                grain_object[scan][each_pot][direction] = case
                grain_object[scan][each_pot]['pH'] = self.phs[self.scans.index(scan)]
        return grain_object

    def reset_meta_data(self):
        self.charge_info = {}
        self.grain_size_info_all_scans = {}
        self.strain_info_all_scans = {}#key is scan_no, each_item is {(pot1,pot2):{"vertical":(abs_value,value_change),"horizontal":(abs_value,value_change)},"pH":pH value}}
        self.tick_label_settings = {}            

    def update_pot_offset(self):
        self.potential_offset = eval(self.lineEdit_pot_offset.text())/1000
        self.append_scans_xrv()
        self.plot_figure_xrv()        

    def make_plot_lib(self):
        self.plot_lib = {}
        if hasattr(self, 'textEdit_plot_lib'):
            info = self.textEdit_plot_lib.toPlainText().rsplit('\n')
            folder = self.lineEdit_cv_folder.text()
            if info==[''] or folder=='':
                return
            for each in info:
                if not each.startswith('#'):
                    # scan, cv, cycle, cutoff,scale,color, ph, func = each.replace(" ","").rstrip().rsplit(',')
                    scan, cv, cycle, scale, length, order, color, ph, func = each.replace(" ","").rstrip().rsplit(',')
                    cv_name = os.path.join(folder,cv)
                    self.plot_lib[int(scan)] = [cv_name,eval(cycle),eval(scale),eval(length), eval(order),color,eval(ph),func]
        if hasattr(self,'tableView_cv_setting'):
            folder = self.lineEdit_cv_folder.text()
            if folder=='':
                return
            for each in range(self.pandas_model_cv_setting._data.shape[0]):
                if self.pandas_model_cv_setting._data.iloc[each,0]:
                    scan, cv, cycle, scale, length, order, color, ph, func = self.pandas_model_cv_setting._data.iloc[each,1:].to_list()
                    cv_name = os.path.join(folder,cv)
                    self.plot_lib[int(scan)] = [cv_name,eval(cycle),eval(scale),eval(length), eval(order),color,eval(ph),func]        

    #data format based on the output of IVIUM potentiostat
    def extract_ids_file(self,file_path,which_cycle=3):
        data = []
        current_cycle = 0
        with open(file_path,encoding="ISO-8859-1") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith('primary_data'):
                    # print(current_cycle)
                    current_cycle=current_cycle+1
                    if current_cycle == which_cycle:
                        for j in range(i+3,i+3+int(lines[i+2].rstrip())):
                            data.append([float(each) for each in lines[j].rstrip().rsplit()])
                        break
                    else:
                        pass
                else:
                    pass
        #return (pot: V, current: mA)
        return np.array(data)[:,0], np.array(data)[:,1]*1000

    #data format based on Fouad's potentiostat
    def extract_cv_file(self,file_path='/home/qiu/apps/048_S221_CV', which_cycle=1):
        #return:time(s), pot(V), current (mA)
        skiprows = 0
        with open(file_path,'r') as f:
            for each in f.readlines():
                if each.startswith('Time(s)'):
                    skiprows+=1
                    break
                else:
                    skiprows+=1
        data = np.loadtxt(file_path,skiprows = skiprows)
        #nodes index saving all the valley pot positions
        nodes =[0]
        for i in range(len(data[:,1])):
            if i!=0 and i!=len(data[:,1])-1:
                if data[i,1]<data[i+1,1] and data[i,1]<data[i-1,1]:
                    nodes.append(i)
        nodes.append(len(data[:,1]))
        if which_cycle>len(nodes):
            print('Cycle number lager than the total cycles! Use the first cycle instead!')
            return data[nodes[1]:nodes[2],0], data[nodes[1]:nodes[2],1],data[nodes[1]:nodes[2],2]
        else:
            return data[nodes[which_cycle]:nodes[which_cycle+1],0],data[nodes[which_cycle]:nodes[which_cycle+1],1],data[nodes[which_cycle]:nodes[which_cycle+1],2]

    def extract_cv_data(self):
        scan_no = int(self.comboBox_scans_3.currentText())
        file_name,which_cycle,cv_scale_factor, smooth_length, smooth_order, color, ph, func_name= self.plot_lib[scan_no]
        func = eval('self.cv_tool.{}'.format(func_name))
        results = func(file_name, which_cycle)
        pot,current = results
        pot_filtered, current_filtered = pot, current
        pot_filtered = RHE(pot_filtered,pH=ph)
        # print(file_name,func_name,pot,current)
        #smooth the current due to beam-induced spikes
        pot_filtered, current_filtered = self.cv_tool.filter_current(pot_filtered, current_filtered*8, smooth_length, smooth_order)
        self.widget.set_data(pot_filtered, current_filtered)
        #return pot_filtered, current_filtered

    def get_integrated_charge(self, pot, current, t, pot_range_full = [1., 1.57], steps = 10,plot= False):
        trans_pot = [1.4,1.42,1.43,1.45,1.5]
        pot_ranges = []
        for each in trans_pot:
            pot_ranges.append([pot_range_full[0],each])
            pot_ranges.append([each,pot_range_full[1]])
        for pot_range in pot_ranges:
            Q_integrated = 0
            pot_step = (pot_range[1] - pot_range[0])/steps
            def _get_index(all_values, current_value, first_half = True):
                all_values = np.array(all_values)
                half_index = int(len(all_values)/2)
                if first_half:
                    return np.argmin(abs(all_values[0:half_index]-current_value))
                else:
                    return np.argmin(abs(all_values[half_index:]-current_value))
            for i in range(steps):
                pot_left, pot_right = pot_range[0] + pot_step*i, pot_range[0] + pot_step*(i+1)
                delta_t = abs(t[_get_index(pot, pot_left)] - t[_get_index(pot, pot_right)])
                i_top_left, i_top_right = current[_get_index(pot, pot_left)], current[_get_index(pot, pot_right)]
                i_bottom_left, i_bottom_right = current[_get_index(pot, pot_left,False)], current[_get_index(pot, pot_right,False)]
                Q_two_triangles = abs(i_top_left - i_top_right)*delta_t/2 + abs(i_bottom_left - i_bottom_right)*delta_t/2
                if i_top_left > i_bottom_left:
                    Q_retangle = abs(abs(min([i_top_left, i_top_right]))-abs(max([i_bottom_left, i_bottom_right])))*delta_t
                else:
                    Q_retangle = abs(abs(min([i_bottom_left, i_bottom_right]))-abs(max([i_top_left, i_top_right])))*delta_t
                Q_integrated = Q_integrated + Q_two_triangles + Q_retangle
            if plot:
                fig = plt.figure()
                plt.plot(t, current)
                plt.show()
            print('Integrated charge between E {} and E {} is {} mC'.format(pot_range[0], pot_range[1], Q_integrated/2))
            #factor of 2 is due to contributions from the anodic and cathodic cycle
        return Q_integrated/2

    def estimate_charge_from_skin_layer_thickness(self, slope, transition_E, pot_range,charge_per_unit_cell = 2, roughness_factor = 1):
        surface_area = 0.125 #cm2
        unitcell_area = 28.3 #A2
        single_layer_thickness = 0.5 #nm
        num_unit_cell_at_surface = surface_area/unitcell_area*10**16*roughness_factor
        single_layer_charge_transfer = num_unit_cell_at_surface * charge_per_unit_cell
        thickness_skin_layer = abs(slope * (pot_range[0] - pot_range[1]))
        percentage_oxidation = thickness_skin_layer/single_layer_thickness
        # print('percentage_oxidation={}'.format(percentage_oxidation))
        charge_transfer = percentage_oxidation * single_layer_charge_transfer * 1.6 * 10**-19 #in C
        print('Charge transfer between E {} and E {} is  {} mC based on skin layer thickness estimation'.format(pot_range[0], pot_range[1], charge_transfer*10**3))
        return charge_transfer*10**3

    #return size/strain change from slope info in vertical or horizontal direction, and the associated absolute size
    def calculate_size_strain_change(self, x0,x1,x2,y1,slope1,slope2,pot_range,roughness_factor = 1):
        y0 = slope1*(x0-x1)+y1
        y2 = slope2*(x2-x1)+y1
        pot_left, pot_right = pot_range

        size_at_pot_left, size_at_pot_right = 0, 0
        if pot_left<=x1:
            size_at_pot_left = slope1*(pot_left-x1)+y1
        else:
            size_at_pot_left = slope2*(pot_left-x1)+y1

        if pot_right<=x1:
            size_at_pot_right = slope1*(pot_right-x1)+y1
        else:
            size_at_pot_right = slope2*(pot_right-x1)+y1
        return max([size_at_pot_left, size_at_pot_right]),abs(size_at_pot_left - size_at_pot_right)

    def calculate_size_strain_change_from_plot_data(self, scan, label, data_range, pot_range, roughness_factor = 1):
        # pot_lf, pot_rt = pot_range
        # pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]]
        # values = np.array(self.data_to_plot[scan][label][data_range[0]:data_range[1]])+self.data_to_plot[scan][label+'_max']
        # values_smooth = signal.savgol_filter(values,41,2)
        # index_lf = np.argmin(np.abs(pots - pot_lf))
        # index_rt = np.argmin(np.abs(pots - pot_rt))
        pot_lf, pot_rt = pot_range
        if data_range[0]==data_range[1]:
            pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1]
        else:
            pots = self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]]
        values = np.array(self.data_to_plot[scan][label])+self.data_to_plot[scan][label+'_max']
        values_smooth = signal.savgol_filter(values,41,2)
        index_lf = np.argmin(np.abs(pots - pot_lf)) + data_range[0]
        index_rt = np.argmin(np.abs(pots - pot_rt)) +data_range[0]
        return max([values_smooth[index_lf],values_smooth[index_rt]]), values_smooth[index_rt]-values_smooth[index_lf]

    def estimate_charge_from_skin_layer_thickness_philippe_algorithm(self, size_info, q0 = 15.15):
        #q0 : number of electron transfered per nm^3 during redox chemistry(15.15 only for Co3O4 material)
        #check the document for details of this algorithm
        vertical, horizontal = size_info['vertical'], size_info['horizontal']
        vertical_size, vertical_size_change = vertical
        horizontal_size, horizontal_size_change = horizontal
        charge_per_electron = 1.6*(10**-19) # C
        v_skin = (vertical_size_change + 2*(horizontal_size_change*vertical_size/horizontal_size))*(10**14)
        q_skin = v_skin * q0 * charge_per_electron * 1000 # mC per m^2
        q_film = vertical_size*(10**14)*q0 * charge_per_electron * 1000
        return q_skin, q_film        

    def _ir_correction(self, Rs):
        text = self.scan_numbers_append.text()
        if text!='':
            scans = list(set([int(each) for each in text.rstrip().rsplit(',')]))
        else:
            return
        scans.sort()
        if not hasattr(self, 'data'):
            return
        self.data['iR'] = 0.0
        assert len(scans)<=len(Rs), 'The shape of total scans and the total Rs does not match'
        for i, scan in enumerate(scans):
            R = Rs[i]
            idx = self.data['potential'][self.data['scan_no']==scan].index
            for id in idx:
                #current in mA, R in om
                self.data.at[id, 'potential']-=abs(self.data['current'][id]*0.001)*R
                self.data.at[id, 'potential_cal']-=abs(self.data['current'][id]*0.001)*R
                #this column is needed for saving the original potential to excel file
                self.data.at[id, 'iR'] = abs(self.data['current'][id]*0.001)*R

    #for each item
    #eg self.data_summary[221]['strain_ip] = [slop_1st_seg, error_1st_seg, slope_2nd_seg, error_2nd_seg]
    # this fun is very much highly customized for one time purpose, not generic at all
    #be careful if you want to use it to extract info from a new fit file
    def make_data_summary_from_external_file(self):
        file = self.lineEdit_slope_file.text()
        if file=="":
            return
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                summary[each][item] = []
                item_values = list(data['scan{}'.format(each)][item])[::-1]
                for each_item in item_values:
                    error = [0.05,0.15][int('size' in item)]
                    summary[each][item] = summary[each][item] + [each_item,error*abs(each_item)]
        self.data_summary = summary

    #extract the potential seperation in the fit file
    def return_seperator_values(self,scan):
        file = self.lineEdit_slope_file.text()
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        summary[scan] = {}
        for_current =[]
        try:
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                #_p1 is the associated potential value at the cross point (seperator)
                summary[scan][item] = [data['scan{}'.format(scan)]['{}_p1'.format(item)]+self.potential_offset]
                for_current.append(summary[scan][item])
            summary[scan]['current'] = for_current
        except:
            summary[scan] = None
        return summary

    #get slope/intercept values from fit files
    #dic of scan_number
    #each item is a dic of structural values, each structure value (eg strain_ip) corresponds to 6 items [p0, p1,p2,y1,a1,a2]
    #p's are potentials at the left bound(p0), right bound (p2), and cross point (p1), y1 is the associated value at p1, a1 and a2 are two slopes
    def return_slope_values(self):
        file = self.lineEdit_slope_file.text()
        data = pd.read_csv(file,sep='\t',comment = '#')
        summary = {}
        for each in self.scans:
            summary[each] = {}
            for item in ['strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                try:
                    summary[each][item] = [data['scan{}'.format(each)]['{}_p{}'.format(item,i)]+self.potential_offset for i in range(3)]
                    summary[each][item] = summary[each][item] + [data['scan{}'.format(each)]['{}_y1'.format(item)]]
                    summary[each][item] = summary[each][item] +  list(data['scan{}'.format(each)]['{}'.format(item)])[::-1]
                except:
                    error_pop_up('Could not extract the slope value from fit file', 'Error')
        #each item = [p0,p1,p2,y1,a1,a2], as are slope values, (p1,y1) transition value, y0 and y2 are end points for potentials
        return summary

    #when using external_slope, the potential range is determined by the fitted cross point and the bounds you provied on the GUI (in tab of More setup, potential_range)
    #If not, just use the ranges specified in tab of Basic setup, potential ranges selector
    def cal_potential_ranges(self,scan):
        f = lambda x:(round(x[0],3),round(x[1],3))
        if self.checkBox_use_external_slope.isChecked():
            slope_info_temp = self.return_slope_values()
        else:
            slope_info_temp = None
        if slope_info_temp == None:
            self.pot_ranges[scan] = [f(each) for each in self.pot_range]
        else:
            _,p1,*_ = slope_info_temp[scan]["strain_ip"]#p1 y1 is the cross point, thus should be the same for all structural pars
            #NOTE: this could be buggy, the text inside the lineEdit has to be like this "1.2,1.5"
            pot_range_specified = eval("({})".format(self.lineEdit_pot_range.text().rstrip()))
            if p1>pot_range_specified[1]:
                p1 = sum(pot_range_specified)/2
            pot_range1 = (pot_range_specified[0], p1)
            pot_range2 = (p1, pot_range_specified[1])
            pot_range3 = pot_range_specified
            self.pot_ranges[scan] = [f(pot_range1),f(pot_range2),f(pot_range3)]

    def _prepare_data_range_and_pot_range(self):
        '''
        data_range = self.lineEdit_data_range.text().rsplit(',')
        if len(data_range) == 1:
            data_range = [list(map(int,data_range[0].rsplit('-')))]*len(self.scans)
        else:
            assert len(data_range) == len(self.scans)
            data_range = [list(map(int,each.rsplit('-'))) for each in data_range]
        self.data_range = data_range
        '''
        # pot_range is a partial set from the specified data_ranges
        # which this, it is more intuitive to pick the data points for variantion calculation (bar chart)
        pot_range = self.lineEdit_potential_range.text().rsplit(',')
        if pot_range == ['']:
            self.pot_range = []
        else:
            self.pot_range = [list(map(float,each.rsplit('-'))) for each in pot_range]
            pot_range_ = []
            for each in self.pot_range:
                if len(each)==1:
                    pot_range_.append(each*2)
                elif len(each)==2:
                    pot_range_.append(each)
            self.pot_range = pot_range_

        temp_pot = np.array(self.pot_range).flatten()
        pot_range_min_max = [min(temp_pot), max(temp_pot)]
        data_range = []
        for scan in self.scans:
            data_range_ = self._get_data_range_auto(scan = scan,
                                                    ref_pot_low = pot_range_min_max[0],
                                                    ref_pot_high = pot_range_min_max[1],
                                                    cycle = self.spinBox_cycle.value(),
                                                    sweep_direction = self.comboBox_scan_dir.currentText(),
                                                    threshold = 10)
            data_range.append(sorted(data_range_))
        self.data_range = data_range


    #given the scan number (scan), automatically locate the data point range corresponding to potential range from
    #ref_pot_low to ref_pot_high at scan cycle of (cycle) with potential sweep direction defined by (sweep_direction)
    #threshold is internally used to seperate potential groups
    #the result could be empty sometimes, especially when the given potential range is larger than the real potential limits in the data
    def _get_data_range_auto(self, scan, ref_pot_low, ref_pot_high,  cycle = -1, sweep_direction = 'down', threshold = 10):
        pot = self.data_to_plot[scan]['potential']
        def _locate_unique_index(ref_pot):
            idxs = sorted(list(np.argpartition(abs(pot-ref_pot), 18)[0:18]))
            sep_index = [0]
            for i, each in enumerate(idxs):
                if i>0 and each-idxs[i-1]>threshold:
                    sep_index.append(i)
            sep_index.append(len(idxs))
            group_idx = {}
            for i in range(len(sep_index)-1):
                group_idx[f'group {i}'] = idxs[sep_index[i]:sep_index[i+1]]
            group_idx_single_rep = []
            for each in group_idx:
                group_idx_single_rep.append(group_idx[each][int(len(group_idx[each])/2.)])
            # print(group_idx_single_rep)
            return group_idx_single_rep
        max_pot_idx = _locate_unique_index(max(pot))
        min_pot_idx = _locate_unique_index(min(pot))
        #note the len of each idx must be >2 for at least one case, not work if both have only one item.
        #let's add one item to either max or min idx to make the following logic work
        if len(max_pot_idx)==1 and len(min_pot_idx)==1:
            if max_pot_idx[0]>min_pot_idx[0]:
                max_pot_idx = [0] + max_pot_idx
                min_pot_idx = min_pot_idx + [len(pot)-1]
            else:
                min_pot_idx = [0] + min_pot_idx
                max_pot_idx = max_pot_idx + [len(pot)-1]

        if ref_pot_high>max(pot):
            target_pot_high_idx = max_pot_idx
        else:
            target_pot_high_idx = _locate_unique_index(ref_pot_high)
        if ref_pot_low<min(pot):
            target_pot_low_idx = min_pot_idx
        else:
            target_pot_low_idx = _locate_unique_index(ref_pot_low)

        #you may have cases where the located idx is too close to the adjacent max(min)_idx
        #in these cases set the idx to the associated max or min idx
        for i, each_ in enumerate(target_pot_high_idx):
            indx = np.argmin(abs(np.array(max_pot_idx)-each_))
            if abs(each_-max_pot_idx[indx])<=threshold:
                target_pot_high_idx[i] = max_pot_idx[indx]
        for i, each_ in enumerate(target_pot_low_idx):
            indx = np.argmin(abs(np.array(min_pot_idx)-each_))
            if abs(each_-min_pot_idx[indx])<=threshold:
                target_pot_low_idx[i] = min_pot_idx[indx]
        cases_map_low_idx = {}
        cases_map_high_idx = {}
        # print(ref_pot_high, max(pot), target_pot_high_idx)
        for each in target_pot_high_idx:
            max_idx = np.argmin(abs(np.array(max_pot_idx) - each))
            min_idx = np.argmin(abs(np.array(min_pot_idx) - each))
            if max_pot_idx[max_idx]>=each>=min_pot_idx[min_idx]:
                cases_map_high_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
            elif min_pot_idx[min_idx]>=each>=max_pot_idx[max_idx]:
                cases_map_high_idx[(max_pot_idx[max_idx],min_pot_idx[min_idx])] = each
            if each in max_pot_idx:#need to make up the other side if each is right on one max idx
                if min_pot_idx[min_idx]>each and (min_idx-1)>=0:
                    cases_map_high_idx[(min_pot_idx[min_idx-1],each)] = each
                elif min_pot_idx[min_idx]<each and (min_idx+1)<len(min_pot_idx):
                    cases_map_high_idx[(each,min_pot_idx[min_idx+1])] = each

            # if each>=min_pot_idx[min_idx] and each<=max_pot_idx[max_idx]:
                # cases_map_high_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
        for each in target_pot_low_idx:
            max_idx = np.argmin(abs(np.array(max_pot_idx) - each))
            min_idx = np.argmin(abs(np.array(min_pot_idx) - each))
            if max_pot_idx[max_idx]>=each>=min_pot_idx[min_idx]:
                cases_map_low_idx[(min_pot_idx[min_idx],max_pot_idx[max_idx])] = each
            elif min_pot_idx[min_idx]>=each>=max_pot_idx[max_idx]:
                cases_map_low_idx[(max_pot_idx[max_idx],min_pot_idx[min_idx])] = each
            if each in min_pot_idx:#need to make up the other side
                if max_pot_idx[max_idx]>each and (max_idx-1)>=0:
                    cases_map_low_idx[(max_pot_idx[max_idx-1],each)] = each
                elif max_pot_idx[max_idx]<each and (max_idx+1)<len(max_pot_idx):
                    cases_map_low_idx[(each,max_pot_idx[max_idx+1])] = each
        print(scan)
        print(min_pot_idx, max_pot_idx)
        print(cases_map_high_idx)
        print(cases_map_low_idx)
        if ref_pot_low == ref_pot_high:
            cases_map_high_idx.update(cases_map_low_idx)
            cases_map_low_idx = cases_map_high_idx

        unique_keys = [each for each in cases_map_low_idx if each in cases_map_high_idx]
        final_group = {'up':[],'down':[]}
        for each in unique_keys:
            temp = [cases_map_low_idx[each], cases_map_high_idx[each]]
            if temp[0]>temp[1]:
                final_group['down'].append(temp)
            elif temp[0]<temp[1]:
                final_group['up'].append(temp)
            else:
                final_group['down'].append(temp)
                final_group['up'].append(temp)

        # print(scan, final_group)
        if ref_pot_low == ref_pot_high:
            #now lets get the scan direction right, at this moment it is a inclusive list of possibility.
            idx_unique = [each_[0] for each_ in final_group['up']]
            #now merge this in the max and min idx container
            full_list = sorted(idx_unique + min_pot_idx + max_pot_idx)
            up_list = []
            down_list = []
            for i, each_ in enumerate(idx_unique):
                idx_in_full_list = full_list.index(each_)
                if idx_in_full_list==0:
                    if each_ in min_pot_idx:
                        up_list.append([each_]*2)
                    elif each_ in max_pot_idx:
                        down_list.append([each_]*2)
                elif idx_in_full_list==len(full_list)-1:
                    if idx_in_full_list==0:
                        if each_ in min_pot_idx:
                            down_list.append([each_]*2)
                        elif each_ in max_pot_idx:
                            up_list.append([each_]*2)
                else:
                    if full_list[idx_in_full_list-1] in min_pot_idx:
                        up_list.append([each_]*2)
                    else:
                        down_list.append([each_]*2)
            final_group = {'up':up_list, 'down': down_list}

        pot_ranges = final_group[sweep_direction]
        assert len(pot_ranges)>0, print('empty list in pot_ranges')
        try:
            return pot_ranges[cycle]
        except:
            print(f'pot_ranges for scan {scan} could not be queried with the cycle index {cycle}, use last cycle instead')
            return pot_ranges[-1]
        '''
        index_list = None
        if type(cycle)==int:
            try:
                pot_range = pot_ranges[cycle]
            except:
                print(f'pot_ranges could not be queried with the cycle index {cycle}, use last cycle instead')
                pot_range = pot_ranges[-1]
            index_list = pot_range
            if ref_pot_low == ref_pot_high:#in this case, we extract only absolute value. And note pot_range[0] = pot_range[1]
                total_change = y_values[pot_range[0]]
                std_val_norm = std_val
            else:
                total_change = abs((y_values[pot_range[1]] - y_values[pot_range[0]])/(pot[pot_range[1]]-pot[pot_range[0]]))
                std_val_norm = std_val/(pot[pot_range[1]]-pot[pot_range[0]])
        else:#here we do average over all cases
            total_change = 0
            for i in range(len(pot_ranges)):
                pot_range = pot_ranges[i]
                if ref_pot_low == ref_pot_high:
                    total_change += y_values[pot_range[0]]
                else:
                    total_change += abs((y_values[pot_range[1]] - y_values[pot_range[0]])/(pot[pot_range[1]]-pot[pot_range[0]]))
            total_change = total_change/len(pot_ranges)
            std_val_norm = std_val/(pot[pot_ranges[0][1]]-pot[pot_ranges[0][0]]) if ref_pot_low != ref_pot_high else std_val
            index_list = pot_ranges[0]#only take the first range
        return total_change, std_val_norm, index_list
        '''

    #cal std for channel centering at point_index with left and right boundary defined such that the span potential_range is reached
    def _cal_std_at_pt(self, scan, channel, point_index, potential_range = 0.02):
        data = self.data_to_plot[scan][channel]
        pot = self.data_to_plot[scan]['potential']
        index_left = point_index
        index_right = point_index
        while True:
            if index_left == 0:
                break
            else:
                if abs(pot[index_left]-pot[point_index])>=potential_range:
                    break
                else:
                    index_left = index_left - 1

        while True:
            if index_right == len(pot)-1:
                break
            else:
                if abs(pot[index_right]-pot[point_index])>=potential_range:
                    break
                else:
                    index_right = index_right + 1

        return np.std(data[index_left:index_right])

    def _cal_structural_change_rate(self,scan, channel,y_values, std_val, data_range, pot_range, marker_index_container, cal_std = True):
        assert 'potential' in list(self.data_to_plot[scan].keys())
        if data_range[0]==data_range[1]:
            index_left = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1] - pot_range[0])) + data_range[0]
            index_right = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]+1] - pot_range[1])) + data_range[0]
        else:            
            index_left = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]] - pot_range[0])) + data_range[0]
            index_right = np.argmin(np.abs(self.data_to_plot[scan]['potential'][data_range[0]:data_range[1]] - pot_range[1])) + data_range[0]
        marker_index_container.append(index_left)
        marker_index_container.append(index_right)
        pot_offset = abs(self.data_to_plot[scan]['potential'][index_left]-self.data_to_plot[scan]['potential'][index_right])
        if cal_std:#use error propogation rule here
            # std_val = max([self._cal_std_at_pt(scan, channel, index_left),self._cal_std_at_pt(scan, channel, index_right)])
            std_val = (self._cal_std_at_pt(scan, channel, index_left)**2+self._cal_std_at_pt(scan, channel, index_right)**2)**0.5
        if pot_offset==0:
            if channel == 'current':
                self.data_summary[scan][channel].append(y_values[index_left])
            else:
                self.data_summary[scan][channel].append(y_values[index_left] + self.data_to_plot[scan][channel+'_max'])
            self.data_summary[scan][channel].append(std_val/(2**0.5))
        else:#calculate the slope here
            self.data_summary[scan][channel].append((y_values[index_right] - y_values[index_left])/pot_offset)
            self.data_summary[scan][channel].append(std_val/pot_offset)
        return marker_index_container            

    def _extract_y_values(self, scan, channel):
        y = self.data_to_plot[scan][channel]
        if channel == 'current':#current --> current density
            y = y*8
        #apply offset, very useful if you want to slightly tweak the channel values for setting a common 0 reference point
        #the offset can be specified from GUI
        y_offset = 0
        if hasattr(self, f'{channel}_offset_{scan}'):
            y_offset = getattr(self, f'{channel}_offset_{scan}')
        #apply the offset to y channel
        y = np.array([each + y_offset for each in y])
        y_smooth_temp = signal.savgol_filter(y,41,2)
        #std is calculated this way for estimation of error bar values
        std_val = np.sum(np.abs(y_smooth_temp - y))/len(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]])
        # std_val = (np.sum((y_smooth_temp - y)**2)/len(self.data_to_plot[scan][self.plot_label_x[self.scans.index(scan)]]-1))**0.5
        return y, y_smooth_temp, std_val

    def _cal_pseudcap_charge(self, scan):
        for each_pot_range in self.pot_ranges[scan]:
            try:
                horizontal = self.grain_size_info_all_scans[scan][each_pot_range]['horizontal']
                vertical = self.grain_size_info_all_scans[scan][each_pot_range]['vertical']
                q_skin,q_film = self.estimate_charge_from_skin_layer_thickness_philippe_algorithm({"horizontal":horizontal,"vertical":vertical})
                if scan not in self.charge_info:
                    self.charge_info[scan] = {}
                    self.charge_info[scan][each_pot_range] = {'skin_charge':q_skin,'film_charge':q_film,'total_charge':0}
                else:
                    self.charge_info[scan][each_pot_range]['skin_charge'] = q_skin
                    self.charge_info[scan][each_pot_range]['film_charge'] = q_film
            except:
                print('Fail to cal charge info. Check! All charges are set to zero!')
                self.charge_info[scan] = {}
                self.charge_info[scan][each_pot_range] = {'skin_charge':0,'film_charge':0,'total_charge':0}

    def _find_reference_at_potential(self, target_pot, pot_array, y_array):
        offset_abs = np.abs(np.array(pot_array)-target_pot)
        #only select three smaller points
        idx = np.argpartition(offset_abs, 4)[0:4]
        y_sub = [y_array[each] for each in idx]
        index_found = idx[y_sub.index(max(y_sub))]
        return y_array[index_found]

    def prepare_data_to_plot_xrv(self,plot_label_list, scan_number):
        if scan_number in self.image_range_info:
            l , r = self.image_range_info[scan_number]
        else:
            l, r = 0, 100000000
        if hasattr(self,'data_to_plot'):
            self.data_to_plot[scan_number] = {}
        else:
            self.data_to_plot = {}
            self.data_to_plot[scan_number] = {}
        if self.checkBox_mask.isChecked():
            condition = (self.data['mask_cv_xrd'] == True)&(self.data['scan_no'] == scan_number)
        else:
            condition = self.data['scan_no'] == scan_number
        #RHE potential, potential always in the y dataset
        self.data_to_plot[scan_number]['potential'] = self.potential_offset + 0.205+np.array(self.data[condition]['potential'])[l:r]+0.059*np.array(self.data[self.data['scan_no'] == scan_number]['phs'])[0]
        for each in plot_label_list:
            if each == 'potential':
                pass
            elif each=='current':#RHE potential
                self.data_to_plot[scan_number][each] = -np.array(self.data[condition][each])[l:r]
            else:
                if each in ['peak_intensity','peak_intensity_error','strain_ip','strain_oop','grain_size_ip','grain_size_oop']:
                    temp_data = np.array(self.data[condition][each])[l:r]
                    y_smooth_temp = signal.savgol_filter(temp_data,41,2)
                    # if scan_number==24000:
                        # y_smooth_temp = temp_data
                    # self.data_to_plot[scan_number][each] = list(temp_data-max(temp_data))
                    #if time_scan, then target_pots are actually image_no
                    target_pots = [float(each) for each in self.lineEdit_reference_potential.text().rstrip().rsplit(',')]
                    if len(target_pots)==1:
                        target_pots = target_pots*2
                    target_pot = target_pots[int('size' in each)]
                    channel = ['potential','image_no'][int(self.checkBox_time_scan.isChecked())]
                    temp_max = self._find_reference_at_potential(target_pot = target_pot, pot_array = self.data_to_plot[scan_number][channel], y_array = y_smooth_temp)
                    if self.checkBox_max.isChecked():
                        self.data_to_plot[scan_number][each] = list(temp_data-temp_max)
                        self.data_to_plot[scan_number][each+'_max'] = temp_max
                    else:
                        self.data_to_plot[scan_number][each] = list(temp_data)
                        self.data_to_plot[scan_number][each+'_max'] = 0
                else:
                    self.data_to_plot[scan_number][each] = list(self.data[condition][each])[l:r]

    def append_scans_xrv(self):
        text = self.scan_numbers_append.text()
        if text!='':
            scans = list(set([int(each) for each in text.rstrip().rsplit(',')]))
        else:
            return
        scans.sort()
        assert (self.lineEdit_x.text()!='' and self.lineEdit_y.text()!=''), 'No channels for plotting have been selected!'
        plot_labels = self.lineEdit_x.text() + ',' + self.lineEdit_y.text()
        plot_labels = plot_labels.rstrip().rsplit(',')
        for scan in scans:
            self.prepare_data_to_plot_xrv(plot_labels,scan)
            print('Prepare data for scan {} now!'.format(scan))
        self.scans = scans
        self._init_meta_data_for_scans(scans)
        self.phs = [self.phs_all[self.scans_all.index(each)] for each in scans]

        #self.plot_label_x = self.lineEdit_x.text()
        self.plot_label_x = self.lineEdit_x.text().rstrip().rsplit(',')
        if len(self.plot_label_x)==1:
            self.plot_label_x = self.plot_label_x*len(scans)
        else:
            if len(self.plot_label_x)<len(scans):
                self.plot_label_x = self.plot_label_x + [self.plot_label_x[-1]]*(len(scans) - len(self.plot_label_x))
        self.plot_labels_y = self.lineEdit_y.text().rstrip().rsplit(',')
        self.comboBox_scans.clear()
        self.comboBox_scans.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_2.clear()
        self.comboBox_scans_2.addItems([str(each) for each in sorted(scans)])
        self.comboBox_scans_3.clear()
        self.comboBox_scans_3.addItems([str(each) for each in sorted(scans)])