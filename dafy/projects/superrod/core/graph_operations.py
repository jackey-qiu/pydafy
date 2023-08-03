import copy
import numpy as np
import logging
import pyqtgraph as pg
from PyQt5 import QtWidgets
# from dafy.core.EnginePool.diffev import ErrorBarError
from dafy.core.util.UtilityFunctions import q_correction_for_one_Bragg_peak, q_correction_for_one_rod 

class GraphOperations(object):
    def generate_gaf_plot(self):
        self.simulate_model(compile = True)
        data_series = sum(self.normalized_datasets,[])
        self.widget_gaf.create_plots(serie_data = data_series, plot_type = self.comboBox_plot_channel.currentText(),relative = self.checkBox_GAF_relatively.isChecked())

    def plot_bar_chart(self, data):
        self.sensitivity_data = list(data)
        par_names = [str(i+1) for i in range(len(data))]
        self.widget_sensitivity_bar.clear()
        bg1 = pg.BarGraphItem(x=list(range(1,len(data)+1)), height=data, width=0.3, brush='g')
        ax_bar = self.widget_sensitivity_bar.addPlot(clear = True)
        ax_bar.addItem(bg1)
        #[list(zip(list(range(1,len(percents)+1)),[str(each) for each in range(1,len(percents)+1)]))]
        ax_bar.getAxis('bottom').setTicks([list(zip(list(range(1,len(data)+1)),par_names))])
        ax_bar.getAxis('bottom').setLabel('parameters')
        ax_bar.getAxis('left').setLabel('Normalized sensitivity')
        ax_bar.setYRange(0, 1, padding = 0.1)
        # ax_bar.autoRange()

    def show_plots_on_next_screen(self):
        """
        show plots on next screen, if one screen is not enough to fill all plots
        """
        if not hasattr(self,"num_screens_plot"):
            return

        if self.num_screens_plot>1:
            if self.current_index_plot_screen<(self.num_screens_plot-1):
                self.update_plot_dimension(self.current_index_plot_screen+1)
                self.update_plot_data_view()
            else:
                pass
        else:
            pass

    def show_plots_on_previous_screen(self):
        """
        show plots on previous screen
        """

        if not hasattr(self,"num_screens_plot"):
            return

        if self.num_screens_plot>1:
            if self.current_index_plot_screen>0:
                self.update_plot_dimension(self.current_index_plot_screen-1)
                self.update_plot_data_view()
            else:
                pass
        else:
            pass

    #do this after model is loaded, so that you know len(data)
    def update_plot_dimension(self, current_index_plot_screen = 0):
        """Setting the layout of data profiles"""
        def _get_index(index_in_use):
            use_or_not = []
            for i in range(len(self.model.data)):
                if self.tableWidget_data.cellWidget(i,2).isChecked():
                    use_or_not.append(True)
                else:
                    use_or_not.append(False)
            index_in_sequence = index_in_use
            total = -1
            for i, each in enumerate(use_or_not):
                if total<index_in_use:
                    if not each:
                        index_in_sequence += 1
                    else:
                        total += 1
            return index_in_sequence
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        height, width = sizeObject.height()*25.4/self.dpi,sizeObject.width()*25.4/self.dpi
        #maximum number of plots allowd to be fit in one screen
        #assuming the minimum plot panel has a size of (w:50mm * h:40mm)
        plot_cols, plot_rows = int(width/50), int(height/40)
        self.max_num_plots_per_screen = plot_cols*plot_rows

        self.widget_data.clear()
        self.widget_data.ci.currentRow = 0
        self.widget_data.ci.currentCol = 0

        self.data_profiles = []
        #only consider the data in use
        total_datasets = len([1 for i in range(len(self.model.data)) if self.tableWidget_data.cellWidget(i,2).isChecked()])
        #total_datasets = len([each for each in self.model.data if each.use])

        if total_datasets<self.max_num_plots_per_screen:
            self.num_screens_plot = 1
        else:
            self.num_screens_plot = int(total_datasets/self.max_num_plots_per_screen)+[0,1][int((total_datasets%self.max_num_plots_per_screen)>0)]
        self.current_index_plot_screen = current_index_plot_screen
        if self.num_screens_plot>1:#more than one screen
            if self.current_index_plot_screen<(self.num_screens_plot-1):
                columns = plot_cols#should be occupied in maximum
                num_plots_on_current_screen = self.max_num_plots_per_screen
            else:#last screen
                num_plots_ = total_datasets%self.max_num_plots_per_screen
                if num_plots_ == 0:
                    columns = plot_cols
                    num_plots_on_current_screen = self.max_num_plots_per_screen
                else:
                    num_plots_on_current_screen = num_plots_
                    if num_plots_>10:
                        columns = 4
                    else:
                        columns = 2
        elif self.num_screens_plot==1:#only one screen
            if total_datasets==self.max_num_plots_per_screen:
                num_plots_on_current_screen = self.max_num_plots_per_screen
                columns = plot_cols
            else:
                num_plots_on_current_screen = total_datasets
                if total_datasets>10:
                    columns = 4
                else:
                    columns = 2

        #current list of ax handle
        self.num_plots_on_current_screen = num_plots_on_current_screen
        offset = self.current_index_plot_screen*self.max_num_plots_per_screen
        for i in range(num_plots_on_current_screen):
            if 1:
                hk_label = '{}{}_{}'.format(str(int(self.model.data[_get_index(i+offset)].extra_data['h'][0])),str(int(self.model.data[_get_index(i+offset)].extra_data['k'][0])),str(self.model.data[_get_index(i+offset)].extra_data['Y'][0]))
                if (i%columns)==0 and (i!=0):
                    self.widget_data.nextRow()
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))
                else:
                    self.data_profiles.append(self.widget_data.addPlot(title=hk_label))

    def setup_plot(self):
        self.fom_evolution_profile = self.widget_fom.addPlot()
        self.par_profile = self.widget_pars.addPlot()
        self.fom_scan_profile = self.widget_fom_scan.addPlot()
        self.fom_scan_profile.getAxis('left').setLabel('Electron denstiy (per water)')
        self.fom_scan_profile.getAxis('bottom').setLabel('Height (Å)')

    def update_plot_data_view(self):
        """update views of all figures if script is compiled, while only plot data profiles if otherwise"""
        def _get_index(index_in_use):
            index_in_sequence = index_in_use
            total = -1
            for i, each in enumerate(self.model.data):
                if total<index_in_use:
                    if not each.use:
                        index_in_sequence += 1
                    else:
                        total += 1
            return index_in_sequence

        if self.model.compiled:
            self.update_data_check_attr()
            self.update_plot_data_view_upon_simulation()
            self.update_electron_density_profile()
        else:
            offset = self.max_num_plots_per_screen*self.current_index_plot_screen
            for i in range(self.num_plots_on_current_screen):
                fmt = self.tableWidget_data.item(i+offset,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                self.data_profiles[i].plot(self.model.data[_get_index(i+offset)].x, self.model.data[_get_index(i+offset)].y,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2], clear = True)
            [each.setLogMode(x=False,y=self.tableWidget_data.cellWidget(_get_index(self.data_profiles.index(each)),1).isChecked()) for each in self.data_profiles]
            [each.autoRange() for each in self.data_profiles]

    def update_electron_density_profile(self):
        if self.lineEdit_z_min.text()!='':
            z_min = float(self.lineEdit_z_min.text())
        else:
            z_min = -20
        if self.lineEdit_z_max.text()!='':
            z_max = float(self.lineEdit_z_max.text())
        else:
            z_max = 100
        raxs_A_list, raxs_P_list = [], []
        #num_raxs = len(self.model.data)-1
        #items for raxs dates have value >=100 in the data_sequence attribute
        num_raxs = sum(np.array(self.model.data.data_sequence)>=100)
        if hasattr(self.model.script_module, "rgh_raxs"):
            for i in range(num_raxs):
                raxs_A_list.append(eval("self.model.script_module.rgh_raxs.getA_{}()".format(i+1)))
                raxs_P_list.append(eval("self.model.script_module.rgh_raxs.getP_{}()".format(i+1)))
        else:
            raxs_A_list.append(0)
            raxs_P_list.append(0)
        # raxs_A_list = raxs_A_list[0:2]
        # raxs_P_list = raxs_P_list[0:2]
        HKL_raxs_list = [[],[],[]]
        for each in self.model.data:
            if each.x[0]>=100:
                HKL_raxs_list[0].append(each.extra_data['h'][0])
                HKL_raxs_list[1].append(each.extra_data['k'][0])
                HKL_raxs_list[2].append(each.extra_data['Y'][0])
        # HKL_raxs_list = [HKL_raxs_list[0][0:2],HKL_raxs_list[1][0:2],HKL_raxs_list[2][0:2]]
        if hasattr(self.model.script_module, "RAXS_EL"):
            raxs_el = getattr(self.model.script_module, "RAXS_EL")
        else:
            raxs_el = None
        try:
            if self.run_fit.running or self.run_batch.running:
                #if model is running, disable showing e profile
                pass
            else:
                self.fom_scan_profile.addLegend(offset = (-10,20))
                label,edf = self.model.script_module.sample.plot_electron_density_superrod(z_min=z_min, z_max=z_max,N_layered_water=500,resolution =1000, raxs_el = raxs_el, use_sym = self.checkBox_symmetry.isChecked())
                #here only plot the total electron density of domain specified by domain_tag
                domain_tag = int(self.spinBox_domain.text())
                self.fom_scan_profile.plot(edf[domain_tag][0],edf[domain_tag][1],pen = {'color': "w", 'width': 1},clear = True)
                self.fom_scan_profile.plot(edf[domain_tag][0],edf[domain_tag][1],fillLevel=0, brush = (0,200,0,100),clear = False, name = 'Total ED')
                if len(edf[domain_tag])==4:
                    self.fom_scan_profile.plot(edf[domain_tag][0],edf[domain_tag][2],fillLevel=0, brush = (200,0,0,80),clear = False)
                    self.fom_scan_profile.plot(edf[domain_tag][0],edf[domain_tag][3],fillLevel=0, brush = (0,0,250,80),clear = False, name = 'Water Layer')
                if hasattr(self.model.script_module, "rgh_raxs"):
                    # print(HKL_raxs_list)
                    # print(raxs_P_list) 
                    # print(raxs_A_list)
                    # z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.raxr_el,resolution=1000,water_scaling=0.33)
                    z_plot,eden_plot,_=self.model.script_module.sample.fourier_synthesis(np.array(HKL_raxs_list),np.array(raxs_P_list).transpose(),np.array(raxs_A_list).transpose(),z_min=z_min,z_max=z_max,resonant_el=self.model.script_module.RAXS_EL,resolution=1000,water_scaling=0.33)
                    self.fom_scan_profile.plot(z_plot,eden_plot,fillLevel=0, brush = (200,0,200,100),clear = False, name = 'ED based on Fourier Synthesis')
                self.fom_scan_profile.autoRange()
        except:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to draw e density profile!')
            logging.getLogger().exception('Fatal error encountered during drawing e density profile!')
            self.tabWidget_data.setCurrentIndex(6)

    def update_plot_data_view_upon_simulation(self, q_correction = False):
        self.normalized_datasets = []
        def _get_index(index_in_use):
            index_in_sequence = index_in_use
            total = -1
            for i, each in enumerate(self.model.data):
                if total<index_in_use:
                    if not each.use:
                        index_in_sequence += 1
                    else:
                        total += 1
            return index_in_sequence

        offset = self.max_num_plots_per_screen*self.current_index_plot_screen
        for i in range(self.num_plots_on_current_screen):
            if 1:
                #plot ideal structure factor
                f_ideal = 1
                scale_factor = 1
                rod_factor = 1
                try:
                    specular_condition = int(round(self.model.data[_get_index(i+offset)].extra_data['h'][0],0))==0 and int(round(self.model.data[_get_index(i+offset)].extra_data['k'][0],0))==0
                    scale_factor = [self.model.script_module.rgh.scale_nonspecular_rods,self.model.script_module.rgh.scale_specular_rod][int(specular_condition)]
                    h_, k_ = int(round(self.model.data[_get_index(i+offset)].extra_data['h'][0],0)),int(round(self.model.data[_get_index(i+offset)].extra_data['k'][0],0))
                    extra_scale_factor = 'scale_factor_{}{}L'.format(h_,k_)
                    if hasattr(self.model.script_module.rgh,extra_scale_factor):
                        rod_factor = getattr(self.model.script_module.rgh, extra_scale_factor)
                    else:
                        rod_factor = 1
                    if self.checkBox_norm.isChecked():
                        f_ideal = self.f_ideal[_get_index(i+offset)]*scale_factor*rod_factor
                    else:
                        self.data_profiles[i].plot(self.model.data[_get_index(i+offset)].x, self.f_ideal[_get_index(i+offset)]*scale_factor*rod_factor,pen = {'color': "w", 'width': 1},clear = True)
                except:
                    pass

                fmt = self.tableWidget_data.item(i+offset,4).text()
                fmt_symbol = list(fmt.rstrip().rsplit(';')[0].rsplit(':')[1])
                line_symbol = list(fmt.rstrip().rsplit(';')[1].rsplit(':')[1])
                if not q_correction:
                    self.data_profiles[i].plot(self.model.data[_get_index(i+offset)].x, self.model.data[_get_index(i+offset)].y/f_ideal,pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2],clear = False)
                else:
                    if self.model.data[_get_index(i+offset)].name == self.comboBox_dataset2.currentText():
                        L_q_correction = self.model.data_original[_get_index(i+offset)].x
                        data_q_correction = self.model.data_original[_get_index(i+offset)].y
                        unitcell = self.model.script_module.unitcell
                        cell = [unitcell.a, unitcell.b, unitcell.c, unitcell.alpha, unitcell.beta, unitcell.gamma]
                        lam = self.model.script_module.wal
                        scale = float(self.lineEdit_scale.text())
                        current_L = int(self.lineEdit_L.text())
                        if not self.fit_q_correction:
                            LL, new_data= q_correction_for_one_Bragg_peak(L = L_q_correction,data = data_q_correction, cell = cell, lam = lam, L_bragg = current_L, scale=scale,delta=0,c_off=0, R_tth = float(self.lineEdit_r_tth.text()))
                        else:
                            LL, new_data= q_correction_for_one_rod(L = L_q_correction, data = data_q_correction, cell = cell, lam = lam, correction_factor_dict = self.q_correction_factor, R_tth = float(self.lineEdit_r_tth.text()))
                            if self.apply_q_correction:
                                self.apply_q_correction_results(_get_index(i+offset), LL)
                        # print(data)
                        # print(scale_factor*rod_factor)
                        #recalculate f_ideal
                        #self.model.script_module.unitcell.set_c(new_c)
                        #self.calc_f_ideal()
                        #f_ideal = self.f_ideal[_get_index(i+offset)]*scale_factor*rod_factor
                        self.data_profiles[i].plot(LL, new_data, pen = None,  symbolBrush=fmt_symbol[1], symbolSize=int(fmt_symbol[0]),symbolPen=fmt_symbol[2],clear = True)
                if self.tableWidget_data.cellWidget(_get_index(i+offset),3).isChecked():
                    #create error bar data, graphiclayout widget doesn't have a handy api to plot lines along with error bars in a log scale
                    #disable this while the model is running
                    if not self.run_fit.solver.optimizer.running:
                        '''#this solution does not work in a log scale
                        x, y, error = self.model.data[_get_index(i+offset)].x, self.model.data[_get_index(i+offset)].y, self.model.data[_get_index(i+offset)].error/2
                        err = pg.ErrorBarItem(x=x, y=y, top=error, bottom=error)
                        self.data_profiles[i].addItem(err)
                        '''
                        x = np.append(self.model.data[_get_index(i+offset)].x[:,np.newaxis],self.model.data[_get_index(i+offset)].x[:,np.newaxis],axis=1)
                        y_d = self.model.data[_get_index(i+offset)].y[:,np.newaxis] - self.model.data[_get_index(i+offset)].error[:,np.newaxis]/2
                        y_u = self.model.data[_get_index(i+offset)].y[:,np.newaxis] + self.model.data[_get_index(i+offset)].error[:,np.newaxis]/2
                        y = np.append(y_d,y_u,axis = 1)
                        for ii in range(len(y)):
                            self.data_profiles[i].plot(x=x[ii],y=y[ii],pen={'color':'w', 'width':1},clear = False)
                

                #plot simulated results
                if not q_correction:
                    if self.tableWidget_data.cellWidget(_get_index(i+offset),2).isChecked():
                        self.data_profiles[i].plot(self.model.data[_get_index(i+offset)].x, self.model.data[_get_index(i+offset)].y_sim/f_ideal,pen={'color': line_symbol[1], 'width': int(line_symbol[0])},  clear = False)
                        # self.normalized_datasets.append(list(np.log10(self.model.data[_get_index(i+offset)].y_sim/f_ideal)))
                        self.normalized_datasets.append(list(self.model.data[_get_index(i+offset)].y_sim/f_ideal))
                    else:
                        pass
        [each.setLogMode(x=False,y=self.tableWidget_data.cellWidget(_get_index(self.data_profiles.index(each)+offset),1).isChecked()) for each in self.data_profiles]
        [each.autoRange() for each in self.data_profiles]
        fom_log = np.array(self.run_fit.solver.optimizer.fom_log)
        self.fom_evolution_profile.plot(fom_log[:,0],fom_log[:,1],pen={'color': 'r', 'width': 2}, clear = True)
        self.fom_evolution_profile.autoRange()
        
    def update_par_bar_during_fit(self):
        """update bar chart during fit, which tells the current best fit and the searching range of each fit parameter"""
        if self.run_fit.running or self.run_batch.running:
            if self.run_fit.running:
                par_max = self.run_fit.solver.optimizer.par_max
                par_min = self.run_fit.solver.optimizer.par_min
                vec_best = copy.deepcopy(self.run_fit.solver.optimizer.best_vec)
                vec_best = (vec_best-par_min)/(par_max-par_min)
                pop_vec = np.array(copy.deepcopy(self.run_fit.solver.optimizer.pop_vec))
            elif self.run_batch.running:
                par_max = self.run_batch.solver.optimizer.par_max
                par_min = self.run_batch.solver.optimizer.par_min
                vec_best = copy.deepcopy(self.run_batch.solver.optimizer.best_vec)
                vec_best = (vec_best-par_min)/(par_max-par_min)
                pop_vec = np.array(copy.deepcopy(self.run_batch.solver.optimizer.pop_vec))

            trial_vec_min =[]
            trial_vec_max =[]
            for i in range(len(par_max)):
                trial_vec_min.append((np.min(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
                trial_vec_max.append((np.max(pop_vec[:,i])-par_min[i])/(par_max[i]-par_min[i]))
            trial_vec_min = np.array(trial_vec_min)
            trial_vec_max = np.array(trial_vec_max)
            bg = pg.BarGraphItem(x=range(len(vec_best)), y=(trial_vec_max + trial_vec_min)/2, height=(trial_vec_max - trial_vec_min)/2, brush='b', width = 0.8)
            self.par_profile.clear()
            self.par_profile.addItem(bg)
            self.par_profile.plot(vec_best, pen=(0,0,0), symbolBrush=(255,0,0), symbolPen='w')
        else:
            pass

    def calculate_error_bars(self):
        """
        cal the error bar for each fit par after fit is completed
        note the error bar values are only estimated from all intermediate fit reuslts from all fit generations,
        and the error may not accutely represent the statistic errors. If you want to get statistical errors of 
        each fit parameter, you can run a further NLLS fit using the the best fit parameters, which is not implemented in the program.
        """
        try:
            try:
                error_bars = self.run_fit.solver.CalcErrorBars()
            except:
                try:
                    error_bars = self.run_batch.solver.CalcErrorBars()
                except:
                    return
            total_num_par = len(self.model.parameters.data)
            index_list = [i for i in range(total_num_par) if self.model.parameters.data[i][2]]
            for i in range(len(error_bars)):
                self.model.parameters.data[index_list[i]][-2] = error_bars[i]
            self.update_par_upon_load()
        except ErrorBarError as e:
            self.statusbar.clearMessage()
            self.statusbar.showMessage('Failure to calculate error bar!')
            logging.getLogger().exception('Fatal error encountered during error calculation!')
            self.tabWidget_data.setCurrentIndex(4)
            _ = QtWidgets.QMessageBox.question(self, 'Runtime error message', str(e), QtWidgets.QMessageBox.Ok)
