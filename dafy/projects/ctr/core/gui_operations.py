import time, os, logging
import pandas as pd
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QCheckBox, QRadioButton, QFileDialog
from dafy.core.util.PlotSetup import overplot_ctr_temp
from dafy.projects.ctr.widgets.custom_widget import DataEditorDialog
from dafy.core.EnginePool.VisualizationEnginePool import plot_bkg_fit_gui_pyqtgraph
from dafy.projects.ctr.core.graph_operations import setup_image


class GuiOperations(object):
    def __init__(self):
        pass

    def setup_image(self):
        setup_image(self)

    def start_dL_BL_editor_dialog(self):
        dlg = DataEditorDialog(self)
        dlg.exec()

    def switch_roi_adjustment_type(self):
        if self.radioButton_roi_position.isChecked():
            self.radioButton_roi_size.setChecked(True)
        else:
            self.radioButton_roi_position.setChecked(True)

    def change_config_layout(self):
        self.widget_config.init_pars(data_type = self.comboBox_beamline.currentText())
        self.app_ctr.beamline = self.comboBox_beamline.currentText()

    def set_tag_process_type(self):
        self.tag_reprocess = False
        
    def reprocess_previous_frame(self):
        t0 = time.time()
        self.tag_reprocess = True
        frame_number = self.app_ctr.current_frame + 1 +int(self.lineEdit_frame_index_offset.text())
        if (frame_number < 0) or (frame_number > self.app_ctr.current_frame):
            self.tag_reprocess = False
            return
        img = self.app_ctr.img_loader.load_one_frame(frame_number = frame_number)
        img = self.app_ctr.create_mask_new.create_mask_new(img = img, img_q_ver = None,
                                  img_q_par = None, mon = self.app_ctr.img_loader.extract_transm_and_mon(frame_number))
        self.app_ctr.bkg_sub.img = img
        self.get_fit_pars_from_frame_index(int(self.lineEdit_frame_index_offset.text()))
        self.set_fit_pars_from_reference()
        self.reset_peak_center_and_width()
        self.update_image()
        
        selected = self.roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
        self.bkg_intensity = selected.mean()
        self.app_ctr.run_update_one_specific_frame(img, self.bkg_intensity, poly_func = ['Vincent','traditional'][int(self.radioButton_traditional.isChecked())], frame_offset = int(self.lineEdit_frame_index_offset.text()))
        # self.update_plot()
        self.update_ss_factor()
        #plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)

        self.statusbar.clearMessage()
        self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,frame_number+1,self.app_ctr.img_loader.total_frame_number))
        self.progressBar.setValue((self.app_ctr.img_loader.current_frame_number+1)/float(self.app_ctr.img_loader.total_frame_number)*100)
        # self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1)
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass
    
    def set_log_image(self):
        if self.checkBox_use_log_scale.isChecked():
            self.image_log_scale = True
            self.update_image()
        else:
            self.image_log_scale = False
            self.update_image()

    #to fold or unfold the config file editor
    def fold_or_unfold(self):
        text = self.pushButton_fold_or_unfold.text()
        if text == "<":
            self.frame.setVisible(False)
            self.pushButton_fold_or_unfold.setText(">")
        elif text == ">":
            self.frame.setVisible(True)
            self.pushButton_fold_or_unfold.setText("<")

    def change_peak_width(self):
        # self.lineEdit_peak_width.setText(str(self.horizontalSlider.value()))
        self.app_ctr.bkg_sub.peak_width = int(self.spinBox_peak_width.value())
        self.updatePlot()

    def save_data(self):
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        try:
            self.app_ctr.save_data_file(data_file)
            self.statusbar.showMessage('Data file is saved as {}!'.format(data_file))
        except:
            self.statusbar.showMessage('Failure to save data file!')
            logging.getLogger().exception('Fatal to save datafile:')

    def remove_data_point(self):
        self.app_ctr.data['mask_ctr'][-1]=False
        self.statusbar.showMessage('Current data point is masked!')
        self.updatePlot2()

    def select_source_for_plot_p3(self):
        self.app_ctr.p3_data_source = self.comboBox_p3.currentText()
        self.updatePlot()

    def select_source_for_plot_p4(self):
        self.app_ctr.p4_data_source = self.comboBox_p4.currentText()
        self.updatePlot()

    def update_poly_order(self, init_step = False):
        ord_total = 0
        i=1
        for each in self.groupBox_2.findChildren(QCheckBox):
            ord_total += int(bool(each.checkState()))*int(each.text())
            i+=i
        self.app_ctr.bkg_sub.update_integration_order(ord_total)
        #print(self.app_ctr.bkg_sub.ord_cus_s)
        
        if not init_step:
            self.updatePlot()

    def update_cost_func(self, init_step = False):
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            if each.isChecked():
                self.app_ctr.bkg_sub.update_integration_function(each.text())
                break
        try:
            self.updatePlot()
        except:
            pass

    def update_ss_factor(self, init_step = False):
        self.app_ctr.bkg_sub.update_ss_factor(self.doubleSpinBox_ss_factor.value())
        #print(self.app_ctr.bkg_sub.ss_factor)
        try:
            self.updatePlot()
        except:
            pass

    def _check_roi_boundary(self,pos,size):
        ver,hor = self.app_ctr.cen_clip
        pos_bound_x = hor*2
        pos_bound_y = ver*2
        pos_return = []
        size_return = []
        if pos[0]<0:
            pos_return.append(0)
        elif pos[0]>pos_bound_x:
            pos_return.append(pos_bound_x-10)
        else:
            pos_return.append(pos[0])

        if pos[1]<0:
            pos_return.append(0)
        elif pos[1]>pos_bound_y:
            pos_return.append(pos_bound_y-10)
        else:
            pos_return.append(pos[1]) 

        if size[0]<1:
            size_return.append(1)
        elif size[0]+pos_return[0]>pos_bound_x:
            size_return.append(pos_bound_x-pos_return[0])
        else:
            size_return.append(size[0])

        if size[1]<1:
            size_return.append(1)
        elif size[1]+pos_return[1]>pos_bound_y:
            size_return.append(pos_bound_y-pos_return[1])
        else:
            size_return.append(size[1])
        return pos_return,size_return

    def load_ref_data(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Data Files (*.xlsx);;text Files (*.csv)", options=options)
        self.lineEdit_ref_data_path.setText(fileName)
        #self.lineEdit_data_file.setText(fileName)
        if fileName != "":
            try:
                self.ref_data = pd.read_excel(fileName)
            except:
                print("Failure to load ref data file!!")

    def get_fit_pars_from_reference(self, match_scan = False):
        if type(self.ref_data) != pd.DataFrame:
            self.ref_fit_pars_current_point = {}
            return
        current_H, current_K, current_L = self.app_ctr.img_loader.hkl
        current_scan = self.app_ctr.img_loader.scan_number
        current_H = int(round(current_H,0))
        current_K = int(round(current_K,0))
        condition = (self.ref_data["H"] == current_H) & (self.ref_data["K"] == current_K)
        if hasattr(self,'checkBox_match_scan'):
            match_scan = self.checkBox_match_scan.isChecked()
        if match_scan:
            condition = (self.ref_data["H"] == current_H) & (self.ref_data["K"] == current_K) & (self.ref_data["scan_no"] == current_scan)
        data_sub = self.ref_data[condition]
        if len(data_sub)!=0:
            which_row = (data_sub['L']-current_L).abs().idxmin()
            f = lambda obj_, str_,row_:obj_[str_][row_]
            for each in ["H", "K", "roi_x", "roi_y", "roi_w", "roi_h", "ss_factor", "peak_width","peak_shift","poly_func", "poly_order", "poly_type"]:
                self.ref_fit_pars_current_point[each] = f(data_sub, each, which_row) 
        else:
            self.ref_fit_pars_current_point = {}
        # print(self.ref_fit_pars_current_point)

    def get_fit_pars_from_frame_index(self,frame_index):
        which_row = frame_index
        f = lambda obj_, str_,row_:obj_[str_][row_]
        for each in ["H", "K", "roi_x", "roi_y", "roi_w", "roi_h", "ss_factor", "peak_width","poly_func", "poly_order", "poly_type"]:
            if each == "peak_width":
                self.ref_fit_pars_current_point[each] = f(self.app_ctr.data, each, which_row)/2 
            else:
                self.ref_fit_pars_current_point[each] = f(self.app_ctr.data, each, which_row) 
        # print(self.ref_fit_pars_current_point)

    def set_fit_pars_from_reference(self):
        if len(self.ref_fit_pars_current_point)==0:
            return
        # print(self.app_ctr.bkg_sub.peak_width)
        # print(self.ref_fit_pars_current_point['peak_width'])
        self.roi.setPos(pos = [self.ref_fit_pars_current_point['roi_x'],self.ref_fit_pars_current_point['roi_y']])
        self.roi.setSize(size = [self.ref_fit_pars_current_point['roi_w'],self.ref_fit_pars_current_point['roi_h']])
        self.doubleSpinBox_ss_factor.setValue(self.ref_fit_pars_current_point['ss_factor'])
        self.spinBox_peak_width.setValue(self.ref_fit_pars_current_point['peak_width']/2)
        self.app_ctr.bkg_sub.peak_shift = self.ref_fit_pars_current_point['peak_shift']
        def _split_poly_order(order):
            if order in [1,2,3,4]:
                return [order]
            else:
                if order == 5:
                    return [1,4]
                elif order == 6:
                    return [2,4]
                elif order == 7:
                    return [2,5]
                elif order == 8:
                    return [1,3,4]
                elif order == 9:
                    return [2,3,4]
                elif order ==10:
                    return [1,2,3,4]
        try:
            eval("self.radioButton_{}.setChecked(True)".format(self.ref_fit_pars_current_point['poly_func']))
        except:
            print("No radioButton named radioButton_{}".format(self.ref_fit_pars_current_point['poly_func']))

        try:
            eval("self.radioButton_{}.setChecked(True)".format(self.ref_fit_pars_current_point['poly_type']))
        except:
            print("No radioButton named radioButton_{}".format(self.ref_fit_pars_current_point['poly_type']))
        
        poly_order_list = _split_poly_order(self.ref_fit_pars_current_point['poly_order'])
        for each in poly_order_list:
            try:
                eval("self.checkBox_order{}.setChecked(True)".format(each))
            except:
                print("No checkBox named checkBox_order{}".format(each))


    def move_roi_left(self):
        if not self.checkBox_big_roi.isChecked():
            self.app_ctr.bkg_sub.peak_shift = self.app_ctr.bkg_sub.peak_shift-int(self.lineEdit_roi_offset.text())
            # self.app_ctr.bkg_sub.peak_shift = -int(self.lineEdit_roi_offset.text())
            self.updatePlot()
        else:
            pos = [int(each) for each in self.roi.pos()] 
            size=[int(each) for each in self.roi.size()]
            if self.radioButton_roi_position.isChecked():
                pos_return,size_return = self._check_roi_boundary([pos[0]-int(self.lineEdit_roi_offset.text()),pos[1]],size)
                self.roi.setPos(pos = pos_return)
                self.roi.setSize(size = size_return)
            else:
                pos_return,size_return = self._check_roi_boundary(pos=[pos[0]-int(self.lineEdit_roi_offset.text()), pos[1]],size=[size[0]+int(self.lineEdit_roi_offset.text())*2, size[1]])
                self.roi.setSize(size=size_return)
                self.roi.setPos(pos = pos_return)

    def move_roi_right(self):
        if not self.checkBox_big_roi.isChecked():
            self.app_ctr.bkg_sub.peak_shift = self.app_ctr.bkg_sub.peak_shift + int(self.lineEdit_roi_offset.text())
            self.updatePlot()
        else:
            pos = [int(each) for each in self.roi.pos()] 
            size=[int(each) for each in self.roi.size()]
            if self.radioButton_roi_position.isChecked():
                pos_return,size_return = self._check_roi_boundary([pos[0]+int(self.lineEdit_roi_offset.text()),pos[1]],size)
                self.roi.setPos(pos = pos_return)
                self.roi.setSize(size = size_return)
            else:
                pos_return,size_return = self._check_roi_boundary(pos=[pos[0]+int(self.lineEdit_roi_offset.text()), pos[1]],size=[size[0]-int(self.lineEdit_roi_offset.text())*2, size[1]])
                self.roi.setSize(size=size_return)
                self.roi.setPos(pos = pos_return)

    def move_roi_down(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        if self.radioButton_roi_position.isChecked():
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]-int(self.lineEdit_roi_offset.text())],size)
            self.roi.setPos(pos_return)
            self.roi.setSize(size_return)
        else:
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]+int(self.lineEdit_roi_offset.text())],[size[0],size[1]-int(self.lineEdit_roi_offset.text())*2])
            self.roi.setPos(pos = pos_return)
            self.roi.setSize(size=size_return)

    def move_roi_up(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        if self.radioButton_roi_position.isChecked():
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]+int(self.lineEdit_roi_offset.text())],size)
            self.roi.setPos(pos_return)
            self.roi.setSize(size_return)
        else:
            pos_return,size_return =self._check_roi_boundary([pos[0], pos[1]-int(self.lineEdit_roi_offset.text())],[size[0],size[1]+int(self.lineEdit_roi_offset.text())*2])
            self.roi.setPos(pos = pos_return)
            self.roi.setSize(size=size_return)

    def display_current_roi_info(self):
        pos = [int(each) for each in self.roi.pos()] 
        size=[int(each) for each in self.roi.size()]
        pos_return,size_return = self._check_roi_boundary(pos,size)
        if not self.checkBox_lock.isChecked():
            self.lineEdit_roi_info.setText(str(pos_return + size_return))
        
    def set_roi(self):
        if self.lineEdit_roi_info.text()=='':
            return
        else:
            roi = eval(self.lineEdit_roi_info.text())
            self.roi.setPos(pos = roi[0:2])
            self.roi.setSize(size = roi[2:])
        

    def find_bounds_of_hist(self):
        bins = 200
        hist, bin_edges = np.histogram(self.app_ctr.bkg_sub.img, bins=bins)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        max_index = np.argmax(hist)
        mean_value = hist.mean()
        index_right = max_index
        for i in range(max_index,bins):
            if abs(hist[i]-mean_value)<mean_value*0.8:
                index_right = i
                break
        index_left = [max_index - (index_right - max_index),0][int((max_index - (index_right - max_index))<0)]
        index_right = min([index_right + int((index_right - index_left)*float(self.lineEdit_tailing_factor.text())),bins-1])
        return bin_centers[index_left], bin_centers[index_right]

    def maximize_roi(self):
        self.roi_pos = self.roi.pos()#save the roi pos first before maximize it
        self.roi_size = self.roi.size()#save the roi pos first before maximize it
        try:
            self.roi.setPos(pos = [int(self.roi.pos()[0]),0])
            self.roi.setSize(size = [int(self.roi.size()[0]),self.app_ctr.bkg_sub.img.shape[0]])
        except:
            logging.getLogger().exception('Error during setting roi to maximum, check the dimension!')
            self.tabWidget.setCurrentIndex(2)

    def track_peak(self):
        self.maximize_roi()
        if self.radioButton_automatic_set_hist.isChecked():
            self.hist.setLevels(*self.find_bounds_of_hist())
        loop_steps = int(self.lineEdit_track_steps.text())
        hist_range = self.hist.region.getRegion()
        left, right = hist_range
        for i in range(loop_steps):
            iso_value_temp = ((right - left)/loop_steps)*i + left + (right - left)*0.3
            self.isoLine.setValue(iso_value_temp)
            self.iso.setLevel(iso_value_temp)
            isocurve_center_x, iso_curve_center_y = self.iso.boundingRect().center().x(), self.iso.boundingRect().center().y()
            isocurve_height, isocurve_width = self.iso.boundingRect().height(),self.iso.boundingRect().width()
            if isocurve_height == 0 or isocurve_width==0:
                pass
            else:
                if (isocurve_height<int(self.lineEdit_track_size.text())) and (isocurve_width<int(self.lineEdit_track_size.text())):
                    break
                else:
                    pass

    def set_peak(self):
        arbitrary_size_offset = 10
        arbitrary_recenter_cutoff = 50
        isocurve_center_x, iso_curve_center_y = self.iso.boundingRect().center().x(), self.iso.boundingRect().center().y()
        isocurve_height, isocurve_width = self.iso.boundingRect().height()+arbitrary_size_offset,self.iso.boundingRect().width()+arbitrary_size_offset
        roi_new = [self.roi.pos()[0] + isocurve_center_x - self.roi.size()[0]/2,self.roi.pos()[1]+(iso_curve_center_y-self.roi.size()[1]/2)+self.roi.size()[1]/2-isocurve_height/2]
        if abs(sum(roi_new) - sum(self.roi_pos))<arbitrary_recenter_cutoff or (self.app_ctr.img_loader.current_frame_number == 0):
            self.roi.setPos(pos = roi_new)
            self.roi.setSize(size = [self.roi.size()[0],isocurve_height])
        else:#if too far away, probably the peak tracking failed to track the right peak. Then reset the roi to what it is before the track!
            self.roi.setPos(pos = self.roi_pos)
            self.roi.setSize(size = self.roi_size)

    def stop_func(self):
        if not self.stop:
            self.stop = True
            self.stopBtn.setText('Resume')
        else:
            self.stop = False
            self.stopBtn.setText('Stop')
        
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Conf Files (*.ini);;text Files (*.txt)", options=options)
        if fileName:
            self.lineEdit.setText(fileName)
            error_msg = self.widget_config.update_parameter(fileName)
            if error_msg!=None:
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Error to load config file!')
                logging.getLogger().exception(error_msg)
                self.tabWidget.setCurrentIndex(2)

    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName))

    def rload_file(self):#to be deleted
        self.save_file()
        #self.region_bounds = [0,1]
        try:
            self.app_ctr.run(self.lineEdit.text())
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000)
            self.plot_()
            self.statusbar.showMessage('Initialization succeed!')
        except:
            self.statusbar.showMessage('Initialization failed!')

    def launch_file(self):
        if not self.lineEdit.text().endswith('_temp.ini'):
            self.lineEdit.setText(self.lineEdit.text().replace('.ini','_temp.ini'))
        self.save_file()
        self.timer_save_data.timeout.connect(self.save_data)
        self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        #update the path to save data
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.data_path = data_file

        try:
            self.app_ctr.run(self.lineEdit.text())
            self.update_poly_order(init_step=True)
            self.update_cost_func(init_step=True)
            if self.launch.text()=='Launch':
                self.setup_image()
            else:
                pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
            self.plot_()
            self.update_ss_factor()
            self.image_set_up = False
            self.launch.setText("Relaunch")
            self.statusbar.showMessage('Initialization succeed!')
            self.image_set_up = True

            self.widget_terminal.update_name_space('data',self.app_ctr.data)
            self.widget_terminal.update_name_space('bkg_sub',self.app_ctr.bkg_sub)
            self.widget_terminal.update_name_space('img_loader',self.app_ctr.img_loader)
            self.widget_terminal.update_name_space('main_win',self)
            self.widget_terminal.update_name_space('overplot_ctr',overplot_ctr_temp)
            self.hist.sigLevelsChanged.connect(self.update_hist_levels)

        except Exception:
            self.image_set_up = False
            try:
                self.timer_save_data.stop()
            except:
                pass
            self.statusbar.showMessage('Initialization failed!')
            logging.getLogger().exception('Fatal error encounter during lauching config file! Check the config file for possible errors.')
            self.tabWidget.setCurrentIndex(2)

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Text documents (*.txt);All files (*.*)")
        #text = self.textEdit.toPlainText()
        #with open(path, 'w') as f:
        #    f.write(text)
        #self.statusbar.showMessage('Config file is saved as {}!'.format(path))
        if path != '':
            self.widget_config.save_parameter(path)
        else:
            self.statusbar.showMessage('Failure to save Config file with the file name of {}!'.format(path))

    def save_file(self):
        #text = self.textEdit.toPlainText()
        #if text=='':
        #    self.statusbar.showMessage('Text editor is empty. Config file is not saved!')
        #else:
        if self.lineEdit.text()!='':
            self.widget_config.save_parameter(self.lineEdit.text())
            self.statusbar.showMessage('Config file is saved with the same file name!')
        else:
            self.statusbar.showMessage('Failure to save Config file with the file name of {}!'.format(self.lineEdit.text()))

    def plot_figure(self):
        self.tag_reprocess = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.plot_)
        self.run_mode = True
        self.timer.start(100)

    def update_image(self):
        int_max,int_min = np.max(self.app_ctr.bkg_sub.img),np.min(self.app_ctr.bkg_sub.img)
        if self.image_log_scale:
            self.img_pyqtgraph.setImage(np.log10(self.app_ctr.bkg_sub.img))
            int_max,int_min = np.log10(int_max),np.log10(int_min)
        else:
            self.img_pyqtgraph.setImage(self.app_ctr.bkg_sub.img)
        self.p1.autoRange() 
        # self.hist.setImageItem(self.img_pyqtgraph)
        # self.hist.setLevels(self.app_ctr.bkg_sub.img.min(), self.app_ctr.bkg_sub.img.mean()*10)
        if self.radioButton_fixed_percent.isChecked():
            offset_ = float(self.lineEdit_scale_factor.text())/100*(int_max-int_min)
            # print(int_min,int_max,offset_)
            self.hist.setLevels(int_min, int_min+offset_)
        elif self.radioButton_fixed_between.isChecked():
            self.hist.setLevels(max([int_min,float(self.lineEdit_left.text())]), float(self.lineEdit_right.text()))
        elif self.radioButton_automatic_set_hist.isChecked(): 
            self.hist.setLevels(*self.find_bounds_of_hist())

    def update_hist_levels(self):
        left,right = self.hist.getLevels()
        self.lineEdit_left.setText(str(round(left,6)))
        self.lineEdit_right.setText(str(round(right,6)))

    def plot_(self):
        #self.app_ctr.set_fig(self.MplWidget.canvas.figure)
        t0 = time.time()
        if self.stop:
            self.timer.stop()
            self.run_mode = False
        else:
            try:
                return_value = self.app_ctr.run_script(poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
                self.get_fit_pars_from_reference()
                self.set_fit_pars_from_reference()
                self.update_plot()
                if self.app_ctr.bkg_sub.img is not None:
                    self.lcdNumber_scan_number.display(self.app_ctr.img_loader.scan_number)
                    self.update_image()
                    if self.image_set_up:
                        self.updatePlot(begin = False)
                    else:
                        self.updatePlot(begin = True)
                if self.checkBox_auto_track.isChecked():
                    self.track_peak()
                    self.set_peak()
                if return_value:
                    self.statusbar.clearMessage()
                    self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.current_frame_number+1,self.app_ctr.img_loader.total_frame_number))
                    self.progressBar.setValue(int((self.app_ctr.img_loader.current_frame_number+1)/self.app_ctr.img_loader.total_frame_number*100))
                    self.lcdNumber_frame_number.display(self.app_ctr.img_loader.current_frame_number+1)
                else:
                    self.timer.stop()
                    self.save_data()
                    self.stop = False
                    self.stopBtn.setText('Stop')
                    self.statusbar.clearMessage()
                    self.statusbar.showMessage('Run for scan{} is finished, {} frames in total have been processed!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.total_frame_number))
                # """
                    #if you want to save the images, then uncomment the following three lines
                    #QtGui.QApplication.processEvents()
                    #exporter = pg.exporters.ImageExporter(self.widget_image.scene())
                    #exporter.export(os.path.join(DaFy_path,'temp','temp_frames','scan{}_frame{}.png'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.frame_number+1)))
            except:
                logging.getLogger().exception('Fatal error encounter during data analysis.')
                self.tabWidget.setCurrentIndex(2)

            # """
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass

    def update_plot(self):
        try:
            img = self.app_ctr.run_update(poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
            if self.tag_reprocess:
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr, int(self.lineEdit_frame_index_offset.text()))
            else:
                plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
            # self.MplWidget.canvas.figure.tight_layout()
            # self.MplWidget.canvas.draw()
        except:
            logging.getLogger().exception('Fatal error encounter during data analysis.')
            self.tabWidget.setCurrentIndex(2)

    def reset_peak_center_and_width(self):
        roi_size = [int(each/2) for each in self.roi.size()][::-1]
        roi_pos = [int(each) for each in self.roi.pos()][::-1]
        #roi_pos[0] = self.app_ctr.cen_clip[0]*2-roi_pos[0]
        #new_center = [roi_pos[0]-roi_size[0],roi_pos[1]+roi_size[1]]
        #roi_pos[0] = self.app_ctr.cen_clip[0]*2-roi_pos[0]
        new_center = [roi_pos[0]+roi_size[0],roi_pos[1]+roi_size[1]]
        self.app_ctr.bkg_sub.center_pix = new_center
        self.app_ctr.bkg_sub.row_width = roi_size[1]
        self.app_ctr.bkg_sub.col_width = roi_size[0]

    def peak_cen_shift_hor(self):
        offset = int(self.spinBox_hor.value())
        self.app_ctr.bkg_sub.update_center_pix_left_and_right(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def peak_cen_shift_ver(self):
        offset = int(self.spinBox_ver.value())
        self.app_ctr.bkg_sub.update_center_pix_up_and_down(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def row_width_shift(self):
        offset = int(self.horizontalSlider.value())
        self.app_ctr.bkg_sub.update_integration_window_row_width(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()

    def col_width_shift(self):
        offset = int(self.verticalSlider.value())
        self.app_ctr.bkg_sub.update_integration_window_column_width(offset)
        #print(self.app_ctr.bkg_sub.center_pix)
        self.update_plot()