import os, time
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog, QCheckBox, QRadioButton
from dafy.core.util.path import DaFy_path
from dafy.projects.xrv.core.util import pixel_to_q
from .graph_operations import setup_image

class GuiOperations(object):
    def __init__(self):
        pass

    def apply_cut_width_offset(self):
        self.app_ctr.peak_fitting_instance.set_cut_width_offset({'hor':self.spinBox_cut_width_offset_hor.value(),'ver':self.spinBox_cut_width_offset_ver.value()})
        self.recenter()

    def apply_center_offset(self):
        self.app_ctr.peak_fitting_instance.set_center_offset([self.spinBox_cen_offset_hor.value(), self.spinBox_cen_offset_ver.value()])
        self.recenter()

    #to fold or unfold the config file editor
    def fold_or_unfold(self):
        text = self.pushButton_fold_or_unfold.text()
        if text == "<":
            self.frame.setVisible(False)
            self.pushButton_fold_or_unfold.setText(">")
        elif text == ">":
            self.frame.setVisible(True)
            self.pushButton_fold_or_unfold.setText("<")

    #fun to save data
    def save_data(self):
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.save_data_file(data_file)
        try:
            self.app_ctr.save_data_file(data_file)
            self.statusbar.showMessage('Data file is saved as {}!'.format(data_file))
        except:
            self.statusbar.showMessage('Failure to save data file!')

    #fun to remove abnormal points (set marsk to false, data is saved acturally!)
    def remove_data_point(self):
        left,right = [int(each) for each in self.region_abnormal.getRegion()]
        self.lineEdit_abnormal_points.setText('Frame {} to Frame {}'.format(left,right))
        first_index_for_current_scan = np.where(np.array(self.app_ctr.data['scan_no'])==self.app_ctr.img_loader.scan_number)[0][0]
        for each_index in range(first_index_for_current_scan+left,first_index_for_current_scan+right+1):
            self.app_ctr.data['mask_cv_xrd'][each_index] = False
            self.app_ctr.data['mask_ctr'][each_index] = False
        self.updatePlot()

    def remove_current_data_point(self):
        #left,right = [int(each) for each in self.region_abnormal.getRegion()]
        #self.lineEdit_abnormal_points.setText('Frame {} to Frame {}'.format(left,right))
        first_index_for_current_scan = np.where(np.array(self.app_ctr.data['scan_no'])==self.app_ctr.img_loader.scan_number)[0][0]
        index = int(first_index_for_current_scan + self.app_ctr.img_loader.current_frame_number)
        self.app_ctr.data['mask_cv_xrd'][index] = False
        self.app_ctr.data['mask_ctr'][index] = False
        self.updatePlot()

    def select_source_for_plot_p2(self):
        self.app_ctr.p2_data_source = self.comboBox_p2.currentText()
        self.updatePlot()

    def update_poly_order(self, init_step = False):
        ord_total = 0
        for each in self.groupBox_2.findChildren(QCheckBox):
            ord_total += int(bool(each.checkState()))*int(each.text())
        self.app_ctr.bkg_sub.update_integration_order(ord_total)
        print('ord_total:',ord_total)
        if not init_step:
            self.updatePlot()

    def update_cost_func(self, init_step = False):
        for each in self.groupBox_cost_func.findChildren(QRadioButton):
            if each.isChecked():
                self.app_ctr.bkg_sub.update_integration_function(each.text())
                if not init_step:
                    self.updatePlot()
                break

    def update_ss_factor(self, init_step = False):
        self.app_ctr.bkg_sub.update_ss_factor(self.doubleSpinBox_ss_factor.value())
        self.updatePlot()

    def setup_image(self):
        setup_image(self)

    def _update_info(self,index):
        self.lcdNumber_potential.display(self.app_ctr.data['potential'][index])
        self.lcdNumber_current.display(self.app_ctr.data['current'][index])
        self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][index])
        self.lcdNumber_strain_par.display(self.app_ctr.data['strain_ip'][index])
        self.lcdNumber_strain_ver.display(self.app_ctr.data['strain_oop'][index])
        self.lcdNumber_size_par.display(self.app_ctr.data['grain_size_ip'][index])
        self.lcdNumber_size_ver.display(self.app_ctr.data['grain_size_oop'][index])
        self.lcdNumber_frame_number.display(self.app_ctr.data['image_no'][index])

    def mouseMoved_in_p2(self, evt):
         pos = evt[0]  ## using signal proxy turns original arguments into a tuple
         current_scan_no = self.app_ctr.data['scan_no'][-1]
         start = self.app_ctr.data['scan_no'].index(current_scan_no)
         if self.p2.sceneBoundingRect().contains(pos):
            mousePoint = self.p2.vb.mapSceneToView(pos)
            index = mousePoint.x()
            which = np.argmin(abs(np.array(self.app_ctr.data['image_no'][start:])-index)) + start
            self._update_info(which)
            if self.app_ctr.p2_data_source == 'vertical':
                strain = self.app_ctr.data['strain_oop'][which]
                size = self.app_ctr.data['grain_size_oop'][which]
                self.single_point_size_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [size])
                self.single_point_strain_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [strain])
            elif self.app_ctr.p2_data_source == 'horizontal':
                strain = self.app_ctr.data['strain_ip'][which]
                size = self.app_ctr.data['grain_size_ip'][which]
                self.single_point_size_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [size])
                self.single_point_strain_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [strain])
            else:
                strain = [self.app_ctr.data['strain_oop'][which]]
                size = [self.app_ctr.data['grain_size_oop'][which]]
                strain.append(self.app_ctr.data['strain_ip'][which])
                size.append(self.app_ctr.data['grain_size_ip'][which])
                self.single_point_size_oop_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [size[0]])
                self.single_point_strain_oop_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [strain[0]])
                self.single_point_size_ip_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [size[1]])
                self.single_point_strain_ip_ax_handle.setData(x = [self.app_ctr.data['image_no'][which]], y = [strain[1]])
            self.vLine_par.setPos(self.app_ctr.data['image_no'][which])
            self.hLine_par.setPos(mousePoint.y())

    def recenter(self):
        self.app_ctr.peak_fitting_instance.recenter = True
        self.updatePlot()
        self._remake_img(self.checkBox_small_cut.isChecked())

    def stop_func(self):
        if not self.stop:
            self.stop = True
            self.stopBtn.setText('Resume')
        else:
            self.stop = False
            self.stopBtn.setText('Stop')
    
    #load config file
    def load_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Config Files (*.ini)", os.path.join(DaFy_path,'scripts','XRV'),options = options)
        if fileName:
            self.lineEdit.setText(fileName)
            self.widget_config.update_parameter(fileName)

    def locate_data_folder(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            self.lineEdit_data_file_path.setText(os.path.dirname(fileName))

    def launch_file(self):
        self.save_file() 
        #update the path to save data
        data_file = os.path.join(self.lineEdit_data_file_path.text(),self.lineEdit_data_file_name.text())
        self.app_ctr.use_q_mapping = self.radioButton_q.isChecked()
        self.app_ctr.img_loader_object = self.comboBox_loader.currentText()
        self.app_ctr.data_path = data_file
        self.app_ctr.run(self.lineEdit.text())
        self.update_poly_order(init_step=True)
        # self.update_cost_func(init_step=True)
        if self.launch.text()=='Launch':
            self.setup_image()
            #ugly way to rescale the widget correctly
            size = self.widget_image.size()
            self.widget_image.resize(size.width()*2,size.height())
            self.widget_image.resize(size.width(),size.height())
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        else:
            pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        #self.timer_save_data.stop()
        self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
        self.plot_()
        self.launch.setText("Relaunch")
        self.statusbar.showMessage('Initialization succeed!') 
        try:
            self.app_ctr.run(self.lineEdit.text())
            self.update_poly_order(init_step=True)
            if self.launch.text()=='Launch':
                self.setup_image()
            else:
                pass
            self.timer_save_data.stop()
            self.timer_save_data.start(self.spinBox_save_frequency.value()*1000*60)
            self.plot_()
            self.launch.setText("Relaunch")
            self.statusbar.showMessage('Initialization succeed!')
        except:
            self.statusbar.showMessage('Initialization failed!')

    def save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save file", "", "Text documents (*.txt);All files (*.*)")
        self.widget_config.save_parameter(path)
        self.statusbar.showMessage('Config file is saved as {}!'.format(path))

    def save_file(self):
        if self.lineEdit.text()=='':
            self.statusbar.showMessage('Text editor is empty. Config file is not saved!')
        else:
            self.widget_config.save_parameter(self.lineEdit.text())
            self.statusbar.showMessage('Config file is saved with the same file name!')

    def plot_figure(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.plot_)
        self.timer.start(50)

    def _remake_img(self, plot_small_cut = False):
        #set image and cut on the image (NOTE:row-major index)
        if plot_small_cut:
            cut_values_hoz=[self.app_ctr.peak_fitting_instance.peak_center[0]+self.app_ctr.peak_fitting_instance.cen_offset[0]-(self.app_ctr.peak_fitting_instance.cut_offset['hor'][-1] +self.app_ctr.peak_fitting_instance.cut_width_offset['hor']),self.app_ctr.peak_fitting_instance.peak_center[0]+self.app_ctr.peak_fitting_instance.cen_offset[0]+(self.app_ctr.peak_fitting_instance.cut_offset['hor'][-1] +self.app_ctr.peak_fitting_instance.cut_width_offset['hor'])]
            cut_values_ver=[self.app_ctr.peak_fitting_instance.peak_center[1]+self.app_ctr.peak_fitting_instance.cen_offset[1]-(self.app_ctr.peak_fitting_instance.cut_offset['ver'][-1] +self.app_ctr.peak_fitting_instance.cut_width_offset['ver']),self.app_ctr.peak_fitting_instance.peak_center[1]+self.app_ctr.peak_fitting_instance.cen_offset[1]+(self.app_ctr.peak_fitting_instance.cut_offset['ver'][-1] +self.app_ctr.peak_fitting_instance.cut_width_offset['ver'])]
        else:
            cut_values_hoz=[self.app_ctr.peak_fitting_instance.peak_center_0[0]-self.app_ctr.peak_fitting_instance.cut_offset['hor'][0],self.app_ctr.peak_fitting_instance.peak_center_0[0]+self.app_ctr.peak_fitting_instance.cut_offset['hor'][0]]
            cut_values_ver=[self.app_ctr.peak_fitting_instance.peak_center_0[1]-self.app_ctr.peak_fitting_instance.cut_offset['ver'][0],self.app_ctr.peak_fitting_instance.peak_center_0[1]+self.app_ctr.peak_fitting_instance.cut_offset['ver'][0]]
        self.img_pyqtgraph.setImage(self.app_ctr.bkg_sub.img)

        self.region_cut_hor.setRegion(cut_values_hoz)
        self.region_cut_ver.setRegion(cut_values_ver)
        #set roi
        size_of_roi = self.roi.size()
        self.roi.setPos([self.app_ctr.peak_fitting_instance.peak_center[1]-size_of_roi[0]/2.,self.app_ctr.peak_fitting_instance.peak_center[0]-size_of_roi[1]/2.])

        if self.app_ctr.img_loader.current_frame_number == 0:
            self.p1.autoRange() 
            #relabel the axis
            if self.radioButton_q.isChecked():
                q_par = self.app_ctr.rsp_instance.q['grid_q_par'][0]
                q_ver = self.app_ctr.rsp_instance.q['grid_q_perp'][:,0]
                scale_ver = (max(q_ver)-min(q_ver))/(len(q_ver)-1)
                shift_ver = min(q_ver)
                scale_hor = (max(q_par)-min(q_par))/(len(q_par)-1)
                shift_hor = min(q_par)
            else:
                scale_hor, shift_hor = 1, 0
                scale_ver, shift_ver = 1, 0
            ax_item_img_hor = pixel_to_q(scale = scale_hor, shift = shift_hor, orientation = 'bottom')
            ax_item_img_ver = pixel_to_q(scale = scale_ver, shift = shift_ver, orientation = 'left')
            ax_item_img_hor.attachToPlotItem(self.p1)
            ax_item_img_ver.attachToPlotItem(self.p1)
        self.hist.setLevels(self.app_ctr.bkg_sub.img.min(), self.app_ctr.bkg_sub.img.mean()*10)

    def plot_(self):
        t0 = time.time()
        if self.stop:
            self.timer.stop()
        else:
            return_value = self.app_ctr.run_script()
            if self.app_ctr.bkg_sub.img is not None:
                self._remake_img(self.checkBox_small_cut.isChecked())
                self.updatePlot(fit = False)

            if return_value:
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Working on scan{}: we are now at frame{} of {} frames in total!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.current_frame_number+1,self.app_ctr.img_loader.total_frame_number))
                self.progressBar.setValue(int((self.app_ctr.img_loader.current_frame_number+1)/float(self.app_ctr.img_loader.total_frame_number)*100))
            else:
                self.timer.stop()
                self.save_data()
                self.stop = False
                self.stopBtn.setText('Stop')
                self.statusbar.clearMessage()
                self.statusbar.showMessage('Run for scan{} is finished, {} frames in total have been processed!'.format(self.app_ctr.img_loader.scan_number,self.app_ctr.img_loader.total_frame_number))
        try:
            self.lcdNumber_speed.display(int(1./(time.time()-t0)))
        except:
            pass

    def reset_peak_center_and_width(self):
        roi_size = [int(each/2) for each in self.roi.size()][::-1]
        roi_pos = [int(each) for each in self.roi.pos()][::-1]
        new_center = [roi_pos[0]+roi_size[0],roi_pos[1]+roi_size[1]]
        self.app_ctr.bkg_sub.center_pix = new_center
        self.app_ctr.bkg_sub.row_width = roi_size[1]
        self.app_ctr.bkg_sub.col_width = roi_size[0]