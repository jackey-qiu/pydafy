class InstrumentOperations(object):

    def __init__(self):
        pass

    def send_hkl_angs(self):
        self.calc_angs()
        all = list([self.comboBox_bragg_peak.itemText(i) for i in range(self.comboBox_bragg_peak.count())])
        h, k = int(float(self.lineEdit_H_calc.text())), int(float(self.lineEdit_K_calc.text()))
        #l = None
        index = None
        for i, each in enumerate(all):
            if each.startswith('['):
                h_, k_, l_ = eval(each)
                if (h == h_) and (k == k_):
                    #l = l_
                    index = i
                    break
        #Bragg_peak = '[{},{},{}]'.format(h, k, l)
        if index==None:
            return
        else:
            self.comboBox_bragg_peak.setCurrentIndex(index)
            # self.radioButton_single_rod.setChecked(True)
            self.extract_peaks_in_zoom_viewer()
            self.lineEdit_delta_angle.setText(self.lineEdit_delta.text())
            self.lineEdit_rot_x.setText(self.lineEdit_eta.text())
            self.lineEdit_gam_angle.setText(self.lineEdit_gam.text())
            self.lineEdit_SN_degree.setText(str(-float(self.lineEdit_phi.text())))
            self.rotate_sample()
            self.send_detector()

    def spin_(self):
        if self.radioButton_continually.isChecked():
            if self.timer_spin_sample.isActive():
                self.timer_spin_sample.stop()
            else:
                self.trajactory_pos = []
                self.timer_spin_sample.start(1000)
        else:
            try:
                self.timer_spin_sample.stop()
            except:
                pass
            self.rotate_sample_SN()

    def rotate_sample(self):
        self.rotate_()
        #update the constrain as well
        self.lineEdit_cons_eta.setText(self.lineEdit_rot_x.text())
        self.set_cons()

    def rotate_sample_SN(self):
        if self.timer_spin_sample.isActive():
            new_theta = self.widget_glview.theta_SN + float(self.lineEdit_speed.text())
            if new_theta > 360:
                self.timer_spin_sample.stop()
            else:
                self.lineEdit_SN_degree.setText(str(new_theta))
                self.rotate_()
        else:
            self.rotate_()

    def rotate_(self):
        #first rotate along x axis to tilt the sample
        theta_x = float(self.lineEdit_rot_x.text())
        self.widget_glview.theta_x_r = theta_x - self.widget_glview.theta_x
        self.widget_glview.theta_x = theta_x
        #then rotate the same alone surface normal direction
        self.widget_glview.theta_SN_r = float(self.lineEdit_SN_degree.text()) - self.widget_glview.theta_SN
        self.widget_glview.theta_SN = float(self.lineEdit_SN_degree.text())
        #update structue in the glviewer
        self.widget_glview.update_structure()
        self.extract_peaks_in_zoom_viewer()
        self.extract_cross_point_info()
        #update detector viewer
        self.simulate_image()

    def send_detector(self):
        self.widget_glview.send_detector(gam = float(self.lineEdit_gam_angle.text()), delta = float(self.lineEdit_delta_angle.text()))

    def simulate_l_scan_live(self):
        ang_list = self.plainTextEdit_cross_points_info.toPlainText().rsplit('\n')
        if (2+self.current_line)==len(ang_list):
            self.timer_l_scan.stop()
            self.current_line = 0
            return
        current_line = ang_list[2+self.current_line].rsplit('\t')
        phi, gam, delta = -float(current_line[6]), float(current_line[3]), float(current_line[2])
        self.lineEdit_delta_angle.setText(str(delta))
        self.lineEdit_gam_angle.setText(str(gam))
        self.lineEdit_SN_degree.setText(str(phi))
        self.pushButton_spin.click()
        self.pushButton_send_detector.click()
        ang_list[2+self.current_line]='>'+ang_list[2+self.current_line]
        self.plainTextEdit_cross_points_info.setPlainText('\n'.join(ang_list))
        self.current_line = self.current_line + 1

    def start_spin(self):
        self.timer_spin.start(100)

    def stop_spin(self):
        self.timer_spin.stop()

    def spin(self):
        #if self.azimuth > 360:
        self.update_camera_position(angle_type="azimuth", angle=self.azimuth_angle)
        self.azimuth_angle = self.azimuth_angle + 1


    