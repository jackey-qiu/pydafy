import numpy as np
from numpy.linalg import inv
from xtal.unitcell import read_cif, write_cif
from xtal.surface import SurfaceCell
from dafy.core.util.DebugFunctions import error_pop_up

class RspaceOperations(object):

    def __init__(self):
        pass

    def get_simulated_hkl(self, x, y):
            
        px, py = self.widget_glview.primary_beam_position
        L = float(self.lineEdit_sample_detector_distance.text())
        ps = float(self.lineEdit_pixel.text())

        delta = np.rad2deg(np.arctan((py-y)*ps/(L**2+(x-px)**2*ps**2)**0.5))
        gam = np.rad2deg(np.arctan((x-px)*ps/L))
        phi = -float(self.lineEdit_SN_degree.text())
        self.energy_keV = float(self.lineEdit_eng.text())
        self.hardware.settings.hardware.energy = self.energy_keV
        angles_string = ['mu', 'delta', 'gam', 'eta', 'chi', 'phi']
        angles = []
        for ag in angles_string:
            angles.append(float(getattr(self, 'lineEdit_{}'.format(ag)).text()))
        angles[1] = delta
        angles[2] = gam
        angles[-1] = phi
        hkls, pars = self.dc.angles_to_hkl(angles, energy=self.energy_keV)
        #hkl = [round(each,3) for each in hkls]
        chi = angles[-2]
        eta = angles[-3]
        return [round(each,3) for each in hkls], phi, delta, gam, chi, eta 

    def generate_surface_unitcell_info(self):
        uc = read_cif(self.structure_container[self.comboBox_substrate_surfuc.currentText()])
        ### this computes the surface cell.  note we pass the TM
        ### matrix so that the bulk hexagonal cell is transformed to a
        ### rhombahedral primitive cell before trying to find the surface
        ### indexing (ie, we want to find the mimimum surface unit cell) 
        #hexagonal to rhombahedral TM [[2/3, 1/3, 1/3], [-1/3, 1/3, 1/3], [-1/3, -2/3, 1/3]]
        #fcc to rhombahedral TM [[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
        TM_ = eval(self.lineEdit_TM.text())
        self.surf = SurfaceCell(uc,hkl=eval(self.lineEdit_hkl.text()),nd=self.spinBox_slab_num.value(),term=self.spinBox_term.value(),bulk_trns=np.array(TM_),use_tradition = self.checkBox_use_tradition.isChecked(), z_start_from_0 = self.checkBox_from_zero.isChecked())
        # self.surf = SurfaceCell(uc,hkl=eval(self.lineEdit_hkl.text()),nd=self.spinBox_slab_num.value(),term=self.spinBox_term.value(),bulk_trns=np.array(TM_))
        info, data = self.surf._write_pandas_df()
        self.textEdit_info.setHtml(info + data.to_html(index = False))

    def display_UB(self):
        if self.comboBox_UB.currentText()=='U matrix':
            matrix = self.get_U()
        elif self.comboBox_UB.currentText()=='B matrix':
            matrix = self.get_B()
        elif self.comboBox_UB.currentText()=='UB matrix':
            matrix = self.get_UB()
        self._set_matrix_values(matrix)

    def get_U(self):
        return self.ub.get_ub_info()[0]

    def get_B(self):
        return self.ub.get_ub_info()[1]

    def get_UB(self):
        return self.ub.get_ub_info()[2]

    def set_UB_matrix(self):
        if self.comboBox_UB.currentText()=='U matrix':
            self.set_U()
            error_pop_up(msg_text = 'U matrix updated!', window_title = 'Information')
        elif self.comboBox_UB.currentText()=='UB matrix':
            self.set_UB()
            error_pop_up(msg_text = 'UB matrix updated!', window_title = 'Information')
        else:
            pass

    def set_U(self):
        self.dc.setu(self._get_matrix_values())

    def set_UB(self):
        self.dc.setub(self._get_matrix_values())

    def _get_matrix_values(self):
        r1=[float(self.lineEdit_U00.text()),float(self.lineEdit_U01.text()),float(self.lineEdit_U02.text())]
        r2=[float(self.lineEdit_U10.text()),float(self.lineEdit_U11.text()),float(self.lineEdit_U12.text())]
        r3=[float(self.lineEdit_U20.text()),float(self.lineEdit_U21.text()),float(self.lineEdit_U22.text())]
        return [r1,r2,r3]

    def _set_matrix_values(self, matrix):
        self.lineEdit_U00.setText(str(round(matrix[0,0],3)))
        self.lineEdit_U01.setText(str(round(matrix[0,1],3)))
        self.lineEdit_U02.setText(str(round(matrix[0,2],3)))
        self.lineEdit_U10.setText(str(round(matrix[1,0],3)))
        self.lineEdit_U11.setText(str(round(matrix[1,1],3)))
        self.lineEdit_U12.setText(str(round(matrix[1,2],3)))
        self.lineEdit_U20.setText(str(round(matrix[2,0],3)))
        self.lineEdit_U21.setText(str(round(matrix[2,1],3)))
        self.lineEdit_U22.setText(str(round(matrix[2,2],3)))

    def add_refs(self):
        self.ub.addref(eval('[{}]'.format(self.lineEdit_hkl_ref1.text())),eval('[{}]'.format(self.lineEdit_angs_ref1.text())),float(self.lineEdit_eng_ref1.text()))
        self.ub.addref(eval('[{}]'.format(self.lineEdit_hkl_ref2.text())),eval('[{}]'.format(self.lineEdit_angs_ref2.text())),float(self.lineEdit_eng_ref2.text()))
        self.ub.calcub()
        self.display_UB()

    def set_cons(self):
        for each in self.cons:
            self.hkl.uncon(each)
        tag = self._get_cons_string()
        cons_string = self.cons
        if tag == 'Failed':
            return
        msg = None
        for each in cons_string:
            if each in ['a_eq_b', 'bin_eq_bout', 'mu_is_gam','bisect']:
                getattr(self, 'checkBox_{}'.format(each)).setChecked(True)
                msg = self.hkl.con(each)
            else:
                msg = self.hkl.con(each,float(getattr(self,'lineEdit_cons_{}'.format(each)).text()))
        error_pop_up('Constraints:\n'+msg,'Information')

    def _get_cons_string(self):
        used_strings = []
        possible_strings = ['a_eq_b', 'bin_eq_bout', 'mu_is_gam','bisect','delta','gam','qaz','naz','alpha','beta','psi','betain','betaout','mu','eta','chi','phi','omega']
        for each in possible_strings:
            if getattr(self, 'checkBox_{}'.format(each)).isChecked():
                used_strings.append(each)
        if len(used_strings)>3:
            error_pop_up('Too many cons. You need only 3 cons!','Error')
            return 'Failed'
        else:
            if len(used_strings)==0:
                error_pop_up('No constraint has been selected!','Error')
            else:
                self.cons = used_strings

    def clear_all_cons(self):
        possible_strings = ['a_eq_b', 'bin_eq_bout', 'mu_is_gam','bisect','delta','gam','qaz','naz','alpha','beta','psi','betain','betaout','mu','eta','chi','phi','omega']
        for each in possible_strings:
            getattr(self, 'checkBox_{}'.format(each)).setChecked(False)

    def set_predefined_cons(self):
        self.clear_all_cons()
        if self.comboBox_predefined_cons.currentText()=='sxrd_ver':
            for each in ['eta','mu','chi']:
                getattr(self, 'checkBox_{}'.format(each)).setChecked(True)
            self.lineEdit_cons_eta.setText('2')
            self.lineEdit_cons_mu.setText('0')
            self.lineEdit_cons_chi.setText('90')
        elif self.comboBox_predefined_cons.currentText()=='reflectivity':
            for each in ['qaz','mu','a_eq_b']:
                getattr(self, 'checkBox_{}'.format(each)).setChecked(True)
            self.lineEdit_cons_qaz.setText('90')
            self.lineEdit_cons_mu.setText('0')
            #self.lineEdit_cons_chi.setText('90')
        self.set_cons()
        
    def _cal_trajactory_pos(self, ax):
        self.trajactory_pos = []
        pilatus_size = [float(self.lineEdit_detector_hor.text()), float(self.lineEdit_detector_ver.text())]
        pixel_size = [float(self.lineEdit_pixel.text())]*2
        try:
            prim_beam = eval(self.lineEdit_prim_beam_pos.text())
        except:
            print('Evaluation error for :{}'.format(self.lineEdit_prim_beam_pos.text()))
            prim_beam = None
        #first rotate along x axis to tilt the sample
        theta_x = float(self.lineEdit_rot_x.text())
        self.widget_glview.theta_x_r = theta_x - self.widget_glview.theta_x
        self.widget_glview.theta_x = theta_x
        self.widget_glview.apply_xyz_rotation()
        #do a 360 degree rotation along SN
        original_text = self.lineEdit_SN_degree.text()
        theta_SN_r_original = self.widget_glview.theta_SN_r
        theta_SN_original = self.widget_glview.theta_SN

        for theta in np.arange(0,360,3):
            self.lineEdit_SN_degree.setText(str(theta))
            #then rotate the same alone surface normal direction
            self.widget_glview.theta_SN_r = float(self.lineEdit_SN_degree.text()) - self.widget_glview.theta_SN
            self.widget_glview.theta_SN = float(self.lineEdit_SN_degree.text())
            self.widget_glview.apply_SN_rotation()
            self.widget_glview.recal_cross_points()
            # self.extract_cross_point_info()    
            self.widget_glview.calculate_index_on_pilatus_image_from_cross_points_info(
                pilatus_size = pilatus_size,
                pixel_size = pixel_size,
                distance_sample_detector = float(self.lineEdit_sample_detector_distance.text()),
                primary_beam_pos = prim_beam)
            for each in self.widget_glview.pixel_index_of_cross_points:
                for i in range(len(self.widget_glview.pixel_index_of_cross_points[each])):
                    pos = self.widget_glview.pixel_index_of_cross_points[each][i]
                    # hkl = self.widget_glview.cross_points_info_HKL[each][i]
                    if len(pos)!=0:
                        if pos not in self.trajactory_pos:
                            self.trajactory_pos.append(pos) 
        #send values back to original ones
        self.lineEdit_SN_degree.setText(original_text)
        self.widget_glview.theta_SN_r = theta_SN_r_original
        self.widget_glview.theta_SN = theta_SN_original
        for each in self.trajactory_pos:
            ax.text(*each[::-1],'.',
            horizontalalignment='center',
            verticalalignment='center',color = 'w',fontsize = 15) 

    #extract cross points between the rods and the Ewarld sphere
    def extract_cross_point_info(self):
        text = ['The cross points between rods and the Ewarld sphere is listed below']
        self.widget_glview.cross_points_info_HKL = {}
        for each in self.widget_glview.cross_points_info:
            self.widget_glview.cross_points_info_HKL[each] = []
            text.append('')
            text.append(each)
            structure = [self.structures[i] for i in range(len(self.structures)) if self.structures[i].name == each][0]
            #apply the rotation matrix due to sample tilting (RM) + sample rotation(RRM)
            RM = self.widget_glview.RRM.dot(self.widget_glview.RM)
            RecTMInv = inv(RM.dot(structure.lattice.RecTM))
            for each_q in self.widget_glview.cross_points_info[each]:
                #H, K, L = structure.lattice.HKL(each_q)
                H, K, L = RecTMInv.dot(each_q)
                self.widget_glview.cross_points_info_HKL[each].append([round(H,2),round(K,2),round(L,2)])
                text.append('HKL:{}'.format([round(H,3),round(K,3),round(L,3)]))
        self.plainTextEdit_cross_points_info.setPlainText('\n'.join(text))

    def compute_angles(self):
        if self.lineEdit_H.text()=='' or self.lineEdit_K.text()=='' or self.lineEdit_L.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        hkl = [float(self.lineEdit_H.text()),float(self.lineEdit_K.text()),float(self.lineEdit_L.text())]
        name = self.comboBox_working_substrate.currentText()
        phi, gamma, delta = self._compute_angles(hkl, name, mu = None)
        self.lineEdit_phi.setText(str(round(phi,3)))
        self.lineEdit_gamma.setText(str(round(gamma,3)))
        self.lineEdit_delta.setText(str(round(delta,3)))

    def _compute_angles(self, hkl, name, mu =0):
        structure = [each for each in self.structures if each.name == name][0]
        energy_kev = float(self.lineEdit_eng.text())
        #if self.comboBox_unit.currentText() != 'KeV':
        #    energy_kev = 12.398/energy_kev
        structure.lattice.set_E_keV(energy_kev)
        #negative because of the rotation sense
        if mu == None:
            mu = -float(self.lineEdit_mu.text())
        else:
            mu = 0
        phi, gamma, delta = structure.lattice.calculate_diffr_angles(hkl,mu)
        return phi, gamma, delta

    def _compute_angles_dc(self, hkl):
        h, k, l = hkl
        try:
            angles, pars = self.dc.hkl_to_angles(h = h, k = k, l = l, energy=self.energy_keV)
            return angles[-1], angles[2], angles[1]
        except:
            if (abs(h)<0.000001) and (abs(k)<0.000001):
                delta = self.dc.c2th(hkl)
                if delta>90:
                    delta = delta - 90
                return 0, 0, delta
            else:
                print('No solution found for', hkl)
                return np.nan, np.nan, np.nan
        #angles_string = ['mu', 'delta', 'gam', 'eta', 'chi', 'phi']

    def cal_qxqyqz(self):
        if self.lineEdit_H.text()=='' or self.lineEdit_K.text()=='' or self.lineEdit_L.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        hkl = [float(self.lineEdit_H.text()),float(self.lineEdit_K.text()),float(self.lineEdit_L.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        qx,qy,qz = structure.lattice.q(hkl)
        self.lineEdit_qx.setText(str(round(qx,4)))
        self.lineEdit_qy.setText(str(round(qy,4)))
        self.lineEdit_qz.setText(str(round(qz,4)))
        self.lineEdit_q_par.setText(str(round((qx**2+qy**2)**0.5,4)))
        self.cal_q_and_2theta()
        sym_hkls = self._find_sym_hkl(name, structure, round((qx**2+qy**2)**0.5,4), round(qz, 4))
        self.comboBox_sym_HKL.clear()
        self.comboBox_sym_HKL.addItems([str(each) for each in sym_hkls])

    def _find_sym_hkl(self, substrate_tag, structure, q_par, qz):
        all_hkls = [list(each[-2]) for each in self.peaks_dict[substrate_tag]]
        target = []
        for hkl in all_hkls:
            qx, qy, qz_ = structure.lattice.q(hkl)
            q_par_ = round((qx**2+qy**2)**0.5,4)
            qz_ = round(qz_, 4)
            if(abs(qz-qz_) + abs(q_par - q_par_))<0.01:
                target.append(hkl)
        return target

    def cal_hkl(self):
        if self.lineEdit_qx.text()=='' or self.lineEdit_qy.text()=='' or self.lineEdit_qz.text()=='':
            error_pop_up('You must fill all qx qy qz blocks for this calculation!')
            return
        qx_qy_qz = [float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        H,K,L = structure.lattice.HKL(qx_qy_qz)
        self.lineEdit_H.setText(str(round(H,3)))
        self.lineEdit_K.setText(str(round(K,3)))
        self.lineEdit_L.setText(str(round(L,3)))
        self.cal_q_and_2theta()

    def cal_xyz(self):
        if self.lineEdit_a.text()=='' or self.lineEdit_b.text()=='' or self.lineEdit_c.text()=='':
            error_pop_up('You must fill all a b c blocks for this calculation!')
            return
        a_b_c = [float(self.lineEdit_a.text()),float(self.lineEdit_b.text()),float(self.lineEdit_c.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        x, y, z = structure.lattice.RealTM.dot(a_b_c)
        self.lineEdit_x.setText(str(round(x,3)))
        self.lineEdit_y.setText(str(round(y,3)))
        self.lineEdit_z.setText(str(round(z,3)))

    def cal_abc(self):
        if self.lineEdit_x.text()=='' or self.lineEdit_y.text()=='' or self.lineEdit_z.text()=='':
            error_pop_up('You must fill all x y z blocks for this calculation!')
            return
        x_y_z = [float(self.lineEdit_x.text()),float(self.lineEdit_y.text()),float(self.lineEdit_z.text())]
        name = self.comboBox_working_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        a, b, c = structure.lattice.RealTMInv.dot(x_y_z)
        self.lineEdit_a.setText(str(round(a,3)))
        self.lineEdit_b.setText(str(round(b,3)))
        self.lineEdit_c.setText(str(round(c,3)))

    def cal_hkl_in_reference(self):
        #name_work = self.comboBox_working_substrate.currentText()
        #structure_work = [each for each in self.structures if each.name == name_work][0]
        name_reference = self.comboBox_reference_substrate.currentText()
        structure_reference = [each for each in self.structures if each.name == name_reference][0]
        self.cal_qxqyqz()
        qx_qy_qz = np.array([float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())])
        hkl = [round(each,3) for each in structure_reference.lattice.HKL(qx_qy_qz)]
        self.lineEdit_hkl_reference.setText('[{},{},{}]'.format(*hkl))

    def cal_q_and_2theta(self):
        qx_qy_qz = [float(self.lineEdit_qx.text()),float(self.lineEdit_qy.text()),float(self.lineEdit_qz.text())]
        q = self._cal_q(qx_qy_qz)
        #energy_anstrom = float(self.lineEdit_energy.text())
        #if self.comboBox_unit.currentText() == 'KeV':
        #    energy_anstrom = 12.398/energy_anstrom
        assert len(self.structures)!=0, 'No substrate structure available!'
        energy_anstrom = 12.398/self.structures[0].energy_keV
        _2theta = self._cal_2theta(q,energy_anstrom)
        self.lineEdit_q.setText(str(round(q,4)))
        self.lineEdit_2theta.setText(str(round(_2theta,2)))
        self.lineEdit_d.setText(str(round(energy_anstrom/2/np.sin(np.deg2rad(_2theta/2)),2)))

    def _cal_q(self,q):
        q = np.array(q)
        return np.sqrt(q.dot(q))

    def _cal_2theta(self,q,wl):
        return np.rad2deg(np.arcsin(q*wl/4/np.pi))*2

    def _collect_Bragg_reflections(self):
        names = list(self.peaks_dict.keys())
        Bragg_peaks_info = {}
        for name in names:
            structure = None
            for each in self.structures:
                if each.name == name:
                    structure = each
                    break
            peaks = self.peaks_dict[name]
            peaks_unique = []
            for i, peak in enumerate(peaks):
                qxqyqz, _, intensity,_,_ = peak
                HKL = [int(round(each_, 0)) for each_ in structure.lattice.HKL(qxqyqz)]
                peaks_unique.append(HKL+[intensity])
            peaks_unique = np.array(peaks_unique)
            peaks_unique = peaks_unique[peaks_unique[:,-1].argsort()[::-1]]
            HKLs_unique = [str(list(map(int,each[:3]))) for each in peaks_unique]
            Bragg_peaks_info[name] = [list(map(int,each[:3])) for each in peaks_unique]
        self.Bragg_peaks_info = Bragg_peaks_info

    def cal_delta_gamma_angles_for_Bragg_reflections(self):
        self._collect_Bragg_reflections()
        Bragg_peaks_detector_angle_info = {}
        Bragg_peaks_info_allowed = {}
        for name in self.Bragg_peaks_info:
            if name == self.structures[0].name:#only reference substrate will be considered here
                Bragg_peaks_detector_angle_info[name] = []
                Bragg_peaks_info_allowed[name] = []
                for hkl in self.Bragg_peaks_info[name]:
                    #_compute_angles return gamma and delta in a tuple
                    # phi, gamma, delta = self._compute_angles(hkl, name, mu = 0)
                    try:
                        phi, gamma, delta = self._compute_angles_dc(hkl)
                        if not (np.isnan(gamma) or np.isnan(delta)):
                            Bragg_peaks_detector_angle_info[name].append([gamma,delta])
                            Bragg_peaks_info_allowed[name].append(hkl)
                    except:
                        print('Unreachable Bragg reflection:{}'.format(hkl))
            else:
                pass
        self.Bragg_peaks_detector_angle_info = Bragg_peaks_detector_angle_info
        self.Bragg_peaks_info = Bragg_peaks_info_allowed 

    def calc_hkl_dc(self):
        self.energy_keV = float(self.lineEdit_eng.text())
        self.hardware.settings.hardware.energy = self.energy_keV
        angles_string = ['mu', 'delta', 'gam', 'eta', 'chi', 'phi']
        angles = []
        for ag in angles_string:
            angles.append(float(getattr(self, 'lineEdit_{}'.format(ag)).text()))
        hkls, pars = self.dc.angles_to_hkl(angles, energy=self.energy_keV)
        self.lineEdit_H_calc.setText(str(round(hkls[0],3)))
        self.lineEdit_K_calc.setText(str(round(hkls[1],3)))
        self.lineEdit_L_calc.setText(str(round(hkls[2],3)))

    def calc_angs(self):
        self.energy_keV = float(self.lineEdit_eng.text())
        self.hardware.settings.hardware.energy = self.energy_keV
        angles, pars = self.dc.hkl_to_angles(h = float(self.lineEdit_H_calc.text()), k = float(self.lineEdit_K_calc.text()), l = float(self.lineEdit_L_calc.text()), energy=self.energy_keV)
        angles_string = ['mu', 'delta', 'gam', 'eta', 'chi', 'phi']
        for ag, val in zip(angles_string, angles):
            getattr(self, 'lineEdit_{}'.format(ag)).setText(str(round(val,3)))

    def _calc_angs_e_scan(self, H, K, L, E_list):
        results = ['energy scan for HKL = {}'.format([H,K,L])]
        results.append('\t'.join(['E(keV)','mu', 'delta', 'gam', 'eta', 'chi', 'phi']))
        for E in E_list:
            self.hardware.settings.hardware.energy = E
            angles, pars = self.dc.hkl_to_angles(h = H, k = K, l = L, energy = E)
            results.append('\t'.join([str(round(E,4))]+[str(round(each,2)) for each in angles]))
        self.hardware.settings.hardware.energy = self.energy_keV
        return results

    def _calc_angs_l_scan(self, H, K, L_list, E):
        results = ['L scan for HK = {} at energy of {} (keV)'.format([H,K], E)]
        results.append('\t'.join(['l','mu', 'delta', 'gam', 'eta', 'chi', 'phi']))
        self.hardware.settings.hardware.energy = E
        for l in L_list:
            angles, pars = self.dc.hkl_to_angles(h = H, k = K, l = l, energy = E)
            results.append('\t'.join([str(round(l,2))]+[str(round(each,2)) for each in angles]))
        return results

    def calc_angs_in_scan(self, scan_type = 'l'):
        if scan_type == 'l':
            H, K = eval(self.lineEdit_HK.text())
            ls = np.arange(float(self.lineEdit_L_begin.text()),float(self.lineEdit_L_end.text()), float(self.lineEdit_L_step.text()))
            self.plainTextEdit_cross_points_info.setPlainText('\n'.join(self._calc_angs_l_scan(H = H, K = K, L_list = ls, E = self.energy_keV)))
            if self.checkBox_live.isChecked():
                self.current_line = 0
                self.timer_l_scan.start(float(self.lineEdit_scan_rate.text())*1000)
        elif scan_type == 'energy':
            H, K, L = eval(self.lineEdit_HKL.text())
            es = np.arange(float(self.lineEdit_E_begin.text()),float(self.lineEdit_E_end.text()), float(self.lineEdit_E_step.text()))
            self.plainTextEdit_cross_points_info.setPlainText('\n'.join(self._calc_angs_e_scan(H = H, K = K,  L = L, E_list = es)))



