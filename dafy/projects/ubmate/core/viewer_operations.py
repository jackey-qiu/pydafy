import numpy as np
import copy
from dafy.projects.ubmate.core.rsp_pkg import reciprocal_space_plot as rsplt

class ViewerOperations(object):

    def __init__(self):
        pass

    def draw_ctrs(self):
        self.widget.canvas.figure.clear()
        num_plot = len(self.peaks_in_zoomin_viewer)
        resolution = 300
        l = np.linspace(0,self.qz_lim_high,resolution)
        intensity_dict = {'total':[self.widget.canvas.figure.add_subplot(num_plot+1,1,num_plot+1),l,np.zeros(resolution),[],[],[]]}
        self.peak_info_selected_rod = {}
        for i in range(num_plot):
            name = list(self.peaks_in_zoomin_viewer.keys())[i]
            self.peak_info_selected_rod[name] = []
            ax = self.widget.canvas.figure.add_subplot(num_plot+1,1,i+1)
            # ax.set_yscale('log')
            intensity_dict[name] = [ax]
            structure = [each_structure for each_structure in self.structures if each_structure.name == name][0]
            I = np.zeros(resolution)
            #l = np.linspace(0,self.qz_lim_high,300)
            l_text = []
            I_text = []
            text = []
            for each_peak in self.peaks_in_zoomin_viewer[name]:
                hkl = structure.lattice.HKL(each_peak)
                text.append([int(round(each,0)) for each in hkl])
                if structure.name != self.structures[0].name:
                    I_this_point = structure.lattice.I(hkl)/10#scaled by 100 to consider for thin film
                else:
                    I_this_point = structure.lattice.I(hkl)
                l_wrt_main_substrate = self.structures[0].lattice.HKL(each_peak)[-1]
                l_text.append(l_wrt_main_substrate)
                self.peak_info_selected_rod[name].append([l_wrt_main_substrate,str(tuple([int(round(each_,0)) for each_ in hkl]))])
                # print(name, hkl, I_this_point)
                #Gaussian expansion, assume sigma = 0.2
                sigma = 0.06
                I_ = I_this_point/(sigma*(2*np.pi)**0.5)*np.exp(-0.5*((l-l_wrt_main_substrate)/sigma)**2)
                I = I + I_
                I_text.append(I_this_point/(sigma*(2*np.pi)**0.5))
            intensity_dict[name].append(l)
            intensity_dict[name].append(I)
            intensity_dict[name].append(l_text)
            intensity_dict[name].append(I_text)
            intensity_dict[name].append(text)
            intensity_dict['total'][2] = intensity_dict['total'][2]+I
            intensity_dict['total'][3]+=l_text
            intensity_dict['total'][4]+=I_text
            intensity_dict['total'][5]+=text
        for each in intensity_dict:
            ax, l, I,l_text, I_text, text = intensity_dict[each]
            ax.plot(l,I,label = each)
            for i in range(len(text)):
                ax.text(l_text[i],I_text[i],str(text[i]),rotation ='vertical')
            ax.set_title(each)
        self.widget.canvas.figure.tight_layout()
        self.widget.canvas.draw()

    def pan_view(self,signs = [1,1,1], widget = 'widget_glview'):
        value = 0.5
        pan_values = list(np.array(signs)*value)
        getattr(self,widget).pan(*pan_values)
        #self.widget_glview.pan(*pan_values)
 
    def update_camera_position(self,widget_name = 'widget_glview', angle_type="azimuth", angle=0):
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def azimuth_0(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=0)

    def azimuth_90(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=90)

    def azimuth_180(self):
        self.update_camera_position(angle_type="elevation", angle=0)
        self.update_camera_position(angle_type="azimuth", angle=180)

    def azimuth_0_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=0)

    def azimuth_90_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=90)

    def azimuth_180_2(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=0)
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="azimuth", angle=180)

    def elevation_90(self):
        self.update_camera_position(widget_name = 'widget_real_space',angle_type="elevation", angle=90)

    def show_structure(self, widget_name = 'widget_glview'):
        getattr(self,widget_name).show_structure()
        if widget_name == 'widget_glview':
            self.extract_cross_point_info()
        self.update_camera_position(widget_name=widget_name, angle_type="elevation", angle=90)
        self.update_camera_position(widget_name=widget_name, angle_type="azimuth", angle=270)

    def extract_Bragg_peaks_along_a_rod(self, substrate_name, hkl_one_Bragg_peak):
        bragg_peaks_list = []
        structure = None
        for each in self.structures:
            if each.name == substrate_name:
                structure = each
                break
        hkl = hkl_one_Bragg_peak
        #hk.append(0)
        qx, qy,_ = structure.lattice.RecTM.dot(hkl)
        for each in self.peaks_dict[substrate_name]:
            if abs(each[0][0]-qx)<0.05 and abs(each[0][1]-qy)<0.05:
                bragg_peaks_list.append(tuple(each[3]))
        return bragg_peaks_list

    def _show_specified_items(self, name = '', sphere_key = [], line_key = []):
        for each in self.widget_glview.line_items_according_to_substrate_and_hkl[name]:
            if each in line_key:
                self.widget_glview.line_items_according_to_substrate_and_hkl[name][each].show()
                [each_item.show() for each_item in self.widget_glview.recreated_items_according_to_substrate_and_hkl[name][each]]
        for each in self.widget_glview.sphere_items_according_to_substrate_and_hkl[name]:
            if each in sphere_key:
                self.widget_glview.sphere_items_according_to_substrate_and_hkl[name][each].show()

    def _hide_all_items(self):
        for name in self.widget_glview.line_items_according_to_substrate_and_hkl:
            for each in self.widget_glview.line_items_according_to_substrate_and_hkl[name]:
                self.widget_glview.line_items_according_to_substrate_and_hkl[name][each].hide()
                [each_item.hide() for each_item in self.widget_glview.recreated_items_according_to_substrate_and_hkl[name][each]]
            for each in self.widget_glview.sphere_items_according_to_substrate_and_hkl[name]:
                self.widget_glview.sphere_items_according_to_substrate_and_hkl[name][each].hide()

    def _show_all_items(self):
        for name in self.widget_glview.line_items_according_to_substrate_and_hkl:
            for each in self.widget_glview.line_items_according_to_substrate_and_hkl[name]:
                self.widget_glview.line_items_according_to_substrate_and_hkl[name][each].show()
                [each_item.show() for each_item in self.widget_glview.recreated_items_according_to_substrate_and_hkl[name][each]]
            for each in self.widget_glview.sphere_items_according_to_substrate_and_hkl[name]:
                self.widget_glview.sphere_items_according_to_substrate_and_hkl[name][each].show()

    def extract_peaks_in_zoom_viewer(self,symHKLs = []):
        structure = None
        for each in self.structures:
            if each.name == self.comboBox_names.currentText():
                structure = each
                break
        hkl = list(eval(self.comboBox_HKs.currentText()))
        if self.radioButton_all_rods.isChecked():
            self._show_all_items()
        else:
            self._hide_all_items()
            sphere_key = self.extract_Bragg_peaks_along_a_rod(self.comboBox_names.currentText(), hkl)
            self._show_specified_items(name = self.comboBox_names.currentText(), sphere_key = sphere_key, line_key = [tuple(hkl)])
        #hk.append(0)
        qx, qy,_ = structure.lattice.RecTM.dot(hkl)
        qxs_sym, qys_sym = [], []
        if type(symHKLs)==list:
            for each in symHKLs:
                if each!=self.comboBox_bragg_peak.currentText():
                    qx_, qy_, _ = structure.lattice.RecTM.dot(list(eval(each)))
                    qxs_sym.append(qx_)
                    qys_sym.append(qy_)
        # print('HK and QX and Qy',hk,qx,qy)
        peaks_temp = []
        text_temp = []
        self.peaks_in_zoomin_viewer = {}
        for key in self.peaks_dict:
            for each in self.peaks_dict[key]:
                if abs(each[0][0]-qx)<0.05 and abs(each[0][1]-qy)<0.05:
                    each_ = copy.deepcopy(each)
                    each_[0][0] = 0
                    each_[0][1] = 0
                    # each_[0][1] = each_[0][1]-0.5

                    # each_[-1] = 0.1
                    #print(each_)
                    peaks_temp.append(each_)
                    structure_temp = [each_structure for each_structure in self.structures if each_structure.name == key]#should be one item list
                    assert len(structure_temp)==1,'duplicate structures'
                    HKL_temp = [int(round(each_item,0)) for each_item in structure_temp[0].lattice.HKL(each[0])]
                    text_temp.append(each_[0]+['{}({})'.format(key,HKL_temp)])
                    if key in self.peaks_in_zoomin_viewer:
                        self.peaks_in_zoomin_viewer[key].append(each[0])
                    else:
                        self.peaks_in_zoomin_viewer[key] = [each[0]]
                    #print(each_)
                else:
                    pass
                    # print(each[0]) 
        # print('peaks',peaks_temp)
        self.widget_glview_zoomin.clear()
        self.widget_glview_zoomin.spheres = peaks_temp
        self.widget_glview_zoomin.texts = text_temp
        self.widget_glview.text_selected_rod = list(self.widget_glview.RM.dot([qx,qy,self.qz_lims[1]]))+['x']
        self.widget_glview.text_sym_rods = [list(self.widget_glview.RM.dot([*each,self.qz_lims[1]]))+['S'] for each in zip(qxs_sym, qys_sym)]

        self.widget_glview.update_text_item_selected_rod()
        # self.widget_glview.update_text_item_sym_rods()

        self.widget_glview_zoomin.show_structure()

    def update_draw_limits(self):
        q_inplane_lim = self.config.get('Plot', 'q_inplane_lim')
        qx_lim_low = self.config.get('Plot', 'qx_lim_low')
        qx_lim_high = self.config.get('Plot', 'qx_lim_high')
        qy_lim_low = self.config.get('Plot', 'qy_lim_low')
        qy_lim_high = self.config.get('Plot', 'qy_lim_high')
        qz_lim_low = self.config.get('Plot', 'qz_lim_low')
        qz_lim_high = self.config.get('Plot', 'qz_lim_high')
        q_mag_lim_low = self.config.get('Plot', 'q_mag_lim_low')
        q_mag_lim_high = self.config.get('Plot', 'q_mag_lim_high')

        self.q_inplane_lim = None if q_inplane_lim == 'None' else float(q_inplane_lim)
        qx_lim_low = None if qx_lim_low == 'None' else float(qx_lim_low)
        qx_lim_high = None if qx_lim_high == 'None' else float(qx_lim_high)
        qy_lim_low = None if qy_lim_low == 'None' else float(qy_lim_low)
        qy_lim_high = None if qy_lim_high == 'None' else float(qy_lim_high)
        qz_lim_low = None if qz_lim_low == 'None' else float(qz_lim_low)
        qz_lim_high = None if qz_lim_high == 'None' else float(qz_lim_high)
        q_mag_lim_low = None if q_mag_lim_low == 'None' else float(q_mag_lim_low)
        q_mag_lim_high = None if q_mag_lim_high == 'None' else float(q_mag_lim_high)

        if qz_lim_high == None:
            self.qz_lim_high = 5
        else:
            self.qz_lim_high = qz_lim_high

        self.qx_lims = [qx_lim_low, qx_lim_high]
        if(self.qx_lims[0] == None or self.qx_lims[1] == None):
            self.qx_lims = None
        self.qy_lims = [qy_lim_low, qy_lim_high]
        if(self.qy_lims[0] == None or self.qy_lims[1] == None):
            self.qy_lims = None
        self.qz_lims = [qz_lim_low, qz_lim_high]
        if(self.qz_lims[0] == None or self.qz_lims[1] == None):
            self.qz_lims = None
        self.mag_q_lims = [q_mag_lim_low, q_mag_lim_high]
        if(self.mag_q_lims[0] == None or self.mag_q_lims[1] == None):
            self.mag_q_lims = None

    def update_draw_limits_new(self):
        q_inplane_lim = self.widget_config.par.names['Plot'].names['q_inplane_lim'].value()
        qx_lim_low = self.widget_config.par.names['Plot'].names['qx_lim_low'].value()
        qx_lim_high = self.widget_config.par.names['Plot'].names['qx_lim_high'].value()
        qy_lim_low = self.widget_config.par.names['Plot'].names['qy_lim_low'].value()
        qy_lim_high = self.widget_config.par.names['Plot'].names['qy_lim_high'].value()
        qz_lim_low = self.widget_config.par.names['Plot'].names['qz_lim_low'].value()
        qz_lim_high = self.widget_config.par.names['Plot'].names['qz_lim_high'].value()
        q_mag_lim_low = self.widget_config.par.names['Plot'].names['q_mag_lim_low'].value()
        q_mag_lim_high = self.widget_config.par.names['Plot'].names['q_mag_lim_high'].value()

        self.q_inplane_lim = None if q_inplane_lim == 'None' else float(q_inplane_lim)
        qx_lim_low = None if qx_lim_low == 'None' else float(qx_lim_low)
        qx_lim_high = None if qx_lim_high == 'None' else float(qx_lim_high)
        qy_lim_low = None if qy_lim_low == 'None' else float(qy_lim_low)
        qy_lim_high = None if qy_lim_high == 'None' else float(qy_lim_high)
        qz_lim_low = None if qz_lim_low == 'None' else float(qz_lim_low)
        qz_lim_high = None if qz_lim_high == 'None' else float(qz_lim_high)
        q_mag_lim_low = None if q_mag_lim_low == 'None' else float(q_mag_lim_low)
        q_mag_lim_high = None if q_mag_lim_high == 'None' else float(q_mag_lim_high)

        if qz_lim_high == None:
            self.qz_lim_high = 5
        else:
            self.qz_lim_high = qz_lim_high

        self.qx_lims = [qx_lim_low, qx_lim_high]
        if(self.qx_lims[0] == None or self.qx_lims[1] == None):
            self.qx_lims = None
        self.qy_lims = [qy_lim_low, qy_lim_high]
        if(self.qy_lims[0] == None or self.qy_lims[1] == None):
            self.qy_lims = None
        self.qz_lims = [qz_lim_low, qz_lim_high]
        if(self.qz_lims[0] == None or self.qz_lims[1] == None):
            self.qz_lims = None
        self.mag_q_lims = [q_mag_lim_low, q_mag_lim_high]
        if(self.mag_q_lims[0] == None or self.mag_q_lims[1] == None):
            self.mag_q_lims = None

    def prepare_objects_for_render(self):
        self.peaks = []
        self.peaks_dict = {}
        # self.peaks_HKLs_dict = {}
        self.HKLs_dict = {}
        self.rods_dict = {}
        self.rods = []
        self.grids = []
        self.axes = []
        self.unit_cell_edges = {}
        space_plots = []
        names = []
        for i in range(len(self.structures)):
            struc = self.structures[i]
            names.append(struc.name)
            space_plots.append(rsplt.space_plot(struc.lattice))

            if(struc.plot_peaks):
                peaks_, HKLs_ = space_plots[i].get_peaks(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, mag_q_lims=self.mag_q_lims, color=struc.color, substrate_name = struc.name)
                if len(peaks_)>0:
                    for each in peaks_:
                        self.peaks.append(each)
                    
                self.peaks_dict[struc.name] = peaks_
                # self.peaks_HKLs_dict[struc.name] = HKLs_
                
            if(struc.plot_rods):
                rods_, HKLs = space_plots[i].get_rods(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, color=struc.color)
                if len(rods_)>0:
                    for each in rods_:
                        self.rods.append(each)
                    self.HKLs_dict[struc.name] = HKLs
                    self.rods_dict[struc.name] = rods_
            if(struc.plot_grid):
                grids_ = space_plots[i].get_grids(qx_lims=self.qx_lims, qy_lims=self.qy_lims, qz_lims=self.qz_lims, q_inplane_lim=self.q_inplane_lim, color=struc.color)
                if len(grids_)>0:
                    for each in grids_:
                        self.grids.append(each)
            if struc.plot_unitcell:
                self.unit_cell_edges[struc.name] = space_plots[i].get_unit_cell_edges(color = struc.color)

        if(self.plot_axes):
            q1 = self.structures[0].lattice.q([1,0,0])
            q2 = self.structures[0].lattice.q([0,1,0])
            q3 = self.structures[0].lattice.q([0,0,1])
            self.axes.append([[0,0,0],q1,0.1,0.2,(0,0,0,0.8)])
            self.axes.append([[0,0,0],q2,0.1,0.2,(0,0,0,0.8)])
            self.axes.append([[0,0,0],q3,0.1,0.2,(0,0,0,0.8)])
            #composer
            qx_min, qy_min = 10000, 10000
            for each in self.rods:
                qx_, qy_ = each[0][0:2]
                if qx_<qx_min:
                    qx_min = qx_
                if qy_<qy_min:
                    qy_min = qy_
            qx_min, qy_min = 0, self.structures[0].lattice.k0
            self.axes.append([[0,-qy_min,0],[1,-qy_min,0],0.1,0.2,(0,0,1,0.8)])
            self.axes.append([[0,-qy_min,0],[0,1-qy_min,0],0.1,0.2,(0,1,0,0.8)])
            self.axes.append([[0,-qy_min,0],[0,-qy_min,1],0.1,0.2,(1,0,0,0.8)])
        if self.checkBox_ewarld.isChecked():
            self.widget_glview.ewarld_sphere = [[0,-self.structures[0].lattice.k0,0],(0,0,1,0.3),self.structures[0].lattice.k0]
        else:
            self.widget_glview.ewarld_sphere = []
        self.comboBox_names.addItems(names)
        self.widget_glview.spheres = self.peaks
        self.widget_glview.lines = self.rods
        self.widget_glview.lines_dict = self.rods_dict
        self.widget_glview.HKLs_dict = self.HKLs_dict
        self.widget_glview.grids = self.grids
        self.widget_glview.arrows = self.axes
        self.widget_glview.unit_cell_edges = self.unit_cell_edges
        self.widget_glview.clear()

    def simulate_image_Bragg_reflections(self):
        self.cal_delta_gamma_angles_for_Bragg_reflections()
        pilatus_size = [float(self.lineEdit_detector_hor.text()), float(self.lineEdit_detector_ver.text())]
        pixel_size = [float(self.lineEdit_pixel.text())]*2
        try:
            prim_beam = eval(self.lineEdit_prim_beam_pos.text())
        except:
            print('Evaluation error for :{}'.format(self.lineEdit_prim_beam_pos.text()))
            prim_beam = None
        self.widget_mpl.canvas.figure.clear()
        ax = self.widget_mpl.canvas.figure.add_subplot(1,1,1)
        # ax.format_coord = Formatter(None)
        self.widget_glview.calculate_index_on_pilatus_image_from_angles(
             pilatus_size = pilatus_size,
             pixel_size = pixel_size,
             distance_sample_detector = float(self.lineEdit_sample_detector_distance.text()),
             angle_info = self.Bragg_peaks_detector_angle_info,
             primary_beam_pos = prim_beam
        )
        ax.imshow(self.widget_glview.cal_simuated_2d_pixel_image_Bragg_peaks(pilatus_size = pilatus_size, pixel_size = pixel_size, gaussian_sim = self.checkBox_gaussian_sim.isChecked()))
        
        pos_all = []
        for each in self.widget_glview.pixel_index_of_Bragg_reflections:
            if each == self.structures[0].name:
                for i in range(len(self.widget_glview.pixel_index_of_Bragg_reflections[each])):
                    pos = self.widget_glview.pixel_index_of_Bragg_reflections[each][i]
                    if pos in pos_all:
                        #pass
                        print('The position for ',self.Bragg_peaks_info[each][i], 'is already occupied!')
                    else:
                        print('Pin the position of ..{}'.format(pos))
                        pos_all.append(pos)
                        hkl = self.Bragg_peaks_info[each][i]
                        if len(pos)!=0:
                            print('Pin the position of {}'.format(hkl))
                            ax.text(*pos[::-1],'x',
                                horizontalalignment='center',
                                verticalalignment='center',color = 'r')
                            ax.text(*(list(pos[::-1]+np.array([20,20]))), '{}{}'.format(each,hkl),color = 'y',rotation = 'vertical',fontsize=8)
            else:
                pass
        
        ax.text(*self.widget_glview.primary_beam_position,'+',color = 'r')
        ax.text(*(self.widget_glview.primary_beam_position+np.array([50,-50])),'Primary Beam Pos',color = 'r')
        if self.checkBox_trajactory.isChecked():
            self._cal_trajactory_pos(ax)
        ax.grid(False)
        self.widget_mpl.fig.tight_layout()
        self.widget_mpl.canvas.draw()
        self.ax_simulation = ax

    def draw_real_space(self):
        super_cell_size = [self.spinBox_repeat_x.value(), self.spinBox_repeat_y.value(), self.spinBox_repeat_z.value()]
        name = self.comboBox_substrate.currentText()
        structure = [each for each in self.structures if each.name == name][0]
        self.widget_real_space.RealRM = structure.lattice.RealTM
        self.widget_real_space.show_structure(structure.lattice.basis, super_cell_size)

    def simulate_image(self):
        pilatus_size = [float(self.lineEdit_detector_hor.text()), float(self.lineEdit_detector_ver.text())]
        pixel_size = [float(self.lineEdit_pixel.text())]*2
        self.widget_mpl.canvas.figure.clear()
        ax = self.widget_mpl.canvas.figure.add_subplot(1,1,1)
        ax.format_coord = Formatter(self)
        try:
            prim_beam = eval(self.lineEdit_prim_beam_pos.text())
        except:
            print('Evaluation error for :{}'.format(self.lineEdit_prim_beam_pos.text()))
            prim_beam = None
        self.widget_glview.calculate_index_on_pilatus_image_from_cross_points_info(
             pilatus_size = pilatus_size,
             pixel_size = pixel_size,
             distance_sample_detector = float(self.lineEdit_sample_detector_distance.text()),
             primary_beam_pos = prim_beam)
        ax.imshow(self.widget_glview.cal_simuated_2d_pixel_image(pilatus_size = pilatus_size, pixel_size = pixel_size, gaussian_sim = self.checkBox_gaussian_sim.isChecked()))
        for each in self.widget_glview.pixel_index_of_cross_points:
            for i in range(len(self.widget_glview.pixel_index_of_cross_points[each])):
                pos = self.widget_glview.pixel_index_of_cross_points[each][i]
                hkl = self.widget_glview.cross_points_info_HKL[each][i]
                if len(pos)!=0:
                    if pos not in self.trajactory_pos:
                        self.trajactory_pos.append(pos)
                    ax.text(*pos[::-1],'x',
                            horizontalalignment='center',
                            verticalalignment='center',color = 'r')
                    ax.text(*(list(pos[::-1]+np.array([20,20]))), '{}{}'.format(each,hkl),color = 'y',rotation = 'vertical',fontsize=8)
        ax.text(*self.widget_glview.primary_beam_position,'x',color = 'w')
        ax.text(*(self.widget_glview.primary_beam_position+np.array([50,-50])),'Primary Beam Pos',color = 'r')
        if self.timer_spin_sample.isActive():
            for each in self.trajactory_pos:
                ax.text(*each[::-1],'.',
                        horizontalalignment='center',
                        verticalalignment='center',color = 'w',fontsize = 15)
        ax.grid(False)
        self.widget_mpl.fig.tight_layout()
        self.widget_mpl.canvas.draw()
        self.ax_simulation = ax

class Formatter(object):
    def __init__(self, main_gui):
        self.main_gui = main_gui

    def __call__(self, x, y):
        hkl, phi, delta, gam, chi, eta = self.main_gui.get_simulated_hkl(x, y)
        return 'hkl = {}, x={:.01f}, y={:.01f}, phi={:.01f}, delta = {:.01f}, gam = {:.01f}, chi = {:.01f}, eta = {:.01f}'.format(hkl, x, y, phi, delta, gam, chi, eta)
