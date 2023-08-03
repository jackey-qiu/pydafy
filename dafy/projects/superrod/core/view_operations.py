from io import StringIO
import traceback
from PyQt5.QtWidgets import QMessageBox

class ViewOperations(object):
    def parallel_projection(self):
        self.widget_edp.opts['distance'] = 2000
        self.widget_edp.opts['fov'] = 1
        # self.widget_msv_top.opts['distance'] = 2000
        # self.widget_msv_top.opts['fov'] = 1
        self.update_structure_view()

    def projective_projection(self):
        self.widget_edp.opts['distance'] = 25
        self.widget_edp.opts['fov'] = 60
        # self.widget_msv_top.opts['distance'] = 25
        # self.widget_msv_top.opts['fov'] = 60
        self.update_structure_view()

    def pan_msv_view(self):
        value = int(self.spinBox_pan_pixel.text())
        self.widget_edp.pan(value*int(self.checkBox_x.isChecked()),value*int(self.checkBox_y.isChecked()),value*int(self.checkBox_z.isChecked()))

    def update_camera_position(self,widget_name = 'widget_edp', angle_type="azimuth", angle=0):
        getattr(self,widget_name).setCameraPosition(pos=None, distance=None, \
            elevation=[None,angle][int(angle_type=="elevation")], \
                azimuth=[None,angle][int(angle_type=="azimuth")])

    def azimuth_0(self):
        self.update_camera_position(angle_type="azimuth", angle=0)

    def azimuth_90(self):
        self.update_camera_position(angle_type="azimuth", angle=90)

    def start_spin(self):
        self.timer_spin_msv.start(100)

    def stop_spin(self):
        self.timer_spin_msv.stop()

    def spin_msv(self):
        #if self.azimuth > 360:
            
        self.update_camera_position(angle_type="azimuth", angle=self.azimuth_angle)
        self.azimuth_angle = self.azimuth_angle + 1


    def elevation_0(self):
        self.update_camera_position(angle_type="elevation", angle=0)

    def elevation_90(self):
        self.update_camera_position(angle_type="elevation", angle=90)

    def init_structure_view(self):
        try:
            domain_tag = int(self.spinBox_domain.text())
        except:
            domain_tag = 0
        size_domain = len(self.model.script_module.sample.domain)
        if size_domain<(1+domain_tag):
            domain_tag = size_domain -1
        else:
            pass
        # self.widget_edp.items = []
        # self.widget_msv_top.items = []
        self.widget_edp.abc = [self.model.script_module.sample.unit_cell.a,self.model.script_module.sample.unit_cell.b,self.model.script_module.sample.unit_cell.c]
        self.widget_edp.T = self.model.script_module.sample.unit_cell.lattice.RealTM
        self.widget_edp.T_INV = self.model.script_module.sample.unit_cell.lattice.RealTMInv
        self.widget_edp.super_cell_size = eval(self.lineEdit_super_cell.text())
        self.widget_edp.show_bond_length = self.checkBox_label.isChecked()
        # self.widget_msv_top.abc = self.widget_edp.abc
        xyz = self.model.script_module.sample.extract_xyz_top(domain_tag, num_of_atomic_layers = self.spinBox_layers.value(), use_sym = self.checkBox_symmetry.isChecked(),size = eval(self.lineEdit_super_cell.text()))
        self.widget_edp.show_structure(xyz)
        try:
            azimuth = self.widget_edp.opts['azimuth']
            elevation = self.widget_edp.opts['elevation']
        except:
            azimuth, elevation = 0, 0
        self.update_camera_position(widget_name = 'widget_edp', angle_type="azimuth", angle=azimuth)
        self.update_camera_position(widget_name = 'widget_edp', angle_type = 'elevation', angle = elevation)
        self.update_electron_density_profile()

        # xyz,_ = self.model.script_module.sample.extract_xyz_top(domain_tag)
        # self.widget_msv_top.show_structure(xyz)
        # self.update_camera_position(widget_name = 'widget_msv_top', angle_type="azimuth", angle=0)
        # self.update_camera_position(widget_name = 'widget_msv_top', angle_type = 'elevation', angle = 90)
        """
        try:
            xyz,_ = self.model.script_module.sample.extract_xyz_top(domain_tag)
            self.widget_msv_top.show_structure(xyz)
            self.update_camera_position(widget_name = 'widget_msv_top', angle_type="azimuth", angle=0)
            self.update_camera_position(widget_name = 'widget_msv_top', angle_type = 'elevation', angle = 90)
        except:
            pass
        """

    def update_structure_view(self, compile = True):
        if hasattr(self.model.script_module,"model_type"):
            if getattr(self.model.script_module,"model_type")=="ctr":
                pass
            else:
                return
        else:
            pass
        try:
            if self.spinBox_domain.text()=="":
                domain_tag = 0
            else:
                domain_tag = int(self.spinBox_domain.text())
            size_domain = len(self.model.script_module.sample.domain)
            if size_domain<(1+domain_tag):
                domain_tag = size_domain -1
            else:
                pass        
            xyz = self.model.script_module.sample.extract_xyz_top(domain_tag, num_of_atomic_layers = self.spinBox_layers.value(),use_sym = self.checkBox_symmetry.isChecked(),size = eval(self.lineEdit_super_cell.text()))
            if self.run_fit.running or (not compile): 
                self.widget_edp.update_structure(xyz)
            else:
                self.widget_edp.clear()
                #self.widget_edp.items = []
                self.widget_edp.abc = [self.model.script_module.sample.unit_cell.a,self.model.script_module.sample.unit_cell.b,self.model.script_module.sample.unit_cell.c]
                self.widget_edp.T = self.model.script_module.sample.unit_cell.lattice.RealTM
                self.widget_edp.T_INV = self.model.script_module.sample.unit_cell.lattice.RealTMInv
                self.widget_edp.super_cell_size = eval(self.lineEdit_super_cell.text())
                self.widget_edp.show_bond_length = self.checkBox_label.isChecked()
                self.widget_edp.show_structure(xyz)
            #let us also update the eden profile
            self.update_electron_density_profile()

            """
            try:
                xyz, _ = self.model.script_module.sample.extract_xyz_top(domain_tag)
                self.widget_msv_top.update_structure(xyz)
            except:
                pass
            """
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            _ = QMessageBox.question(self, "",'Runtime error message:\n{}'.format(str(val)), QMessageBox.Ok)