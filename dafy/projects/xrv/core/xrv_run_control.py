import os
import numpy as np
import pandas as pd
from dafy.core.util.path import *
from dafy.core.EnginePool.GrainAnalysisEnginePool import cal_strain_and_grain
from dafy.core.FilterPool.DataFilterPool import create_mask, merge_data
from dafy.core.EnginePool.FitEnginePool import model,Reciprocal_Space_Mapping, XRD_Peak_Fitting, background_subtraction_single_img
from dafy.core.util.XRD_tools import reciprocal_space_v3 as rsp
from dafy.core.util.UtilityFunctions import image_generator, scan_generator, extract_vars_from_config, edf_image_loader, nexus_image_loader
from dafy.core.util.DebugFunctions import error_pop_up

class RunApp(object):
    def __init__(self, use_q_mapping = True, filter_order_object = None, img_loader = 'nexus_image_loader'):
        self.stop = True
        self.conf_file = None
        self.data = {}
        self.model = model
        self.data_path = str(user_data_path)
        self.use_q_mapping = use_q_mapping
        self.filter_order_object = filter_order_object
        self.img_loader_object = img_loader

    def run(self, config):
        self.conf_file = config
        #extract global vars from config
        self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
        for each in self.kwarg_global:
            setattr(self,each,self.kwarg_global[each])

        #pars lib for everything else
        self.kwarg_visulization = extract_vars_from_config(self.conf_file, section_var ='Visulization')
        self.kwarg_film = extract_vars_from_config(self.conf_file, section_var ='Film_Lattice')
        self.kwarg_data = extract_vars_from_config(self.conf_file, section_var ='Data_Storage')
        self.kwarg_peak_fit = extract_vars_from_config(self.conf_file, section_var = 'Peak_Fit')
        self.kwarg_rsp = extract_vars_from_config(self.conf_file, section_var = 'Reciprocal_Mapping')
        self.kwarg_image = extract_vars_from_config(self.conf_file, section_var = 'Image_Loader')
        self.kwarg_mask = extract_vars_from_config(self.conf_file,section_var = 'Mask')

        #recal crop_boundary and cen
        self.crop_boundary = {"ver":[max([self.cen[0]-self.clip_width['ver'],0]),min([self.cen[0]+self.clip_width['ver']+1,self.kwarg_global['dim_detector'][0]])],
                        "hor":[max([self.cen[1]-self.clip_width['hor'],0]),min([self.cen[1]+self.clip_width['hor']+1,self.kwarg_global['dim_detector'][1]])]}
        self.cen_clip = [int(abs(self.crop_boundary['ver'][1]-self.crop_boundary['ver'][0])/2), int(abs(self.crop_boundary['hor'][1]-self.crop_boundary['hor'][0])/2)]       
        offset = [self.cen_clip[0] - self.clip_width['ver'], self.cen_clip[1] - self.clip_width['hor']]
        #apply offset now
        self.cen_clip = [self.cen_clip[0] - abs(offset[0]),self.cen_clip[1] - abs(offset[1])]

        #data file
        for key in self.data_keys:
            self.data[key]=[]

        #init peak fit, bkg subtraction and reciprocal space and image loader instance
        self.bkg_sub = background_subtraction_single_img(self.cen_clip, self.conf_file, sections = ['Background_Subtraction'])
        self.rsp_instance = Reciprocal_Space_Mapping(img =None, cen=self.cen_clip, kwarg = self.kwarg_rsp)
        self.peak_fitting_instance = XRD_Peak_Fitting(img = None, cen=self.cen_clip, kwarg = self.kwarg_peak_fit, use_q_mapping = self.use_q_mapping, rsp_mapping = self.rsp_instance)
        self.img_loader = eval(self.img_loader_object)(crop_boundary = self.crop_boundary, kwarg = self.kwarg_image)
        self.create_mask_new = create_mask(kwarg = self.kwarg_mask)
        self.lattice_skin = rsp.lattice.from_cif(os.path.join(DaFy_path, 'resources','cif',"{}".format(self.kwarg_film['film_material_cif'])),
                                            HKL_normal=self.kwarg_film['film_hkl_normal'],\
                                            HKL_para_x=self.kwarg_film['film_hkl_x'],\
                                            E_keV=self.rsp_instance.e_kev, offset_angle=0)

        self._scans = scan_generator(scans = self.scan_nos)
        self._images = image_generator(self._scans,self.img_loader,self.rsp_instance,self.peak_fitting_instance,self.create_mask_new, use_q_mapping = self.use_q_mapping, filter_order_object = self.filter_order_object)

    def run_script(self, bkg_intensity = 0):
        try:
            img = next(self._images)
            if hasattr(self,'current_scan_number'):
                if self.current_scan_number!=self.img_loader.scan_number:
                    self.save_data_file(self.data_path)
                    self.current_scan_number = self.img_loader.scan_number
            else:
                setattr(self,'current_scan_number',self.img_loader.scan_number)
            self.current_frame = self.img_loader.current_frame_number
            self.img = img
            good_check = self.peak_fitting_instance.reset_fit(img, check = True, first_frame = self.current_frame==0)
            if good_check[0]:
                self.bkg_sub.fit_background(None, img, plot_live = False, freeze_sf = True)
                self.data = merge_data(self.data, self.img_loader, self.peak_fitting_instance, self.bkg_sub, self.kwarg_global, tweak = False)
                self.data = cal_strain_and_grain(self.data,HKL = self.kwarg_film['film_hkl_bragg_peak'][0], lattice = self.lattice_skin)
                self.data['bkg'].append(bkg_intensity)
            else:
                error_pop_up(good_check[1]+'\nPress recenter to continue!!')
            return True
        except StopIteration:
            self.save_data_file(self.data_path)
            return False

    def run_update(self, bkg_intensity = 0):
        self.peak_fitting_instance.reset_fit(self.img, check = False)
        self.bkg_sub.fit_background(None, self.img, plot_live = False, freeze_sf = True)
        self.data = merge_data(self.data, self.img_loader, self.peak_fitting_instance, self.bkg_sub, self.kwarg_global, tweak = True)
        self.data = cal_strain_and_grain(self.data,HKL = self.kwarg_film['film_hkl_bragg_peak'][0], lattice = self.lattice_skin)
        self.data['bkg'][-1] = bkg_intensity

    def save_data_file(self, path):
        self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        with self.writer as writer:
            pd.DataFrame(self.data).to_excel(writer,sheet_name='CV_XRD_data',columns = self.kwarg_global['data_keys'])
            # writer.save()



if __name__ == "__main__":
    RunApp()

