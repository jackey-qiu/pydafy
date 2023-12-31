import sys, os 
import numpy as np
import pandas as pd
from dafy.core.util.path import *
import matplotlib
matplotlib.use("TkAgg")
from dafy.core.FilterPool.DataFilterPool import create_mask, merge_data_bkg, update_data_bkg, update_data_bkg_previous_frame, merge_data_image_loader, merge_data_image_loader_gsecars,make_data_config_file, cal_ctot_stationary
from dafy.core.EnginePool.FitEnginePool import background_subtraction_single_img
from dafy.core.util.UtilityFunctions import scan_generator,image_generator_bkg, image_generator_bkg_gsecars,nexus_image_loader, gsecars_image_loader,extract_vars_from_config

#make compatibility of py 2 and py 3#
if (sys.version_info > (3, 0)):
    raw_input = input

class RunApp(object):
    def __init__(self, beamline = 'PETRA3_P23'):
        self.stop = True
        self.current_frame = 0
        self.conf_file = None
        self.bkg_clip_image = None
        self.beamline = beamline
        self.data_path = str(user_data_path)
        self.conf_path_temp = str(ctr_path / 'config' / 'config_ctr_analysis_standard.ini')

    def run(self, config = None):
        #extract global vars from config
        if config == None:
            pass
        else:
            self.conf_file = config
            self.kwarg_global = extract_vars_from_config(self.conf_file, section_var ='Global')
        for each in self.kwarg_global:
            #flatten [[1,5],15] to [1,2,3,4,5,15] if necessarily
            if each=='scan_nos':
                temp_scans = []
                for each_item in self.kwarg_global['scan_nos']:
                    if type(each_item)==list:
                        temp_scans = temp_scans + list(range(each_item[0],each_item[1]+1))
                    else:
                        temp_scans.append(each_item)
                setattr(self,'scan_nos',temp_scans)
            else:
                setattr(self,each,self.kwarg_global[each])

        #pars lib for everything else
        self.kwarg_image = extract_vars_from_config(self.conf_file, section_var = 'Image_Loader')
        self.kwarg_mask = extract_vars_from_config(self.conf_file,section_var = 'Mask')

        #recal clip_boundary and cen
        self.clip_boundary = {"ver":[self.cen[0]-self.clip_width['ver'],self.cen[0]+self.clip_width['ver']+1],
                        "hor":[self.cen[1]-self.clip_width['hor'],self.cen[1]+self.clip_width['hor']+1]}     
        self.cen_clip = [self.clip_width['ver'],self.clip_width['hor']]

        self.img = None
        #data file
        self.data = {}
        if 'noise' not in self.data_keys:
            self.data_keys.append('noise')
        if 'ctot' not in self.data_keys:
            self.data_keys.append('ctot')
        if 'peak_shift' not in self.data_keys:
            self.data_keys.append('peak_shift')
        for key in self.data_keys:
            self.data[key]=[]
        # print(data)
        #init peak fit, bkg subtraction and reciprocal space and image loader instance
        self.bkg_sub = background_subtraction_single_img(self.cen_clip, self.conf_file, sections = ['Background_Subtraction'])
        self.bkg_sub.update_col_row_width(clip_lib = self.kwarg_global['clip_width'], default_padding_rate = 0.8)
        if self.beamline == 'PETRA3_P23':
            self.img_loader = nexus_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image)
        elif self.beamline == 'APS_13IDC':
            self.img_loader = gsecars_image_loader(clip_boundary = self.clip_boundary, kwarg = self.kwarg_image, scan_numbers= self.scan_nos)
        self.create_mask_new = create_mask(kwarg = self.kwarg_mask)
        self.setup_frames()

    def setup_frames(self):
        #build generator funcs
        self._scans = scan_generator(scans = self.scan_nos)
        if self.beamline == 'PETRA3_P23':
            self._images = image_generator_bkg(self._scans,self.img_loader,self.create_mask_new)
        elif self.beamline == 'APS_13IDC':
            self._images = image_generator_bkg_gsecars(self._scans,self.img_loader,self.create_mask_new)

    def run_script(self,bkg_intensity = 0,poly_func = 'Vincent'):
        try:
            # t0 = time.time()
            img = next(self._images)
            #img = img/self.bkg_clip_image
            if hasattr(self,'current_scan_number'):
                if self.current_scan_number!=self.img_loader.scan_number:
                    # self.save_data_file(self.data_path)
                    self.current_scan_number = self.img_loader.scan_number
            else:
                setattr(self,'current_scan_number',self.img_loader.scan_number)
            self.current_frame = self.img_loader.frame_number
            self.img = img
            if self.beamline == 'PETRA3_P23':
                self.data = merge_data_image_loader(self.data, self.img_loader)
            elif self.beamline == 'APS_13IDC':
                self.data = merge_data_image_loader_gsecars(self.data, self.img_loader)
            self.bkg_sub.fit_background(None, img, self.data, plot_live = True, freeze_sf = True,poly_func = poly_func)
            # print(self.bkg_sub.fit_results)
            ctot = 1
            if 'incidence_ang' in self.kwarg_global and 'det_ang_ver' in self.kwarg_global and 'det_ang_hor' in self.kwarg_global:
                ctot = cal_ctot_stationary(incidence_ang = self.data[self.kwarg_global['incidence_ang']][-1],
                                           det_ang_ver = self.data[self.kwarg_global['det_ang_ver']][-1],
                                           det_ang_hor = self.data[self.kwarg_global['det_ang_hor']][-1])
            # assert ctot > 0, 'Correction factor is a negative value, check it out.'
            if ctot<0:
                ctot = abs(ctot)
                print(f'Correction factor of frame {self.current_frame} is a negative value at this point, check it out.')
            self.data = merge_data_bkg(self.data, self.bkg_sub, ctot)
            self.data['bkg'].append(bkg_intensity)
            self.data['ctot'].append(ctot)
            # print(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4)
            return True
        except StopIteration:
            self.save_data_file(self.data_path)
            return False

    def run_update(self,bkg_intensity = 0,begin = False, poly_func = 'Vincent'):
        if not begin:
            self.bkg_sub.fit_background(None, self.img, self.data, plot_live = True, freeze_sf = True, poly_func = poly_func)
        self.data = update_data_bkg(self.data, self.bkg_sub)
        self.data['bkg'][-1] = bkg_intensity

    def run_update_one_specific_frame(self, img, bkg_intensity, poly_func = 'Vincent', frame_offset = -1):
        self.bkg_sub.fit_background(None, img, self.data, plot_live = True, freeze_sf = True, poly_func = poly_func)
        # print(self.data['peak_intensity'])
        self.data = update_data_bkg_previous_frame(self.data, self.bkg_sub, frame_offset)
        # print(self.data['peak_intensity'])
        self.data['bkg'][frame_offset] = bkg_intensity

    def save_data_file(self,path):
        #update path for saving data
        if path == self.data_path:
            pass
        else:
            self.data_path = path
        #path_ = path.replace('.xlsx','_.xlsx')
        #writer_ = pd.ExcelWriter(path_,engine = 'openpyxl',mode = 'w')
        self.writer = pd.ExcelWriter([path+'.xlsx',path][int(path.endswith('.xlsx'))],engine = 'openpyxl',mode ='w')
        with self.writer as writer:
            pd.DataFrame(self.data).to_excel(writer,sheet_name='CTR_data',columns = self.data_keys)
            # writer.save()
        #now empty the data container
        #for key in self.data_keys:
        #    self.data[key]=[self.data[key][-1]]

    #export csv data to be imported in SuPerRod software
    def save_rod_data(self, path, conditions_df):
        '''
        conditions_df is a pandas dataframe containing columns of save, scan_no, H, K, dL, BL and escan
        '''
        data_saved_rod = pd.DataFrame(np.zeros([1,8])[0:0],columns=["L","H","K","na","peak_intensity","peak_intensity_error","BL","dL"])
        data_saved_raxs = pd.DataFrame(np.zeros([1,8])[0:0],columns=["E","H","K","L","peak_intensity","peak_intensity_error","BL","dL"])
        data_df = pd.DataFrame(self.data)
        for i in range(conditions_df.shape[0]):
            current_row = conditions_df.iloc[i]
            if current_row['save']:
                escan = current_row['escan']
                H, K, dL, BL, scan_no = map(lambda x: int(current_row[x]), ['H', 'K', 'dL', 'BL', 'scan_no'])
                if not escan:
                    subset_df = data_df.loc[(data_df['scan_no']==scan_no) & (data_df['H']==H) & (data_df['K']==K)].reset_index()
                    dL_BL_df = pd.DataFrame({'dL': [dL]*subset_df.shape[0], 'BL': [BL]*subset_df.shape[0],'na': [0]*subset_df.shape[0]})
                    data_merged = pd.concat([subset_df, dL_BL_df], axis = 1)
                    data_selected = data_merged.loc[:,["L","H","K","na","peak_intensity","peak_intensity_error","BL","dL"]]
                    data_saved_rod = pd.concat([data_saved_rod,data_selected], ignore_index = True)
                else:
                    subset_df = data_df.loc[(data_df['scan_no']==scan_no) & (data_df['H']==H) & (data_df['K']==K)].reset_index()
                    dL_BL_df = pd.DataFrame({'dL': [dL]*subset_df.shape[0], 'BL': [BL]*subset_df.shape[0],'na': [0]*subset_df.shape[0]})
                    data_merged = pd.concat([subset_df, dL_BL_df], axis = 1)
                    data_selected = data_merged.loc[:,["E","H","K","L","peak_intensity","peak_intensity_error","BL","dL"]]
                    data_saved_rod = pd.concat([data_saved_rod,data_selected], ignore_index = True)
        if data_saved_rod.shape[0]!=0:
            data_saved_rod.to_csv(path, header = False, sep =' ', index=False)
        if data_saved_raxs.shape[0]!=0:
            data_saved_raxs.to_csv(path, header = False, sep =' ', index=False)


if __name__ == "__main__":
    RunApp()
