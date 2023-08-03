import pyqtgraph as pg
from dafy.core.EnginePool.VisualizationEnginePool import plot_bkg_fit_gui_pyqtgraph

def setup_image(self):
    # Interpret image data as row-major instead of col-major
    global img, roi, roi_bkg, data, p2, isoLine, iso
    win = self.widget_image
    # print(dir(win))
    win.setWindowTitle('pyqtgraph example: Image Analysis')

    # A plot area (ViewBox + axes) for displaying the image
    p1 = win.addPlot()

    # Item for displaying image data
    img = pg.ImageItem()
    self.img_pyqtgraph = img
    p1.addItem(img)

    # Custom ROI for selecting an image region
    ver_width,hor_width = self.app_ctr.cen_clip

    roi = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.6, ver_width*2])
    #roi = pg.ROI([100, 100], [100, 100])
    self.roi = roi
    roi.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi.addScaleHandle([0, 0.5], [0.5, 0.5])
    roi.addRotateHandle([0., 0.], [0.5, 0.5])
    p1.addItem(roi)
    
    roi_peak = pg.ROI(pos = [hor_width-self.app_ctr.bkg_sub.peak_width, 0.], size = [self.app_ctr.bkg_sub.peak_width*2, ver_width*2], pen = 'g')
    #roi = pg.ROI([100, 100], [100, 100])
    self.roi_peak = roi_peak
    p1.addItem(roi_peak)

    # Custom ROI for monitoring bkg
    roi_bkg = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.1, ver_width*2],pen = 'r')
    # roi_bkg = pg.ROI([0, 100], [100, 100],pen = 'r')
    self.roi_bkg = roi_bkg
    # roi_bkg.addScaleHandle([0.5, 1], [0.5, 0.5])
    # roi_bkg.addScaleHandle([0, 0.5], [0.5, 0.5])
    p1.addItem(roi_bkg)
    #roi.setZValue(10)  # make sure ROI is drawn above image

    # Isocurve drawing
    iso = pg.IsocurveItem(level=0.8, pen='g')
    iso.setParentItem(img)
    self.iso = iso
    
    #iso.setZValue(5)

    # Contrast/color control
    hist = pg.HistogramLUTItem()
    self.hist = hist
    hist.setImageItem(img)
    win.addItem(hist)

    # Draggable line for setting isocurve level
    isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
    self.isoLine = isoLine
    hist.vb.addItem(isoLine)
    hist.vb.setMouseEnabled(y=True) # makes user interaction a little easier
    isoLine.setValue(0.8)
    isoLine.setZValue(100000) # bring iso line above contrast controls

    # Another plot area for displaying ROI data
    win.nextRow()
    p2 = win.addPlot(colspan=2, title = 'ROI image profile')
    p2.setMaximumHeight(200)
    p2.setLabel('left','Intensity', units='c/s')
    p2.setLabel('bottom','Pixel number')

    #p2.setLogMode(y = True)


    # plot to show intensity over time
    win.nextRow()
    p3 = win.addPlot(colspan=2)
    p3.setMaximumHeight(200)
    p3.setLabel('left','Integrated Intensity', units='c/s')

    # plot to show intensity over time
    win.nextRow()
    p4 = win.addPlot(colspan=2)
    p4.setMaximumHeight(200)
    p4.setLabel('bottom','frame number')

    region_roi = pg.LinearRegionItem()
    region_roi.setZValue(10)
    region_roi.setRegion([10, 15])

    # Generate image data
    #data = np.random.normal(size=(500, 600))
    #data[20:80, 20:80] += 2.
    #data = pg.gaussianFilter(data, (3, 3))
    #data += np.random.normal(size=(500, 600)) * 0.1
    #img.setImage(data)
    ##hist.setLevels(data.min(), data.max())

    # build isocurves from smoothed data
    ##iso.setData(pg.gaussianFilter(data, (2, 2)))

    # set position and scale of image
    #img.scale(0.2, 0.2)
    #img.translate(-50, 0)

    # zoom to fit imageo
    self.p1 = p1
    self.p2 = p2
    self.p3 = p3
    self.p4 = p4
    p1.autoRange()  

    def update_bkg_signal():
        selected = roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
        self.bkg_intensity = selected.mean()
        #self.bkg_clip_image = selected
        #self.app_ctr.bkg_clip_image = selected

    def update_bkg_clip():
        selected = roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
        #self.bkg_intensity = selected.sum()
        #self.bkg_clip_image = selected
        self.app_ctr.bkg_clip_image = selected

    # Callbacks for handling user interaction
    def updatePlot(begin = False):
        # t0 = time.time()
        update_bkg_signal()
        #global data
        try:
            selected = roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
        except:
            #selected = roi.getArrayRegion(data, self.img_pyqtgraph)
            pass

        self.p3.setLabel('left',self.comboBox_p3.currentText())
        self.p4.setLabel('left',self.comboBox_p4.currentText())
        
        # p2.plot(selected.sum(axis=int(self.app_ctr.bkg_sub.int_direct=='y')), clear=True)
        self.reset_peak_center_and_width()
        if self.tag_reprocess:
            self.app_ctr.run_update_one_specific_frame(self.app_ctr.bkg_sub.img, self.bkg_intensity, poly_func = ['Vincent','traditional'][int(self.radioButton_traditional.isChecked())], frame_offset = int(self.lineEdit_frame_index_offset.text()))
        else:
            self.app_ctr.run_update(bkg_intensity=self.bkg_intensity,begin = begin,poly_func=['Vincent','traditional'][int(self.radioButton_traditional.isChecked())])
        # t1 = time.time()
        ##update iso curves
        x, y = [int(each) for each in self.roi.pos()]
        w, h = [int(each) for each in self.roi.size()]
        self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
        self.iso.setPos(x,y)
        #update peak roi
        self.roi_peak.setSize([self.app_ctr.bkg_sub.peak_width*2,h])
        self.roi_peak.setPos([x+w/2.-self.app_ctr.bkg_sub.peak_width+self.app_ctr.bkg_sub.peak_shift,y])
        #update bkg roi
        self.roi_bkg.setSize([w/2-self.app_ctr.bkg_sub.peak_width+self.app_ctr.bkg_sub.peak_shift,h])
        self.roi_bkg.setPos([x,y])
        self.display_current_roi_info()
        # t2 = time.time()
        # print(t1-t0,t2-t1)
        if self.app_ctr.img_loader.frame_number ==0:
            isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
        else:
            pass
        #print(isoLine.value(),self.current_image_no)
        #plot others
        #plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
        if self.tag_reprocess:
            index_frame = int(self.lineEdit_frame_index_offset.text())
            plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr,index_frame)
            self.lcdNumber_frame_number.display(self.app_ctr.img_loader.frame_number+1+index_frame+1)
            try:
                self.lcdNumber_potential.display(self.app_ctr.data['potential'][index_frame])
                self.lcdNumber_current.display(self.app_ctr.data['current'][index_frame])
            except:
                pass
            self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][index_frame])
            self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][index_frame]/self.app_ctr.data['noise'][index_frame])
        else:
            plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
            try:
                self.lcdNumber_potential.display(self.app_ctr.data['potential'][-1])
                self.lcdNumber_current.display(self.app_ctr.data['current'][-1])
            except:
                pass
            self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-1])
            self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['noise'][-1])
        self.lcdNumber_iso.display(isoLine.value())
        # if self.run_mode and ((self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['peak_intensity_error'][-1])<1.5):
        if self.run_mode and ((self.app_ctr.data['peak_intensity'][-1]/self.app_ctr.data['noise'][-1])<self.doubleSpinBox_SN_cutoff.value()):
            self.pushButton_remove_current_point.click()

    def updatePlot_after_remove_point():
        #global data
        try:
            selected = roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
        except:
            #selected = roi.getArrayRegion(data, self.img_pyqtgraph)
            pass
        p2.plot(selected.sum(axis=0), clear=True)
        self.reset_peak_center_and_width()
        #self.app_ctr.run_update()
        ##update iso curves
        x, y = [int(each) for each in self.roi.pos()]
        w, h = [int(each) for each in self.roi.size()]
        self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
        self.iso.setPos(x,y)
        if self.app_ctr.img_loader.frame_number ==0:
            isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
        else:
            pass
        #print(isoLine.value(),self.current_image_no)
        #plot others
        plot_bkg_fit_gui_pyqtgraph(self.p2, self.p3, self.p4,self.app_ctr)
        try:
            self.lcdNumber_potential.display(self.app_ctr.data['potential'][-2])
            self.lcdNumber_current.display(self.app_ctr.data['current'][-2])
        except:
            pass
        self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-2])
        self.lcdNumber_signal_noise_ratio.display(self.app_ctr.data['peak_intensity'][-2]/self.app_ctr.data['noise'][-2])
        self.lcdNumber_iso.display(isoLine.value())

    self.updatePlot = updatePlot
    self.updatePlot2 = updatePlot_after_remove_point
    self.update_bkg_clip = update_bkg_clip
    #roi.sigRegionChanged.connect(updatePlot)
    roi.sigRegionChanged.connect(self.update_ss_factor)

    def updateIsocurve():
        global isoLine, iso
        iso.setLevel(isoLine.value())
        self.lcdNumber_iso.display(isoLine.value())

    self.updateIsocurve = updateIsocurve

    isoLine.sigDragged.connect(updateIsocurve)