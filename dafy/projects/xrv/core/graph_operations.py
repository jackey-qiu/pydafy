import pyqtgraph as pg
from dafy.core.EnginePool.VisualizationEnginePool import plot_xrv_gui_pyqtgraph

def update_roi(self):
    ver_width,hor_width = self.app_ctr.cen_clip
    self.roi.setSize([hor_width*2*0.9, ver_width*2])
    self.roi.setPos([hor_width*2*0.2, 0.])
    self.roi_bkg.setSize([hor_width*2*0.09, ver_width*2])
    self.roi_bkg.setPos([hor_width*2*0.2, 0.])

def setup_image(self):
    win = self.widget_image

    # Contrast/color control
    self.hist = pg.HistogramLUTItem()
    win.addItem(self.hist,row=0,col=0,rowspan=1,colspan=2)

    # A plot area (ViewBox + axes) for displaying the image
    p1 = win.addPlot(row=0,col=2,rowspan=1,colspan=3)
    # Item for displaying image data
    img = pg.ImageItem()
    p1.getViewBox().invertY(False)
    self.img_pyqtgraph = img
    p1.addItem(img)
    self.hist.setImageItem(img)
    
    # Custom ROI for selecting an image region
    ver_width,hor_width = self.app_ctr.cen_clip
    roi = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.9, ver_width*2])
    # roi = pg.ROI([100, 100], [200, 200])
    self.roi = roi
    roi.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi.addScaleHandle([1, 0.5], [0.5, 0.5])
    p1.addItem(roi)

    # Custom ROI for monitoring bkg
    roi_bkg = pg.ROI(pos = [hor_width*2*0.2, 0.], size = [hor_width*2*0.09, ver_width*2],pen = 'r')
    # roi_bkg = pg.ROI([0, 100], [200, 200],pen = 'r')
    self.roi_bkg = roi_bkg
    roi_bkg.addScaleHandle([0.5, 1], [0.5, 0.5])
    roi_bkg.addScaleHandle([0, 0.5], [0.5, 0.5])
    p1.addItem(roi_bkg)

    # Isocurve drawing
    iso = pg.IsocurveItem(level=0.8, pen='g')
    iso.setParentItem(img)
    self.iso = iso
    
    # Draggable line for setting isocurve level
    isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
    self.isoLine = isoLine
    self.hist.vb.addItem(isoLine)
    self.hist.vb.setMouseEnabled(y=True) # makes user interaction a little easier
    isoLine.setValue(0.8)
    isoLine.setZValue(100000) # bring iso line above contrast controls

    #set up the region selector for peak fitting
    self.region_cut_hor = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Horizontal)
    self.region_cut_ver = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
    self.region_cut_hor.setRegion([180,220])
    self.region_cut_ver.setRegion([180,220])
    p1.addItem(self.region_cut_hor, ignoreBounds = True)
    p1.addItem(self.region_cut_ver, ignoreBounds = True)

    # double-y axis plot for structure data (strain and size)
    p2 = win.addPlot(row=1,col=1,colspan=3,rowspan=1, title = 'film structure parameters')
    self.vLine_par = pg.InfiniteLine(angle=90, movable=False)
    self.hLine_par = pg.InfiniteLine(angle=0, movable=False)
    self.vLine_par.setZValue(100)
    self.hLine_par.setZValue(100)
    # p2 = win.addPlot(row=1,col=1,colspan=2,rowspan=1)
    p2.setLabel('bottom','x channel')
    p2.getAxis('left').setLabel('Strain (%)')
    p2.getAxis('left').setPen(pg.mkPen('w', width=2))

    self.proxy_p2_panel = pg.SignalProxy(p2.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved_in_p2)
    p2_r = pg.ViewBox()
    p2.showAxis('right')
    p2.setLabel('right','grain size (nm)')
    p2.getAxis('right').setPen(pg.mkPen('g', width=2))
    p2.scene().addItem(p2_r)
    p2.getAxis('right').linkToView(p2_r)
    p2_r.setXLink(p2)
    ## Handle view resizing
    def updateViews_p2():
        ## view has resized; update auxiliary views to match
        p2_r.setGeometry(p2.vb.sceneBoundingRect())
        p2_r.linkedViewChanged(p2.vb, p2_r.XAxis)
    updateViews_p2()
    p2.vb.sigResized.connect(updateViews_p2)

    # plot to show intensity(sig and bkg) over time
    p3 = win.addPlot(row=2,col=1,colspan=3,rowspan=1,title = 'CTR intensity')
    p3.getAxis('left').setLabel('Peak intensity')
    p3.getAxis('left').setPen(pg.mkPen('w', width=2))
    #p3.setLabel('left','Integrated Intensity', units='c/s')
    p3.setLabel('bottom','x channel')
    p3_r = pg.ViewBox()
    p3.showAxis('right')
    p3.setLabel('right','bkg intensity')
    p3.getAxis('right').setPen(pg.mkPen('g', width=2))
    p3.scene().addItem(p3_r)
    p3.getAxis('right').linkToView(p3_r)
    p3_r.setXLink(p3)
    #p3.getAxis('right').setLabel('bkg', color='b')
    ## Handle view resizing
    def updateViews_p3():
        ## view has resized; update auxiliary views to match
        p3_r.setGeometry(p3.vb.sceneBoundingRect())
        ## need to re-update linked axes since this was called
        ## incorrectly while views had different shapes.
        ## (probably this should be handled in ViewBox.resizeEvent)
        p3_r.linkedViewChanged(p3.vb, p3_r.XAxis)
    updateViews_p3()
    p3.vb.sigResized.connect(updateViews_p3)

    # plot to show current/potential over time
    p4 = win.addPlot(row=3,col=1,colspan=3,rowspan=1,title = 'CV data')
    p4.getAxis('left').setLabel('Potential_RHE (V)')
    p4.getAxis('left').setPen(pg.mkPen('w', width=2))
    #p4.setMaximumHeight(200)
    p4.setLabel('bottom','x channel')
    p4_r = pg.ViewBox()
    p4.showAxis('right')
    p4.setLabel('right','current (mA)')
    p4.getAxis('right').setPen(pg.mkPen('g', width=2))
    p4.scene().addItem(p4_r)
    p4.getAxis('right').linkToView(p4_r)
    p4_r.setXLink(p4)
    #p4.getAxis('right').setLabel('bkg', color='b')
    ## Handle view resizing
    def updateViews_p4():
        ## view has resized; update auxiliary views to match
        p4_r.setGeometry(p4.vb.sceneBoundingRect())
        ## need to re-update linked axes since this was called
        ## incorrectly while views had different shapes.
        ## (probably this should be handled in ViewBox.resizeEvent)
        p4_r.linkedViewChanged(p4.vb, p4_r.XAxis)
    updateViews_p4()
    p4.vb.sigResized.connect(updateViews_p4)

    #plot the peak fit results(horizontally)
    p5 = win.addPlot(row=1,col=0,colspan=1,rowspan=1,title = 'peak fit result_horz')
    p6 = win.addPlot(row=2,col=0,colspan=1,rowspan=1,title = 'peak fit result_vert')
    p6.setLabel('bottom','q')
    p7 = win.addPlot(row=3,col=0,colspan=1,rowspan=1,title = 'Peak intensity')
    p7.setLabel('bottom','pixel index')

    #add slider to p2 to exclude the abnormal points
    self.region_abnormal = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
    self.region_abnormal.setZValue(100)
    self.region_abnormal.setRegion([0, 5])
    p2.addItem(self.region_abnormal, ignoreBounds = True)

    self.p1 = p1
    self.p2 = p2
    self.p2_r = p2_r
    self.p3 = p3
    self.p3_r = p3_r
    self.p4 = p4
    self.p4_r = p4_r
    self.p5 = p5
    self.p6 = p6
    self.p7 = p7

    # zoom to fit image
    p1.autoRange()  

    def update_bkg_signal():
        selected = self.roi_bkg.getArrayRegion(self.app_ctr.img, self.img_pyqtgraph)
        self.bkg_intensity = selected.sum()
        #self.bkg_clip_image = selected
        #self.app_ctr.bkg_clip_image = selected

    self.update_bkg_signal = update_bkg_signal

    # Callbacks for handling user interaction
    def updatePlot(fit = True):
        #global data
        try:
            selected = self.roi.getArrayRegion(self.app_ctr.bkg_sub.img, self.img_pyqtgraph)
        except:
            pass

        self.reset_peak_center_and_width()
        self.update_bkg_signal()
        if fit:
            self.app_ctr.run_update(bkg_intensity=self.bkg_intensity)
        else:
            self.app_ctr.data['bkg'][-1] = self.bkg_intensity
        ##update iso curves
        x, y = [int(each) for each in self.roi.pos()]
        w, h = [int(each) for each in self.roi.size()]
        self.iso.setData(pg.gaussianFilter(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)], (2, 2)))
        self.iso.setPos(x,y)

        if self.app_ctr.img_loader.current_frame_number ==0:
            isoLine.setValue(self.app_ctr.bkg_sub.img[y:(y+h),x:(x+w)].mean())
        else:
            pass

        #plot others
        handles = plot_xrv_gui_pyqtgraph(self.p1,[self.p2,self.p2_r], [self.p3,self.p3_r], [self.p4,self.p4_r],self.p5, self.p6, self.p7,self.app_ctr, self.checkBox_x_channel.isChecked(), plot_small_cut_result = self.checkBox_small_cut.isChecked())
        if len(handles) == 2:
            self.single_point_strain_ax_handle, self.single_point_size_ax_handle = handles
        elif len(handles) == 4:
            self.single_point_strain_oop_ax_handle, self.single_point_strain_ip_ax_handle, self.single_point_size_oop_ax_handle, self.single_point_size_ip_ax_handle = handles

        # add slider afterwards to have it show up on th plot
        self.p2.addItem(self.region_abnormal, ignoreBounds = True)
        self.p2.addItem(self.vLine_par, ignoreBounds = True)
        self.p2.addItem(self.hLine_par, ignoreBounds = True)

        #update roi
        x, y = [int(each) for each in self.roi.pos()]
        w, h = [int(each) for each in self.roi.size()]
        self.roi_bkg.setSize([w*0.1,h])
        self.roi_bkg.setPos([x,y])

        #show values for current status
        self.lcdNumber_potential.display(self.app_ctr.data['potential'][-1])
        self.lcdNumber_current.display(self.app_ctr.data['current'][-1])
        self.lcdNumber_intensity.display(self.app_ctr.data['peak_intensity'][-1])
        self.lcdNumber_strain_par.display(self.app_ctr.data['strain_ip'][-1])
        self.lcdNumber_strain_ver.display(self.app_ctr.data['strain_oop'][-1])
        self.lcdNumber_size_par.display(self.app_ctr.data['grain_size_ip'][-1])
        self.lcdNumber_size_ver.display(self.app_ctr.data['grain_size_oop'][-1])
        self.lineEdit_peak_center.setText(str(self.app_ctr.peak_fitting_instance.peak_center))
        self.lineEdit_previous_center.setText(str(self.app_ctr.peak_fitting_instance.previous_peak_center))
        self.lcdNumber_iso.display(isoLine.value())
        self.lcdNumber_scan_number.display(self.app_ctr.img_loader.scan_number)
        self.lcdNumber_frame_number.display(self.app_ctr.img_loader.current_frame_number+1)

    roi.sigRegionChanged.connect(updatePlot)
    #roi_bkg.sigRegionChanged.connect(updatePlot)
    self.updatePlot = updatePlot

    def updateIsocurve():
        # global isoLine, iso
        self.iso.setLevel(isoLine.value())
        self.lcdNumber_iso.display(isoLine.value())
    self.updateIsocurve = updateIsocurve
    isoLine.sigDragged.connect(updateIsocurve)