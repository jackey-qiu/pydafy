"""
Optical system design demo
"""

import numpy as np
from ..resources.optics import *

import pyqtgraph as pg
from pyqtgraph import Point


class LenseWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent = parent,show=True, border=0.5)
        self.phase = 0.0
        self.show()
        self.add_timer()
        self.add_three_viewbox()

    def add_three_viewbox(self):
        self.view1 = self.addViewBox()
        self.view1.setAspectLocked()
        self.view2 = self.addViewBox()
        self.view2.setAspectLocked()
        self.nextRow()
        self.view3 = self.addViewBox(colspan=2)
        self.prepare_view1()
        self.prepare_view2()
        self.prepare_view3()
        self.start_timer()

    def prepare_view1(self):
        self.optics1 = []
        rays = []
        m1 = Mirror(r1=-100, pos=(5,0), d=5, angle=-15)
        self.optics1.append(m1)
        m2 = Mirror(r1=-70, pos=(-40, 30), d=6, angle=180-15)
        self.optics1.append(m2)

        self.allRays1 = []
        for y in np.linspace(-10, 10, 21):
            r = Ray(start=Point(-100, y))
            self.view1.addItem(r)
            self.allRays1.append(r)

        for o in self.optics1:
            self.view1.addItem(o)
            
        self.t1 = Tracer(self.allRays1, self.optics1)    

    def prepare_view2(self):
        ### Dispersion demo

        self.optics2 = []
        self.l1 = Lens(r1=20, r2=20, d=10, angle=8, glass='Corning7980')
        self.optics2.append(self.l1)

        self.allRays2 = []
        for wl in np.linspace(355,1040, 25):
            for y in [10]:
                r = Ray(start=Point(-100, y), wl=wl)
                self.view2.addItem(r)
                self.allRays2.append(r)

        for o in self.optics2:
            self.view2.addItem(o)

        self.t2 = Tracer(self.allRays2, self.optics2)            

    def prepare_view3(self):
        ### Scanning laser microscopy demo

        #view.setAspectLocked()
        self.view3.setRange(QtCore.QRectF(200, -50, 500, 200))

        ## Scan mirrors
        scanx = 250
        scany = 20
        self.m1 = Mirror(dia=4.2, d=0.001, pos=(scanx, 0), angle=315)
        self.m2 = Mirror(dia=8.4, d=0.001, pos=(scanx, scany), angle=135)

        ## Scan lenses
        self.l3 = Lens(r1=23.0, r2=0, d=5.8, pos=(scanx+50, scany), glass='Corning7980')  ## 50mm  UVFS  (LA4148)
        self.l4 = Lens(r1=0, r2=69.0, d=3.2, pos=(scanx+250, scany), glass='Corning7980')  ## 150mm UVFS  (LA4874)

        ## Objective
        self.obj = Lens(r1=15, r2=15, d=10, dia=8, pos=(scanx+400, scany), glass='Corning7980')

        self.IROptics = [self.m1, self.m2, self.l3, self.l4, self.obj]

        ## Scan mirrors
        scanx = 250
        scany = 30
        self.m1a = Mirror(dia=4.2, d=0.001, pos=(scanx, 2*scany), angle=315)
        self.m2a = Mirror(dia=8.4, d=0.001, pos=(scanx, 3*scany), angle=135)

        ## Scan lenses
        self.l3a = Lens(r1=46, r2=0, d=3.8, pos=(scanx+50, 3*scany), glass='Corning7980') ## 100mm UVFS  (LA4380)
        self.l4a = Lens(r1=0, r2=46, d=3.8, pos=(scanx+250, 3*scany), glass='Corning7980') ## 100mm UVFS  (LA4380)

        ## Objective
        self.obja = Lens(r1=15, r2=15, d=10, dia=8, pos=(scanx+400, 3*scany), glass='Corning7980')

        self.IROptics2 = [self.m1a, self.m2a, self.l3a, self.l4a, self.obja]



        for o in set(self.IROptics+self.IROptics2):
            self.view3.addItem(o)
            
        self.IRRays = []
        self.IRRays2 = []

        for dy in [-0.4, -0.15, 0, 0.15, 0.4]:
            self.IRRays.append(Ray(start=Point(-50, dy), dir=(1, 0), wl=780))
            self.IRRays2.append(Ray(start=Point(-50, dy+2*scany), dir=(1, 0), wl=780))
            
        for r in set(self.IRRays+self.IRRays2):
            self.view3.addItem(r)

        self.IRTracer = Tracer(self.IRRays, self.IROptics)
        self.IRTracer2 = Tracer(self.IRRays2, self.IROptics2)

    def update(self):
        if self.phase % (8*np.pi) > 4*np.pi:
            self.m1['angle'] = 315 + 1.5*np.sin(self.phase)
            self.m1a['angle'] = 315 + 1.5*np.sin(self.phase)
        else:
            self.m2['angle'] = 135 + 1.5*np.sin(self.phase)
            self.m2a['angle'] = 135 + 1.5*np.sin(self.phase)
        self.phase += 0.2
    
    def start_timer(self):
        self.timer.start(40)

    def add_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
