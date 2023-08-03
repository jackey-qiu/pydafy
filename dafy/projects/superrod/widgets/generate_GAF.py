import sys,os
import math
import numpy as np
from PyQt5.QtWidgets import*
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

# Tools
def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))

def create_time_serie(size, time):
    """Generate a time serie of length size and dynamic with respect to time."""
    # Generating time-series
    support = np.arange(0, size)
    serie = np.cos(support + float(time))
    return(support, serie)

def compute_GAF(serie):
    """Compute the Gramian Angular Field of an image"""
    # Min-Max scaling
    serie = np.array(serie)
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)
    # Note! The computation of r is not necessary
    r = np.linspace(0, 1, len(scaled_serie))
    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return(gaf, phi, r, scaled_serie)

class GAF_Widget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent) 
        self.parent = None
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig) 
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.navi_toolbar)
        # self.canvas.ax_img = self.canvas.figure.add_subplot(121)
        # self.canvas.ax_profile = self.canvas.figure.add_subplot(322)
        # self.canvas.ax_ctr = self.canvas.figure.add_subplot(324)
        # self.canvas.ax_pot = self.canvas.figure.add_subplot(326)
        self.ax = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        self.gaf = 0

    def update_canvas(self, fig_size):
        self.canvas = FigureCanvas(Figure(fig_size)) 
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        vertical_layout.addWidget(self.navi_toolbar)
        # self.canvas.ax_profile = self.canvas.figure.add_subplot(322)
        # self.canvas.ax_ctr = self.canvas.figure.add_subplot(324)
        # self.canvas.ax_pot = self.canvas.figure.add_subplot(326)
        #self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def reset(self):
        self.canvas.figure.clear()
        self.canvas.draw()
        self.data = {}
        self.ax_handle_ed = None
        self.ax_handles_ctr = None

    def clear_plot(self):
        self.canvas.figure.clear()
        self.canvas.draw()

    def make_gaf_data(self, serie_data = np.cos(np.arange(0,100))):
        return compute_GAF(serie_data)

    def create_plots(self, serie_data = np.cos(np.arange(0,100)), plot_type = 'GAF diagram', relative = False):
        gaf, phi, r, scaled_time_serie = self.make_gaf_data(serie_data)
        if relative:
            gaf_ = gaf - self.gaf
        self.gaf, self.phi, self.r, self.scaled_time_serie = gaf, phi, r, scaled_time_serie
        # self.update_canvas(eval(self.parent.lineEdit_fig_size.text()))
        self.canvas.figure.clear()
        if plot_type == 'polar encoding':
            self.ax = self.canvas.figure.add_subplot(111,polar = True)
            self.ax.plot(phi, r)
            self.ax.set_title("Polar Encoding")
            self.ax.set_rticks([0, 1])
            self.ax.set_rlabel_position(-22.5)
            self.ax.grid(True)
        elif plot_type == 'GAF diagram':
            self.ax = self.canvas.figure.add_subplot(111)
            if relative:
                self.ax.matshow(gaf_)
            else:
                self.ax.matshow(gaf)
            self.ax.set_title("Gramian Angular Field")
            self.ax.set_yticklabels([])
            self.ax.set_xticklabels([])
        elif plot_type == 'CTR series':
            self.ax = self.canvas.figure.add_subplot(111)
            self.ax.plot(range(len(scaled_time_serie)), scaled_time_serie)
            self.ax.set_title("Scaled CTR Serie")
        self.canvas.draw()