import pyqtgraph as pg

#customize the label of image axis (q basis instead of index basis)
class pixel_to_q(pg.AxisItem):
    def __init__(self, scale, shift, *args, **kwargs):
        super(pixel_to_q, self).__init__(*args, **kwargs)
        self.scale = scale
        self.shift = shift

    def tickStrings(self, values, scale, spacing):
        return [round(value*self.scale+self.shift,3) for value in values]

    def attachToPlotItem(self, plotItem):
        """Add this axis to the given PlotItem
        :param plotItem: (PlotItem)
        """
        self.setParentItem(plotItem)
        viewBox = plotItem.getViewBox()
        self.linkToView(viewBox)
        self._oldAxis = plotItem.axes[self.orientation]['item']
        self._oldAxis.hide()
        plotItem.axes[self.orientation]['item'] = self
        pos = plotItem.axes[self.orientation]['pos']
        plotItem.layout.addItem(self, *pos)
        self.setZValue(-1000)