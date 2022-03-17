from matplotlib import pyplot as plt


class PlotClass:
    def __init__(self):
        self.title = "None"
        self.titleZoom = "NoneZoom"
        self.xlabel = "X"
        self.ylabel = "Y"
        self.loc = "lower right"
        self.color = "b"
        self.yscale = "linear"
        self.xscale = "linear"
        self.xminval = 0
        self.xmaxval = 50
        self.yminval = 0
        self.ymaxval = 50
        self.dataX = []
        self.dataY = []
        self.dataLegends = []
        self.dataColor = []

    def setTitle(self, title):
        self.title = title

    def setZoomTitle(self, titleZoom):
        self.titleZoom = titleZoom

    def setXLabel(self, xlabel):
        self.xlabel = xlabel

    def setYLabel(self, ylabel):
        self.ylabel = ylabel

    def setLoc(self, loc):
        self.loc = loc

    def setColor(self, color):
        self.color = color

    def setYScale(self, yscale):
        self.yscale = yscale

    def setXScale(self, xscale):
        self.xscale = xscale

    def setXMin(self, xminval):
        self.xminval = xminval

    def setXMax(self, xmaxval):
        self.xmaxval = xmaxval

    def setYMin(self, yminval):
        self.yminval = yminval

    def setYMax(self, ymaxval):
        self.ymaxval = ymaxval

    def appendSeries(self, legend, xserie, yserie, color):
        if len(xserie) == len(yserie):
            self.dataX.append(xserie)
            self.dataY.append(yserie)
            self.dataLegends.append(legend)
            self.dataColor.append(color)
        else:
            print("Not a valid list")
            print("Len X = ", len(xserie), " Len Y= ", len(yserie))

    def show(self):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        #print(range(len(self.dataX)))
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i], self.dataColor[i])
            plt.legend(self.dataLegends[i], loc=self.loc)
            #print('DataLegends %i: ' % i, str(self.dataLegends[i]))
            #print('DataLegends', str(self.dataLegends[i]))
        plt.show()

    def save(self, filename):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        #print(range(len(self.dataX)))
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i], self.dataColor[i])
            plt.legend(self.dataLegends[i], loc=self.loc)
            #print('DataLegends Save: ' % i, str(self.dataLegends[i]))
            #print('DataLegends', str(self.dataLegends[i]))
        plt.savefig(filename)

    def zoomSave(self, filename):
        plt.clf()
        plt.title(self.titleZoom)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i], self.dataColor[i])
            plt.legend(self.dataLegends[i], loc=self.loc)
            plt.xlim(self.xminval, self.xmaxval)
            plt.ylim(self.yminval, self.ymaxval)
        plt.savefig(filename)

    def zoomShow(self):
        plt.clf()
        plt.title(self.titleZoom)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.yscale(self.yscale)
        plt.xscale(self.xscale)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i], self.dataColor[i])
            plt.legend(self.dataLegends[i], loc=self.loc)
            plt.xlim(self.xminval, self.xmaxval)
            plt.ylim(self.yminval, self.ymaxval)
        plt.show()

    def printSeries(self):
        print("Legend", self.dataLegends)
        print("X series", self.dataX)
        print("Y series", self.dataY)
