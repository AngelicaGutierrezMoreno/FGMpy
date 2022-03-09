from matplotlib import pyplot as plt


class PlotClass:
    def __init__(self):
        self.title = "None";
        self.xlabel = "X"
        self.ylabel = "Y"
        self.loc = "lower right"
        self.color = "b"
        self.dataX = []
        self.dataY = []
        self.dataLegends = []
        self.dataColor = []

    def setTitle(self, title):
        self.title = title

    def setXLabel(self, xlabel):
        self.xlabel = xlabel

    def setYLabel(self, ylabel):
        self.ylabel = ylabel

    def setLoc(self, loc):
        self.loc = loc

    def setColor(self, color):
        self.color = color

    def appendSeries(self, legend, xserie, yserie, color):
        if len(xserie) == len(yserie):
            self.dataX.append(xserie)
            self.dataY.append(yserie)
            self.dataLegends.append(legend)
            self.dataColor.append(color)
        else:
            print("Not a valid list")

    def show(self):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i], self.dataColor[i])
            plt.legend(self.dataLegends[i], loc=self.loc)
        plt.show()

    def save(self, filename):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        #ax = plt.subplot(2)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i])
            plt.legend(self.dataLegends[i])
            #
            # plt.subplot(212)
            # plt.plot(self.dataX[i], self.dataY[i])
            # plt.legend(self.dataLegends[i])
            # plt.xlim(1.3, 4.0)
        plt.savefig(filename)



    def printSeries(self):
        print("Legend", self.dataLegends)
        print("X series", self.dataX)
        print("Y series", self.dataY)