from matplotlib import pyplot as plt


class PlotClass:
    def __init__(self):
        self.title = "None";
        self.xlabel = "X"
        self.ylabel = "Y"
        self.dataX = []
        self.dataY = []
        self.dataLegends = []

    def setTitle(self, title):
        self.title = title

    def setXLabel(self, xlabel):
        self.xlabel = xlabel

    def setYLabel(self, ylabel):
        self.ylabel = ylabel

    def appendSeries(self, legend, xserie, yserie):
        if(len(xserie) == len(yserie)):
            self.dataX.append(xserie)
            self.dataY.append(yserie)
            self.dataLegends.append(legend)
        else:
            print("Not a valid list")

    def show(self):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i])
            plt.legend(self.dataLegends[i])
        plt.show()

    def save(self, filename):
        plt.clf()
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for i in range(len(self.dataX)):
            plt.plot(self.dataX[i], self.dataY[i])
            plt.legend(self.dataLegends[i])
        plt.savefig(filename)



    def printSeries(self):
        print("Legend", self.dataLegends)
        print("X series", self.dataX)
        print("Y series", self.dataY)