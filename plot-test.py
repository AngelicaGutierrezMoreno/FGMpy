from plotting import PlotClass


def main():
    pa = PlotClass()
    pa.appendSeries("Serie1", [1,2,3,4,5],[2,2,3,4,5])
    pa.appendSeries("Serie2", [1, 2, 3, 4, 5], [5, 8, 3, 2, 1])
    pa.appendSeries("Serie3", [1, 2, 3, 4, 5], [4, 5, 10, 14, 7])
    pa.show()

    pb = PlotClass()
    pb.setTitle("Graph 2")
    pb.setXLabel("X axis")
    pb.setYLabel("Y axis")
    pb.appendSeries("Serie1b", [1, 2, 3], [2, 2, 3])
    pb.appendSeries("Serie2b", [1, 2, 3], [5, 8, 3])
    pb.appendSeries("Serie3b", [1, 2, 3, 4], [4, 5, 10, 14])
    pb.save("Example1")


if __name__ == '__main__':
    main()