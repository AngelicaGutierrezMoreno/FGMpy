import csv
from matplotlib import pyplot as plt
from plotting import PlotClass
import algorithms


def graph_fitnessVSgenerations(fitness_values, generations, f):
    # Compute the x and y coordinates
    plt.title("FITNESS vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    # plt.show()
    # file_name = 'fitnessvsgen_' + f
    # plt.savefig(file_name)
    # plt.clf() #cleans the graph


def graph_distanceVSgenerations(distance_values, generations, f):
    # Compute the x and y coordinates
    plt.title("DISTANCE vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, distance_values)
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    # plt.show()
    file_name = 'distancevsgen_' + f
    plt.savefig(file_name)
    # plt.clf()


def graph_numGensVSgenerations(gen_size, generations, f):
    # Compute the x and y coordinates
    plt.title("GEN SIZE vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, gen_size)
    plt.xlabel('Generation')
    plt.ylabel('Gen Size')
    # plt.show()
    file_name = 'numgensvsgen_' + f
    plt.savefig(file_name)
    # plt.clf()


def gen_graphs(graphs_gen_val, graphs_generation_val, f, color_values, dim):
    genplot = PlotClass()
    genplot.setTitle('Number of genes comparation %i D' % dim)
    #genplot.setTitle('Number of genes comparation')
    genplot.setXLabel('Generations')
    genplot.setYLabel('Number of genes')
    genplot.setLoc('lower right')
    for i in range(int(len(graphs_gen_val))):
        genplot.appendSeries(f, graphs_generation_val[i], graphs_gen_val[i], color_values[i])
    genplot.show()
    # genplot.printSeries()
    #genplot.save('Number of genes comparation')
    genplot.save("GenSize_comparation%iD_log" % dim)


def fitness_graphs(graphs_fitness_val, graphs_generation_val, f, color_values, dim):
    fitplot = PlotClass()
    fitplot.setTitle('Distance to the optimum comparation %iD' % dim)
    #fitplot.setTitle('Distance to the optimum comparation models')
    fitplot.setZoomTitle('Distance to the optimum comparation %iD- ZOOM' % dim)
    fitplot.setXLabel('Generations')
    fitplot.setYLabel('Distance')
    fitplot.setLoc('upper right')
    fitplot.setYScale('log')
    fitplot.setXMax(600)
    fitplot.setXMin(400)
    fitplot.setYMax(1)
    for i in range(int(len(graphs_fitness_val))):
        fitplot.appendSeries(f, graphs_generation_val[i], graphs_fitness_val[i], color_values[i])
    #fitplot.printSeries()
    fitplot.show()
    fitplot.zoomShow()
    fitplot.zoomSave("GenSize_comparation%iD-Zoom_log" % dim)
    # fitplot.printSeries()
    fitplot.save("DistanceToTptimum_comparation%iD_log" % dim)
    #fitplot.save("DistanceToTptimum_comparationMODELS")


def main():

    filename = "results-2d-log.csv"
    header = ['Inicial point', 'Dimensions', 'Mutation rate', 'Dup rate', 'Del. rate', 'Epsilon',
              'Best phenotype', 'Best genotype', 'Distance to optimum', 'Total generations', 'Gen size', 'PlotName']
    # ['id', 'fgm_mode', 'gen_mode', 'inicial_point', 'n_dim', 'mutation_rate', 'gen_duplication_rate', 'gen_deletion_rate', 'n_generations', 'epsilon]
    experiments = [
        # ## ------------------------- 1D - plot ----------------------------------------
        # # FGM
        # ["plot1D1-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        # ["plot1D2-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        # ["plot1D3-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        # ["plot1D4-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        # ["plot1D5-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        # ["plot1D6-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 700, 0.001, 'b'],
        #
        # # Model
        # ["plot1D1-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r'],
        # ["plot1D2-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r'],
        # ["plot1D3-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r'],
        # ["plot1D4-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r'],
        # ["plot1D5-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r'],
        # ["plot1D6-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 700, 0.001, 'r']

        ## ------------------------- 2D - plot ----------------------------------------
        # FGM
        ["plot2D1-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        ["plot2D2-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        ["plot2D3-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        ["plot2D4-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        ["plot2D5-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        ["plot2D6-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],

        # Model
        ["plot2D1-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        ["plot2D2-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        ["plot2D3-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        ["plot2D4-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        ["plot2D5-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        ["plot2D6-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r']

        # ## ------------------------- 3D - plot ----------------------------------------
        # # FGM
        # ["plot3D1-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot3D2-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot3D3-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot3D4-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot3D5-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot3D6-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        #
        # # Model
        # ["plot3D1-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot3D2-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot3D3-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot3D4-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot3D5-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot3D6-Model", False, False, [10.0, 0.0, 0.0], 3, 0.05, 0.5, 0.5, 500, 0.001, 'r']

        # ## ------------------------- 4D - plot ----------------------------------------
        # # FGM
        # ["plot4D1-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot4D2-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot4D3-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot4D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot4D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot4D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        #
        # # Model
        # ["plot4D1-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot4D2-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot4D3-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot4D4-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot4D5-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot4D6-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.05, 0.5, 0.5, 500, 0.001, 'r']

        # ## ------------------------- 5D - plot ----------------------------------------
        #
        # # FGM
        # ["plot5D1-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot5D2-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot5D3-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot5D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot5D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot5D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        #
        # # Model
        # ["plot5D1-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot5D2-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot5D3-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot5D4-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot5D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot5D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r']

        # ## ------------------------- 6D - plot ----------------------------------------
        #
        # # FGM
        # ["plot6D1-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot6D2-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot6D3-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot6D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot6D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        # ["plot6D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        #
        # # Model
        # ["plot6D1-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot6D2-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot6D3-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot6D4-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot6D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        # ["plot6D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.05, 0.5, 0.5, 600, 0.001, 'r']

        # ## ------------------------- Dif dimention - plot ----------------------------------------
        # # FGM
        # ["D1-FGM", False, False, [10.0                           ], 1, 0.07, 0.0, 0.0, 1000, 0.001, 'c'],
        # ["D1-Model", False, False, [10.0                         ], 1, 0.07, 0.5, 0.5, 1000, 0.001, 'c'],
        # ["D2-FGM", False, False, [10.0, 0.0                      ], 2, 0.07, 0.0, 0.0, 1000, 0.001, 'b'],
        # ["D2-Model", False, False, [10.0, 0.0                    ], 2, 0.07, 0.5, 0.5, 1000, 0.001, 'b'],
        # ["D3-FGM", False, False, [10.0, 0.0, 0.0                 ], 3, 0.07, 0.0, 0.0, 1000, 0.001, 'm'],
        # ["D3-Model", False, False, [10.0, 0.0, 0.0               ], 3, 0.07, 0.5, 0.5, 1000, 0.001, 'm'],
        # ["D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0            ], 4, 0.07, 0.0, 0.0, 1000, 0.001, 'r'],
        # ["D4-Model", False, False, [10.0, 0.0, 0.0, 0.0          ], 4, 0.07, 0.5, 0.5, 1000, 0.001, 'r'],
        # ["D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0       ], 5, 0.07, 0.0, 0.0, 1000, 0.001, 'y'],
        # ["D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0     ], 5, 0.07, 0.5, 0.5, 1000, 0.001, 'y'],
        # ["D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0  ], 6, 0.07, 0.0, 0.0, 1000, 0.001, 'k'],
        # ["D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.07, 0.5, 0.5, 1000, 0.001, 'k']

        # ## ------------------------- Dif dimention 0.2 - plot ----------------------------------------
        # # FGM
        # ["D1-FGM", False, False, [10.0]                           , 1, 0.2, 0.0, 0.0, 1000, 0.001, 'c'],
        # ["D1-Model", False, False, [10.0]                         , 1, 0.2, 0.5, 0.5, 1000, 0.001, 'c'],
        # ["D2-FGM", False, False, [10.0, 0.0]                      , 2, 0.2, 0.0, 0.0, 1000, 0.001, 'b'],
        # ["D2-Model", False, False, [10.0, 0.0]                    , 2, 0.2, 0.5, 0.5, 1000, 0.001, 'b'],
        # ["D3-FGM", False, False, [10.0, 0.0, 0.0]                 , 3, 0.2, 0.0, 0.0, 1000, 0.001, 'm'],
        # ["D3-Model", False, False, [10.0, 0.0, 0.0]               , 3, 0.2, 0.5, 0.5, 1000, 0.001, 'm'],
        # ["D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0]            , 4, 0.2, 0.0, 0.0, 1000, 0.001, 'r'],
        # ["D4-Model", False, False, [10.0, 0.0, 0.0, 0.0]          , 4, 0.2, 0.5, 0.5, 1000, 0.001, 'r'],
        # ["D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0]       , 5, 0.2, 0.0, 0.0, 1000, 0.001, 'y'],
        # ["D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0]     , 5, 0.2, 0.5, 0.5, 1000, 0.001, 'y'],
        # ["D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  , 6, 0.2, 0.0, 0.0, 1000, 0.001, 'k'],
        # ["D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.2, 0.5, 0.5, 1000, 0.001, 'k']

        # ## ------------------------- Dif dimention - plot ----------------------------------------
        # # FGM
        # ["D1-FGM", False, False, [10.0], 1, 0.2, 0.0, 0.0, 1000, 0.001, 'c'],
        # ["D1-Model", False, False, [10.0], 1, 0.2, 0.5, 0.5, 1000, 0.001, 'c'],
        # ["D1-Model", False, False, [10.0], 1, 0.2, 0.2, 0.8, 1000, 0.001, 'c'],
        # ["D1-Model", False, False, [10.0], 1, 0.2, 0.8, 0.2, 1000, 0.001, 'c'],
        # ["D2-FGM", False, False, [10.0, 0.0], 2, 0.2, 0.0, 0.0, 1000, 0.001, 'b'],
        # ["D2-Model", False, False, [10.0, 0.0], 2, 0.2, 0.5, 0.5, 1000, 0.001, 'b'],
        # ["D3-FGM", False, False, [10.0, 0.0, 0.0], 3, 0.2, 0.0, 0.0, 1000, 0.001, 'm'],
        # ["D3-Model", False, False, [10.0, 0.0, 0.0], 3, 0.2, 0.5, 0.5, 1000, 0.001, 'm'],
        # ["D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.2, 0.0, 0.0, 1000, 0.001, 'r'],
        # ["D4-Model", False, False, [10.0, 0.0, 0.0, 0.0], 4, 0.2, 0.5, 0.5, 1000, 0.001, 'r'],
        # ["D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.2, 0.0, 0.0, 1000, 0.001, 'y'],
        # ["D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.2, 0.5, 0.5, 1000, 0.001, 'y'],
        # ["D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.2, 0.0, 0.0, 1000, 0.001, 'k'],
        # ["D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 0.2, 0.5, 0.5, 1000, 0.001, 'k']
    ]

    results = []
    graphs_fitness_val = []
    graphs_gen_val = []
    graphs_generation_val = []
    epsilon_vector = []
    color_values = []
    name_graph = []
    epsilon_generation = []

    for e in experiments:
        model = algorithms.Organism(
            fgm_mode=e[1],  # True = FG model, False = proposed model
            gen_mode=e[2],  # True = number of generations , False = until optimum is reached
            initial_point=e[3],  # Initial point
            n_dim=e[4],
            mutation_rate=e[5],  # keep rates minimum
            gen_duplication_rate=e[6],
            gen_deletion_rate=e[7],
            n_generations=e[8],
            epsilon=e[9]
        )

        best_phenotype, best_genotype, fitness_value, distance_value, i, generations, fitness_values, distance_values, \
        gen_size, gen_length = model.run()
        #print("Best phenotype: ", best_phenotype, " Best genotype:", best_genotype, " Fitness_value:", fitness_value,
        #      " Generations", i)

        results.append([
            e[3],
            e[4],
            e[5],
            e[6],
            e[7],
            e[9],
            best_phenotype,
            best_genotype,
            fitness_value,
            i,
            gen_length,
            e[0]
        ])

        graphs_fitness_val.append(fitness_values)
        graphs_gen_val.append(gen_size)
        graphs_generation_val.append(generations)
        epsilon_vector = [e[9]] * len(generations)
        #print(len(epsilon_vector))
        epsilon_generation = list(range(len(generations)))
        #print(len(epsilon_generation))
        color_values += [e[10]]
        name_graph += [e[0]]
        # print(color_values)
    #print(epsilon_vector)
    #print(epsilon_generation)
        ## Save graphs
    graphs_fitness_val.append(epsilon_vector)
    graphs_gen_val.append(epsilon_vector)
    graphs_generation_val.append(epsilon_generation)
    name_graph.append('Epsilon')
    color_values.append('g')
    fitness_graphs(graphs_fitness_val, graphs_generation_val, name_graph, color_values, e[4])
    gen_graphs(graphs_gen_val, graphs_generation_val, name_graph, color_values, e[4])
    # graph_fitnessVSgenerations(fitness_values, generations, e[0])
    # graph_distanceVSgenerations(distance_values, generations, e[0])
    # graph_numGensVSgenerations(gen_size, generations, e[0])

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write results
        writer.writerows(results)


if __name__ == '__main__':
    main()
