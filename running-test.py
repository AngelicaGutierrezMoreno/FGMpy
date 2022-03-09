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


def gen_graphs(graphs_gen_val, graphs_generation_val, f, color_values):
    genplot = PlotClass()
    genplot.setTitle('Gen size comparation 5D')
    genplot.setXLabel('Generations')
    genplot.setYLabel('Fitness')
    genplot.setLoc('lower right')
    for i in range(int(len(graphs_gen_val))):
        genplot.appendSeries(f, graphs_generation_val[i], graphs_gen_val[i], color_values[i])
    genplot.show()
    #genplot.printSeries()
    genplot.save("GenSize_comparation 5D")

def fitness_graphs(graphs_fitness_val, graphs_generation_val, f, epsilon_vector, color_values):
    fitplot = PlotClass()
    fitplot.setTitle('Fitness comparation 5D')
    fitplot.setXLabel('Generations')
    fitplot.setYLabel('Fitness')
    fitplot.setLoc('upper right')
    for i in range(int(len(graphs_fitness_val))):
        fitplot.appendSeries(f, graphs_generation_val[i], graphs_fitness_val[i], color_values[i])
    fitplot.appendSeries('Epsilon', graphs_generation_val[-1], epsilon_vector, 'g')
    fitplot.show()
    #fitplot.printSeries()
    fitplot.save("Fitness_comparation 5D")


def main():
    filename = "results-5d.csv"
    header = ['Inicial point', 'Dimensions', 'Mutation rate', 'Dup rate', 'Del. rate', 'Epsilon',
              'Best phenotype', 'Best genotype', 'Distance to optimum', 'Total generations', 'Gen size', 'PlotName']
    # ['id', 'fgm_mode', 'gen_mode', 'inicial_point', 'n_dim', 'mutation_rate', 'gen_duplication_rate', 'gen_deletion_rate', 'n_generations', 'epsilon]
    experiments = [
        # ## ------------------------- 1D - plot ----------------------------------------
        # #FGM
        # ["plot1D1-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot1D2-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot1D3-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot1D4-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot1D5-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot1D6-FGM", False, False, [10.0], 1, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        #
        # #Model
        # ["plot1D1-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot1D2-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot1D3-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot1D4-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot1D5-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot1D6-Model", False, False, [10.0], 1, 0.05, 0.5, 0.5, 500, 0.001, 'r']

        # ## ------------------------- 2D - plot ----------------------------------------
        # # FGM
        # ["plot2D1-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot2D2-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot2D3-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot2D4-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot2D5-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        # ["plot2D6-FGM", False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.0, 500, 0.001, 'b'],
        #
        # # Model
        # ["plot2D1-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot2D2-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot2D3-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot2D4-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot2D5-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r'],
        # ["plot2D6-Model", False, False, [10.0, 0.0], 2, 0.05, 0.5, 0.5, 500, 0.001, 'r']

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

        ## ------------------------- 5D - plot ----------------------------------------

        # FGM
        ["plot5D1-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        ["plot5D2-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        ["plot5D3-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        ["plot5D4-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        ["plot5D5-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],
        ["plot5D6-FGM", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.0, 0.0, 600, 0.001, 'b'],

        # Model
        ["plot5D1-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        ["plot5D2-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        ["plot5D3-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        ["plot5D4-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        ["plot5D5-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r'],
        ["plot5D6-Model", False, False, [10.0, 0.0, 0.0, 0.0, 0.0], 5, 0.05, 0.5, 0.5, 600, 0.001, 'r']


    ]

    results = []
    graphs_fitness_val = []
    graphs_gen_val = []
    graphs_generation_val = []
    epsilon_vector = []
    color_values = []

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

        best_phenotype, best_genotype, fitness_value, distance_value, i, generations, fitness_values, distance_values, gen_size, gen_length = model.run()
        print("Best phenotype: ", best_phenotype, " Best genotype:", best_genotype, " Fitness_value:", fitness_value, " Generations", i)

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
        color_values += [e[10]]
        #print(color_values)
        #print(epsilon_vector)
        ## Save graphs

    fitness_graphs(graphs_fitness_val, graphs_generation_val, e[0], epsilon_vector, color_values)
    gen_graphs(graphs_gen_val, graphs_generation_val, e[0], color_values)
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
