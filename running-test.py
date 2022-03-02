import csv
from matplotlib import pyplot as plt
import algorithms


def graph_fitnessVSgenerations(fitness_values, generations, f):
    # Compute the x and y coordinates
    plt.title("FITNESS vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    # plt.show()
    file_name = 'fitnessvsgen_' + f
    plt.savefig(file_name)
    plt.clf()


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
    plt.clf()


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
    plt.clf()


def main():
    filename = "results.csv"
    header = ['Inicial point', 'Dimensions', 'Mutation rate', 'Dup rate', 'Del. rate', 'Epsilon',
              'Best phenotype', 'Best genotype', 'Distance to optimum', 'Total generations', 'Gen size', 'PlotName']
    # ['id', 'fgm_mode', 'gen_mode', 'inicial_point', 'n_dim', 'mutation_rate', 'gen_duplication_rate', 'gen_deletion_rate', 'n_generations', 'epsilon]
    experiments = [
        # # To debug
        # ['plot1D1', False, False, [10.0], 1, 0.0, 0.5, 0.5, 10, 0.001],
        # ['plot1D2', False, False, [10.0], 1, 0.0, 0.5, 0.5, 10, 0.001],
        # ['plot1D3', False, False, [10.0], 1, 0.0, 0.0, 0.5, 10, 0.001],
        # ['plot1D4', False, False, [10.0], 1, 0.0, 0.0, 0.5, 10, 0.001],
        # ['plot1D5', False, False, [10.0], 1, 0.0, 0.5, 0.0, 10, 0.001],
        # ['plot1D6', False, False, [10.0], 1, 0.0, 0.5, 0.0, 10, 0.001],

        # 2D - changes
        ['plot2D1-1mut', False, False, [10.0, 0.0], 2, 0.01, 0.5, 0.5, 10000, 0.001],
        ['plot2D2-0mut', False, False, [10.0, 0.0], 2, 0.01, 0.0, 0.0, 10000, 0.001],
        #['plot2D3-0dup', False, False, [10.0, 0.0], 2, 0.01, 0.0, 0.5, 10, 0.001],
        #['plot2D4-0dup', False, False, [10.0, 0.0], 2, 0.05, 0.0, 0.5, 10, 0.001],
        #['plot2D5-0del', False, False, [10.0, 0.0], 2, 0.1, 0.5, 0.0, 10, 0.001],
        #['plot2D6-0del', False, False, [10.0, 0.0], 2, 0.15, 0.5, 0.0, 10, 0.001]

    ]

    results = []

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
        print("Results:")
        print("Best phenotype:", best_phenotype)
        print("Best genotype:", best_genotype)
        print("Fitness_value:", fitness_value)
        print("Generations", i)

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

        ## Save graphs
        graph_fitnessVSgenerations(fitness_values, generations, e[0])
        graph_distanceVSgenerations(distance_values, generations, e[0])
        graph_numGensVSgenerations(gen_size, generations, e[0])

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write results
        writer.writerows(results)


if __name__ == '__main__':
    main()
