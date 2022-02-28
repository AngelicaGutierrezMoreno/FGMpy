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
              'Best phenotype', 'Best genotype', 'Distance to optimum', 'Total generations','Gen size', 'PlotName']
    # ['id', 'fgm_mode', 'gen_mode', 'inicial_point', 'n_dim', 'mutation_rate', 'gen_duplication_rate', 'gen_deletion_rate', 'n_generations', 'epsilon]
    experiments = [
        # 2 Dim - sample
        ['plot2D1', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D2', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D3', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D4', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D5', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D6', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],

        # 2 Dim - mutation changes
        ['plot2D-mut1', False, False, [10.0, 10.0], 2, 0.01, 0.5, 0.5, 10, 0.5],
        ['plot2D-mut2', False, False, [10.0, 10.0], 2, 0.05, 0.5, 0.5, 10, 0.5],
        ['plot2D-mut3', False, False, [10.0, 10.0], 2, 0.1, 0.5, 0.5, 10, 0.5],
        ['plot2D-mut4', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.5, 10, 0.5],
        ['plot2D-mut5', False, False, [10.0, 10.0], 2, 0.3, 0.5, 0.5, 10, 0.5],
        ['plot2D-mut6', False, False, [10.0, 10.0], 2, 0.4, 0.5, 0.5, 10, 0.5],

        # 2 Dim - duplication changes
        ['plot2D-del1', False, False, [10.0, 10.0], 2, 0.2, 0.2, 0.5, 10, 0.5],
        ['plot2D-del2', False, False, [10.0, 10.0], 2, 0.2, 0.3, 0.5, 10, 0.5],
        ['plot2D-del3', False, False, [10.0, 10.0], 2, 0.2, 0.4, 0.5, 10, 0.5],
        ['plot2D-del4', False, False, [10.0, 10.0], 2, 0.2, 0.6, 0.5, 10, 0.5],
        ['plot2D-del5', False, False, [10.0, 10.0], 2, 0.2, 0.7, 0.5, 10, 0.5],
        ['plot2D-del6', False, False, [10.0, 10.0], 2, 0.2, 0.8, 0.5, 10, 0.5],

        # 2 Dim - deletion changes
        ['plot2D1', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.2, 10, 0.5],
        ['plot2D2', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.3, 10, 0.5],
        ['plot2D3', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.4, 10, 0.5],
        ['plot2D4', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.6, 10, 0.5],
        ['plot2D5', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.7, 10, 0.5],
        ['plot2D6', False, False, [10.0, 10.0], 2, 0.2, 0.5, 0.8, 10, 0.5],

        #2 Dim - changes
        ['plot2D-ch1', False, False, [10.0, 10.0], 2, 0.1, 0.7, 0.3, 10, 0.5],
        ['plot2D-ch2', False, False, [10.0, 10.0], 2, 0.1, 0.2, 0.6, 10, 0.5],
        ['plot2D-ch3', False, False, [10.0, 10.0], 2, 0.3, 0.8, 0.2, 10, 0.5],

        # # 3 Dim - sample
        # ['plot3D1', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D2', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D3', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D4', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D5', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D6', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        #
        # # 3 Dim - mutation
        # ['plot3D-mut1', False, False, [10.0, 10.0, 10.0], 3, 0.01, 0.5, 0.5, 10, 0.5],
        # ['plot3D-mut2', False, False, [10.0, 10.0, 10.0], 3, 0.05, 0.5, 0.5, 10, 0.5],
        # ['plot3D-mut3', False, False, [10.0, 10.0, 10.0], 3, 0.1, 0.5, 0.5, 10, 0.5],
        # ['plot3D-mut4', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot3D-mut5', False, False, [10.0, 10.0, 10.0], 3, 0.3, 0.5, 0.5, 10, 0.5],
        # ['plot3D-mut6', False, False, [10.0, 10.0, 10.0], 3, 0.4, 0.5, 0.5, 10, 0.5],
        #
        # # 3 Dim - duplication changes
        # ['plot3D-dup1', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.2, 0.5, 10, 0.5],
        # ['plot3D-dup2', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.3, 0.5, 10, 0.5],
        # ['plot3D-dup3', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.4, 0.5, 10, 0.5],
        # ['plot3D-dup4', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.6, 0.5, 10, 0.5],
        # ['plot3D-dup5', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.7, 0.5, 10, 0.5],
        # ['plot3D-dup6', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.8, 0.5, 10, 0.5],
        #
        # # 3 Dim - deletion changes
        # ['plot3D-del1', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.2, 10, 0.5],
        # ['plot3D-del2', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.3, 10, 0.5],
        # ['plot3D-del3', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.4, 10, 0.5],
        # ['plot3D-del4', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.6, 10, 0.5],
        # ['plot3D-del5', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.7, 10, 0.5],
        # ['plot3D-del6', False, False, [10.0, 10.0, 10.0], 3, 0.2, 0.5, 0.8, 10, 0.5],
        #
        # # 3 Dim - changes
        # ['plot3D-ch1', False, False, [10.0, 10.0, 10.0], 3, 0.1, 0.7, 0.3, 10, 0.5],
        # ['plot3D-ch2', False, False, [10.0, 10.0, 10.0], 3, 0.1, 0.2, 0.6, 10, 0.5],
        # ['plot3D-ch3', False, False, [10.0, 10.0, 10.0], 3, 0.3, 0.8, 0.2, 10, 0.5],
        #
        # # 4 Dim - sample
        # ['plot4D1', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D2', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D3', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D4', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D5', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D6', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        #
        # # 4 Dim - mutation
        # ['plot4D-mut1', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.01, 0.5, 0.5, 10, 0.5],
        # ['plot4D-mut2', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.05, 0.5, 0.5, 10, 0.5],
        # ['plot4D-mut3', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.1, 0.5, 0.5, 10, 0.5],
        # ['plot4D-mut4', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.5, 10, 0.5],
        # ['plot4D-mut5', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.3, 0.5, 0.5, 10, 0.5],
        # ['plot4D-mut6', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.4, 0.5, 0.5, 10, 0.5],
        #
        # # 4 Dim - duplication changes
        # ['plot4D-dup1', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.2, 0.5, 10, 0.5],
        # ['plot4D-dup2', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.3, 0.5, 10, 0.5],
        # ['plot4D-dup3', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.4, 0.5, 10, 0.5],
        # ['plot4D-dup4', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.6, 0.5, 10, 0.5],
        # ['plot4D-dup5', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.7, 0.5, 10, 0.5],
        # ['plot4D-dup6', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.8, 0.5, 10, 0.5],
        #
        # # 4 Dim - deletion changes
        # ['plot4D-del1', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.2, 10, 0.5],
        # ['plot4D-del2', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.3, 10, 0.5],
        # ['plot4D-del3', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.4, 10, 0.5],
        # ['plot4D-del4', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.6, 10, 0.5],
        # ['plot4D-del5', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.7, 10, 0.5],
        # ['plot4D-del6', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.2, 0.5, 0.8, 10, 0.5],
        #
        # # 4 Dim - changes
        # ['plot4D-ch1', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.1, 0.7, 0.3, 10, 0.5],
        # ['plot3D-ch2', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.1, 0.2, 0.6, 10, 0.5],
        # ['plot4D-ch3', False, False, [10.0, 10.0, 10.0, 10.0], 4, 0.3, 0.8, 0.2, 10, 0.5]

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

        best_phenotype, best_genotype, fitness_value, distance_value, i, generations, fitness_values, distance_values, \
        gen_size = model.run()
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
            gen_size,
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
