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
    #plt.show()
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
    header = [ 'Inicial point', 'Dimensions', 'Mutation rate', 'Dup rate', 'Del. rate', 'Epsilon',
               'Best genotype', 'Distance to optimum']
    #['id', 'fgm_mode', 'gen_mode', 'inicial_point', 'n_dim', 'mutation_rate', 'gen_duplication_rate', 'gen_deletion_rate', 'n_generations', 'epsilon]
    experiments = [
        ['plot1', False, False, [10.0], 1, 0.5, 0.5, 0.5, 10, .5],
        ['plot2', False, False, [10.0], 1, 0.5, 0.5, 0.5, 20, .5],
        ['plot3', False, False, [10.0], 1, 0.5, 0.5, 0.5, 30, .5],
        ['plot4', False, False, [10.0], 1, 0.5, 0.5, 0.5, 40, .5],
        ['plot5', False, False, [10.0], 1, 0.5, 0.5, 0.5, 50, .5],
        ['plot6', False, False, [20.0], 1, 0.5, 0.5, 0.5, 50, .5]
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

        father_genotype, fitness_value, distance_value, i, generations, fitness_values, distance_values, gen_size = model.run()
        print("Results:")
        print("Father genotype:", father_genotype)
        print("Fitness_value:", fitness_value)
        print("Distance to optimum", distance_value)
        print("Generations", i)

        results.append([
            e[3],
            e[4],
            e[5],
            e[6],
            e[7],
            e[9],
            father_genotype,
            fitness_value,
            distance_value,
            i
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