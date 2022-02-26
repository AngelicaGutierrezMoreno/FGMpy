"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""
import functools
import math
import statistics
import sys
# import statistics as stat
from scipy.stats import multivariate_normal  # multivariate normal distribution
from statistics import NormalDist  # Normal distribution

# importing operator for operator functions
import operator

import numpy as np
import random
from scipy.spatial import distance
from matplotlib import pyplot as plt


def exist(genotype):
    # print('Genotypes length : ' + str(len(genotype)))
    # check_dim(genotype)
    if len(genotype) == 0:
        sys.exit("Genotype does't exist anymore")
    else:
        # print('Length genotype is ' + str(len(genotype)) + 'in exist(genotype) function')
        return True


def check_dim(genotype):
    print('Check dim')
    if type(genotype) == np.float64:
        print('Type = ' + str(type(genotype)))
        genotype = [genotype]
        print('New type = ' + str(type(genotype)))
        print(genotype)
        return genotype
    else:
        return genotype


def sum_genes(genotype):
    # print('Initializing sum_genes')
    # print('Genotype sum ' + str(genotype))
    if len(genotype) > 0:
        phenotype = np.add.reduce(genotype)
        # print(type(phenotype))
        # print('suma de genes : ' + str(phenotype))
    elif len(genotype) == 0:
        sys.exit("organism doesn't exist anymore")
    else:
        # print('Genotype 0 = ' + str(genotype))
        phenotype = genotype
    # print('Ending sum_genes')
    return phenotype

    # ----Declaration of events


def duplication(genotype, i):
    while exist(genotype):
        #print(genotype[i])
        genotype = list(genotype)
        genotype.append(genotype[i])
        #print('After duplication' + str(genotype))
        break
    return genotype


def deletion(genotype, i):
    while exist(genotype):
        print(genotype[i])
        genotype = list(genotype)
        genotype.pop(i)
        print('After deletion' + str(genotype))
        break
    return genotype


def fitness_function(phenotype):
    """pdf of the multivari-ate normal distribution."""
    fitness_value = np.mean(multivariate_normal.pdf(phenotype))
    return fitness_value


def duplication_loop(genotype, num_repeticiones):
    i = 1
    if num_repeticiones > 0:
        print("Iniciando gen duplication")
        while i <= num_repeticiones:
            if len(genotype) == 1:
                genotype = duplication(genotype, 0)
            else:
                pos = random.randint(0, len(genotype) - 1)
                # print('Position to duplicate ' + str(pos))
                genotype = duplication(genotype, pos)
                # print('Genotype after duplication : ' + str(genotype))
            i = i + 1
        print('Genotype after duplication : ' + str(genotype))
    else:
        print('No gens will be duplicated')
    return genotype


def deletion_loop(genotype, num_repeticiones):
    i = 1
    # check_dim(genotype)
    if num_repeticiones > 0:
        print("Iniciando gen deletion")
        while i <= num_repeticiones:
            pos = random.randint(0, len(genotype) - 1)
            # print('Position to delete ' + str(pos))
            genotype = deletion(genotype, pos)
            # print('Genotype after deletion : ' + str(genotype))
            i = i + 1
        print('Genotype after deletion : ' + str(genotype))
    else:
        print('No gens will be deleted')
    return genotype


def graph_fitnessVSgenerations(fitness_values, generations):
    # Compute the x and y coordinates
    plt.title("FITNESS vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, fitness_values)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()


def graph_distanceVSgenerations(distance_values, generations):
    # Compute the x and y coordinates
    plt.title("DISTANCE vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, distance_values)
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.show()


def graph_numGensVSgenerations(gen_size, generations):
    # Compute the x and y coordinates
    plt.title("GEN SIZE vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, gen_size)
    plt.xlabel('Generation')
    plt.ylabel('Gen Size')
    plt.show()


def graph_distanceVSoptimum(distance_values, optimum):
    # Compute the x and y coordinates
    plt.title("DISTANCE vs OPTIMUM")
    # Plot the points using matplotlib
    plt.plot(optimum, distance_values)
    plt.xlabel('Optimum')
    plt.ylabel('Distance')
    plt.show()


def graph_numMutVSgenerations(number_mutations, generations):
    # Compute the x and y coordinates
    plt.title("NUMBER OF ACUMULATED MUTATIONS vs GENERATIONS")
    # Plot the points using matplotlib
    plt.plot(generations, number_mutations)
    plt.xlabel('Generation')
    plt.ylabel('Acumulated mutations')
    plt.show()


def print_graphs(generations, fitness_values, distance_values, gen_size): #, number_mutations):
    graph_fitnessVSgenerations(fitness_values, generations)
    graph_distanceVSgenerations(distance_values, generations)
    graph_numGensVSgenerations(gen_size, generations)
    #graph_numMutVSgenerations(number_mutations, generations)


def print_FGMgraphs(generations, fitness_values, distance_values):#, number_mutations):
    graph_fitnessVSgenerations(fitness_values, generations)
    graph_distanceVSgenerations(distance_values, generations)
    #graph_numMutVSgenerations(number_mutations, generations)


def add_to_elements(generations, i, fitness_values, fitness_value, distance_values, distance_value,
                    gen_size, gen_len): #, number_mutations, nm):
    generations.append(i + 1)
    fitness_values.append(fitness_value)
    distance_values.append(distance_value)
    gen_size.append(gen_len)
    #number_mutations.append(nm)


def add_to_elements_mutation(generations, i, fitness_values, fitness_value, distance_values, distance_value,
                             number_mutations, nm):
    generations.append(i)
    fitness_values.append(fitness_value)
    distance_values.append(distance_value)
    number_mutations.append(nm)


def is_favorable(son_genotype, father_genotype, organism):
    son_distance = Organism.distance_optimum(organism, son_genotype)
    father_distance = Organism.distance_optimum(organism, father_genotype)
    if son_distance < father_distance:
        return True
    else:
        # create new father
        new_father = Organism.create_father(organism)
        return new_father


class Organism:
    def __init__(self, fgm_mode, gen_mode, initial_point, n_dim, mutation_rate, gen_duplication_rate, gen_deletion_rate,
                 n_generations, epsilon):
        self.fgm_mode = fgm_mode  # Determinate if we're using FGM or the model proposed
        self.gen_mode = gen_mode  # Determinate if we continue the evaluation until the optimum is reached or if we
        # continue until the number of generations is met
        self.initial_point = initial_point  # initial point where the model will part
        self.n_dim = n_dim  # number of dimensions of the phenotype
        self.mutation_rate = mutation_rate  # keep to minimum
        self.gen_duplication_rate = gen_duplication_rate  # keep to minimum
        self.gen_deletion_rate = gen_deletion_rate  # keep to minimum
        self.n_generations = n_generations  # number of generations to be evaluated
        self.epsilon = epsilon  # maximum range of distance to the optimum

    # ----------------------------------FUNCTION DECLARATION-----------------------------------------#

    # ----Creation of static vectors

    def get_optimum(self):  # optimum will always be at 0
        optimum = [0.0] * self.n_dim
        # print('Optimum vector : ' + str(optimum))
        return optimum

    def initial_genotype(self):
        genotype = NormalDist(0.0, 0.5).samples(self.n_dim)
        # print(len(genotype))
        return genotype

    def get_mutation_vector(self, deviation):
        mutation_vector = NormalDist(0.0, deviation).samples(self.n_dim)
        # print('Mutation vector: ' + str(mutation_vector))
        return mutation_vector

    def mutation(self, genotype, deviation): #, nm):
        if exist(genotype):
            print("Iniciando mutacion")
            i = 0
            while i <= len(genotype) - 1:
                if self.fgm_mode:
                    genotype = np.subtract(genotype, self.get_mutation_vector(deviation))
                else:
                    genotype[i] = np.subtract(genotype[i], self.get_mutation_vector(deviation))
                i += 1
            #nm += 1
            print('Mutated genotype: ' + str(genotype))
        else:
            print('Cannot do mutation')
        return genotype #, nm

    def event_provability(self, genotype):
        number_events = np.random.binomial(len(genotype), 0.5)
        print('number events = ' + str(number_events))
        number_duplications = math.ceil(number_events*self.gen_duplication_rate)
        print('num dup: ' + str(number_duplications))
        number_deletions = math.ceil(number_events*self.gen_deletion_rate)
        print('Num delet ' + str(number_deletions))
        return number_duplications, number_deletions

    def event_selection(self, genotype, number_events):
        number_duplications = np.random.binomial(len(genotype), self.gen_duplication_rate)
        # if not (number_events-number_duplications) > 0:
        #    number_deletions = np.random.binomial((number_events-number_duplications), self.gen_deletion_rate)
        number_deletions = np.random.binomial(len(genotype), self.gen_deletion_rate)
        print('# dup = ' + str(number_duplications) + ' # delet = ' + str(number_deletions))
        return number_duplications, number_deletions

    def distance_optimum(self, phenotype):
        distance_optimum = distance.euclidean(self.get_optimum(), phenotype)
        # print('Distance = ' + str(distance_optimum))
        return distance_optimum

    def reproduction(self, genotype): #, nm):
        # check_dim(genotype)
        if self.fgm_mode:
            while exist(genotype):
                genotype = self.mutation(genotype, self.mutation_rate)  # , nm)
                break
        elif not self.fgm_mode:
            if exist(genotype):
                # number_events = self.event_provability(genotype)
                # [n_dup, n_del] = self.event_selection(number_events)
                # rate = self.gen_duplication_rate
                [n_dup, n_del] = self.event_provability(genotype)
                # print(n_dup)
                genotype = duplication_loop(genotype, n_dup)
                print('------')
                genotype = deletion_loop(genotype, n_del)
                print('------')
                # print(len(genotype))
                genotype = self.mutation(genotype, self.mutation_rate) #, nm)
                print('------')
                # print(len(genotype))
            else:
                sys.exit("Organism doesn't exist")
        return genotype #, nm

    def selection(self, son_genotype, father_genotype):
        """
        si la evaluación del fitness hijo es menor a la del fitness padre, hacer seleccion
        """
        [fitness_son, son_distance, size_son] = self.evaluate(son_genotype)
        [fitness_father, father_distance, size_father] = self.evaluate(father_genotype)
        print('Son genotype: ' + str(son_genotype))
        print('Son fitness : ' + str(fitness_son))
        print('Son distance to optimum ' + str(son_distance))
        print('Son size: ' + str(size_son))
        print('Father genotype: ' + str(father_genotype))
        print('Father fitness : ' + str(fitness_father))
        print('Father distance to optimum ' + str(father_distance))
        print('Father size: ' + str(size_father))
        # if fitness_son < fitness_father:
        if son_distance <= father_distance:
            # print("Son's phenotype " + str(son_genotype))
            return son_genotype, fitness_son, son_distance
        else:
            # print("Father's phenotype" + str(father_phenotype))
            return father_genotype, fitness_father, father_distance

    def create_phenotype(self, initial_point, genotype):
        # phenotype = np.add.reduce( genotype)
        if self.fgm_mode:
            phenotype = np.add(initial_point, genotype)
        else:
            phenotype = np.add(initial_point, sum_genes(genotype))
        return phenotype

    def create_father(self):
        if self.fgm_mode:
            father_genotype = self.initial_genotype()
        else:
            father_genotype = [self.initial_genotype()]
        return father_genotype

    def get_father_data(self, initial_point):
        father_genotype = self.create_father()
        father_phenotype = self.create_phenotype(initial_point, father_genotype)
        return father_genotype, father_phenotype

    def evaluate(self, genotype):
        print('Entering evaluate function')
        print('Genotype to evaluate : ' + str(genotype))
        fitness_value = fitness_function(self.create_phenotype(self.initial_point, genotype))
        distance_value = self.distance_optimum(self.create_phenotype(self.initial_point, genotype))
        size_value = len(genotype)
        return fitness_value, distance_value, size_value

    def initial_evaluation(self, initial_point):
        fitness_value = fitness_function(initial_point)
        distance_value = self.distance_optimum(initial_point)
        return fitness_value, distance_value

    def model_FGM(self, i, father_phenotype, generations, fitness_values, distance_values, number_mutations): #,nm
        print('Generacion: ', i)
        #son_phenotype, nm = self.mutation(father_phenotype, self.mutation_rate, nm)
        son_phenotype = self.mutation(father_phenotype, self.mutation_rate)
        print('Son ' + str(son_phenotype))
        print('Number_mut = ' + str(number_mutations))
        [selected_phenotype, fitness_value, distance_value] = self.selection(son_phenotype, father_phenotype)
        print('Selected phenotype : ' + str(selected_phenotype))
        print('Fitness of selected phenotype : ' + str(fitness_value))
        print('Distance to optimum : ' + str(distance_value))

        father_phenotype = selected_phenotype
        distance_value = distance_value
        add_to_elements_mutation(generations, i, fitness_values, fitness_value, distance_values, distance_value,
                                 number_mutations) #, nm)
        return generations, fitness_values, distance_values, number_mutations

    def model(self, i, father_genotype, generations, fitness_values, distance_values, gen_size, number_mutations): #, nm):
        print('#######################################################')
        print('Generation: ', i)
        #son_genotype, nm = self.reproduction(father_genotype, nm)
        son_genotype = self.reproduction(father_genotype)
        selected_genotype, fitness_value, distance_value = self.selection(son_genotype, father_genotype)
        selected_phenotype = self.create_phenotype(self.initial_point, selected_genotype)
        father_genotype = selected_genotype
        gen_len = len(selected_genotype)
        i += 1
        add_to_elements(generations, i, fitness_values, fitness_value, distance_values,
                        distance_value, gen_size, gen_len) #, number_mutations #, nm)
        #print('Number_ mutations' + str(nm))
        print('Son genotype ' + str(son_genotype))
        print('Best suited genotype is : ' + str(selected_genotype))
        print('Best suited phenotype is : ' + str(selected_phenotype))
        print('Distancia del óptimo: ' + str(distance_value))
        initial_point = selected_phenotype
        father_phenotype = selected_phenotype
        print('Point in the graph : ' + str(initial_point))
        print('Generations: ' + str(generations))
        print('Gen size vector : ' + str(gen_size))
        return generations, fitness_values, distance_values, gen_size, number_mutations

    # ----Declaration of the algo
    def run(self):
        initial_point = self.initial_point
        [initial_fitness_value, initial_distance_value] = self.initial_evaluation(initial_point)
        print('Starting point: ' + str(initial_point))
        print('Fitness initial point: ' + str(initial_fitness_value))
        print('Distance to optimum from initial point: ' + str(initial_distance_value))

        [father_genotype, father_phenotype] = self.get_father_data(initial_point)
        [fitness_value, distance_value, size_value] = self.evaluate(father_genotype)

        print('Father genotype : ' + str(father_genotype))
        print('Father phenotype : ' + str(father_phenotype))
        print('Father fitness : ' + str(fitness_value))
        print('Distance to the optimum = ' + str(distance_value))
        i = 0
        nm = 1
        number_mutations = [nm]
        print('num_mut = ' + str(number_mutations))
        generations = [i]
        fitness_values = [fitness_value]
        distance_values = [distance_value]
        gen_size = [size_value]

        if self.fgm_mode:
            # Inicia FGM
            while not father_phenotype.any():
                if self.gen_mode:
                    # Initialize FGM for determined number of generations
                    for i in range(self.n_generations):
                        generations, fitness_values, distance_values, number_mutations = \
                            self.model_FGM(i, father_phenotype, generations, fitness_values, distance_values,
                                           number_mutations)  # , nm)
                        print_FGMgraphs(generations, fitness_values, distance_values)#, number_mutations)
                else:
                    # Initialize FGM until optimum is reached
                    while self.epsilon <= distance_value:
                        generations, fitness_values, distance_values, number_mutations = \
                            self.model_FGM(i, father_phenotype, generations, fitness_values, distance_values,
                                           number_mutations) #, nm)
                        print_FGMgraphs(generations, fitness_values, distance_values)#, number_mutations)
        else:
            # Inicia modelo propuesto
            while exist(father_genotype):
                if self.gen_mode:
                    # Initialize model for determined number of generations
                    for i in range(self.n_generations):
                        generations, fitness_values, distance_values, gen_size, number_mutations = \
                            self.model(i, father_genotype, generations, fitness_values, distance_values, gen_size,
                                       number_mutations) #, nm)
                        print_graphs(generations, fitness_values, distance_values, gen_size)#, number_mutations)
                else:
                    # Initialize model until optimum is reached
                    while self.epsilon <= distance_value:
                        generations, fitness_values, distance_values, gen_size, number_mutations = \
                            self.model(i, father_genotype, generations, fitness_values, distance_values, gen_size,
                                       number_mutations) #, nm)
                        print_graphs(generations, fitness_values, distance_values, gen_size)#, number_mutations)

        # if is_favorable(distance_value, initial_distance_value, self):
        #
        # else:
        #     sys.exit('Not favorable')


def main():
    model = Organism(
        fgm_mode=False,  # True = FG model, False = proposed model
        gen_mode=False,  # True = number of generations , False = until optimum is reached
        initial_point=[10.0, 10.0, 10.0],  # Inital point in FGM
        n_dim=3,
        mutation_rate=0.8,  # keep rates minimum
        gen_duplication_rate=0.9,
        gen_deletion_rate=0.4,
        n_generations=50,
        epsilon=5
    )

    model.run()


if __name__ == '__main__':
    main()
