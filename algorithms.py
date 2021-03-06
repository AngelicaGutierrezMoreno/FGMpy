"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""
import copy
import functools
import math
import statistics
import sys
# import statistics as stat
from scipy.stats import multivariate_normal  # multivariate normal distribution
from statistics import NormalDist  # Normal distribution

import pandas as pd
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
        return False
    else:
        # print('Length genotype is ' + str(len(genotype)) + 'in exist(genotype) function')
        return True

    # ----Declaration of events


def duplication(genotype, i):
    while exist(genotype):
        # print(genotype[i])
        genotype = list(genotype)
        genotype.append(genotype[i])
        # print('After duplication' + str(genotype))
        break
    return genotype


def deletion(genotype, i):
    while exist(genotype):
        # print(genotype[i])
        genotype = list(genotype)
        genotype.pop(i)
        # print('After deletion' + str(genotype))
        break
    return genotype


def fitness_function(phenotype):
    """pdf of the multivari-ate normal distribution."""
    # fitness_value = np.mean(multivariate_normal.pdf(phenotype))
    fitness_value = np.linalg.norm(phenotype)
    return fitness_value


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


def print_graphs(generations, fitness_values, distance_values, gen_size):  # , number_mutations):
    graph_fitnessVSgenerations(fitness_values, generations)
    graph_distanceVSgenerations(distance_values, generations)
    graph_numGensVSgenerations(gen_size, generations)
    # graph_numMutVSgenerations(number_mutations, generations)


def print_FGMgraphs(generations, fitness_values, distance_values):  # , number_mutations):
    graph_fitnessVSgenerations(fitness_values, generations)
    graph_distanceVSgenerations(distance_values, generations)
    # graph_numMutVSgenerations(number_mutations, generations)


def add_to_elements(fitness_values, fitness_value, distance_values,
                    distance_value, generations, gen_size, gen_len, i):  # , number_mutations #, nm):
    generations.append(i)
    fitness_values.append(fitness_value)
    distance_values.append(distance_value)
    gen_size.append(gen_len)
    # number_mutations.append(nm)
    # print('Generations: ' + str(generations))
    # print('Gen size vector : ' + str(gen_size))


def add_to_elements_mutation(fitness_values, fitness_value, distance_values, distance_value, generations,
                             i):  # ,number_mutations, nm):
    generations.append(i)
    fitness_values.append(fitness_value)
    distance_values.append(distance_value)
    # number_mutations.append(nm)
    # print('Generations: ' + str(generations))


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
        genotype = NormalDist(0.0, 0.1).samples(self.n_dim)
        # print(len(genotype))
        return genotype

    def get_mutation_vector(self, deviation):
        mutation_vector = NormalDist(0.0, deviation).samples(self.n_dim)
        # print('Mutation vector: ' + str(mutation_vector))
        return mutation_vector

    def mutation(self, genotype, deviation):  # , nm):
        if exist(genotype):
            # print("Iniciando mutacion")
            i = 0
            while i <= len(genotype) - 1:
                if self.fgm_mode:
                    genotype = np.subtract(genotype, self.get_mutation_vector(deviation))
                else:
                    genotype[i] = np.subtract(genotype[i], self.get_mutation_vector(deviation))
                i += 1
            # nm += 1
            # print('Mutated genotype: ' + str(genotype))
        return genotype  # , nm

    def sum_genes(self, genotype):
        # print('Initializing sum_genes')
        # print('Genotype sum ' + str(genotype))
        if len(genotype) >= 1:
            phenotype = np.add.reduce(genotype)
            # print(type(phenotype))
            # print('suma de genes : ' + str(phenotype))
        elif len(genotype) == 0:
            # print("organism doesn't exist anymore")
            phenotype = self.initial_point
            # quit()
        else:
            # print('Genotype 0 = ' + str(genotype))
            phenotype = genotype
        # print('Ending sum_genes')
        return phenotype

    def event_provability(self, genotype):
        probability = (self.gen_duplication_rate + self.gen_deletion_rate) # * len(genotype)
        #print(probability)
        number_events = np.random.binomial(len(genotype), probability)
        #print('number events = ' + str(number_events))
        # number_duplications = math.ceil(number_events * self.gen_duplication_rate)
        # print('num dup: ' + str(number_duplications))
        # number_deletions = math.ceil(number_events * self.gen_deletion_rate)
        # print('Num delet ' + str(number_deletions))
        return number_events

    def distance_optimum(self, phenotype):
        distance_optimum = distance.euclidean(self.get_optimum(), phenotype)
        # print('Distance = ' + str(distance_optimum))
        return distance_optimum

    def reproduction(self, x_genotype):  # , nm):
        # check_dim(genotype)
        # print('Iniciando reproducci??n')
        genotype = x_genotype
        if self.fgm_mode:
            while exist(genotype):
                genotype = self.mutation(genotype, self.mutation_rate)  # , nm)
                break
        elif not self.fgm_mode:
            if exist(genotype):
                # number_events = self.event_provability(genotype)
                # [n_dup, n_del] = self.event_selection(number_events)
                # rate = self.gen_duplication_rate
                number_events = self.event_provability(genotype)
                # print('Number of events to do: ' + str(number_events))
                e = 1
                while e <= number_events:
                    pos = random.randint(0, len(genotype) - 1)
                    # print('gen position to duplicate: ' + str(pos))
                    prob = random.uniform(0, np.add(self.gen_duplication_rate, self.gen_deletion_rate))
                    # print('Probability random: ' + str(prob))
                    if prob <= self.gen_deletion_rate:
                        # print('Deletion won')
                        genotype = deletion(genotype, pos)
                    else:
                        # print('Duplication won')
                        genotype = duplication(genotype, pos)
                    e += 1
                    # print('Genotype ' + str(genotype))
                # print(n_dup)
                # for each event, choose random gene from the the father (so no duplication of gene twice)
                # the choose which event to do.
                # Suma los rates de los eventos y si es menor que deletion rate es una deletion if its higher es una duplicacion
                # genotype = duplication_loop(genotype, n_dup)
                # print('------')
                # genotype = deletion_loop(genotype, n_del)
                # print('------')
                # print(len(genotype))
                genotype = self.mutation(genotype, self.mutation_rate)  # , nm)
                # print('------')
                # print(len(genotype))
            else:
                # print("Organism doesn't exist")
                # sys.exit("Organism doesn't exist")
                genotype = [self.initial_point]
        return genotype  # , nm

    def selection(self, son_genotype, father_genotype):
        """
        si la evaluaci??n del fitness hijo es menor a la del fitness padre, hacer seleccion
        """
        [fitness_son, son_distance, size_son] = self.evaluate(son_genotype)
        [fitness_father, father_distance, size_father] = self.evaluate(father_genotype)
        # print('Son genotype: ' + str(son_genotype))
        # print('Son fitness : ' + str(fitness_son))
        # print('Son distance to optimum ' + str(son_distance))
        # print('Son size: ' + str(size_son))
        # print('Father genotype: ' + str(father_genotype))
        # print('Father fitness : ' + str(fitness_father))
        # print('Father distance to optimum ' + str(father_distance))
        # print('Son distance to optimum ' + str(son_distance))
        # print('Father size: ' + str(size_father))
        # if fitness_son < fitness_father:
        if son_distance <= father_distance:
            # print("Son's phenotype " + str(son_genotype))
            # print("Keep son")
            return son_genotype, fitness_son, son_distance, size_son
        else:
            # print("Father's phenotype" + str(father_genotype))
            # print("keep father")
            return father_genotype, fitness_father, father_distance, size_father

    def selection_FGM(self, son_phenotype, father_phenotype):
        """
        si la evaluaci??n del fitness hijo es menor a la del fitness padre, hacer seleccion
        """
        fitness_son, son_distance = self.evaluate_FGM(son_phenotype)
        fitness_father, father_distance = self.evaluate_FGM(father_phenotype)
        # print('Son genotype: ' + str(son_phenotype))
        # print('Son fitness : ' + str(fitness_son))
        # print('Son distance to optimum ' + str(son_distance))
        # print('Father genotype: ' + str(father_phenotype))
        # print('Father fitness : ' + str(fitness_father))
        # print('Father distance to optimum ' + str(father_distance))
        # print('Son distance to optimum ' + str(son_distance))
        # if fitness_son < fitness_father:
        if son_distance <= father_distance:
            # print("Son's phenotype " + str(son_genotype))
            return son_phenotype, fitness_son, son_distance
        else:
            # print("Father's phenotype" + str(father_phenotype))
            return father_phenotype, fitness_father, father_distance

    def create_phenotype(self, initial_point, genotype):
        # phenotype = np.add.reduce( genotype)
        if self.fgm_mode:
            phenotype = np.add(initial_point, genotype)
        else:
            if exist(genotype):
                phenotype = np.add(initial_point, self.sum_genes(genotype))
            else:
                phenotype = self.initial_point
        return phenotype

    def create_father(self):
        if self.fgm_mode:
            father_genotype = self.initial_genotype()
        else:
            father_genotype = [self.initial_genotype()]
            # father_genotype = [[-1.0]]
        return father_genotype

    def get_father_data(self, initial_point):
        father_genotype = self.create_father()
        father_phenotype = self.create_phenotype(initial_point, father_genotype)
        return father_genotype, father_phenotype

    def evaluate(self, genotype):
        # print('Entering evaluate function')
        # print('Genotype to evaluate : ' + str(genotype))
        fitness_value = fitness_function(self.create_phenotype(self.initial_point, genotype))
        distance_value = self.distance_optimum(self.create_phenotype(self.initial_point, genotype))
        size_value = len(genotype)
        return fitness_value, distance_value, size_value

    def evaluate_FGM(self, phenotype):
        # print('Entering evaluate function')
        # print('Genotype to evaluate : ' + str(genotype))
        fitness_value = fitness_function(phenotype)
        distance_value = self.distance_optimum(phenotype)
        return fitness_value, distance_value

    def initial_evaluation(self, initial_point):
        fitness_value = fitness_function(initial_point)
        distance_value = self.distance_optimum(initial_point)
        return fitness_value, distance_value

    def is_favorable(self, son_genotype, father_genotype):
        son_distance = self.distance_optimum(son_genotype)
        father_distance = Organism.distance_optimum(father_genotype)
        if son_distance < father_distance:
            return True
        else:
            # create new father
            new_father = self.create_father()
            return new_father

    def model_FGM(self, father_phenotype):  # , number_mutations ,nm
        # son_phenotype, nm = self.mutation(father_phenotype, self.mutation_rate, nm)
        son_phenotype = self.mutation(father_phenotype, self.mutation_rate)
        # print('Son ' + str(son_phenotype))
        # print('Number_mut = ' + str(number_mutations))
        selected_phenotype, fitness_value, distance_value = self.selection_FGM(son_phenotype,
                                                                               father_phenotype)
        # print('Selected phenotype : ' + str(selected_phenotype))
        # print('Fitness of selected phenotype : ' + str(fitness_value))
        # print('Distance to optimum : ' + str(distance_value))

        return selected_phenotype, fitness_value, distance_value  # , number_mutations

    def model(self, father_genotype):  # , nm):
        # son_genotype, nm = self.reproduction(father_genotype, nm)
        son_genotype = self.reproduction(copy.deepcopy(father_genotype))
        # print("Father---", father_genotype)
        # print("Son-----", son_genotype)
        selected_genotype, fitness_selected, distance_selected, size_selected = self.selection(son_genotype,
                                                                                               father_genotype)
        selected_phenotype = self.create_phenotype(self.initial_point, selected_genotype)
        # print('Number_ mutations' + str(nm))
        # print('Son genotype ' + str(son_genotype))
        # print('Best suited genotype is : ' + str(selected_genotype))
        # print('Best suited phenotype is : ' + str(selected_phenotype))
        # print('Distancia del ??ptimo: ' + str(distance_selected))
        # print('Point in the graph : ' + str(initial_point))
        return selected_genotype, fitness_selected, distance_selected, size_selected

    def get_favorable_gene(self, initial_fitness_value):
        # print('initial fitness: ' + str(initial_fitness_value))
        while True:
            [father_genotype, father_phenotype] = self.get_father_data(self.initial_point)
            [fitness_value, distance_value, size_value] = self.evaluate(father_genotype)
            # print('Try Father genotype  : ' + str(father_genotype))
            # print('Try Father phenotype : ' + str(father_phenotype))
            # print('TRy Father fitness : ' + str(fitness_value))
            # print('Try Distance to the optimum = ' + str(distance_value))
            if fitness_value < initial_fitness_value:
                # print('Gene is favorable')
                break
        return father_genotype, father_phenotype, fitness_value, distance_value, size_value

    # ----Declaration of the algo
    def run(self):
        initial_point = self.initial_point
        [initial_fitness_value, initial_distance_value] = self.initial_evaluation(initial_point)
        #print('Starting point: ' + str(initial_point))
        #print('Fitness initial point: ' + str(initial_fitness_value))
        #print('Distance to optimum from initial point: ' + str(initial_distance_value))

        father_genotype, father_phenotype, fitness_value, distance_value, size_value = self.get_favorable_gene(
            initial_fitness_value)

        #print('Father genotype : ' + str(father_genotype))
        #print('Father phenotype : ' + str(father_phenotype))
        #print('Father fitness : ' + str(fitness_value))
        #print('Distance to the optimum = ' + str(distance_value))
        i = 0
        # print('num_mut = ' + str(number_mutations))
        generations = [i]
        fitness_values = [fitness_value]
        distance_values = [distance_value]
        gen_size = [size_value]

        if self.fgm_mode:
            # Inicia FGM
            # print('FGM selected')
            while father_phenotype.all():
                # print('Exist fath_gen')
                if self.gen_mode:
                    # Initialize FGM for determined number of generations
                    for i in range(self.n_generations):
                        # print('########################################')
                        # print('Generacion: ', i)
                        selected_phenotype, fitness_selected, distance_selected = self.model_FGM(father_phenotype)
                        father_phenotype = selected_phenotype
                        # print('Father: ' + str(father_genotype))
                        fitness_value = fitness_selected
                        distance_value = distance_selected
                        i += 1
                        add_to_elements_mutation(fitness_values, fitness_value, distance_values, distance_value,
                                                 generations, i)  # , number_mutations, nm)
                        print('Generation: ' + str(i) + ', fitness: ' + str(fitness_value) + str(fitness_value))
                    break
                else:
                    # Initialize FGM until optimum is reached
                    while self.epsilon <= distance_value:
                        # print('########################################')
                        # print('Generacion: ', i)
                        selected_phenotype, fitness_selected, distance_selected = self.model_FGM(father_phenotype)
                        father_phenotype = selected_phenotype
                        print('Father: ' + str(father_genotype))
                        fitness_value = fitness_selected
                        distance_value = distance_selected
                        i += 1
                        add_to_elements_mutation(fitness_values, fitness_value, distance_values, distance_value,
                                                 generations, i)  # , number_mutations, nm)
                        print('Generation: ' + str(i) + ', fitness: ' + str(fitness_value))
                    break

            print_FGMgraphs(generations, fitness_values, distance_values)  # , number_mutations)
        else:
            # Inicia modelo propuesto
            while exist(father_genotype):
                if self.gen_mode:
                    # Initialize model for determined number of generations
                    for i in range(self.n_generations):
                        # print('########################################')
                        # print('Generacion: ', i)
                        selected_genotype, fitness_selected, distance_selected, size_selected = self.model(
                            father_genotype)
                        father_genotype = selected_genotype
                        # print('Father: ' + str(father_genotype))
                        fitness_value = fitness_selected
                        distance_value = distance_selected
                        gen_len = size_selected
                        i += 1
                        add_to_elements(fitness_values, fitness_value, distance_values, distance_value, generations,
                                        gen_size, gen_len, i)  # , number_mutations #, nm)
                        #print('Generation: ' + str(i) + ', fitness: ' + str(fitness_value)
                        #      + ', number of genes ' + str(gen_len))
                    break

                else:
                    # Initialize model until optimum is reached
                    while self.epsilon <= distance_value:
                        # print('########################################')
                        # print('Generacion: ', i)
                        selected_genotype, fitness_selected, distance_selected, size_selected = self.model(
                            father_genotype)  # , nm)
                        father_genotype = selected_genotype
                        # print('Father: ' + str(father_genotype))
                        fitness_value = fitness_selected
                        distance_value = distance_selected
                        gen_len = size_selected
                        i += 1
                        add_to_elements(fitness_values, fitness_value, distance_values, distance_value, generations,
                                        gen_size, gen_len, i)  # , number_mutations #, nm)
                        if i == self.n_generations:
                            break
                        #print('Generation: ' + str(i) + ', fitness: ' + str(fitness_value)
                        #      + ', number of genes: ' + str(gen_len) + ', father_phenotype: ' + str(father_phenotype) +
                        #      ', father_genotype: ' + str(father_genotype))
                    break

            # df1 = pd.DataFrame()

            # print_graphs(generations, fitness_values, distance_values, gen_size)  # , number_mutations)
            best_genotype = father_genotype
            gen_length = size_selected
            if (exist(best_genotype)):
                best_phenotype = self.create_phenotype(self.initial_point, best_genotype)
            else:
                best_phenotype = self.initial_point

            return best_phenotype, best_genotype, fitness_value, distance_value, i, generations, fitness_values, distance_values, gen_size, gen_length


def main(_fgm_mode,
         _gen_mode,
         _initial_point,
         _n_dim,
         _mutation_rate,
         _gen_duplication_rate,
         _gen_deletion_rate,
         _n_generations,
         _epsilon):
    model = Organism(
        # fgm_mode=False,  # True = FG model, False = proposed model
        # gen_mode=False,  # True = number of generations , False = until optimum is reached
        # initial_point=[10.0],  # Initial point
        # n_dim=1,
        # mutation_rate=0.5,  # keep rates minimum
        # gen_duplication_rate=0.5,
        # gen_deletion_rate=0.5,
        # n_generations=10,
        # epsilon=0.5
        fgm_mode=_fgm_mode,
        gen_mode=_gen_mode,
        initial_point=_initial_point,
        n_dim=_n_dim,
        mutation_rate=_mutation_rate,
        gen_duplication_rate=_gen_duplication_rate,
        gen_deletion_rate=_gen_deletion_rate,
        n_generations=_n_generations,
        epsilon=_epsilon
    )

    model.run()


if __name__ == '__main__':
    main(
        _fgm_mode=True,  # True = FG model, False = proposed model
        _gen_mode=False,  # True = number of generations , False = until optimum is reached
        _initial_point=[10.0, 0.0, 0.0, 0.0, 0.0],  # Initial point
        _n_dim=5,
        _mutation_rate=0.2,  # keep rates minimum
        _gen_duplication_rate=0.0,
        _gen_deletion_rate=0.0,
        _n_generations=10000,
        _epsilon=0.001
    )
