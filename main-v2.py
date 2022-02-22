"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""
import functools
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
    # print('Genotypes length : ' + len(genotype))
    #genotype = check_dim(genotype)
    if len(genotype) == 0:
        sys.exit("Genotype does't exist anymore")
    else:
        return True, len(genotype)


def check_dim(genotype):
    print('Check dim')
    if type(genotype) == np.float64:
        #print('Type = ' + str(type(genotype)))
        genotype = [genotype]
        #if
        #print('New type = ' + str(type(organism)))
        print(genotype)
        return genotype
    else:
        return genotype


class Organism:
    def __init__(self, fgm_mode, gen_mode, initial_point, n_dim, mutation_rate, gen_duplication_rate, gen_deletion_rate,
                 n_generations, epsilon, mutation_type):
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
        # self.mutation_type = mutation_type  # def. if only mutate phenotype, genotype, both or randomly mutate one gene

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

    def sum_genes(self, genotype):
        #print('Initializing sum_genes')
        # print('length '+ str(len(genotype)))
        #genotype = check_dim(genotype)
        if len(genotype) > 0:
            organism = np.add.reduce(genotype)
            #print(type(organism))
            #print('suma de genes : ' + str(organism))
        elif len(genotype) == 0:
            sys.exit("organism doesn't exist anymore")
        else:
            print('Genotype 0 = ' + str(genotype))
            organism = genotype
        #print('Ending sum_genes')
        return organism
    # ----Declaration of events

    def get_mutation_vector(self, deviation):
        mutation_vector = NormalDist(0.0, deviation).samples(self.n_dim)
        print('Mutation vector: ' + str(mutation_vector))
        return mutation_vector

    def mutation(self, genotype, deviation):
        if exist(genotype):
            print("Iniciando mutacion")
            #check_dim(genotype)
            genotype = np.subtract(genotype, self.get_mutation_vector(deviation))
            # genotype = np.subtract(genotype, NormalDist(0.0, deviation).samples(self.n_dim))
            print('Mutated genotype: ' + str(genotype))
        else:
            print('Cannot do mutation')
        return genotype

    def duplication(self, genotype, i):
        while exist(genotype):
            # print(genotype[i])
            genotype = list(genotype)
            genotype.append(genotype[i])
            # print('After duplication' + str(genotype))
            break
        return genotype

    def deletion(self, genotype, i):
        while exist(genotype):
            # print(genotype[i])
            genotype = list(genotype)
            genotype.pop(i)
            # print('After deletion' + str(genotype))
            break
        return genotype

    def fitness_function(self, phenotype):
        """pdf of the multivari-ate normal distribution."""
        # fitness_function = multivariate_normal(0.0, 0.15)
        # fitness_value = fitness_function.pdf(phenotype)
        # fitness_value = multivariate_normal.pdf(phenotype, 0.0, 0.15)
        # NormalDist(0.0, 0.15)
        fitness_value = np.mean(multivariate_normal.pdf(phenotype))
        # statistics.mean(fitness_value)
        #print('Fitness = ' + str(fitness_value))
        return fitness_value

    def event_provability(self, rate):
        number_events = np.random.binomial(2, rate, 2)  # probability vector that indicates how many
        # duplications/deletions should happen in the genotype // binomial(n [>= 0], p [>= 0 and <=1], size=None)
        print(number_events)
        return number_events

    def duplication_loop(self, genotype, num_repeticiones):
        i = 1
        #check_dim(genotype)
        if num_repeticiones > 0:
            print("Iniciando gen duplication")
            while i <= num_repeticiones:
                if len(genotype) == 1:
                    genotype = self.duplication(genotype, 0)
                else:
                    pos = random.randint(0, len(genotype) - 1)
                    print('Position to duplicate ' + str(pos))
                    genotype = self.duplication(genotype, pos)
                    # print('Genotype after duplication : ' + str(genotype))
                i = i + 1
            print('Genotype after duplication : ' + str(genotype))
        else:
            print('No gens will be duplicated')
        return genotype

    def deletion_loop(self, genotype, num_repeticiones):
        i = 1
        #check_dim(genotype)
        if num_repeticiones > 0:
            print("Iniciando gen deletion")
            while i <= num_repeticiones:
                pos = random.randint(0, len(genotype) - 1)
                print('Position to delete ' + str(pos))
                genotype = self.deletion(genotype, pos)
                # print('Genotype after deletion : ' + str(genotype))
                i = i + 1
            print('Genotype after deletion : ' + str(genotype))
        else:
            print('No gens will be deleted')
        return genotype

    def distance_optimum(self, phenotype):
        distance_optimum = distance.euclidean(self.get_optimum(), phenotype)
        #print('Distance = ' + str(distance_optimum))
        return distance_optimum

    def reproduction(self, genotype):
        #check_dim(genotype)
        if self.fgm_mode:
            while exist(genotype):
                genotype = self.mutation(genotype, self.mutation_rate)
                break
        elif not self.fgm_mode:
            if exist(genotype):
                [n_dup, n_del] = self.event_provability(self.gen_duplication_rate)
                # print(n_dup)
                genotype = self.duplication_loop(genotype, n_dup)
                print('------')
                genotype = self.deletion_loop(genotype, n_del)
                print('------')
                # print(len(genotype))
                genotype = self.mutation(genotype, self.mutation_rate)
                print('------')
                # print(len(genotype))
            else:
                sys.exit("Organism doesn't exist")
        return genotype

    def selection(self, son_genotype, father_genotype):
        """
        si la evaluación del fitness hijo es menor a la del fitness padre, hacer seleccion
        """
        # score_son = self.fitness(np.add.reduce(son_genotype))
        fitness_son = self.fitness_function(self.create_phenotype(self.initial_point, son_genotype))
        #fitness_son = self.fitness_function(son_genotype)
        print('Son fitness : ' + str(fitness_son))
        fitness_father = self.fitness_function(self.create_phenotype(self.initial_point, father_genotype))
        #fitness_father = self.fitness_function(father_genotype)
        print('Father fitness : ' + str(fitness_father))

        son_distance = self.distance_optimum(self.create_phenotype(self.initial_point, son_genotype))
        father_distance = self.distance_optimum(self.create_phenotype(self.initial_point, father_genotype))

        #if fitness_son < fitness_father:
        if son_distance < father_distance:
            # print("Son's phenotype " + str(son_genotype))
            return son_genotype, fitness_son, son_distance
        else:
            # print("Father's phenotype" + str(father_phenotype))
            return father_genotype, fitness_father, father_distance

    def create_phenotype(self, initial_point, genotype):
        # phenotype = np.add.reduce( genotype)
        phenotype = np.subtract(initial_point, self.sum_genes(genotype))
        return phenotype

    def graph_fitnessVSgenerations(self, fitness_Values, generations):
        # Compute the x and y coordinates
        plt.title("FITNESS vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, fitness_Values)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    # ----Declaration of the algo
    def run(self):
        initial_point = self.initial_point
        print('Starting point: ' + str(initial_point))
        father_genotype = [self.initial_genotype()]
        print('Father genotpe : ' + str(father_genotype))
        self.fitness_function(initial_point)
        self.distance_optimum(initial_point)
        father_phenotype = self.create_phenotype(initial_point, father_genotype)
        # phenotype = np.add.reduce(np.array(initial_point), ph)
        # print(phenotype)
        fitness_value = self.fitness_function(father_phenotype)
        print('Father fitness : ' + str(fitness_value))
        distance_optimum = self.distance_optimum(father_phenotype)
        print('Distance to the optimum = ' + str(distance_optimum))
        i = 0
        generations = [i]
        fitness_Values = [fitness_value]

        if self.gen_mode:
            for i in range(self.n_generations):
                print('_______________________')
                print('Generacion: ', i)
                son_genotype = self.reproduction(father_genotype)
                selected_genotype, fitness_value = self.selection(son_genotype, father_genotype)
                # selected_organism = self.create_organism(selected_genotype)
                selected_phenotype = self.sum_genes(selected_genotype)
                #selected_genotype = check_dim(selected_genotype)
                # print('Best suited organism is : ' + str(selected_organism))
                print('Best suited genotype is : ' + str(selected_genotype))
                print('Best suited phenotype is : ' + str(selected_phenotype))
                father_genotype = selected_genotype
                # father_genotype = selected_phenotype
                initial_point = selected_phenotype
                # father_genotype =
                print('Point in the graph : ' + str(initial_point))
                generations.append(i + 1)
                # print('Generations: ' + str(generations))
                fitness_Values.append(fitness_value)
                # print('Fitness values : ' + str(fitness_Values))
                # fitness_Values += fitness_value
        elif not self.gen_mode:

            while self.epsilon <= distance_optimum:  # father_phenotype == self.get_optimum() or
                print(
                    '#####################################################################################################')
                print('Generacion: ', i)
                #check_dim(father_genotype)
                son_genotype = self.reproduction(father_genotype)
                #check_dim(son_genotype)
                #son_phenotype = self.sum_genes(son_genotype)
                print('Son genotype ' + str(son_genotype))
                selected_genotype, fitness_value, selected_distance = self.selection(son_genotype, father_genotype)
                #selected_genotype = check_dim(selected_genotype)
                # selected_organism = self.create_organism(selected_genotype)
                selected_phenotype = self.sum_genes(selected_genotype)
                # print('Best suited organism is : ' + str(selected_organism))
                print('Best suited genotype is : ' + str(selected_genotype))
                # print('Best suited phenotype is : ' + str(selected_phenotype))
                father_genotype = selected_genotype
                #initial_point = selected_phenotype
                #father_phenotype = selected_phenotype
                #print('Point in the graph : ' + str(initial_point))
                i += 1
                generations.append(i + 1)
                # print('Generations: ' + str(generations))
                fitness_Values.append(fitness_value)


# =========== Main definition of the

def main():
    model = Organism(
        fgm_mode=False,  # True = FG model, False = proposed model
        gen_mode=False,  # True = number of generations , False = until optimum is reached
        initial_point=[10.0,10.0,10.0],  # Inital point in FGM
        n_dim=3,
        mutation_rate=0.8,  # keep rates minimum
        gen_duplication_rate=0.9,
        gen_deletion_rate=0.0000,
        n_generations=50,
        epsilon=10,
        mutation_type=0,  # 0 -> phenotype, 1 -> genotype, 2 -> one random gene, 3 -> both
    )

    model.run()


if __name__ == '__main__':
    main()
