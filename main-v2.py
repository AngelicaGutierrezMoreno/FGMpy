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
    if len(genotype) == 0:
        sys.exit("Genotype does't exist anymore")
    else:
        return True, len(genotype)




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

    def sum_genes(self, genotype):
        # print('Initializing sum_genes')
        # print('length '+ str(len(genotype)))
        if len(genotype) > 0:
            organism = np.add.reduce(genotype)
            # print(type(organism))
            # print('suma de genes : ' + str(organism))
        elif len(genotype) == 0:
            sys.exit("organism doesn't exist anymore")
        else:
            # print('Genotype 0 = ' + str(genotype))
            organism = genotype
        # print('Ending sum_genes')
        return organism

    # ----Declaration of events

    def get_mutation_vector(self, deviation):
        mutation_vector = NormalDist(0.0, deviation).samples(self.n_dim)
        # print('Mutation vector: ' + str(mutation_vector))
        return mutation_vector

    def mutation(self, genotype, deviation): #,nm):
        if exist(genotype):
            print("Iniciando mutacion")
            i = 0
            while i <= len(genotype) - 1:
                genotype[i] = np.subtract(genotype[i], self.get_mutation_vector(deviation))
                i += 1
                #nm += 1
            # genotype = np.subtract(genotype, self.get_mutation_vector(deviation))
            print('Mutated genotype: ' + str(genotype))
        else:
            print('Cannot do mutation')
        return genotype #, nm

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
        # print('Fitness = ' + str(fitness_value))
        return fitness_value

    def event_provability(self, genotype):
        # number_events = np.random.binomial(2, rate, 2)  # probability vector that indicates how many
        # number_events = np.random.binomial(len(genotype), 0.5)
        # duplications/deletions should happen in the genotype // binomial(n [>= 0], p [>= 0 and <=1], size=None)
        # rate = np.mean([self.gen_duplication_rate, self.gen_deletion_rate])
        # print(len(genotype))
        # print(rate)
        number_events = np.random.binomial(len(genotype), 0.5)
        print(number_events)
        number_duplications = np.random.binomial(len(genotype), self.gen_duplication_rate)
        print(number_duplications)
        number_deletions = np.random.binomial(len(genotype), self.gen_deletion_rate)
        print(number_deletions)
        if (number_duplications + number_deletions) > number_events:
            number_deletions = number_events - number_duplications
        print(number_deletions)

        # return number_events
        return number_duplications, number_deletions

    def event_selection(self, genotype, number_events):
        number_duplications = np.random.binomial(len(genotype), self.gen_duplication_rate)
        # if not (number_events-number_duplications) > 0:
        #    number_deletions = np.random.binomial((number_events-number_duplications), self.gen_deletion_rate)
        number_deletions = np.random.binomial(len(genotype), self.gen_deletion_rate)
        print('# dup = ' + str(number_duplications) + ' # delet = ' + str(number_deletions))
        return number_duplications, number_deletions

    def duplication_loop(self, genotype, num_repeticiones):
        i = 1
        if num_repeticiones > 0:
            print("Iniciando gen duplication")
            while i <= num_repeticiones:
                if len(genotype) == 1:
                    genotype = self.duplication(genotype, 0)
                else:
                    pos = random.randint(0, len(genotype) - 1)
                    #print('Position to duplicate ' + str(pos))
                    genotype = self.duplication(genotype, pos)
                    # print('Genotype after duplication : ' + str(genotype))
                i = i + 1
            print('Genotype after duplication : ' + str(genotype))
        else:
            print('No gens will be duplicated')
        return genotype

    def deletion_loop(self, genotype, num_repeticiones):
        i = 1
        if num_repeticiones > 0:
            print("Iniciando gen deletion")
            while i <= num_repeticiones:
                pos = random.randint(0, len(genotype) - 1)
                # print('Position to delete ' + str(pos))
                genotype = self.deletion(genotype, pos)
                # print('Genotype after deletion : ' + str(genotype))
                i = i + 1
            print('Genotype after deletion : ' + str(genotype))
        else:
            print('No gens will be deleted')
        return genotype

    def distance_optimum(self, phenotype):
        distance_optimum = distance.euclidean(self.get_optimum(), phenotype)
        # print('Distance = ' + str(distance_optimum))
        return distance_optimum

    def reproduction(self, genotype): #, nm):
        if self.fgm_mode:
            while exist(genotype):
                genotype = self.mutation(genotype, self.mutation_rate) #, nm)
                break
        elif not self.fgm_mode:
            if exist(genotype):
                # number_events = self.event_provability(genotype)
                # [n_dup, n_del] = self.event_selection(number_events)
                # rate = self.gen_duplication_rate
                [n_dup, n_del] = self.event_provability(genotype)
                # print(n_dup)
                genotype = self.duplication_loop(genotype, n_dup)
                print('------')
                genotype = self.deletion_loop(genotype, n_del)
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
        # score_son = self.fitness(np.add.reduce(son_genotype))
        fitness_son = self.fitness_function(self.create_phenotype(self.initial_point, son_genotype))
        # fitness_son = self.fitness_function(son_genotype)
        print('Son fitness : ' + str(fitness_son))
        fitness_father = self.fitness_function(self.create_phenotype(self.initial_point, father_genotype))
        # fitness_father = self.fitness_function(father_genotype)
        print('Father fitness : ' + str(fitness_father))

        son_distance = self.distance_optimum(self.create_phenotype(self.initial_point, son_genotype))
        print('Son distance to optimum ' + str(son_distance))
        father_distance = self.distance_optimum(self.create_phenotype(self.initial_point, father_genotype))
        print('Father distance to optimum ' + str(father_distance))
        # if fitness_son < fitness_father:
        if son_distance < father_distance:
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
            phenotype = np.add(initial_point, self.sum_genes(genotype))
        return phenotype

    def graph_fitnessVSgenerations(self, fitness_Values, generations):
        # Compute the x and y coordinates
        plt.title("FITNESS vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, fitness_Values)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    def graph_distanceVSgenerations(self, distance_Values, generations):
        # Compute the x and y coordinates
        plt.title("DISTANCE vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, distance_Values)
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.show()

    def graph_numGensVSgenerations(self, gen_Size, generations):
        # Compute the x and y coordinates
        plt.title("GEN SIZE vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, gen_Size)
        plt.xlabel('Generation')
        plt.ylabel('Gen Size')
        plt.show()

    def graph_distanceVSoptimum(self, distance_Values, optimum):
        # Compute the x and y coordinates
        plt.title("DISTANCE vs OPTIMUM")
        # Plot the points using matplotlib
        plt.plot(optimum, distance_Values)
        plt.xlabel('Optimum')
        plt.ylabel('Distance')
        plt.show()

    def graph_numMutVSgenerations(self, number_mutations, generations):
        # Compute the x and y coordinates
        plt.title("NUMBER OF ACUMULATED MUTATIONS vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, number_mutations)
        plt.xlabel('Generation')
        plt.ylabel('Acumulated mutations')
        plt.show()
    # def get_genSize(self, genotype):
    #     for i in genotype:

    # ----Declaration of the algo
    def run(self):
        # if self.fgm_mode:
        #     initial_point = self.initial_point
        #     print('Starting point: ' + str(initial_point))
        #     self.fitness_function(initial_point)
        #     self.distance_optimum(initial_point)
        #
        #     father_phenotype = self.initial_genotype()
        #     print('Father genotpe : ' + str(father_phenotype))
        #     fitness_value = self.fitness_function(father_phenotype)
        #     distance_value = self.distance_optimum(father_phenotype)
        #     # print('Father fitness : ' + str(fitness_value))
        #     distance_optimum = self.distance_optimum(father_phenotype)
        #     print('Distance to the optimum = ' + str(distance_optimum))
        #     father_size = len(father_genotype)
        #     i = 0
        #     nm = 0
        #     number_mutations = [nm]
        #     generations = [i]
        #     fitness_Values = [fitness_value]
        #     distance_Values = [distance_value]
        #     gen_Size = [father_size]
        #
        # else:

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
        distance_value = self.distance_optimum(father_phenotype)
        # print('Father fitness : ' + str(fitness_value))
        distance_optimum = self.distance_optimum(father_phenotype)
        print('Distance to the optimum = ' + str(distance_optimum))
        father_size = len(father_genotype)
        i = 0
        nm = 0
        number_mutations = [nm]
        generations = [i]
        fitness_Values = [fitness_value]
        distance_Values = [distance_value]
        gen_Size = [father_size]

        if self.gen_mode:
            for i in range(self.n_generations):
                print(
                    '#####################################################################################################')
                print('Generacion: ', i)
                son_genotype, nm = self.reproduction(father_genotype, nm)
                print('Son genotype ' + str(son_genotype))
                print('Number_ mutations' + str(nm))
                selected_genotype, fitness_value, distance_value = self.selection(son_genotype, father_genotype)
                # selected_phenotype = self.sum_genes(selected_genotype)
                selected_phenotype = self.create_phenotype(self.initial_point, selected_genotype)
                # print('Best suited organism is : ' + str(selected_organism))
                print('Best suited genotype is : ' + str(selected_genotype))
                print('Best suited phenotype is : ' + str(selected_phenotype))
                father_genotype = selected_genotype
                distance_optimum = distance_value
                print('Distancia del óptimo: ' + str(distance_optimum))
                gen_len = len(selected_genotype)
                # initial_point = selected_phenotype
                # father_phenotype = selected_phenotype
                # print('Point in the graph : ' + str(initial_point))
                i += 1
                generations.append(i + 1)
                # print('Generations: ' + str(generations))
                fitness_Values.append(fitness_value)
                distance_Values.append(distance_value)
                gen_Size.append(gen_len)
                number_mutations.append(nm)
                # print('Gen size vector : ' + str(gen_Size))
            self.graph_fitnessVSgenerations(fitness_Values, generations)
            self.graph_distanceVSgenerations(distance_Values, generations)
            self.graph_numGensVSgenerations(gen_Size, generations)
            # print('Valores de distancia ' + str(distance_Values))
            # self.graph_distanceVSoptimum(distance_Values, self.get_optimum())
        elif not self.gen_mode:

            while self.epsilon <= distance_optimum:  # father_phenotype == self.get_optimum() or
                print(
                    '#####################################################################################################')
                print('Generacion: ', i)
                #son_genotype, nm = self.reproduction(father_genotype) #, nm)
                son_genotype = self.reproduction(father_genotype)
                print('Son genotype ' + str(son_genotype))
                #print('Number_ mutations' + str(nm))
                selected_genotype, fitness_value, distance_value = self.selection(son_genotype, father_genotype)
                # selected_phenotype = self.sum_genes(selected_genotype)
                selected_phenotype = self.create_phenotype(self.initial_point, selected_genotype)
                # print('Best suited organism is : ' + str(selected_organism))
                print('Best suited genotype is : ' + str(selected_genotype))
                print('Best suited phenotype is : ' + str(selected_phenotype))
                father_genotype = selected_genotype
                distance_optimum = distance_value
                print('Distancia del óptimo: ' + str(distance_optimum))
                gen_len = len(selected_genotype)
                # initial_point = selected_phenotype
                # father_phenotype = selected_phenotype
                # print('Point in the graph : ' + str(initial_point))
                i += 1
                generations.append(i + 1)
                # print('Generations: ' + str(generations))
                fitness_Values.append(fitness_value)
                distance_Values.append(distance_value)
                gen_Size.append(gen_len)
                number_mutations.append(nm)
                # print('Gen size vector : ' + str(gen_Size))
            self.graph_fitnessVSgenerations(fitness_Values, generations)
            self.graph_distanceVSgenerations(distance_Values, generations)
            self.graph_numGensVSgenerations(gen_Size, generations)
            self.graph_numMutVSgenerations(number_mutations, generations)
            # print('Valores de distancia ' + str(distance_Values))
            # self.graph_distanceVSoptimum(distance_Values, self.get_optimum())


# =========== Main definition of the

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
        epsilon=1
    )

    model.run()


if __name__ == '__main__':
    main()
