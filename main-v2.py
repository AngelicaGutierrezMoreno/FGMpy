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


def is_empty(genotype):
    # for i in len(genotype) :

    return True


def exist(genotype):
    # print('Genotypes length : ' + len(genotype))
    if len(genotype) == 0:
        sys.exit("Genotype does't exist anymore")
    else:
        return True, len(genotype)


class Organism:
    def __init__(self, fgm_mode, gen_mode, initial_point, n_organisms, mu_mean, sigma_covariance, n_dim, mutation_rate,
                 gen_duplication_rate, gen_deletion_rate, n_generations, epsilon, mutation_type):
        self.fgm_mode = fgm_mode  # Determinate if we're using FGM or the model proposed
        self.gen_mode = gen_mode  # Determinate if we continue the evaluation until the optimum is reached or if we
        # continue until the number of generations is met
        self.initial_point = initial_point  # initial point where the model will part
        self.n_organisms = n_organisms  # number of organisms to be used in the model (leave it at 1)
        self.mu_mean = mu_mean  #
        self.sigma_covariance = sigma_covariance  #
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
        # print('length '+ str(len(genotype)))
        if len(genotype) > 0:
            organism = np.add.reduce(genotype)
            print('sum genes : ' + str(organism))
        elif len(genotype) == 0:
            sys.exit("organism doesn't exist anymore")
        else:
            print('Genotype 0 = ' + str(genotype))
            organism = np.add.reduce(genotype)
        return organism

    # ----Declaration of events

    def get_mutation_vector(self, deviation):
        mutation_vector = NormalDist(0.0, deviation).samples(self.n_dim)
        print('Mutation vector: ' + str(mutation_vector))
        return mutation_vector

    def mutation(self, genotype, deviation):
        if exist(genotype):
            print("Iniciando mutacion")
            genotype = np.subtract(genotype, self.get_mutation_vector(deviation))
            # genotype = np.subtract(genotype, NormalDist(0.0, deviation).samples(self.n_dim))
            print('Mutated genotype: ' + str(genotype))
        else:
            print('Cannot do mutation')
        return genotype

    def duplication(self, genotype, i):

        while exist(genotype):
            #print(genotype[i])
            genotype = list(genotype)
            genotype.append(genotype[i])
            #print('After duplication' + str(genotype))
            break
        return genotype

    def deletion(self, genotype, i):
        print("Iniciando gen deletion")
        if exist(genotype):
            #print(genotype[i])
            genotype = list(genotype)
            genotype.pop(i)
            print('After deletion' + str(genotype))
        else:
            print('Cannot do deletion')
        return genotype

    def fitness_function(self, phenotype):
        """pdf of the multivari-ate normal distribution."""
        print('Phenotype : ' + str(phenotype))
        # fitness_function = multivariate_normal(0.0, 0.15)
        # fitness_value = fitness_function.pdf(phenotype)
        # fitness_value = multivariate_normal.pdf(phenotype, 0.0, 0.15)
        # NormalDist(0.0, 0.15)
        fitness_value = np.mean(multivariate_normal.pdf(phenotype))
        # statistics.mean(fitness_value)
        print('Fitness = ' + str(fitness_value))
        return fitness_value

    def event_provability(self, genotype):
        number_events = np.random.binomial(2, 0.5, len(genotype)) # probability vector that indicates how many
        # duplications/deletions should happend in the genotype
        print(number_events)
        return number_events

    def duplication_loop(self, genotype, num_repeticiones):
        i = 0
        if num_repeticiones > 0:
            print("Iniciando gen duplication")
            while i <= num_repeticiones:
                pos = random.randint(0, len(genotype)-1)
                genotype = self.duplication(genotype, pos)
                i = i + 1
                print(i)
                break
            print(genotype)
        else:
            print('No gens will be duplicated')
        return genotype

    def deletion_loop(self, genotype, num_repeticiones):
        i = 0
        while i < num_repeticiones:
            pos = random.randint(0, len(genotype))
            genotype = self.deletion(genotype, pos)
            i = i + 1
            break
        print(genotype)
        return genotype

    def distance_optimum(self, phenotype):
        distance_optimum = distance.euclidean(self.get_optimum(), phenotype)
        print('Distance = ' + str(distance_optimum))
        return distance_optimum

    def reproduction(self, genotype):

        if self.fgm_mode:
            while exist(genotype):
                self.mutation(genotype, self.mutation_rate)
                break
        elif not self.fgm_mode:
            if exist(genotype):
                [n_dup, n_del] = self.event_provability(genotype)
                #print(n_dup)
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

    def create_phenotype(self, initial_point, genotype):
        # phenotype = np.add.reduce( genotype)
        phenotype = np.subtract(initial_point, self.sum_genes(genotype))
        return phenotype

    def graph_fitnessVSgenerations(self, fitnessValues, generations):
        # Compute the x and y coordinates
        plt.title("FITNESS vs GENERATIONS")
        # Plot the points using matplotlib
        plt.plot(generations, fitnessValues)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()

    # ----Declaration of the algo
    def run(self):
        initial_point = self.initial_point
        print('Starting point: ' + str(initial_point))
        father_genotype = self.initial_genotype(), self.initial_genotype()
        # print(len(father_genotype))
        # print(is_empty(father_genotype))
        #father_genotype = self.initial_genotype()
        print('Father genotpe : ' + str(father_genotype))
        self.fitness_function(initial_point)
        self.distance_optimum(initial_point)
        father_genotype = self.reproduction(father_genotype)
        #print(father_genotype)
        # print(gens)
        phenotype = self.create_phenotype(initial_point, father_genotype)
        # phenotype = np.add.reduce(np.array(initial_point), ph)
        #print(phenotype)
        self.fitness_function(phenotype)
        self.distance_optimum(phenotype)


# =========== Main definition of the

def main():
    model = Organism(
        fgm_mode=False,  # True = FG model, False = proposed model
        gen_mode=False,  # True = number of generations , False = until optimum is reached
        initial_point=[10.0],  # Inital point in FGM
        n_organisms=2,
        mu_mean=[1.0, 1.0, 1.0],
        sigma_covariance=[[2000.0, -1000.0, 0.0], [-1000, 2000.0, -1000.0], [0.0, -1000.0, 2000.0]],
        # the matrix covariance must be a positive semidefinite symmetric one
        n_dim=1,
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
