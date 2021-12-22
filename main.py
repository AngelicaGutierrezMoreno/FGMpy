# # This is a sample Python script.
#
# # Press Mayús+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""

import numpy as np
import random
# import GA
#import matplotlib.pyplot as plot


class Organism:
    def __init__(self, optimum, n_organisms, mu_mean, sigma_covariance, n_dim, mutation_rate, gen_duplication_rate,
                 gen_deletion_rate, n_generations, verbose=True):
        self.optimum = optimum
        self.n_organisms = n_organisms
        self.mu_mean = mu_mean
        self.sigma_covariance = sigma_covariance
        self.n_dim = n_dim
        self.mutation_rate = mutation_rate
        self.gen_duplication_rate = gen_duplication_rate
        self.gen_deletion_rate = gen_deletion_rate
        self.n_generations = n_generations
        self.verbose = verbose

    # creacion del organismo
    def create_organism(self, min=0,
                        max=500):  # largest number supported by the machine (extracted used import sys sys.maxsize in python terminal)
        # organism = {'mu' : np.random.uniform(min, max), 'sigma' : [np.random.uniform(min, max)]}
        # mu = [np.random.randint(min, max, self.n_dim)]
        # mu = [None]*self.n_dim
        # mu.append(np.random.randint(min, max))
        # mu = random.sample(range(min, max), self.n_dim)
        mu = [np.random.randint(min, max)] * self.n_dim
        sigma = [np.random.randint(min, max)]
        organism = [mu, sigma]
        # organism = [[np.random.randint(min, max)]*self.n_dim, [np.random.randint(min, max)]*self.n_dim]
        # [np.random.randint(min, max) for _ in range(len(self.optimum))]
        return organism

    # mutación, duplicación o deletion of the gene
    def mutation(self, organism):
        print("Iniciando mutacion")
        print('Organismo : ' + str(organism))
        for i in range(len(organism)):
            if random.random() <= self.mutation_rate:
                # point = np.random.randint(len(self.optimum)) #pick random mutation position
                new_value = np.random.randint(0.00, 500.00)
                organism[i][i] -= new_value
                organism[i + 1][i] -= new_value
                # =============================================================================
                #                 while new_value == organism[i][point]:
                #                     new_value = np.random.randint(0.00, 500.00)
                #
                #                 organism[i][point] = new_value
                # =============================================================================
                # organism[i] = int(organism[i]) + new_value
                # print(new_value)
                # print('Sigma : '+ str(organism[i]))
                print('New organism ' + str(organism))
            else:
                print('No se realiza accion')
                # print('Organism ' + str(organism))
                # organism = organism[:]
            return organism

    def gene_duplication(self, organism):
        print("Iniciando gen duplication")
        print('Organismo : ' + str(organism))
        if random.random() <= self.gen_duplication_rate:
            point = np.random.randint(len(organism[1]))
            # point = len(organism[1])
            new_gene = organism[1][point]
            organism[1].append(new_gene)
            # print('New organism ' + str(organism[1]))
            # print(point)
            # print(new_gene)
            # print(len(organism[1]))
            print('Posicion a duplicar : ' + str(point))
            print('New organism ' + str(organism))
        else:
            print('No se realiza accion')
            # print('Organism ' + str(organism))
            # organism = organism[:]
        return organism

    def gene_deletion(self, organism):
        print("Iniciando gen deletion")
        print('Organismo : ' + str(organism))
        if random.random() <= self.gen_deletion_rate:
            point = np.random.randint(len(organism[1]))
            selected_gene = organism[1][point]
            organism[1].remove(selected_gene)
            print('Posicion a eliminar : ' + str(point))
            print('New organism ' + str(organism))
        else:
            print('No se realiza accion')
            # print('Organism ' + str(organism))
            # organism = organism[:]
        return organism

    def fitness(self, organism):
        """
        Funcion que determina cuántos valores son iguales a los del óptimo
        """

        fitness = 0

        for i in range(len(organism)):
            if organism[i] == self.optimum[i]:
                fitness += 1
            return fitness

    def reproduction(self, organism, father_organism):

        father_organism = organism[:]

        organism = self.mutation(organism)
        print('------')
        organism = self.gene_duplication(organism)
        print('------')
        organism = self.gene_deletion(organism)
        print('------')

        print('Father : ' + str(father_organism))
        print('Son : ' + str(organism))

        return organism

    def selection(self, organism, father_organism):
        """
        si la evaluación del fitness hijo es mayor a la del fitness padre, hacer seleccion
        """
        scores = [(self.fitness(i), i) for i in organism]
        scores = [i[1] for i in sorted(scores)]

        return scores[len(scores) - self.n_selection:]

        # evaluación fitness padre e hijo
        # scores = [self.fitness(organism), organism]
        # comparar organismo padre e hijo
        # seleccionar el que tenga mejor fitness
        # designarlo como el nuevo organismo padre
        print(scores)

    # =============================================================================
    #     def create_population(self):
    #         return [self.create_organism() for _ in range(self.n_organisms)]
    # =============================================================================

    def fitness_function(self, organism):
        """pdf of the multivariate normal distribution."""
        x_m = int(str(organism)) - self.mu_mean
        fitness_value = (1. / (np.sqrt((2 * np.pi) ** self.n_dim * np.linalg.det(self.sigma_covariance))) * np.exp(
            -(np.linalg.solve(self.sigma_covariance, x_m).T.dot(x_m)) / 2))
        print(fitness_value)
        return fitness_value
        # https://peterroelants.github.io/posts/multivariate-normal-primer/

    def run(self):
        father_organism = self.create_organism()
        # print(father_organism)

        organism = father_organism[:]
        for i in range(self.n_generations):
            # print('_______________________')
            # print('Generacion: ', i)
            # =============================================================================
            #             organism = self.mutation(organism)
            #
            #             organsm = self.gene_duplication(organism)
            #             organism = self.gene_deletion(organism)
            # =============================================================================
            organism = self.reproduction(organism, father_organism)
            # son_fitness = self.fitness_function(organism,self.n_dim,self.mu_mean, self.sigma_covariance)
            # organism_after_mutation = self.mutation(organism)
            # organism_after_gen_dup = self.gene_duplication(organism)
            # organism_after_gen_del = self.gene_deletion(organism)
            # fitness_value_father = self.fitness_function(father_organism)
            # fitness_value = self.fitness_function(organism)
            # print(organism_after_mutation)
            # print(organism_after_gen_dup)
            # print(organism_after_gen_del)
            # print(father_organism)
            # print(organism_son)
            # print(organism)
            # print(fitness_value_father)
            # print(fitness_value)


# =============================================================================
#             if self.verbose:
#                 print('___________')
#                 print('Generacion: ', i)
#                 print('Organismo padre: ', father_organism)
#                 print('Organismo hijo: ', organism)
#                 print()
# =============================================================================

def main():
    optimum = [0.0, [0.0, 0.0, 0.0, 0.0]]
    model = Organism(
        optimum=optimum,
        n_organisms=1,
        mu_mean=100.0,
        sigma_covariance=600.0,
        n_dim=2,
        mutation_rate=0.5,  # keep rates minimum
        gen_duplication_rate=0.3,
        gen_deletion_rate=20,
        n_generations=1,
        verbose=True)
    # model.create_organism()
    # model.selection(model.create_organism())
    # model.fitness(model.create_organism())
    model.run()
    # model.fitness_function(2, 20, 10)
    # model.mutation(model.create_organism())
    # model.gene_duplication(model.create_organism())
    # model.gene_deletion(model.create_organism())


if __name__ == '__main__':
    main()
