"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""
import sys

import numpy as np
import random
from scipy.spatial import distance
from matplotlib import pyplot as plt


# import GA


class Organism:
    def __init__(self, optimum, n_organisms, mu_mean, sigma_covariance, n_dim, mutation_rate, gen_duplication_rate,
                 gen_deletion_rate, n_generations, limit_min, limit_max, epsilon, mutation_type, fgm_mode, gen_mode, initial_point,
                 verbose=True):
        self.optimum = optimum  # ideal vector (set to 0)
        self.n_organisms = n_organisms
        self.mu_mean = mu_mean
        self.sigma_covariance = sigma_covariance
        self.n_dim = n_dim
        self.mutation_rate = mutation_rate  # keep to minimum
        self.gen_duplication_rate = gen_duplication_rate  # keep to minimum
        self.gen_deletion_rate = gen_deletion_rate  # keep to minimum
        self.n_generations = n_generations
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.epsilon = epsilon  # maximum range of distance to the optimum
        self.mutation_type = mutation_type  # def. if only mutate phenotype, genothype, both or randomly mutate one gene
        self.fgm_mode = fgm_mode  # Determinates if we're using FGM or the model proposed
        self.gen_mode = gen_mode  # Determinates if we continue the evaluation until the optimum is reached or if we
        # continue until the number of generations is met
        self.initial_point = initial_point
        self.verbose = verbose

    # ================================= creacion del organismo ====================================================

    def initial_genotype(self):
        genotype = np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')
        # print(len(genotype))
        return genotype

    def create_organism(self, genotype):
        # print('length '+ str(len(genotype)))
        if len(genotype) > self.n_dim:# or len(genotype) < self.n_dim:
            # if (len(genotype) > 1):
            #organism = np.add.reduce(genotype)
            organism = self.create_organism(genotype)
        elif len(genotype) == 1 or len(genotype) == self.n_dim:
            organism = genotype
        elif len(genotype) == 0:
            sys.exit("organism doesn't exist anymore")
        else:
            #print('Genotype 0 = ' + str(genotype))
            organism = np.add.reduce(genotype)
            #organism = self.create_organism(genotype)
        return organism

    # ========================== mutación, duplicación o deletion of the gene ========================================

    def mutation(self, genotype):
        print("Iniciando mutacion")
        organism = np.add.reduce(genotype)
        if random.random() <= self.mutation_rate:
            #print('Organismo a mutar: ' + str(organism) )
            print('Genotype a mutar ' + str(genotype))
            vector_substract_value = [
                np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')]
            print('vector a substraer = ' + str(vector_substract_value))
            #Sólo se va a mutar un gen del genotipo
            if len(genotype) == 0:
                print('No organism to mutate')
            elif len(genotype) == self.n_dim:
                mutated_gen = np.subtract(genotype, vector_substract_value)
                print('Mutated gen : ' + str(mutated_gen))
                genotype = list(genotype)
                genotype = mutated_gen
                print('Mutated genotype : ' + str(genotype))
                #print('TYPES : ' + ' Vector = ' + str(type(vector_substract_value)))
                print('Organismo mutado: ' + str(genotype))
            else:
                gen_position = random.randint(0, len(genotype) - 1)
                print('Selected position of gen to mutate: ' + str(gen_position))
                gen_to_mutate = genotype[gen_position]
                print('Selected gen : ' + str(gen_to_mutate))
                mutated_gen = np.subtract(gen_to_mutate, vector_substract_value)
                print('Mutated gen : ' + str(mutated_gen))
                genotype = list(genotype)
                genotype[gen_position] = mutated_gen
                #print('Mutated genotype : ' + str(genotype)) + print('Matrix value :' + ' GEN = ' + str(np.shape(gen_to_mutate)) + ' VECTOR = ' + str( np.shape(vector_substract_value)) + ' Mutated GEN = ' + str(np.shape(mutated_gen)) + ' Mutated genotype : ' + str(np.shape(genotype))) + print('TYPES : ' + 'GEN to mutate type = ' + str(type(gen_to_mutate)) + ' Vector = ' + str( type(vector_substract_value)) +' mix ' + str(type(genotype[gen_position])))
                #organism = np.add.reduce(genotype)
                organism = self.create_organism(genotype)
                print('Organismo mutado: ' + str(organism))
        else:
            print('No se realiza mutación')
        return genotype

    def gene_duplication(self, genotype):
        print("Iniciando gen duplication")
        # print('Length ' + str(len(genotype)))
        #print('Genotipo antes de duplicar : ' + str(genotype))
        if random.random() <= self.gen_duplication_rate:
            if len(genotype) == 0:
                print('No gen to duplicate')
            elif len(genotype) == 1 or len(genotype) == self.n_dim:
                genotype = genotype, genotype
                print('Duplicated genotype ' + str(genotype))
                # organism = self.create_organism(genotype)
                organism = np.add.reduce(genotype)
                print('Organism ' + str(organism))
            else:
                # print(len(genotype))
                point = np.random.randint(len(genotype))
                print("Point " + str(point))
                duplicated_gen = genotype[point]
                print('Gen to duplicate : ' + str(duplicated_gen))
                genotype = genotype.append(duplicated_gen)
                #genotype = genotype.extend(duplicated_gen)
                print('Posicion a duplicar : ' + str(point))
                print('Genotype after duplication : ' + str(genotype))
                #organism = self.create_organism(genotype)
                organism = np.add.reduce(genotype)
                print('New organism ' + str(organism))
        else:
            print('No se realiza duplicación del gen')
        return genotype

    def gene_deletion(self, genotype):
        print("Iniciando gen deletion")
        # print('Genotipo antes de eliminación: ' + str(genotype))
        if random.random() <= self.gen_deletion_rate:
            print('Genotype before deletion ' + str(genotype))
            if len(genotype) == 0:
                print("No gen to delete")
                # organism = self.create_organism(genotype)
                # print('New organism ' + str(organism))
            else:
                point = np.random.randint(len(genotype))
                print('Point ' + str(point))
                selected_gene = genotype[point]
                print('Selected gene: ' + str(selected_gene))
                genotype = np.delete(genotype, point, 0)
                print('Genotype after deletion ' + str(genotype))
                #organism = self.create_organism(genotype)
                #organism = np.add.reduce(genotype)
                organism = self.create_organism(genotype)
                print('Organism ' + str(organism))
        else:
            print('No se realiza supresión del gen')
        return genotype

    # =================================== Fitness function evaluation ===============================================

    def reproduction(self, genotype):

        if self.fgm_mode:
            genotype = self.mutation(genotype)
        elif not self.fgm_mode:
            genotype = self.gene_duplication(genotype)
            print('------')
            genotype = self.gene_deletion(genotype)
            print('------')
            genotype = self.mutation(genotype)
            print('------')
        return genotype

    def selection(self, son_genotype, father_genotype):
        """
        si la evaluación del fitness hijo es mayor a la del fitness padre, hacer seleccion
        """
        #score_son = self.fitness(np.add.reduce(son_genotype))
        score_son = self.fitness(self.create_organism(son_genotype))
        print('Son fitness : ' + str(score_son))
        score_father = self.fitness(np.add.reduce(father_genotype))
        print('Father fitness : ' + str(score_father))

        if score_son < score_father:
            #print("Son's genotype " + str(son_genotype))
            return son_genotype
        else:
            #print("Father's genotype" + str(father_genotype))
            return father_genotype

    def fitness(self, genotype):
        """pdf of the multivari-ate normal distribution."""
        # organism = self.create_organism(genotype)
        organism = np.add.reduce(genotype)
        #print('Organism to evaluate : ' + str(organism))
        fitness_value = distance.euclidean(organism, self.optimum)
        #print('Fitness = ' + str(fitness_value))
        return fitness_value
        # https://peterroelants.github.io/posts/multivariate-normal-primer/


    # ======================================== Flux ================================================================
    def run(self):

        initial_point = self.initial_point
        print('Starting point: ' + str(initial_point))
        fitness_value = self.fitness(initial_point)
        print('Fitness starting point = ' + str(fitness_value))
        #father_genotype = self.initial_genotype()
        father_genotype = self.initial_genotype() - initial_point
        #father_genotype = self.initial_genotype(), initial_point
        print('Padre genotype ' + str(father_genotype))
        #print('Initial genotype ' + str(initial_genotype))
        #father_genotype = np.add.reduce(initial_genotype, self.initial_point)
        #father_genotype = initial_point-initial_genotype
        #father_phenotype = [self.create_organism(initial_genotype)]
        #father_phenotype = initial_point + father_genotype
        # print('Padre ' + str(father_organism))
        #father_genotype = initial_point, initial_genotype
        #print('Padre genotype ' + str(father_genotype))
        #fitness_value = self.fitness(father_phenotype)
        #print('Initial fitness = ' + str(fitness_value))
        i = 0

        # genotype = initial_genotype[:]
        if self.gen_mode:
            for i in range(self.n_generations):
                if self.verbose:
                    son_genotype = self.reproduction(father_genotype)
                    selected_genotype = self.selection(son_genotype, father_genotype)
                    father_genotype = selected_genotype
                    print('___________')
                    print('Generacion: ', i)
                    print('Organismo padre: ', father_phenotype)
                    print('Organismo hijo: ', self.create_organism(selected_genotype))
                    print()


                elif not self.verbose:
                    #print('Padre genotype ' + str(father_genotype))
                    print('_______________________')
                    print('Generacion: ', i)
                    print('Point : ' + str(initial_point))
                    son_genotype = self.reproduction(father_genotype)
                    print('Son genotype ' + str(son_genotype))
                    #son_phenotype = np.add.reduce(son_genotype) + self.initial_point
                    #son_phenotype = np.add.reduce(son_genotype)
                    #print('Son phenotype : ' + str(son_phenotype))
                    #father_phenotype = np.add.reduce(father_genotype) + self.initial_point
                    #father_phenotype = np.add.reduce(father_genotype)
                    #print('Father phenotype : ' + str(father_phenotype))
                    #selected_phenotype = self.selection(son_phenotype, father_phenotype)
                    selected_genotype = self.selection(son_genotype, father_genotype)
                    #selected_organism = self.create_organism(selected_genotype)
                    selected_phenotype = self.create_organism(selected_genotype)
                    #print('Best suited organism is : ' + str(selected_organism))
                    print('Best suited genotype is : ' + str(selected_genotype))
                    #print('Best suited genotype is : ' + str(selected_phenotype))
                    #father_genotype = selected_genotype
                    initial_point = selected_phenotype
                    #father_genotype =
                    print('Point in the graph : ' + str(initial_point))

                    # Compute the x and y coordinates
                    plt.title("Progreción")
                    # Plot the points using matplotlib
                    plt.plot(i, fitness_value)
                    plt.show()

        elif not self.gen_mode:

            while self.epsilon <= fitness_value: #father_phenotype == self.optimum or
                print(
                    '#####################################################################################################')
                print('Generacion: ', i)
                print('Point : ' + str(initial_point))
                son_genotype = self.reproduction(father_genotype)
                print('Son genotype ' + str(son_genotype))
                # son_phenotype = np.add.reduce(son_genotype) + self.initial_point
                # son_phenotype = np.add.reduce(son_genotype)
                # print('Son phenotype : ' + str(son_phenotype))
                # father_phenotype = np.add.reduce(father_genotype) + self.initial_point
                # father_phenotype = np.add.reduce(father_genotype)
                # print('Father phenotype : ' + str(father_phenotype))
                # selected_phenotype = self.selection(son_phenotype, father_phenotype)
                selected_genotype = self.selection(son_genotype, father_genotype)
                # selected_organism = self.create_organism(selected_genotype)
                selected_phenotype = self.create_organism(selected_genotype)
                # print('Best suited organism is : ' + str(selected_organism))
                print('Best suited genotype is : ' + str(selected_genotype))
                # print('Best suited genotype is : ' + str(selected_phenotype))
                # father_genotype = selected_genotype
                initial_point = selected_phenotype
                # father_genotype =
                print('Point in the graph : ' + str(initial_point))
                i += 1
                # Compute the x and y coordinates
                plt.title("Progreción")
                # Plot the points using matplotlib
                plt.plot(i, fitness_value)
                plt.show()


# =========== Main definition of the

def main():
    model = Organism(
        optimum=[0.0, 0.0, 0.0],  # first vector most match the n_dim
        n_organisms=1,
        mu_mean=[1.0, 1.0, 1.0],
        sigma_covariance=[[2000.0, -1000.0, 0.0], [-1000, 2000.0, -1000.0], [0.0, -1000.0, 2000.0]],
        # the matrix covariance must be a positive semidefinite symmetric one
        n_dim=3,
        mutation_rate=0.8,  # keep rates minimum
        gen_duplication_rate=0.9,
        gen_deletion_rate=0.0000,
        n_generations=3,
        limit_min=0.00,  # limite inferior para los valores del gen
        limit_max=500.00,  # limite superior para los valores del gen
        epsilon=8,
        mutation_type=0,  # 0 -> phenotype, 1 -> genotype, 2 -> one random gene, 3 -> both
        initial_point=[10.0, 10.0, 10.0], #Iniital point in FGM
        fgm_mode=False,  # True = FG model, False = proposed model
        gen_mode=False,  # True = number of generations , False = until optimum is reached
        verbose=False)
    model.run()


if __name__ == '__main__':
    main()

    #
    # while True:
    #     do_something()
    #     if condition():
    #         break
