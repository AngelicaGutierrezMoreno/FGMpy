"""
Created on Tue Nov 30 16:58:58 2021

@author: angie
"""

import numpy as np
import random


# import GA
# import matplotlib.pyplot as plot


class Organism:
    def __init__(self, optimum, n_organisms, mu_mean, sigma_covariance, n_dim, mutation_rate, gen_duplication_rate,
                 gen_deletion_rate, n_generations, limit_min, limit_max, epsilon, mutation_type, fgm_mode, verbose=True):
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
        self.mutation_type = mutation_type # def. if only mutate phenotype, genothype, both or randomly mutate one gene
        self.fgm_mode = fgm_mode #Determinates if we're using FGM or the model proposed
        self.verbose = verbose

    # ================================= creacion del organismo ====================================================
    def initial_genotype(self):
        genotype = [np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')]
        return genotype


    def create_organism(self, genotype):
        organism = np.add.reduce(genotype)
        return organism

    # ========================== mutación, duplicación o deletion of the gene ========================================

    def mutation(self, genotype):
        print("Iniciando mutacion")
        # print('Organismo : ' + str(organism))
        #organism = np.add.reduce(genotype)
        if random.random() <= self.mutation_rate:
            #print('Organismo a mutar: ' + str(organism))
            print('Genotype ' + str(genotype))
            vector_substract_value = [np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')]
            print('vector a substraer = ' + str(vector_substract_value))
            #Sólo se va a mutar un gen del genotipo
            gen_position = random.randint(0, len(genotype)-1)
            print('Selected position of gen to mutate: ' + str(gen_position))
            gen_to_mutate = genotype[gen_position]
            print('Selected gen : ' + str(gen_to_mutate))
            mutated_gen = np.subtract(gen_to_mutate, vector_substract_value)
            #mutated_gen = genotype[gen_position] - vector_substract_value
            print('Mutated gen : ' + str(mutated_gen))
            #print('Matrix value :' + ' GEN = ' + str(np.shape(gen_to_mutate)) + ' VECTOR = ' + str(
            #    np.shape(vector_substract_value)) + ' Mutated GEN = ' + str(np.shape(mutated_gen)))
            #genotype[genotype.] = mutated_gen
            #genotype[np.where(genotype == genotype[gen_position])] = mutated_gen
            #print('TYPES : ' + 'GEN to mutate type = ' + str(type(gen_to_mutate)) + ' Vector = ' + str(
            #    type(vector_substract_value)) + ' mix ' + str(type(genotype[gen_position])))
            genotype = list(genotype)
            genotype[gen_position] = np.subtract(gen_to_mutate, vector_substract_value)
            genotype = tuple(genotype)
            #genotype[gen_position] = np.subtract(gen_to_mutate, vector_substract_value)
            #genotype = np.where(genotype[gen_position] == gen_position, mutated_gen, genotype)
            print('Mutated genotype : ' + str(genotype))

            organism = np.add.reduce(genotype)
            print('Organismo mutado desenlazado: ' + str(organism))
        else:
            organism = np.add.reduce(genotype)
            print('No se realiza mutación')
        return organism

    def gene_duplication(self, genotype):
        print("Iniciando gen duplication")
        print('Genotipo : ' + str(genotype))
        if random.random() <= self.gen_duplication_rate:
            if len(genotype) == 0:
                print('No gen to duplicate')
            elif len(genotype) == 1:
                genotype = genotype, genotype
                print('Genotype ' + str(genotype))
                organism = np.add.reduce(genotype)
                print('New organism ' + str(organism))
            else:
                print(len(genotype))
                point = np.random.randint(len(genotype))
                print("Point " + str(point))
                genotype = genotype, genotype[point]
                print('Posicion a duplicar : ' + str(point))
                organism = np.add.reduce(genotype)
                print('New organism ' + str(organism))
        else:
            print('No se realiza duplicación del gen')
        return genotype

    def gene_deletion(self, genotype):
        print("Iniciando gen deletion")
        print('Genotipo : ' + str(genotype))
        if random.random() <= self.gen_deletion_rate:
            print('Genotype before deletion ' + str(genotype))
            if len(genotype) == 0:
                print("No gen to delete")
                organism = np.add.reduce(genotype)
                print('New organism ' + str(organism))
            else:
                print(len(genotype))
                point = np.random.randint(len(genotype))
                print('Point ' + str(point))
                selected_gene = genotype[point]
                print('Selected gene: ' + str(selected_gene))
                genotype = np.delete(genotype, point, 0)
                print('Genotype after deletion ' + str(genotype))
                organism = np.add.reduce(genotype)
                print('Posicion a eliminar : ' + str(point))
                print('New organism ' + str(organism))
        else:
            print('No se realiza supresión del gen')
        return genotype

    # =================================== Fitness function evaluation ===============================================

    def fitness(self, organism):
        """
        Funcion que determina cuántos valores son iguales a los del óptimo
        """
        fitness_value = self.fitness_function(organism)
        print('Fitness value = ' + str(fitness_value))
        return fitness_value

    def reproduction(self, genotype, father_organism):

        #organism = self.gene_duplication(genotype)
        genotype = self.gene_duplication(genotype)
        print('------')
        genotype = self.gene_deletion(genotype)
        print('------')
        organism = self.mutation(genotype)
        print('------')

        print('Father : ' + str(father_organism))
        print('Son : ' + str(organism))

        return organism

    def selection(self, organism, father_organism):
        """
        si la evaluación del fitness hijo es mayor a la del fitness padre, hacer seleccion
        """
        # scores = [(self.fitness(i), i) for i in organism]
        # scores = [i[1] for i in sorted(scores)]

        # return scores[len(scores) - self.n_selection:]

        # evaluación fitness padre e hijo
        # scores = [self.fitness(organism), organism]
        # comparar organismo padre e hijo
        # seleccionar el que tenga mejor fitness
        # designarlo como el nuevo organismo padre
        # print(scores)

        score_son = self.fitness(organism)
        # print('Score son : ' + str(score_son))
        score_father = self.fitness(father_organism)
        # print('Score father : ' + str(score_father))

        if score_son > score_father:
            return organism
        else:
            return father_organism

    # =============================================================================
    #     def create_population(self):
    #         return [self.create_organism() for _ in range(self.n_organisms)]
    # =============================================================================

    def fitness_function(self, organism):
        """pdf of the multivariate normal distribution."""
        #organism = organism.squeeze(organism)

        ### COMPROBAR MATRICES !!!!!!!!!!!!!!!!!! ####
        #mu_mean = [self.mu_mean]
        #x_m = np.subtract(organism, mu_mean)
        #x_m = np.subtract(organism, np.transpose(self.mu_mean))
        x_m = np.subtract(organism, self.mu_mean)
        print('Mu = ' + str(organism) + ' mu mean = ' + str(self.mu_mean) + ' x_m = ' + str(x_m))
        print('SHAPES : Mu : ' + str(np.shape(organism)) + ' mu mean :' + str(np.shape(self.mu_mean)) + ' x_m : ' + str(
            np.shape(x_m)))
        #print('SHAPES : Mu : ' + str(np.shape(organism)) + ' mu mean :' + str(np.shape(mu_mean)) + ' x_m : ' + str(
        #    np.shape(x_m)))
        fitness_value = (1. / (np.sqrt((2 * np.pi) ** self.n_dim * np.linalg.det(self.sigma_covariance))) * np.exp(
            -(np.linalg.solve(self.sigma_covariance, x_m).T.dot(x_m)) / 2))
        # print('Fitness = ' + str(fitness_value))
        return fitness_value
        # https://peterroelants.github.io/posts/multivariate-normal-primer/

    # ======================================== Flux ================================================================
    def run(self):
        genotype = self.initial_genotype()
        print('Initial genotype ' + str(genotype))
        father_organism = [self.create_organism(genotype)]
        print('Padre ' + str(father_organism))

        epsilon = [self.epsilon for x in range(self.n_dim)]
        print('Epsilon' + str(epsilon))
        #zona_tolerancia = np.subtract(self.optimum - epsilon)
        #print('Resta ' + str(zona_tolerancia))

        #while father_organism <= epsilon:
        for i in range(self.n_generations):
            # print(father_organism)
            #organism = father_organism
            if self.verbose:
                print('___________')
                #print('Generacion: ', i)
                print('Organismo padre: ', father_organism)
                print('Organismo hijo: ', organism)
                print()


            else:
                print('_______________________')
                print('Generacion: ', i)
                organism = self.reproduction(genotype, father_organism)
                selected_organism = self.selection(organism, father_organism)
                print('Best suited organism is : ' + str(selected_organism))
                father_organism = selected_organism
                #father_organism = organism
            # =============================================================================
            #             organism = self.mutation(organism)
            #
            #             organsm = self.gene_duplication(organism)
            #             organism = self.gene_deletion(organism)
            # =============================================================================

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


# =========== Main definition of the

def main():
    model = Organism(
        optimum=[0.0, 0.0, 0.0],  # first vector most match the n_dim
        n_organisms=1,
        mu_mean=[0.0, 0.0, 0.0],
        sigma_covariance=[[2000.0, -1000.0, 0.0], [-1000, 2000.0, -1000.0], [0.0, -1000.0, 2000.0]],
        # the matrix covariance must be a positive semidefinite symmetric one
        n_dim=3,
        mutation_rate=10,  # keep rates minimum
        gen_duplication_rate=10,
        gen_deletion_rate=0.0,
        n_generations=1,
        limit_min=0.00,  # limite inferior para los valores del gen
        limit_max=500.00,  # limite superior para los valores del gen
        epsilon=10.0,
        mutation_type=0, # 0 -> phenotype, 1 -> genotype, 2 -> one random gene, 3 -> both
        fgm_mode = False, #True = normal model, False = proposed model
        verbose=False)
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