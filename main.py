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
                 gen_deletion_rate, n_generations, limit_min, limit_max, epsilon, verbose=True):
        self.optimum = optimum #ideal vector (set to 0)
        self.n_organisms = n_organisms
        self.mu_mean = mu_mean
        self.sigma_covariance = sigma_covariance
        self.n_dim = n_dim
        self.mutation_rate = mutation_rate #keep to minimum
        self.gen_duplication_rate = gen_duplication_rate #keep to minimum
        self.gen_deletion_rate = gen_deletion_rate #keep to minimum
        self.n_generations = n_generations
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.epsilon = epsilon #maximum range of distance to the optimum
        self.verbose = verbose

    #================================= creacion del organismo ====================================================
    def create_organism(self):

        #mean = np.random.uniform(self.limit_min, self.limit_min, self.n_dim)
        #cov = [[0 for x in range(self.n_dim)] for y in range(self.n_dim)] #create a matrix of NxN with 0 values
        #cov_row = np.random.uniform(self.limit_min, 1.0, self.n_dim)
        #covariance_matrix = np.tril(cov_row) + np.tril(cov_row, -1).T
        #symm = covariance_matrix @ covariance_matrix.T
        #print('Row : ' + str(cov_row))
        #print('Matrix : ' + str(covariance_matrix))

        # test for symmetry
        #print(symm == symm.T)

        #===== crear función que compruebe que una matriz sea simétrica y positiva-seidefinida
        mu_gauss = np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')
        #mu = np.random.uniform(self.limit_min, self.limit_max, self.n_dim)
        sigma = np.random.uniform(self.limit_min, self.limit_max, 1) #start genotype with onl 1 gene
        organism = mu_gauss, sigma
        #organism = mu, sigma
        return organism

    # def create_optimum(self, organism):
    #     [mu_org, sigma_org] = organism
    #     mu = np.random.uniform(self.limit_min, self.limit_min, self.n_dim)
    #     sigma = sigma_org * 0
    #     optimum = mu, sigma
    #     return optimum

    #========================== mutación, duplicación o deletion of the gene ========================================
    def mutation(self, organism):
        print("Iniciando mutacion")
        #print('Organismo : ' + str(organism))
        if random.random() <=self.mutation_rate:
            [mu, sigma] = organism
            print('Organismo : ' + str([mu, sigma]))
            substract_value = random.uniform(self.limit_min, self.limit_max)
            vector_substract_value = np.random.multivariate_normal(self.mu_mean, self.sigma_covariance, check_valid='raise')
            print ('vector a substraer = ' + str(vector_substract_value))
            print('valor a substraer = ' + str(substract_value))
            #mu = [x - substract_value for x in mu]
            mu = mu - vector_substract_value
            sigma = [x - substract_value for x in sigma]
            organism = [mu, sigma]
            print('Organismo mutado: ' + str(organism))
        else:
            print('No se realiza mutación')
            # print('Organism ' + str(organism))
            # organism = organism[:]
        return organism

    def gene_duplication(self, organism):
        print("Iniciando gen duplication")
        print('Organismo : ' + str(organism))
        [mu, sigma] = organism
        if random.random() <= self.gen_duplication_rate:
            point = np.random.randint(len(sigma))
            #new_gene = sigma[point]
            #sigma.append(new_gene)
            sigma.append(sigma[point])
            #sigma.append(sigma[np.random.randint(len(sigma))])
            print('Posicion a duplicar : ' + str(point))
            #mu = mu_array.tolist()
            organism = [mu, sigma]
            print('New organism ' + str(organism))
        else:
            print('No se realiza duplicación del gen')
            # print('Organism ' + str(organism))
            # organism = organism[:]
        return organism

    def gene_deletion(self, organism):
        print("Iniciando gen deletion")
        print('Organismo : ' + str(organism))
        [mu, sigma] = organism
        if random.random() <= self.gen_deletion_rate:
            point = np.random.randint(len(sigma))
            selected_gene = sigma[point]
            print('Selected gene: ' + selected_gene)
            sigma.remove(selected_gene)
            organism = [mu, sigma]
            print('Posicion a eliminar : ' + str(point))
            print('New organism ' + str(organism))
        else:
            print('No se realiza supresión del gen')
            # print('Organism ' + str(organism))
            # organism = organism[:]
        return organism

    #=================================== Fitness function evaluation ===============================================

    def fitness(self, organism):
        """
        Funcion que determina cuántos valores son iguales a los del óptimo
        """
        [mu, sigma] = organism
        [mu_optimum, sigma_optimum] = self.optimum
        mu_fitness = np.subtract(mu, mu_optimum)
        sigma_fitness = np.subtract(sigma, sigma_optimum)
        #print('Mu substraction : ' + str(mu) + ' - ' + str(mu_optimum) + ' = ' + str(mu_fitness))
        #print('Sigma substraction : ' + str(sigma) + ' - ' + str(sigma_optimum) + ' = ' + str(sigma_fitness))
        fitness = [mu_fitness, sigma_fitness]
        print('Fitness : ' + str(fitness))

        fitness_value = self.fitness_function(organism)
        print('Fitness = ' + str(fitness_value))
        return fitness_value
        # fitness = 0
        #
        # for i in range(len(organism)):
        #     if organism[i] == self.optimum[i]:
        #         fitness += 1
        #     return fitness

    def reproduction(self, organism, father_organism):

        #father_organism = organism[:]

        organism = self.gene_duplication(organism)
        print('------')
        organism = self.gene_deletion(organism)
        print('------')
        organism = self.mutation(organism)
        print('------')

        print('Father : ' + str(father_organism))
        print('Son : ' + str(organism))

        return organism

    def selection(self, organism, father_organism):
        """
        si la evaluación del fitness hijo es mayor a la del fitness padre, hacer seleccion
        """
        #scores = [(self.fitness(i), i) for i in organism]
        #scores = [i[1] for i in sorted(scores)]

        #return scores[len(scores) - self.n_selection:]

        # evaluación fitness padre e hijo
        # scores = [self.fitness(organism), organism]
        # comparar organismo padre e hijo
        # seleccionar el que tenga mejor fitness
        # designarlo como el nuevo organismo padre
        #print(scores)

        score_son = self.fitness(organism)
        #print('Score son : ' + str(score_son))
        score_father = self.fitness(father_organism)
        #print('Score father : ' + str(score_father))

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
        [mu, sigma] = organism
        x_m = mu - self.mu_mean
        print('Mu = ' + str(mu) + ' mu mean = ' + str(self.mu_mean) + ' x_m = ' + str(x_m))
        fitness_value = (1. / (np.sqrt((2 * np.pi) ** self.n_dim * np.linalg.det(self.sigma_covariance))) * np.exp(
            -(np.linalg.solve(self.sigma_covariance, x_m).T.dot(x_m)) / 2))
        #print('Fitness = ' + str(fitness_value))
        return fitness_value
        # https://peterroelants.github.io/posts/multivariate-normal-primer/

    #======================================== Flux ================================================================
    def run(self):
        father_organism = self.create_organism()
        epsilon =
        for i in range(self.n_generations):
            # print(father_organism)
            organism = father_organism
            if self.verbose:
                print('___________')
                print('Generacion: ', i)
                print('Organismo padre: ', father_organism)
                print('Organismo hijo: ', organism)
                print()


            else:
                print('_______________________')
                print('Generacion: ', i)
                organism = self.reproduction(organism, father_organism)
                selected_organism = self.selection(organism, father_organism)
                print('Best suited organism is : ' + str(selected_organism))
                father_organism = selected_organism
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



#=========== Main definition of the
def main():
    optimum = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] #first vector most match the n_dim
    model = Organism(
        optimum = optimum,
        n_organisms = 1,
        mu_mean = [0.0, 0.0, 0.0], #the vector must match number of dimentions selected
        sigma_covariance = [[2000.0, -1000.0, 0.0],[-1000, 2000.0, -1000.0],[0.0, -1000.0, 2000.0]], #the matrix covariance must be a positive semidefinite symmetric one
        n_dim = 3,
        mutation_rate = 20,  # keep rates minimum
        gen_duplication_rate = 0.1,
        gen_deletion_rate=0.1,
        n_generations=1,
        limit_min= 0.00, #limite inferior para los valores del gen
        limit_max= 500.00, #limite superior para los valores del gen
        epsilon = 10.0,
        verbose=False)
    #model.create_organism()
    #model.selection(model.create_organism())
    #model.fitness(model.create_organism())
    model.run()
    # model.fitness_function(2, 20, 10)
    #model.mutation(model.create_organism())
    # model.gene_duplication(model.create_organism())
    # model.gene_deletion(model.create_organism())


if __name__ == '__main__':
    main()
