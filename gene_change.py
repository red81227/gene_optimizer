"""
This scrip define GA crossover, mutation and repair 
"""
import random
from typing import Tuple
import numpy as np

class GeneChange:
    """
    crossover: 將選出來的解部分互換 
    mutation: 將選出來的解部分突變
    """
    def __init__(
        self,
        accuracy: np.array,
        future_step: int
    ) -> None:
        self.accuracy = accuracy
        self.future_step = future_step
    
    def crossover(self, pop: np.array, crossover_rate: float)-> np.array:
        """crossover part population chromosome"""
        for individual in range(len(pop)-1):
            if(random.random() < crossover_rate):
                genotype, genotype2 = self.crossover_each_chromsome(pop, individual)
                pop[individual] = genotype
                pop[individual+1] = genotype2
        return pop
    

    def crossover_each_chromsome(self, pop: np.array, individual: int)-> Tuple[np.array, np.array]:
        """crossover this genotype cromosome"""
        genotype = pop[individual]
        genotype2 = pop[individual+1]
        
        for chrom in range(len(genotype)-1):
            chromosome1 = genotype[chrom]
            chromosome2 = genotype2[chrom]
            cross_over_point = random.randint(0,len(chromosome1))
            new_chromosome1 = []
            new_chromosome2 = []
            new_chromosome1.extend(chromosome1[0:cross_over_point])
            new_chromosome1.extend(chromosome2[cross_over_point:len(chromosome1)])
            new_chromosome2.extend(chromosome2[0:cross_over_point])
            new_chromosome2.extend(chromosome1[cross_over_point:len(chromosome1)])
            new_chromosome1 = np.asarray(new_chromosome1)
            new_chromosome2 = np.asarray(new_chromosome2)
            genotype[chrom] = new_chromosome1
            genotype2[chrom] = new_chromosome2
        return genotype, genotype2

    def mutation(self, pop: np.array, mutation_rate: float)-> np.array:
        """mutation part population chromosome"""
        for individual in range(len(pop)):
            if(random.random() < mutation_rate):
                genotype = pop[individual]
                mutation_genotype = self.mutation_each_chromosome(genotype)
                pop[individual] = mutation_genotype
        return pop
    
    def mutation_each_chromosome(self, genotype: np.array)-> np.array:
        """mutation this genotype chromosome"""
        for chrom in range(len(self.accuracy)):
            mutation_point = random.randint(0, self.future_step-1)
            
            empty_genotype = np.zeros(self.accuracy[chrom], int)
            empty_genotype[random.randint(0, self.accuracy[chrom]-1)] = 1
            genotype[chrom][mutation_point] = empty_genotype
        return genotype


    def repair_ac_status(self, pop: np.array, ac_opend: list, current_ac_status: int)-> np.array:
        """repair ac status, only open or close once"""

        for individual in range(len(pop)-1):
            genotype = pop[individual]
            ac_array = genotype[1]
            #ac opend and all time step at ac open time 
            if current_ac_status == 1:
                if all(elem == 1 for elem in ac_opend):
                    genotype[1] = np.tile([0,1], (len(ac_opend), 1))
                    pop[individual] = genotype
                    continue
            #ac opend and all time step at ac close time 
            if current_ac_status == 0:
                if all(elem == 0 for elem in ac_opend):
                    genotype[1] = np.tile([1,0], (len(ac_opend), 1))
                    pop[individual] = genotype
                    continue

            #not all time step at ac open/close time
            if current_ac_status == 1:
                current_ac = np.array([0, 1])
            else:
                current_ac = np.array([1, 0])
            
            ac_array_with_current = np.insert(ac_array, 0, current_ac, axis=0)
            for element in range(len(ac_array_with_current)-1):
                if ac_array_with_current[element][0] != ac_array_with_current[element+1][0]:
                    break
            head_array = ac_array_with_current[:element+1,:]
            add_len = len(ac_array_with_current) -1 - element
            tail_array = np.tile(ac_array_with_current[element+1], (add_len, 1))
            ac_repair_array = np.concatenate((head_array, tail_array), axis=0)
            ac_repair_array = ac_repair_array[1:]#remove current ac status

            for element in range(len(ac_repair_array)-1):
                if current_ac_status == 1:
                    if ac_opend[element] == 1:#ac opend and at ac open time 
                        ac_status = [0, 1]
                    else:
                        ac_status = ac_repair_array[element]#ac opend and at ac close time, model decide
                else:
                    if ac_opend[element] == 1:
                        ac_status = ac_repair_array[element]#ac close and at ac open time, model decide
                    else:
                        ac_status = [1, 0] #ac close and at ac close time
                ac_repair_array[element] = ac_status

            genotype[1] = ac_repair_array
            pop[individual] = genotype
        return pop
        
