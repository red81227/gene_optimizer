"""
This scrip is for encoding and decoding the phenotype and genotype

step_solution -> chromosome -> genotype -> pop

"""
from typing import Dict
import random
import numpy as np

class GeneTranslation():
    """This class is to define model structure, and training model."""
    def __init__(self,
        pop_size: int = None,
        number_of_features: int = None,
        future_step: int = None,
        accuracy: list = None
    ) -> None:
        self.pop_size = pop_size
        self.number_of_features = number_of_features
        self.future_step = future_step
        self.accuracy = accuracy
        # self.empty_step_solution = np.zeros(self.accuracy,int)
        
    def create_encoded_pop(self)-> np.array:
        """Create population genotype"""
        pop = []
        for _ in range(self.pop_size):
            genotype = self.get_genotype()
            pop.append(genotype)
        return pop
    
    def get_genotype(self)-> list:
        """get genotype for each pop"""
        genotype = []
        for feature in range(self.number_of_features):
            chromosome_length = self.accuracy[feature]
            upper_bound = chromosome_length-1
            empty_step_solution = np.zeros(chromosome_length,int)
            chromosomes = self.get_chromosome(upper_bound, empty_step_solution)
            chromosomes = np.asarray(chromosomes)
            genotype.append(chromosomes)
        return genotype 


    def get_chromosome(self, upper_bound, empty_step_solution)-> list:
        """get chromosome for each genotype"""
        chromosomes = []
        for _ in range(self.future_step):
            lower_bound = 0
            chromosome = empty_step_solution.copy()
            new_position = random.randint(lower_bound, upper_bound)#數值位置
            chromosome[new_position] = 1 #random give  1
            chromosomes.append(chromosome)
        return chromosomes


    def decode_chrom(self, pop: np.array, actions: list) -> list:
        """decode genotype to phenotype"""
        pop_phenotype = []
        for individual in range(len(pop)):
            phenotype = self.get_phenotype(actions, pop, individual)
            pop_phenotype.append(phenotype)
        return pop_phenotype
    
    @staticmethod
    def get_phenotype(actions: list, pop: np.array, individual: int)-> np.array:
        """get individual phenotype"""
        
        phenotype = []
        for action in range(len(actions)):
            action_array = np.array(actions[action], dtype=float)
            phenotype_array = np.asarray(pop[individual][action][:, np.newaxis] * action_array)
            action_phenotype = [np.sum(array) for array in phenotype_array]

            # for i in range(phenotype_array.shape[0]):
            #     print([np.sum(array) for array in phenotype_array])
                
            #     action_phenotype.append([np.sum(array) for array in phenotype_array])        
            phenotype.append(action_phenotype)
        return np.array(phenotype)

