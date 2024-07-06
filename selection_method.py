"""
GA 選擇 : 保留最佳解 
"""
from typing import List, Tuple
import random
import numpy as np

class SelectionMethod:
    """This class is to define selection method."""

    def __init__(self,
            pop_size: int = None) -> None:
        self.pop_size = pop_size

    @staticmethod
    def get_best_individual(pop, fit_value)-> Tuple[list, float]:
        """get best individual"""
        best_individual = pop[0]
        best_fit = fit_value[0]#in case = 0
        for i in range(1, len(pop)):
            if(fit_value[i] < best_fit) and (fit_value[i] > 0):
                best_fit = fit_value[i]
                best_individual = pop[i]
        return best_individual, best_fit

    @staticmethod
    def get_best_individual_for_contract(pop, fit_value)-> Tuple[list, float]:
        """get best individual"""
        best_individual = pop[0]
        best_fit = fit_value[0]#in case = 0
        for i in range(1, len(pop)):
            if(fit_value[i] <= best_fit) and (fit_value[i] > 0):
                if (fit_value[i] == best_fit) and (fit_value[i] > 0):
                    if pop[i][0][0][0] > best_individual[0][0][0]:#normal_demend_contract bigger is better
                        best_individual = pop[i]
                else:
                    best_fit = fit_value[i]
                    best_individual = pop[i]
        return best_individual, best_fit

    def selection(self, fit_value: List, parent_pop: List, fit_value_parent: List, pop: List)-> Tuple[list, list]:
        """Roulette Wheel Selection"""
        probability = self.convert_fit_value_to_probability(fit_value)
        random_select_value = self.get_random_select_value()
        return self.roulette_wheel_selection(parent_pop, fit_value_parent, pop, fit_value, probability, random_select_value)
    
    def convert_fit_value_to_probability(self, fit_value: List)-> List:
        """convert fit value to probability"""
        probability = []
        for i in range(len(fit_value)):
            if fit_value[i] <= 0 :
                probability.append([0])
            else:
                probability.append(1/fit_value[i])
        probability = probability/np.sum(probability) 
        return np.cumsum(probability)

    def get_random_select_value(self)-> List:
        """get random select value to select wheel"""
        random_select_value = []
        for i in range((self.pop_size -1)):
            random_select_value.append(random.random())
        random_select_value.sort()
        return random_select_value
    
    def roulette_wheel_selection(self, parent_pop: List, fit_value_parent: List, pop: List, fit_value: List, probability: List, random_select_value: List)-> Tuple[list, list]:
        """process of roulette wheel selection"""
        fitin = 0
        newin = 0
        new_pop = parent_pop.copy()[:-1]# 保留給最好的
        new_fit_value_parent = fit_value_parent.copy()[:-1]# 保留給最好的
        # 转轮盘选择法
        while newin <  (self.pop_size-1):
            if(random_select_value[newin] < probability[fitin]):
                new_pop[newin] = pop[fitin]
                new_fit_value_parent[newin] = fit_value[fitin]
                newin = newin + 1
            else:
                fitin = fitin + 1
        return new_pop.copy(), new_fit_value_parent.copy()