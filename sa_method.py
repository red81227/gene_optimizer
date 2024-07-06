"""This scrip define SA method"""

import random
import numpy as np


class SA:
    """ SA method."""
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def anneal_prob(delta_c, temperature)-> float:
        """Anneal prob""" 
        return min(1, np.clip(np.exp(1*float(delta_c/temperature)), -1e6, 1e6))

    @staticmethod
    def anneal_stride(solution)-> list:
        """Anneal stride"""
        
        solution_temp = solution[:][0] #提取array出來走下一步

        #紀錄原狀態的pos
        pos = []
        for i  in range(len(solution_temp)):
            pos2 = []
            for j  in range(len(solution_temp[i])):
                pos2.append(np.where(solution_temp[i][j] != 0)[0][0])
            pos.append(pos2)

        #往前or往後跨出solution
        for i  in (range(len(pos))):
            for j  in (range(len(pos[i]))):
                            
                if(random.random() > 0.5):  #>往前Or往後各占一半機率
                    if pos[i][j] < len(solution_temp[i][j])-1 :
                        pos[i][j]=pos[i][j] + random.randint(1,1)
                else:
                    if pos[i][j] > 0:
                        pos[i][j]=pos[i][j] - random.randint(1,1)
                    
        #填回solution
        temp=[]
        for i in range(len(solution_temp)):
            temp2 = []
            
            for j in range(len(solution_temp[i])):              
                dna1 = np.zeros(len(solution_temp[i][j]),int)              
                dna1[pos[i][j]] = 1
                temp2.append(dna1)        
            temp2 = np.asarray(temp2)# a dicision at one feature at all time step
            temp.append(temp2)
        # temp = np.asarray(temp)# a dicision of all feature  
        output = []
        output.append(temp);

        return output