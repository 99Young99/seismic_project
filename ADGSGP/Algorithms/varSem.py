import math
import random
import sys

import numpy as np

def varSem(population, toolbox, tarSem, library):
    """

    """
    # Clone the selected individuals
    offspring = [toolbox.clone(ind) for ind in population]

    # use ADS to select a list of pairs，这里是一对一对的， ADS 一次同时选择出 p1 与 p2 ，作为一个元组（p1, p2）, 总共选出了 np 对
    angle_list = toolbox.ADS(pop=offspring, tarSem=tarSem, np=len(offspring), nt=10, ta=math.pi/2)

    # Apply crossover and mutation on the selected pairs, get the desired semantics
    for i in range(len(offspring)):
        if i < len(angle_list):
            if random.random() < 0.5: # 0.5 的概率进行交叉变异
                dsr_sem = toolbox.PC(parents=angle_list[i], tarSem=tarSem)
            else:
                dsr_sem = toolbox.RSM(parent=angle_list[i][0], tarSem=tarSem)
        else:
            dsr_sem = toolbox.RSM(parent=offspring[i], tarSem=tarSem)

        offspring[i], = toolbox.SCR(offspring[i], tarSem=dsr_sem, library=library) # 利用当前的期望语义，在 library 库中进行查找替换，限制每颗树的深度
        del offspring[i].fitness.values

    return offspring

