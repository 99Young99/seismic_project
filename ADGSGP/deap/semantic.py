"""The :mod:`semantic` module provides the methods and classes to perform
Genetic Programming with Semantic backpropagation with DEAP. It essentially contains the classes to
evaluate the gp tree, maintain the semantic library, and variate the gp tree.
"""
import sys

from Algorithms.subTree import subTree
from deap import gp
import math
import random
import numpy
from deap import creator
from deap import base
from Algorithms.ADS import angle_dis, angleDrivenSel
from Algorithms.RSM import randSegMut
from Algorithms.SCP import *


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

class Node(object):
    content =[]  # primitive/terminal/ephemeral
    index=[]
    subSem0=[]  # numpy.ndarray  sub semantic
    subSem1=[]

    def __init__(self, con, index, subSem0, subSem1):
        self.content=con
        self.index=index
        self.subSem0=subSem0
        self.subSem1=subSem1

    def Invert(self, con, k, tarSem):
        """
        con: the node content
        k: the index of param 0 / 1
        tarSem: desired semantic
        """
        dsr_sem=[]
        if con.name() == "add":
            if k==0:
                dsr_sem = tarSem - self.subSem1
            else:
                dsr_sem = tarSem - self.subSem0
        elif con.name()=="sub":
            if k==0:
                dsr_sem = tarSem + self.subSem1
            else:
                dsr_sem = self.subSem0 - tarSem
        if con.name() == "mul":
            if k==0:
                for i in range(len(tarSem)):
                    if self.subSem1[i]!=0:
                        numpy.hstack([dsr_sem, tarSem[i]/self.subSem1[i]])
                    if self.subSem1[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem1[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
            else:
                for i in range(len(tarSem)):
                    if self.subSem0[i]!=0:
                        numpy.hstack([dsr_sem, tarSem[i]/self.subSem0[i]])
                    if self.subSem0[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
        if con.name()=="div":
            if k==0:
                for i in range(len(tarSem)):
                    if math.isfinite(self.subSem1[i]):
                        numpy.hstack([dsr_sem, tarSem[i]*self.subSem1[i]])
                    if math.isinf(self.subSem1[i]) and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if math.isinf(self.subSem1[i]) and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
            else:
                for i in range(len(tarSem)):
                    if self.subSem0[i]!=0:
                        numpy.hstack([dsr_sem, self.subSem0[i]/tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])

        return dsr_sem


def perpendicularCX(parents, tarSem):

    """
    :param parents:
    :param tarSem:
    :return:  O 向量的语义值
    """
    p1=parents[0]
    p2=parents[1]
    relSV1 = tarSem - p1.sem_vec # list(tarSem[i] - p1.sem_vec[i] for i in range(len(tarSem))) T - P1
    relSV2 = tarSem - p2.sem_vec # list(tarSem[i] - p2.sem_vec[i] for i in range(len(tarSem))) T - P2
    relatSV1 = p2.sem_vec - p1.sem_vec  # list(p2.sem_vec[i] - p1.sem_vec[i] for i in range(len(tarSem))) P2 - P1
    relatSV2 = p1.sem_vec - p2.sem_vec  # list(p1.sem_vec[i] - p2.sem_vec[i] for i in range(len(tarSem))) # P1 - P2

    alpha = angle_dis(relSV1, relatSV1)
    beta = angle_dis(relSV2, relatSV2)

    relaNorm = math.sqrt((relatSV1**2).sum()) # relatSV1 向量的模
    if alpha <= math.pi / 2 and beta < math.pi / 2:
        roNorm = math.sqrt(((p1.sem_vec - tarSem)**2).sum())*math.cos(alpha)
        ov = p1.sem_vec + (roNorm /relaNorm) * relatSV2
    elif alpha > math.pi / 2:
        roNorm = math.sqrt(((p1.sem_vec - tarSem) ** 2).sum()) * math.cos(math.pi-alpha)
        ov = p1.sem_vec - (roNorm / relaNorm) * relatSV2
    elif beta >= math.pi / 2:
        roNorm =  math.sqrt(((p2.sem_vec - tarSem) ** 2).sum()) * math.cos(math.pi-beta)
        ov = p2.sem_vec + (roNorm /relaNorm) * relatSV2
    else:
        ov = randSegMut(p1, tarSem)

    return ov





class library(object):

    expr_pool=[]  # list of subTree type items
    toolbox=[]
    pset=[]
    points=[]

    def __init__(self, toolbox, pset, points):
        self.toolbox=toolbox
        self.pset=pset
        self.points=points
        print("init library")


    def similarity(self, sv1, sv2):
        """compare the similarity between two semantic vector"""
        return ((sv1-sv2)**2).sum()/sv1.size

    def insert_lib(self, subt):
        """subt is the subTree type"""
        self.expr_pool.append(subt)

    def lib_clear(self):
        self.expr_pool.clear()

    def lib_maintain(self, pop):
        """pop: the population , type: deap.gp.Primitive object
        """
        self.lib_clear()


        for chro in pop:

            """chro: 子代个体，形式为函数表达式形式
            
                extract any sub tree from chro， check semantically unique， and insert into the library
            """


            for inde in range(1, chro.__len__()):
                # print('inde：', inde, ", chro.searchSubtree(inde):", chro[chro.searchSubtree(inde)])


                sub_expr=creator.Individual(chro[chro.searchSubtree(inde)])


                # print(np.shape(self.points))
                # exit(0)
                sub_tree=subTree(self.toolbox, sub_expr, self.points)

                if len(self.expr_pool)==0: # 子树池中为空则直接插入当前子树到 library 中，即
                    should_insert = True
                else:

                    should_insert=all((self.similarity(sub_tree.sem_vec, st.sem_vec) for st in self.expr_pool))
                if should_insert:
                    self.insert_lib(sub_tree)