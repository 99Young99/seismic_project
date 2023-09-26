import decimal
import operator

import numpy

from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import semantic as sm

from OverloadedOperator import *

def inital(parame):

    decimal.getcontext().prec = 100

    pset = gp.PrimitiveSet("MAIN", parame) # 2 为输入变量的个数

    # ===================== 加入操作符 ======================================
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    #pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedMul, 2, name="mul")
    pset.addPrimitive(protectedDiv, 2, name="div")
    #pset.addPrimitive(operator.neg, 1)
    # pset.addPrimitive(math.cos, 1, name="cos")
    # pset.addPrimitive(math.sin, 1, name="sin")
    # pset.addPrimitive(protectedExp, 1, name= "exp")
    # pset.addPrimitive(protectedLog, 1, name="ln")
    #pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))

    # =========================== 重命名变量 ====================================
    pset.renameArguments(ARG0='x0')
    pset.renameArguments(ARG1='x1')

    # 定义问题
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # # 单目标，weights 指定 -1.0 表示最小化问题
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, sem_vec=numpy.zeros(180)) #  #创建 Individual 类，继承 gp.PrimitiveTree

    # =========================  创建个体 ===========================
    toolbox = base.Toolbox() # 实例化一个 Toolbox

    # 注册 toolbox.expr 个体编码方式为树形表达式, 并通过注册 toolbox.individual 来集成
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6) # gp.genHalfAndHalf： 利用 half and half 的形式产生树结构，树形结果的最小深度为 2， 最大深度为 6
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 注册初始化种群的工作 toolbox.population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 注册 toolbox.compile 对个体树形表达式转化为可编译函数
    toolbox.register("compile", gp.compile, pset=pset)

    # ======================== 定义评价函数 evalSymbReg ===============================


    # =================================== 定义种群演化操作 ============================================

    # 选择操作
    # toolbox.register("select", tools.selTournament, tournsize=2) # 锦标赛
    toolbox.register("select", tools.selAutomaticEpsilonLexicase) # 选择 AutomaticEpsilonLexicase

    # 为保持种群规模，需要将育种后代插入到父代中,去掉父代或子代的一部分个体，这里选择 selBest 插入，组合新的种群
    toolbox.register("selectBest", tools.selBest)

    # 角度驱动的选择操作
    toolbox.register("ADS", sm.angleDrivenSel, toolbox=toolbox)

    # 角度交叉
    toolbox.register("PC", sm.perpendicularCX)

    # 随机片段变异
    toolbox.register("RSM", sm.randSegMut)

    #  ================================ 注册计算过程中需要记录的数据 ==============================
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values) # 模板
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)


    # SF = StatisticFile.StatisticFile()


