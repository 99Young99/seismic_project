import decimal
import operator
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import semantic as sm
from OverloadedOperator import *


decimal.getcontext().prec = 100

pset = gp.PrimitiveSet("MAIN", 22)

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

pset.renameArguments(ARG0='x0')
pset.renameArguments(ARG1='x1')

# 定义问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, sem_vec=numpy.zeros(180)) #
toolbox = base.Toolbox()


toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)


toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("compile", gp.compile, pset=pset)

# toolbox.register("select", tools.selAutomaticEpsilonLexicase)
toolbox.register("select", tools.selTournament, tournsize=2)


toolbox.register("selectBest", tools.selBest)

toolbox.register("ADS", sm.angleDrivenSel, toolbox=toolbox)

toolbox.register("PC", sm.perpendicularCX)

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


