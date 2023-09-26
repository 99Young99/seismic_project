import pandas as pd
import random
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from deap import algorithms
from initial import *
from calculation_error import evalSymbReg
import re

# import AdaBoost
X_train = pd.read_excel(r'./datasets/CCost.xlsx', header=0, usecols="A:V")
y_train = pd.read_excel(r'./datasets/CCost.xlsx', header=0, usecols="W")

X_train = np.array(X_train)
y_train = np.array(y_train).flatten()

X_train=(X_train-X_train.mean())/(X_train.std())
y_train=(y_train-y_train.mean())/(y_train.std())

# print(X_train.shape)
# sys.exit(0)
# 注册个体的评价函数 evalSymbReg，最终使用 toolbox.evaluate 来计算 fitness
toolbox.register("evaluate", evalSymbReg, points=X_train, outputs=y_train)

# 通过角度选择，PC, RSM 得到期望语义后，执行 Semantic Context Replacement
toolbox.register("SCR", sm.semConRep, toolbox=toolbox, pset=pset, points=X_train)

# 装饰器，限制树的最大深度为 17 ，
toolbox.decorate("SCR", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

time_start = time.time()

pop = toolbox.population(n=100)  # 初始化种群数量
hof = tools.HallOfFame(1)  # 名人堂最多存储一个个体

for i in range(5):

    lib = sm.library(toolbox, pset=pset, points=X_train)

    # 目标语义就是训练数据的值
    pop, log, eva_times = algorithms.semanticGP(pop, toolbox, tarSem=y_train, library=lib, ngen=50,
                                                stats=mstats, halloffame=hof, verbose=True, output_file=None)
    time_end = time.time()

    # ================================== train time =================================
    traintime = time_end - time_start

    func = toolbox.compile(expr=hof.items[0])
    # Make predictions on test set

    print(hof.items[0])

    Train_pred = numpy.array(list(func(*x) for x in X_train))
    Train_MSE = mean_squared_error(y_train, Train_pred)
    r2 = r2_score(y_train, Train_pred)

    print('训练误差：', Train_MSE, '， 回归系数：', r2)
    print()