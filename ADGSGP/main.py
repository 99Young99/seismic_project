import random
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

from deap import algorithms
from initial import *
from calculation_error import evalSymbReg

avg_fit = 0
suc_rate = 0
avg_size = 0

for run in range(0,1): # 跑几轮算法

    for ei in range(0, 1): # 每轮跑多少次

        random.seed(103 + ei)

        rng = check_random_state(0) # 确保结果可以复现

        # Training samples
        X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
        y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1

        # Testing samples
        X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
        y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

        toolbox.register("evaluate", evalSymbReg, points=X_train, outputs=y_train)

        toolbox.register("SCR", sm.semConRep, toolbox=toolbox, pset=pset, points=X_train)

        # 装饰器，限制树的最大深度为 17 ，
        toolbox.decorate("SCR", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        time_start = time.time()

        pop = toolbox.population(n=100) # 初始化种群数量
        hof = tools.HallOfFame(1) # 名人堂最多存储一个个体

        lib = sm.library(toolbox, pset=pset, points=X_train)

        # 目标语义就是训练数据的值
        pop, log, eva_times = algorithms.semanticGP(pop, toolbox, tarSem=y_train, library=lib, ngen=50,
                                         stats=mstats, halloffame=hof, verbose=True, output_file=None)

        time_end = time.time()

        traintime = time_end - time_start

        func = toolbox.compile(expr=hof.items[0])
        # Make predictions on test set

        y_pred = numpy.array(list(func(*x) for x in X_test))

        y_Trainpred = numpy.array(list(func(*x) for x in X_train))


        # Show mean squared error
        print('Train MSE:', np.mean(np.square(y_train - y_Trainpred)))
        print('Test MSE:', np.mean(np.square(y_test - y_pred)))

        # Evaluate performance using R-squared score
        r2_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"R-squared score: {r2:.2f}")

        print(hof.items[0])

        print('Train time: ', traintime)

