import csv
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state
from deap import algorithms
from initial import *
from calculation_error import evalSymbReg

Algorithm = 'ADGSGP.csv'
output_file = './result/{}'.format(Algorithm)

header = ['Algorithm', 'Dataset', 'trial', 'Training time', 'Training MSE', 'Test MSE', 'R2']

with open(output_file,'w', newline='',encoding='utf8') as out:
    # 1:创建writer对象
    writer = csv.writer(out)

    # 2:写表头
    writer.writerow(header)

# 所有数据集
Train_total_Mse = 0
Test_total_Mse = 0
Train_total_Time = 0
R2_total = 0

X = pd.read_excel(r'./datasets/CCost.xlsx', header=0, usecols="A:V")
y = pd.read_excel(r'./datasets/CCost.xlsx', header=0, usecols="W")

X = np.array(X)
y = np.array(y).flatten()

# print(X)
# sys.exit(0)

# dataset = "高温材料"

for run in range(0, 1): # trial number

    rng = check_random_state(run) # Make sure the algorithm is reproducible

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=run + 1)

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

    lib = sm.library(toolbox, pset=pset, points=X_train)

    # 目标语义就是训练数据的值
    pop, log, eva_times = algorithms.semanticGP(pop, toolbox, tarSem=y_train, library=lib, ngen=50,
                                                stats=mstats, halloffame=hof, verbose=True, output_file=None)
    time_end = time.time()

    # ================================== train time =================================
    traintime = time_end - time_start

    func = toolbox.compile(expr=hof.items[0])
    # Make predictions on test set

    Train_pred = numpy.array(list(func(*x) for x in X_train))
    Test_pred = numpy.array(list(func(*x) for x in X_test))

    Train_MSE = mean_squared_error(y_train, Train_pred)
    Test_MSE = mean_squared_error(y_test, Test_pred)

    # ============================= Show R squared  ===================================
    r2 = r2_score(y_test, Test_pred)

    # ============================== store date to csv file ============================
    result = [(Algorithm, dataset, run + 1, traintime, Train_MSE, Test_MSE, r2)]

    Train_total_Mse = Train_total_Mse + Train_MSE
    Test_total_Mse = Test_total_Mse + Test_MSE
    Train_total_Time = Train_total_Time + traintime
    R2_total = R2_total + r2

    with open(output_file, 'a', newline='', encoding='utf8') as out:
        # 1:创建writer对象
        writer = csv.writer(out)

        # 2:遍历列表，将每一行的数据写入csv
        for p in result:
            writer.writerow(p)

run = run + 1
Avg_result = [(Algorithm, dataset, run, Train_total_Time/run, Train_total_Mse/run, Test_total_Mse/run, R2_total/run)]

print('Dataset:', dataset, ', Training time:', Train_total_Time/run, ', Training MSE:', Train_total_Mse/run, ', Test MSE:',Test_total_Mse/run,  'R2:', R2_total/run)
with open(output_file, 'a', newline='', encoding='utf8') as out:
    # 1:创建writer对象
    writer = csv.writer(out)

    # 2:遍历列表，将每一行的数据写入csv
    for p in Avg_result:
        writer.writerow(p)
    writer.writerow('\n')