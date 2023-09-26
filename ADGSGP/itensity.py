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
X = pd.read_excel(r'./datasets/基于神经网络的烈度衰减融合模型研究_数据集.xlsx', header=0, usecols="B, A")
y = pd.read_excel(r'./datasets/基于神经网络的烈度衰减融合模型研究_数据集.xlsx', header=0, usecols="C")

y = np.array(y).flatten()

# data = pd.DataFrame()
data = {'地震时间、地点、震级': [],
        '烈度': []}

# 遍历每一行数据
for index, row in X.iterrows():
    # 在这里可以访问每一行的数据
    # print(f"Row {index + 1}:")

    Moment_magnitu = row['地震时间、地点、震级']
    if isinstance(Moment_magnitu, str):
        pattern = r'M=(\d+(\.\d+)?)'  # 匹配 M 后面的数字（包括整数和浮点数）
        matches = re.findall(pattern, Moment_magnitu)
        M = matches[0][0] # 震中距
        M = float(M)

    data['地震时间、地点、震级'].append(M)

    # print(type(M))


    data['烈度'].append(row['烈度'])

    # print(Moment_magnitu)

# exit(0)
data = pd.DataFrame(data)
data = np.array(data)
# y    = np.array(y)

# print(data.head(5))
#
# print(y.head(5))
Algorithm = '你好'
dataset = 'without'

# 所有数据集
Train_total_Mse = 0
Test_total_Mse = 0
Train_total_Time = 0
R2_total = 0

for run in range(0, 1): # trial number

    rng = check_random_state(run) # Make sure the algorithm is reproducible

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=run + 1)

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
    pop, log, eva_times = algorithms.semanticGP(pop, toolbox, tarSem=y_train, library=lib, ngen=100,
                                                stats=mstats, halloffame=hof, verbose=True, output_file=None)
    time_end = time.time()

    # ================================== train time ==============================================
    traintime = time_end - time_start

    func = toolbox.compile(expr=hof.items[0])
    print(hof.items[0])
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

    run += 1
    print('Dataset:', dataset, ', Training time:', Train_total_Time / run, ', Training MSE:', Train_total_Mse / run,
          ', Test MSE:', Test_total_Mse / run, 'R2:', R2_total / run)

    # with open(output_file, 'a', newline='', encoding='utf8') as out:
    #     # 1:创建writer对象
    #     writer = csv.writer(out)
    #
    #     # 2:遍历列表，将每一行的数据写入csv
    #     for p in result:
    #         writer.writerow(p)
