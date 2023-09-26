import re

import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.utils import check_random_state
import csv
import pandas as pd

from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# import AdaBoost
X = pd.read_excel(r'./datasets/基于神经网络的烈度衰减融合模型研究_数据集.xlsx', header=0, usecols="B, A")
y = pd.read_excel(r'./datasets/基于神经网络的烈度衰减融合模型研究_数据集.xlsx', header=0, usecols="C")

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
# data = np.array(data)
# y    = np.array(y)

Algorithm = 'gplearn'
dataset = 'jj'
Avg_result = 0
r2 = 0
# 所有数据集
Train_total_Mse = 0
Test_total_Mse = 0
Train_total_Time = 0
R2_total = 0

for run in range(0,1): # trial number

    rng = check_random_state(run) # Make sure the algorithm is reproducible

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=run + 1)

    time_start = time.time()

    est = SymbolicRegressor(population_size=500,
                               generations=50, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=0, metric='mse')
    est.fit(X_train, y_train)

    time_end = time.time()

    # ================================== train time =================================
    traintime = time_end - time_start

    Train_pred = est.predict(X_train)
    Test_pred = est.predict(X_test)

    Train_MSE = mean_squared_error(y_train, Train_pred)
    Test_MSE = mean_squared_error(y_test, Test_pred)

    # ============================= Show R squared  ===================================
    y_pred = est.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # ============================== store date to csv file ============================
    result = [(Algorithm, dataset, run + 1, traintime, Train_MSE, Test_MSE, r2)]

    Train_total_Mse = Train_total_Mse + Train_MSE
    Test_total_Mse = Test_total_Mse + Test_MSE
    Train_total_Time = Train_total_Time + traintime
    R2_total = R2_total + r2

    run = run + 1
    Avg_result = [(Algorithm, dataset, run, Train_total_Time/run, Train_total_Mse/run, Test_total_Mse/run, R2_total/run)]

    print('Dataset:', '地震项目', ', Training time:', Train_total_Time/run, ', Training MSE:', Train_total_Mse/run, ', Test MSE:',Test_total_Mse/run,  'R2:', R2_total/run)

