import numpy as np


class subTree(object):
    expr=[]
    sem_vec=[]
    angle_dis=-1

    def __init__(self, toolbox, expr_st, points):
        """ expr_st: the sub tree individual (Individual object) """

        self.expr = expr_st

        # 将子树转化为可以传入数据的函数
        func = toolbox.compile(expr=expr_st)

        # print(np.shape(points))

        # exit(0)

        # for x in points:
        #     print(type(x))
        sqerrors = np.array(list(func(*x) for x in points))

        # exit(0)

        self.sem_vec=sqerrors # 将所有训练数据带入子树，计算结果作为当前子树的语义向量, 训练数据仅带入子树有的变量，子树没有的变量在计算时不带入

        # print(self.sem_vec)