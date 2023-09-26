import numpy as np

from initial import *

# def evalSymbReg(individual, points, outputs):
#     # Transform the tree expression in a callable function
#     func = toolbox.compile(expr=individual)
#     # Evaluate the mean squared error between the expression
#     # and the real function : x**4 + x**3 + x**2 + x
#     #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
#     #semanVe = numpy.array(list(func(x[0], x[1]) for x in points))
#
#     semanVe = numpy.array(list(func(*x) for x in points)) # 计算语义向量
#
#     sqerrors = []
#
#     for x, y in zip(points, outputs):
#
#         sqerrors.append((decimal.Decimal(func(*x)) - decimal.Decimal(y)) ** 2)
#
#     return (math.sqrt(math.fsum(sqerrors) / len(points)),semanVe), # RMSE


def evalSymbReg(individual, points, outputs):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    #semanVe = numpy.array(list(func(x[0], x[1]) for x in points))

    # for x in points

    semanVe = numpy.array(list(func(*x) for x in points)) # 计算语义向量
    # semanVe = 0

    # for x in points:
    #     print(x)

    # for x, y in zip(points, outputs):
    #
    #     sqerrors.append((decimal.Decimal(func(*x)) - decimal.Decimal(y)) ** 2)

    t = np.mean(np.square(outputs - semanVe))

    return (np.mean(np.square(outputs - semanVe)),semanVe), # RMSE

