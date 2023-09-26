
import math

# Define new functions

# '\'
def protectedDiv(left, right):
    if right == 0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res

# '*'
def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

# 'log'
def protectedLog(arg):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))

# 'exp'
def protectedExp(arg):
    if arg > 10:
        arg = 10
    return math.exp(arg)