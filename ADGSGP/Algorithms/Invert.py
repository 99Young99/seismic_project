import math

import numpy


def Invert(con, k, tarSem, subSem0=numpy.zeros(1), subSem1=numpy.zeros(1)):
    """
    con: the node content
    k: the index of param 0 / 1
    tarSem: desired semantic

    algorithim principle: 见博文最后算法原理部分 https://blog.csdn.net/qq_46450354/article/details/128683758?spm=1001.2014.3001.5502

    """
    dsr_sem=numpy.zeros(tarSem.size)
    if con.name == "add":
        if k==0:
            dsr_sem = tarSem - subSem1
        else:
            dsr_sem = tarSem - subSem0
    elif con.name=="sub":
        if k==0:
            dsr_sem = tarSem + subSem1
        else:
            dsr_sem = subSem0 - tarSem
    if con.name == "mul":
        if k==0:
            for i in range(len(tarSem)):
                if subSem1[i]!=0:
                    dsr_sem[i] = tarSem[i] / subSem1[i]
                if subSem1[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem1[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0:
                    dsr_sem[i] = tarSem[i] / subSem0[i]
                if subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
    if con.name =="div":
        if k==0:
            for i in range(len(tarSem)):
                if math.isfinite(subSem1[i]):
                    dsr_sem[i] = tarSem[i]*subSem1[i]
                if math.isinf(subSem1[i]) and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if math.isinf(subSem1[i]) and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0 and tarSem[i]!=0:
                    dsr_sem[i] = subSem0[i]/tarSem[i]
                elif subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                elif subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
    if con.name == "sin":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.asin(tarSem[i])
    if con.name == "cos":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.acos(tarSem[i])
    if con.name == "exp":
        for i in range(len(tarSem)):
            if tarSem[i] <= 0:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.log(tarSem[i])
    if con.name == "ln":
        for i in range(len(tarSem)):
            if tarSem[i] > 10:
                tarSem[i] = 10
            dsr_sem[i] = math.exp(tarSem[i])

    return dsr_sem