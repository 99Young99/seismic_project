import math

from Algorithms.angle_dis import *



def angleDrivenSel(toolbox, pop, tarSem, np, nt, ta):
    """pop: population
        tarSem: target semantics， 即真实值
        np: the number of pairs, 即 offspring 个体数
        nt: maximum number of trials
        ta: the thresold of angle distance， 即角度驱动的阈值
        return: a list of selected pairs， 选择出的语义值
    """
    output_list=[]
    for i in range(0, np):
        flag = False
        p1 = toolbox.select(pop, k=1)[0] # 选择一个个体
        cp2_use = toolbox.select(pop, k=1)[0] # 预备 cp2， 当下面产生的 cp2 不满足时，利用这个进行新的 cp2 选择
        maxangle = -500
        for j in range(0, nt):
            cp2 = toolbox.select(pop, k=1)[0] # 候选个体 cp2
            relSV1 = tarSem - p1.sem_vec # 语义向量： T - P1
            relSV2 = tarSem - cp2.sem_vec # 语义向量： T - P2
            gamma = angle_dis(relSV1, relSV2)

            # 大于阈值，则选择
            if gamma > ta:
                p2=cp2
                flag=True
                break
            else:
                if gamma > maxangle:
                    cp2_use=cp2 #更新
                    maxangle=gamma # 保存当前最大的 maxangle
                #else:
                    #print(gamma, " ", cp2.sem_vec)

        if flag==False:
            p2=cp2_use

        if (abs(p2.sem_vec - p1.sem_vec) < 1e-4).all(): # 判断如果两个向量相同，则继续下一次迭代，否则加入 output_list 中
            continue
        output_list.append((p1,p2))

    return output_list