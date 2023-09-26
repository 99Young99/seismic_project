
import math

from Algorithms.RSM import randSegMut
from Algorithms.angle_dis import *


def perpendicularCX(parents, tarSem):
    p1=parents[0]
    p2=parents[1]
    relSV1 = tarSem - p1.sem_vec # list(tarSem[i] - p1.sem_vec[i] for i in range(len(tarSem)))
    relSV2 = tarSem - p2.sem_vec # list(tarSem[i] - p2.sem_vec[i] for i in range(len(tarSem)))
    relatSV1 = p2.sem_vec - p1.sem_vec  # list(p2.sem_vec[i] - p1.sem_vec[i] for i in range(len(tarSem)))
    relatSV2 = p1.sem_vec - p2.sem_vec  # list(p1.sem_vec[i] - p2.sem_vec[i] for i in range(len(tarSem)))
    alpha = angle_dis(relSV1, relatSV1)
    beta = angle_dis(relSV2, relatSV2)

    relaNorm = math.sqrt((relatSV1**2).sum())
    if alpha <= math.pi / 2 and beta < math.pi / 2:
        roNorm = math.sqrt(((p1.sem_vec - tarSem)**2).sum())*math.cos(alpha)
        ov = p1.sem_vec + (roNorm /relaNorm) * relatSV2
    elif alpha > math.pi / 2:
        roNorm = math.sqrt(((p1.sem_vec - tarSem) ** 2).sum()) * math.cos(math.pi-alpha)
        ov = p1.sem_vec - (roNorm / relaNorm) * relatSV2
    elif beta >= math.pi / 2:
        roNorm =  math.sqrt(((p2.sem_vec - tarSem) ** 2).sum()) * math.cos(math.pi-beta)
        ov = p2.sem_vec + (roNorm /relaNorm) * relatSV2
    else:
        ov = randSegMut(p1, tarSem)

    return ov