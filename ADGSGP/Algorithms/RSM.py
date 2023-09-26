import random


def randSegMut(parent, tarSem):
    relaSV = tarSem - parent.sem_vec
    k = random.random()
    return parent.sem_vec + k*relaSV
