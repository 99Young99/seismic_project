import math

def angle_dis(vec1, vec2):
    if len(vec1) != len(vec2):
        print("the dimension of two vectors must be consistent in angle_dis function")
        return 0
    if (vec1 == 0).all():
        vec1 = vec1 + 1e-4
    if (vec2 == 0).all():
        vec2 = vec2 + 1e-4
    vec1_sig = vec1.sum()  # math.fsum(vec1)
    vec2_sig = vec2.sum()  # math.fsum(vec2)
    vec_sum = (vec1*vec2).sum() # 计算向量 V1 * V2
    norm_vec1 = math.sqrt((vec1*vec1).sum())  # math.fsum((vi**2 for vi in vec1)) ||V1|| 计算向量 V1 的模
    norm_vec2 = math.sqrt((vec2*vec2).sum())  # math.fsum((vi ** 2 for vi in vec2)) ||V2|| 计算向量 V2 的模
    res = vec_sum/(norm_vec1*norm_vec2) # 两个向量的模相乘
    # res = (vec1_sig*vec2_sig) / (norm_vec1 * norm_vec2)
    if res>1:
        res=1
    elif res<-1:
        res=-1
    return math.acos(res)
