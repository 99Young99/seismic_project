import decimal
import sys

from deap.semantic import *
from deap import gp
from Algorithms.subTree import *
from Algorithms.Decode import *
from Algorithms.Invert import *
from Algorithms.angle_dis import *

def semConRep(par, tarSem, toolbox, pset, points, library):
    """ par: the parent individual
    MD: the maximum depth
    AL: the angle distance list
    tarSem: the final desired semantic
    """

    # get current parent individual depth
    curDep = par.__len__()
    spt = random.randint(0, curDep - 1)  # spliting point

    # print the substree result
    # for inde in range(1, par.__len__()):
    #     # 提取子树，通过 toolbox.compile 可以编译出运行的函数
    #     print('par.searchSubtree(inde)', par.searchSubtree(inde)) # slice(1, 12, None)
    #     sub_expr = creator.Individual(par[par.searchSubtree(inde)])
    #     print('inde：', inde, ", sub_expr:", sub_expr)

    # arity is paremeter, if arity is 0, par[spt] is terminal, else is primitive
    while par[spt].arity==0:
        spt = random.randint(0, curDep - 1)  # spliting point until select the primitive

    node = par[spt] # 子节点， 类型为 <deap.gp.Primitive object at 0x0000023FA88C0540>

    # print(node.ret)
    #
    # print(node.args) # 参数类型
    #
    # print(node.arity) # 参数数量

    # for p in pset.primitives[node.ret]:
    #     print(p)


    # ====replace the spliting point====
    prims = [p for p in pset.primitives[node.ret] if p.args == node.args] # pset.primitives[node.ret] 提取同 node.ret 同类型的节点,且参数相同

    # perhaps change the one function in the par， eg. sin -> cos
    par[spt] = random.choice(prims)

    #===== excute the inverse operation ====
    # decode the individual, parent_arr reprent every index for parent, index is start from 0, -1 is root,
    # children reprent every index for children, [-1, -1] is terminal
    parent_arr, children_arr=decode(par)


    # identify the path from root to R
    path=[spt] # store the node index from the root to index spt
    travel=spt
    while parent_arr[travel] != -1: # -1 is reprent root
        path.append(parent_arr[travel])
        travel = parent_arr[travel]
    path.reverse()

    # based on the children_arr, construct the other subtree and get its semantic
    dsr_sem=tarSem
    if len(path)>1:
        # use invert compute current semantic of path
        for j in path[1:len(path)]:
            if len(children_arr[parent_arr[j]]) == 1:
                dsr_sem = Invert(par[parent_arr[j]], 0, dsr_sem)
            else:
                # R in left sub tree
                if j == children_arr[parent_arr[j]][0]:
                    # calculate the semantic of right sub tree
                    fix_st=children_arr[parent_arr[j]][1] # get the index of right sub tress
                    sub_tree=subTree(toolbox,creator.Individual(par[par.searchSubtree(fix_st)]),points)
                    dsr_sem = Invert(par[parent_arr[j]], 0, dsr_sem, subSem1=sub_tree.sem_vec)
                else: # R in right sub tree
                    # calculate the semantic of left sub tree
                    fix_st=children_arr[parent_arr[j]][0]
                    sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
                    dsr_sem = Invert(par[parent_arr[j]], 1, dsr_sem, subSem0=sub_tree.sem_vec)

    # randomly select a child serving as T, if node R is a primitive.
    # determine the sub-semantic
    if node.arity>0:
        if node.arity == 1: # unary function
            Snode = children_arr[spt][0]
            dsr_sem = Invert(node, 0, dsr_sem)
        else: # binary function
            prefix = random.randint(0, node.arity - 1)
            # Snode = children_arr[parent_arr[path[-1]]][prefix]
            Snode = children_arr[spt][abs(1-prefix)]
            if prefix == 1:   # left subtree is T
                # fix_st = children_arr[parent_arr[path[-1]]][1]
                fix_st = children_arr[spt][1] # comput right subtree
                sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
                dsr_sem = Invert(node, 0, dsr_sem, subSem1=sub_tree.sem_vec)
            else:
                # fix_st = children_arr[parent_arr[path[-1]]][0]
                fix_st = children_arr[spt][0]
                sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
                dsr_sem = Invert(node, 1, dsr_sem, subSem0=sub_tree.sem_vec)
    else:
        return par   # cause the node R (a termnal / ephemeral) has been modified by a random operation at the beginning

    #==== construct the angle list from the library ====
    min_angle=500
    a=b=float()
    min_st = library.expr_pool[0].expr # get the first expression of library

    # find the most min angle of libary's all expression and dsr_sem
    for st in library.expr_pool:
        # compute the angle
        st.angle_dis = angle_dis(st.sem_vec, dsr_sem)
        if st.angle_dis < min_angle:
            min_angle = st.angle_dis
            min_st = st.expr
            min_st.sem_vec = st.sem_vec

    # ============================ Linear scaling to output ===================
    # dsr_sem = dsr_sem.reshape(dsr_sem.shape[0], 1)
    # obtain the coefficient b, dsr_sem.sum()/dsr_sem.size is the mean(dsr_sem)
    # if ((min_st.sem_vec - min_st.sem_vec.sum()/min_st.sem_vec.size)**2).sum() != 0:
    #     b = ((dsr_sem - dsr_sem.sum()/dsr_sem.size)*(min_st.sem_vec - min_st.sem_vec.sum()/min_st.sem_vec.size)).sum()/((min_st.sem_vec - min_st.sem_vec.sum()/min_st.sem_vec.size)**2).sum()
    # else:
    #     b = ((dsr_sem - dsr_sem.sum() / dsr_sem.size) * (min_st.sem_vec - min_st.sem_vec.sum() / min_st.sem_vec.size)).sum()

    # # obtain the coefficient a
    # a = dsr_sem.sum() / dsr_sem.size - b * min_st.sem_vec.sum() / min_st.sem_vec.size


    t = dsr_sem
    ct = min_st.sem_vec

    Norm_t = math.sqrt(math.fsum(t * t))

    Norm_ct = math.sqrt(math.fsum(ct * ct))

    if sum((ct - Norm_ct)*(ct - Norm_ct)) != 0:
        b = sum((t - Norm_t) * (ct - Norm_ct)) / sum((ct - Norm_ct) * (ct - Norm_ct))
        a = Norm_t - b * Norm_ct
    else:
        b = 1
        a = 0


    # ====crossover into the subtree====
    # use a, b construct the new sub tree , 对选出来的子树乘以 b + a
    nst =[pset.mapping['add'], gp.Constant(a), pset.mapping['mul'], gp.Constant(b)]
    nst.extend(min_st[:])
    new_subtree = creator.Individual(nst)

    CT_slice = par.searchSubtree(Snode)
    # par[CT_slice] = creator.Individual(min_st)
    par[CT_slice] = new_subtree

    return par,