import sys


def decode(ind):
    """
    ind: type individual
    return:
    parent list (each item: Node type (primitive/terminal, index))
    children list (Node type)

    algorithim principle:
    ind: mul(sub(x0, x1), add(x1, x1))
    the final parent: [-1, 0, 1, 1, 0, 4, 4]
    the final children : [[1, 4], [2, 3], [-1, -1], [-1, -1], [5, 6], [-1, -1], [-1, -1]]
    explain parent:
                                 [-1, 0, 1, 1, 0, 4, 4]
                    -1 represent current index of 0 is root, (0 => mul)
                    index 0 is the parent of current index 1, 4,  (1 => sub, 4 => add)
                    index 1 is the parent of current index 2, 3, (2 => x0,  3 => x1)
                    the final index 4 is the parent of current index 5, 6 ,(5 => x1, 6 => x1)

    explain children:
                        [[1, 4], [2, 3], [-1, -1], [-1, -1], [5, 6], [-1, -1], [-1, -1]]
                    the first [1, 4] is the index 0'children, (0 => mul, 1 => sub, 4 => add)
                    [2, 3] is the index 1's children, (1 => sub, 2 => x0, 3 =>  x1)
                    [-1, -1] is terminal located index 2, 3, (2 => x0, 3 =>  x1)
                    Residual empathy

    """

    # print(ind)
    parent= [] # 1 dimension list
    children= []  # 2 dimension list
    pi=-1   # pi: the processing index;   usi: the last unsatisfied index
    parent.append(pi)
    pi = usi = pi + 1

    # print(len(ind))


    if len(ind) == 1:
        return parent, children

    for i in range(1, len(ind)):

        # update the parent
        parent.append(pi)    # record the parent of current node ，individual in combined by list， so the first index 0 is root
        #pi = pi + 1          # update the parent

        # update the children
        if pi + 1 > len(children):  # if the parent node is new, append new item to the tail of children
            children.append([i]) # add [i], represent two dimension

        # if the parent node has been met, append the item to the existing item of children
        else:
            children[pi].append(i)

        # update the usi
        # if pi is satisfied
        if ind[pi].arity == len(children[pi]):
            # find the last unsatisfied parent
            while ind[usi].arity <= len(children[usi]):
                if usi >= 1:
                    usi = usi - 1
                else:
                    break
        # if pi hasn't been satisfied
        else:
            usi = pi

        # update the pi
        # if i is the primitive, pi follows i
        if ind[i].arity > 0:
            pi = usi = i

        # else if i is the terminal / ephemeral && pi is not satisfied, pi keeps still
        # else if i is the terminal / ephemeral && pi is satisfied, pi follows usi
        elif ind[i].arity==0 and ind[pi].arity == len(children[pi]):
            pi = usi

        # if i is the terminal / ephemeral, insert the placeholder into children
        if ind[i].arity == 0:
            children.append([-1,-1]) # [-1, -1] is terminal
    # print(parent)
    # print('=======================================')
    # print(children)
    return parent, children