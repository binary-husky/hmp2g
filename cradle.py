import numpy as np
def get_rank(score_list):
    print('各队伍得分:', score_list)
    rank = np.array([sum([ s2 > s1 for s2 in score_list ]) for s1 in score_list])
    if (rank == 0).all():  # if all team draw, then all team lose
        rank[:] = -1
    print('各队伍排名:', rank)
    return rank
get_rank([1,2,2,0,2])
get_rank([9,9,9,9,9])