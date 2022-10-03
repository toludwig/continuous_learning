# Model Based (MB) solution for the shopping task

# Assuming only 3 goals + a starting point, the task is easy:
# Go from the start to the closer one of the extreme points
# (i.e. the ones on a bounding box = with the highest norm after centering),
# then via the middle point to the other extreme point.

import numpy as np

def MB_shopping(env, task, start=[0,0]):

    def manhattan(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # find start location
    start = env.start

    n_goals = sum(task)

    # find the goal locations
    goals = []
    for i,t in enumerate(task):
        if t == 1:
            goals.append(np.nonzero(env.item[:,:,i] == 1))

    if n_goals == 1:
        min_dist = manhattan(start, goals[0])
        min_path = [start, min_dist, goals[0]]
    elif n_goals == 2:
        # measure distance from start to these extreme points
        dist = [manhattan(start, goals[g]) for g in [0,1]]
        sa = manhattan(start, goals[0])
        ab = manhattan(goals[0], goals[1])
        min_dist = sa + ab
        min_path = [start, sa, goals[0], ab, goals[1]]
    elif n_goals == 3:
        # find the two most extreme points (z-transform)
        x, y = zip(*goals)
        x = np.stack(x)
        y = np.stack(y)
        z1 = x - np.mean(x)
        z2 = y - np.mean(y)

        znorm = np.abs(z1) + np.abs(z2)
        znorm = znorm[:,0] # to make it a vector

        extremity = np.argsort(znorm)
        extreme2 = extremity[-2:] # the two most extreme points

        # measure distance from start to these extreme points
        dist = [manhattan(start, goals[g]) for g in extreme2]

        # go to the extreme goal which is closest
        tmp = extreme2[np.argsort(dist)]
        a = tmp[0][0] # define a as the closer of the two extremes
        c = tmp[1][0] # and c as the other
        b = extremity[0]

        sa = manhattan(start, goals[a])
        ab = manhattan(goals[a], goals[b])
        bc = manhattan(goals[b], goals[c])
        min_dist = sa + ab + bc
        min_path = [start, sa, goals[a], ab, goals[b], bc, goals[c]]
    else: # n_goals > 3:
        raise NotImplementedError("More than 3 goals are not supported by the MB agent yet.")

    return (min_dist, min_path)
