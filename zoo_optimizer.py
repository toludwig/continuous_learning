"""
Optimizes the zoo for our experiment, so to increase power.
E.g. discriminability between MB and SFGPI solutions is optimized
by minimizing the distance between the solution paths.

"""
import numpy as np
import pandas as pd
import random
import copy
import pickle
# env and models
from zoo import Zoo
from sfgpi import SFGPI
from zoo_feeding import init_curriculum, run_training_test

#########################
# SIMULATION-PARAMETERS #
#########################
N = 1 # repetitions / subjects
B = 8 # number of blocks
T = 100 # trials per block
M = 2 # number of concurrent tasks (learned alternatingly)
tags = ""

# tasks are tuples of -1,0,1
TASKS = [[0,1], [1,0], [1,1], [-1,0], [0,-1], [-1,-1], [-1,1], [1,-1]]


def hill_climbing(curriculum, cost_functions, constraints, convergence_crit=10):
    """
    Simulate training-test runs on a single block of tasks and
    minimize the sum of the cost functions using hill-climbing.
    They are functions that take a simulation dataframe and return a cost
    (e.g. 1st-trial similarity between mb and sfgpi paths).
    Minimization is subject to additional constraints, given as a function
    of the environment itself (e.g. to have unique mb-paths for each task).
    We use random tweaks of features to iteratively improve the environment
    until the convergence criterium is fulfilled (i.e. no change for x times).
    Returns the final environment and its cost score.
    """
    assert(cost_functions != []) # we need at least one cost function to compute the score

    env1 = Zoo()           # old
    # at the start, assign random features
    env1.assign_features_randomly()
    env2 = copy.copy(env1) # new

    # now, do hill-climbing, i.e. compute the cost of the solution,
    # then tweak a single phi and keep the solution if it's better
    cost1 = 1000 # the lower the better
    cost2 = 0
    cost_improved     = 0 # count of how often cost improved
    cost_not_improved = 0 # count of how often in a row cost did not improve
    constraint_failed = 0
    while(cost_improved < 2 and cost1 > 0): # stop if improved at least twice or if cost==0
          #cost_not_improved < convergence_crit): # or stop, if it takes too long
        #print(constraint_failed)

        # step 1: mutate env2
        if constraint_failed > 20: # if constraints cannot be satisfied
            env2.assign_features_randomly() # restart with a new set of features
            constraint_failed = 0
        else: # otherwise, just swap two features
            env2.mutate_features_randomly()

        # step 2: check if all constraints are fulfilled, if not, back to 1
        if not all([c(env2, curriculum) for c in constraints]):
            constraint_failed += 1
            continue

        # step 3: compute cost2 and keep env2 if better
        df = run_training_test(env2, curriculum)
        cost2 = max([cf(df) for cf in cost_functions]) # important: max!

        if cost2 < cost1:
            env1 = copy.copy(env2) # improve env1
            cost1 = cost2
            print("Cost improved: " + str(cost1))
            cost_improved += 1
            cost_not_improved = 0
        else:
            env2 = copy.copy(env1)
            cost_not_improved += 1

        #print(df.loc[0, "mb_path"])
        #print(df.loc[0, "sf_path"])
        print(df.loc[0, "mb_sim"])
        print(df.loc[0, "mf_sim"])
        #print(env1.phi)

    return (env1, cost1, df)


def meta_hill_climbing(cost_functions, constraints, restarts=2): #100):
    """
    Restarts hill climbing with different initialisations.
    """
    env  = None
    cost = 1000
    for r in range(restarts):
        env1, cost1 = hill_climbing(tasks, cost_functions, constraints)
        if cost1 < cost:
            env  = env1
            cost = cost1
            print(cost)
    return env, cost


###########################################################################
# COST FUNCTIONS AND CONSTRAINTS
###########################################################################

# possible cost functions
# each function should have a cost between 0 and 1
mb_sim_1st = lambda df: df.loc[df["trial"] == 0, "mb_sim"].mean() / 3
mf_sim_1st = lambda df: df.loc[df["trial"] == 0, "mf_sim"].mean() / 3
#mb_npaths  = lambda df: df.loc[df["trial"] < M, "mb_npaths"].sum() / M / B / 8


def unique_mb_paths_constraint(env, curriculum):
    training, test = curriculum
    tasks = [*training, test]
    mb = np.zeros([len(tasks), env.n_paths])
    for t, w in enumerate(tasks):
        _, mb[t,:] = env.mb_paths(w)

    return np.all(np.sum(mb, axis=1) == 1)


def diverse_mb_paths_constraint(env, curriculum):
    """
    Constraint is fulfilled if there is only one MB path per task
    and all of them are disjoint.
    """
    training, test = curriculum
    tasks = [*training, test]
    mb = np.zeros([len(tasks), env.n_paths])
    for t, w in enumerate(tasks):
        _, mb[t,:] = env.mb_paths(w)

    return np.all(np.sum(mb, axis=0) <= 1) and np.all(np.sum(mb, axis=1) == 1)


###########################################################################
# OPTIMIZE BLOCKS
###########################################################################

def optimize_blocks():
    """
    Precomputes blocks for an experiment.
    A block is a tuple of environment and curriculum.
    Saves the result of each optimization in a pickle file.
    """

    # there are 4 possible compositions
    curricula = [
        [[[1,0], [0,1]], [1,1]],
        [[[1,0], [0,-1]], [1,-1]],
        [[[-1,0], [0,1]], [-1,1]],
        [[[-1,0], [0,-1]], [-1,-1]],
    ]

    for b, cur in enumerate(curricula):
        env, _, df = hill_climbing(cur,
                                   cost_functions=[mb_sim_1st, mf_sim_1st],
                                   constraints=[diverse_mb_paths_constraint])
        block = (env, cur, df)

        with open(f"./zoo/block_{b}.pickle", "wb") as f:
            pickle.dump(block, f)



if __name__ == "__main__":
    optimize_blocks()
    #env, score = meta_hill_climbing(cost_functions, constraints)
    # print(str(env.phi))
    # print(str(env.phi).replace("\n", "").replace(" ", ","))
    # print(score)
