"""
Implements the zoo-feeding task:
Given an animal and its vitamin demands (task), find the right fruits for it.
TODO The arrangement of fruits in the market changes in each block.
"""

import numpy as np
import pandas as pd
import random
import copy
import sys
# progress bar
from time import sleep
from tqdm import tqdm
# env and models
from zoo import Zoo
from sfgpi import SFGPI

#########################
# SIMULATION-PARAMETERS #
#########################
N = 10 # repetitions / subjects
B = 3 # number of blocks
T = 100 # trials per block
M = 2 # number of concurrent tasks (learned alternatingly)
tags = ""


#########################
# TASKS                 #
#########################
"""
Tasks are tuples of -1,0,1.
There is two types of tasks, cardinal and diagonal ones
corresponding to the vectors on the unit square [-1,1]^2.
Notice, that the diagonal ones are summations of the cardinal ones,
hence, there are 4 "compositions" of cardinals that lead to the diagonals
(we only allow summations that lead to tasks in the square, excluding [0,0]).
There are also 4 cases of summations of a cardinal with a diagonal task
which lead back to a cardinal one (e.g. [1,1] + [-1,0] = [0,1]).
We call these "decompositions" or "projections".
Decomposition means that one dimension can now be disregarded
which means that the policy of the diagonal task could be simply reused.
"""
CARDINAL = [[0,1], [1,0], [0,-1], [-1,0]]
DIAGONAL = [[1,1], [-1,1], [1,-1], [-1,-1]]
TASKS    = [CARDINAL, DIAGONAL]

#########################
# model hyperparams     #
#########################
ALPHA = .5 #0.75       # learning rate for TD
GAMMA = .90           # discount rate for TD
BUFFER_CAPACITY = 250 # how many transitions can be stored for offline learning
BATCH_SIZE = 50       # from how many transitions to learn at a time
RESET_BUFFER = True   # whether to empty the buffer at the beginning of a block
# TODO always true?

#########################
# exploration           #
#########################
# dynamic exploration schedule
EPS_START = .8
EPS_DECAY = 10 # the lower, the steeper
EPS_END   = .05
def epsilon_greedy(Q, trial):
    """
    Returns an epsilon-greedy policy and epsilon for trial x within a block.
    Epsilon follows an exponential decay.
    """
    n_actions = len(Q)
    if trial == 0:
        eps = 0 # be greedy on first trial
    else:
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-trial/ EPS_DECAY)
    pi = np.ones(n_actions) * eps / n_actions
    pi[np.argmax(Q)] += 1 - eps
    return (pi, eps)


def ucb_exploration(Q, counts, trial):
    """
    Upper confidence bound exploration, i.e. to the q-value
    we add an uncertainty bonus depending on how often the action was taken.
    The bonus is also dependent on time, in later trials, we exploit more.
    Returns a policy, i.e. distribution over actions.
    """
    n_actions = len(Q)
    if trial == 0:
        beta = 0 # be greedy on the first trial
    else:
        beta = BETA
    pi = Q + beta * np.sqrt(np.log(trial) / counts)
    pi /= np.sum(pi)
    return pi


#########################
# init curriculum       #
#########################

def init_circular_curriculum():
    # this function creates a sequence of blocks,
    # alternating compositions and decompositions in a circular fashion
    # following the pattern: c c c d c c c d c c c d ...
    curriculum = [#[[0,0], [0,0]],   # random
                  [[1,0], [0,1]],   # c c
                  [[1,1], [-1,0]],  # d c
                  [[0,1], [-1,0]],  # c c
                  [[-1,1], [0,-1]], # d c
                  [[-1,0], [0,-1]], # c c
                  [[-1,-1], [1,0]], # d c
                  [[0,-1], [1,0]],  # c c
                  [[1,-1], [0,1]]]  # d c

    curriculum_len = len(curriculum)

    # repeat the curriculum B times
    if curriculum_len < B:
        curriculum *= B // curriculum_len
        curriculum += curriculum[0:B % curriculum_len]
    else:
        curriculum = curriculum[0:B]
    return curriculum


def init_curriculum(kind="composition"):
    """
    Initialises a curriculum, i.e. a tuple of training tasks and test task.
    There are different kinds of curricula:
    - unrelated: the test task is not informed by the training tasks
    - projection: the test task can be solved in the same way as one of the training tasks
    - composition: the test task is a summation of the training tasks
    TODO generalize to M
    The training-test task distance is defined as
    D = min_{w in training} d_1(w, w_test)
    where d_1 is the manhattan distance.
    It is 0 for projections, 1 for compositions and 3 for unrelated. # TODO 2 or 3?
    """
    if kind == "composition":
        test = random.choice(DIAGONAL)
        w1 = [test[0], 0]
        w2 = [0, test[1]]
    elif kind == "projection":
        # pick one cardinal and one diagonal one
        w1 = random.choice(DIAGONAL)
        axis = random.choice([0,1]) # on which axis to project
        w2 = [0,0]
        w2[axis] = -w1[axis]
        test = [0,0]
        test[1-axis] = w1[1-axis]
    elif kind == "unrelated":
        # there is two ways to achieve a distance of D=3 on the square:
        # T type or L type?
        subkind = random.choice(["T", "L"])
        if subkind == "T":
            test = random.choice(CARDINAL) # bottom of T stem
            axis = 0 if test[0] == 0 else 1 # on which axis the stem of the T is
            w1 = [0,0]
            w1[1-axis] = -test[1-axis]
            w1[axis] = -1
            w2 = [0,0]
            w2[1-axis] = -test[1-axis]
            w2[axis] = 1
        else: # L type
            test = random.choice(DIAGONAL) # long end of the L
            w1 = [0,0]
            w1[0] = -test[0]
            w2 = [0,0]
            w2[1] = -test[1]
    else:
        raise ValueError("Not a valid curriculum kind")

    # def manhattan(v, w):
    #     return abs(v[0]-w[0]) + abs(v[1]-w[1])

    # def task_distance(training, test):
    #     return min(map(lambda w: manhattan(w, test), training))

    return ([w1, w2], test) # training, test


def init_blocks():
    """
    Inits a list of blocks, each is a tuple of a curriculum and environment. TODO
    """
    pass



def init_mb_mf(env, curriculum):
    """
    Precomputes MB and MF paths for a curriculum.
    Returns a tuple of mb_reward, mb_paths and mf_paths.
    """
    mb_reward = np.zeros([B,M])               # MB reward
    mb_paths  = np.zeros([B,M,8], dtype=bool) # MB paths as a binary vector
    mf_paths  = np.zeros([B,8], dtype=bool)   # MF paths as a binary vector
    for b in range(B):
        for m in range(M):
            mb_reward[b,m], mb_paths[b,m,:] = env.mb_paths(curriculum[b][m])
        if b > 0:
            m = np.argmax(mb_reward[b-1,:]) # which mb path was best in b-1
            mf_paths[b,:] = mb_paths[b-1,m]

    return (mb_reward, mb_paths, mf_paths)


def run_trial(env, sfgpi, t, w, her):
    trial_path = []
    trial_reward = 0
    env.reset()
    done = False
    s = 0
    while not done:
        Q = sfgpi.predict_Q(s, w)
        # TODO if t == 1: print(Q)
        pi,_ = epsilon_greedy(Q, t)
        a = np.nonzero(np.random.multinomial(1, pi))[0]

        # transition
        observation, reward, done = env.step(a, w)
        s_next, phi_next = observation

        # update path and reward info
        trial_path   += [s_next]
        trial_reward += reward

        # store transition in replay buffer and transit
        sfgpi.store_transition(w, s, a, s_next, phi_next)
        s = s_next

        # train online (only this one transition)
        sfgpi.train_online(w, her=her)

    # update SFs after each trial
    sfgpi.train_offline(w, her=her)

    # path in index form
    trial_path = Zoo.path_items2idx(trial_path)

    return (trial_path, trial_reward)


def run_training_test(env, curriculum, n_repeat=10,
                      ensure_training_converged=True, her=True):
    """
    Given a curriculum of M training tasks and 1 test task,
    repeat training and test n_repeat times
    and return the most common test path.
    If ensure_training_converged == True, repeat even more often,
    such that training is always convered to the MB solution before testing.
    """
    w_training, w_test = curriculum

    sf_training = np.zeros((M, n_repeat), dtype=int)
    mb_training = np.zeros((M, env.n_paths), dtype=int)
    for m in range(M):
        mb_training[m,:] = env.mb_paths(w_training[m])[1] # only path
    sf_test = np.zeros(n_repeat, dtype=int)
    reward_test = np.zeros(n_repeat, dtype=int)
    mb_reward_test, p_mb = env.mb_paths(w_test)
    _, p_mf = env.mf_paths(w_training)

    r = 0 # repetition counter
    while r < n_repeat:
        # init sfgpi with no memory across repetitions
        sfgpi = SFGPI(env.n_states, env.n_actions, env.n_features,
                      ALPHA, GAMMA, BUFFER_CAPACITY, BATCH_SIZE,
                      TASK_CAPACITY=4) # XXX two previous and two current tasks

        # training
        for t in range(T):
            w = w_training[t % M]
            run_trial(env, sfgpi, t, w, her)

        # last training path
        for m in range(M):
            sf_training[m,r], _ = run_trial(env, sfgpi, 0, w_training[m], her) # deterministic at t=0

        # check, if all training tasks converged to their MB solution
        if ensure_training_converged:
            if not sum(mb_training[range(M), sf_training[range(M),r]]) == M:
                continue # if not, train again

        # test
        sf_test[r], reward_test[r] = run_trial(env, sfgpi, 0, w_test, her) # deterministic at t=0

        # next repetition
        r += 1

    #print(path_training)
    #print(path_test)

    # find most common test path
    unique_paths, path_count = np.unique(sf_test, return_counts=True)
    most_common = np.argmax(path_count)
    if np.size(most_common) > 1:
        most_common = most_common[0]
    p_sf = int(unique_paths[most_common])
    reward_sf = reward_test[most_common]

    # compute the similarity to the closest mf/mb path
    mb_sim = env.max_path_similarity(p_sf, np.nonzero(p_mb)[0])
    mf_sim = env.max_path_similarity(p_sf, np.nonzero(p_mf)[0])
    # compute the similarity to the mb paths of previous block
    p0_mb = np.nonzero(mb_training[0,:])[0]
    p1_mb = np.nonzero(mb_training[1,:])[0]
    p0_sim = env.max_path_similarity(p_sf, p0_mb)
    p1_sim = env.max_path_similarity(p_sf, p1_mb)
    # is p_sf a composition of the previous mb paths?
    sf_comp = env.any_path_composed(p_sf, p0_mb, p1_mb)

    # make data frame for the last M training trials and the test trial
    df = pd.DataFrame( # TODO generalize to M
            data = [{"trial": -2, "task": str(w_training[0]),
                     "sf_path": sf_training[0, most_common],
                     "mb_path": np.nonzero(mb_training[0,:])[0][0],
                     "mb_paths": str(mb_training[0,:]),
                     "mb_npaths": sum(mb_training[0,:])},
                    {"trial": -1, "task": str(w_training[1]),
                     "sf_path": sf_training[1, most_common],
                     "mb_path": np.nonzero(mb_training[1,:])[0][0],
                     "mb_paths": str(mb_training[1,:]),
                     "mb_npaths": sum(mb_training[1,:])},
                    {"trial": 0, "task": str(w_test),
                     "reward": reward_sf, "regret": mb_reward_test - reward_sf,
                     "sf_path": p_sf,
                     "mb_path": np.nonzero(p_mb)[0][0],
                     "mf_path": np.nonzero(p_mf)[0][0],
                     "mb_paths": str(p_mb), "mf_paths": str(p_mf),
                     "mb_npaths": sum(p_mb), "mf_npaths": sum(p_mf),
                     "mb_sim": mb_sim, "mf_sim": mf_sim,
                     "p0_sim": p0_sim, "p1_sim": p1_sim,
                     "p01_sim": (p0_sim + p1_sim) / 2,
                     "sf_comp": sf_comp}],
        index=[-2,-1,0])
    return df


def run_curriculum(env, curriculum, mb_reward, mb_paths, mf_paths, her=True):
    """
    Implements SFGPI zoo-feeding for a whole curriculum.
    Returns a dataframe with the records (e.g. regret, paths, MF and MB paths).
    """

    # init a single sfgpi with memory across blocks
    sfgpi = SFGPI(env.n_states, env.n_actions, env.n_features,
                  ALPHA, GAMMA, BUFFER_CAPACITY, BATCH_SIZE,
                  TASK_CAPACITY=4) # XXX two previous and two current tasks

    # init records
    df = pd.DataFrame()

    for b in tqdm(range(B)):
        for t in range(T):
            w = curriculum[b][t % M]

            # simulate a trial
            trial_path, trial_reward = run_trial(env, sfgpi, t, w, her)

            # fill record of trial
            reward_mb = mb_reward[b,t%M]
            regret = reward_mb - trial_reward
            p_mb = mb_paths[b,t%M,:]
            p_mf = mf_paths[b,:]
            p_sf = Zoo.path_items2idx(trial_path)

            # compute the similarity to the closest mf/mb path
            mb_sim = env.max_path_similarity(p_sf, p_mb)
            mf_sim = env.max_path_similarity(p_sf, p_mf)
            # compute the similarity to the mb paths of previous block
            p0_mb = np.nonzero(mb_paths[b-1,0,:])[0]
            p1_mb = np.nonzero(mb_paths[b-1,1,:])[0]
            p0_sim = env.max_path_similarity(p_sf, p0_mb)
            p1_sim = env.max_path_similarity(p_sf, p1_mb)
            # is p_sf a composition of the previous mb paths?
            sf_comp = env.any_path_composed(p_sf, p0_mb, p1_mb)

            row = {
                "block":      b,
                "trial":      t,
                "trialblock": b*T+t,
                "task":       t % M,
                "reward":     trial_reward,
                "regret":     regret,
                "sf_path":    p_sf,
                "mb_paths":   str(p_mb),
                "mf_paths":   str(p_mf),
                "mb_sim":     mb_sim,
                "mf_sim":     mf_sim,
                "mb_npaths":  sum(p_mb),
                "mf_npaths":  sum(p_mf),
                "p0_sim":     p0_sim,
                "p1_sim":     p1_sim,
                "p01_sim":    (p0_sim + p1_sim) / 2,
                "sf_comp":    sf_comp,
            }

            df = pd.concat([df, pd.DataFrame(row, index=[b*T+t])])

    return df


def simulate_and_save():
    df_subjects = pd.DataFrame()

    for subject in range(N):
        print(f"Subject {subject}")

        # every subject gets its own env
        # (so seed is reset and scrambling is the same across subjects)
        env = Zoo()
        curriculum = init_circular_curriculum()
        mb_reward, mb_paths, mf_paths = init_mb_mf(env, curriculum)
        df = run_curriculum(env, curriculum, mb_reward, mb_paths, mf_paths,
                            her=False) # TODO True
        df.insert(0, "subject", subject)
        df_subjects = pd.concat([df_subjects, df])

    filename = f"./zoo/sim_N{N}_B{B}_T{T}_M{M}_{tags}.csv"

    df_subjects.to_csv(filename, index=False)
    return df


if __name__ == "__main__":
    #df = simulate_and_save()

    env = Zoo()
    df = run_training_test(env, [[0,1], [1,0]], [1,1])
    print(df)
