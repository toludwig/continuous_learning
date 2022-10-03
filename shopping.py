"""
Implements the shopping task:
Given a shopping list / recipe, the agent has to collect items
from the supermarket on the shortest possible route.
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
from supermarket import Supermarket
from sfgpi import SFGPI
from mb_shopping import MB_shopping
# TODO TEST
from plot_shopping import plot_path, plot_value_map

#########################
# SIMULATION-PARAMETERS #
#########################
N = 20 # repetitions / subjects
B = 3 # number of blocks
T = 200 # trials per block
M = 4 # number of concurrent tasks (learned alternatingly)
G = n_goals = 1 # number of goals per task
recipes = [[2], [3], [4], [5]]
#recipes = [[2,3], [3,12], [9,12], [2,6]] # tuples
#recipes = [[2,3,9], [3,12,15], [6,9,12], [2,6,15]] # XXX triples are recipes
tags = "alpha0.5_gamma0.9" # randomstart


#########################
# environment           #
#########################
# States are uniquely identified by their features and the current goal state,
# both are encoded in the observation as [1-hot-item; goal-state].
# where goal state is a vector with 1 bit per goal saying if it was reached.
n_grid = 4 # size of the grid (side of square)
n_states = (n_grid**2)*(2**n_goals) # state encodes item + 1 bit per goal
n_actions = 4 # how many actions are possible (north, east, south, west)
n_features = n_grid**2 # as many as there are items / cells

def _item2integer(item):
    return np.nonzero(item)[0][0]

def _task2binary(task):
    # recipe is an integer, task is a 3-hot vector
    binary = [0]*n_grid**2
    for t in task:
        binary[t] = 1
    return binary

def _dual2dec(goal_state):
    # transform dual goal_state (least-significant bit first) to decimal
    dec = 0
    for i,s in enumerate(goal_state):
        dec += s*2**i
    return dec

def _parse_observation(observation):
    """
    The observation is a concatenated vector of the 1-hot encoded observed item
    and the goal state, i.e. 1 bit per goal indicating if it was collected.
    Here, we call the item part "features" or "phi",
    and the "state" is a unique integer representing the whole observation.
    Returns a tuple of (state, phi).
    """
    phi = observation[0:n_features]
    goal_state = observation[n_features:]
    phi_int = _item2integer(phi)
    dec = _dual2dec(goal_state)
    state = int(phi_int + n_features * dec)
    return (state, phi)

def _state2item(state):
    """
    Transforms a state index to an item, ignoring the goal state.
    This is as simple as taking modulo the number of goal_states.
    """
    return state % n_features


#########################
# exploration           #
#########################
# dynamic exploration schedule
EPS_START = .8
EPS_DECAY = 20
EPS_END   = .05
def epsilon_greedy(Q, trial):
    """
    Returns an epsilon-greedy policy and epsilon for a given block and trial.
    Epsilon follows an exponential decay.
    """
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-trial/ EPS_DECAY)
    pi = np.ones(n_actions) * eps / n_actions
    pi[np.argmax(Q)] += 1 - eps
    return (pi, eps)


#########################
# model hyperparams     #
#########################
ALPHA = .5 #0.75       # learning rate for TD
GAMMA = .90           # discount rate for TD
BUFFER_CAPACITY = 250 # how many transitions can be stored for offline learning
BATCH_SIZE = 50       # from how many transitions to learn at a time


def run_sfgpi(env, her=True, verbose=False):
    """
    Implements SFGPI shopping.
    """

    # init records
    regret = np.zeros([B,T]) # record of all regret across trials and blocks
    path = np.empty([B,T], dtype=list) # record of all paths
    optimal_dist = np.zeros([B,T])
    #optimal_path = # TODO array of lists?

    for b in range(B):

        if verbose:
            print("new block: {}/{}".format(b, B))

        # at the beginning of a block scramble the market
        env.scramble()

        # init a new SFGPI because the transitions have changed completely
        sfgpi = SFGPI(n_states, n_actions, n_features,
                      ALPHA, GAMMA, BUFFER_CAPACITY, BATCH_SIZE)

        # HACK prevent cylces: if state is the same as one of the last 4, take a random action
        cycle_queue = []

        for t in tqdm(range(T)): # trials
            w = _task2binary(recipes[t % M]) # alternate tasks

            # simulate trajectory
            observation = env.reset(w)
            s, phi = _parse_observation(observation)

            trial_path = [_item2integer(phi)] # start item
            trial_reward = 0
            done = False
            while not done:
                Q = sfgpi.predict_Q(s, w)
                pi,_ = epsilon_greedy(Q, b, t)

                # sample next action / state
                if s in cycle_queue: # if cycle
                    a = np.random.choice([0,1,2,3]) # pick random action
                else:
                    a = np.nonzero(np.random.multinomial(1, pi))[0]

                cycle_queue.append(s) # enqueue
                if len(cycle_queue) > 4: # if capacity reached
                    cycle_queue.pop(0) # dequeue first

                # transition
                observation, reward, done = env.step(a, w)
                s_next, phi_next = _parse_observation(observation)

                # update path and reward info
                trial_path   += [_item2integer(phi_next)]
                trial_reward += reward

                # store transition in replay buffer and transit
                sfgpi.store_transition(w, s, a, s_next, phi_next)
                s = s_next

                # train online (only this one transition)
                sfgpi.train_online(w, her=her)

            # update the neural network after each trial
            sfgpi.train_offline(w, her=True)

            # fill record of trial
            optimal_dist[b,t], _ = MB_shopping(env, w)
            path_length = -(trial_reward - n_goals*env.REWARD_SCALE)
            regret[b,t] = path_length - optimal_dist[b,t]
            #print(regret[b,t])
            path[b,t] = trial_path

        # TODO test path
        plot_path(path[b,t], w, env.item2cell)

        # TODO at the end of each block, plot value map for the last task w
        plot_value_map(sfgpi.sf_by_task[t % M], w, env.item2cell)


    path = None # TODO
    optimal_path = None # TODO
    return (regret, path, optimal_dist, optimal_path)




def collect_trials(regret, path, optimal_dist, optimal_path):
    """
    Make a dataframe with all trials of each block.
    """
    df = pd.DataFrame()
    idx = 0 # only for enumerating rows

    for b in range(B):
        for t in range(T):
            row = {
                "block":   b,
                "trial":   t,
                "task":    t % M,
                "regret":  regret[b,t],
                "mb_dist": optimal_dist[b,t],
                #"path":    path[b,t],
                #"mb_path": optimal_path[b,t],
            }

            df = pd.concat([df, pd.DataFrame(row, index=[idx])])
            idx += 1 # only for enumerating rows

    return df


def simulate_and_save():
    df_subjects = pd.DataFrame()

    for subject in range(N):
        print(f"Subject {subject}")

        # every subject gets its own env
        # (so seed is reset and scrambling is the same across subjects)
        env = Supermarket(n_goals, n_grid)
        regret, path, optimal_dist, optimal_path = run_sfgpi(env, her=False, verbose=False)

        n_trials = 10
        df = collect_trials(regret, path, optimal_dist, optimal_path)
        df.insert(0, "subject", subject)
        df_subjects = pd.concat([df_subjects, df])

    filename = f"./supermarket/sim_N{N}_B{B}_T{T}_M{M}_G{G}_{tags}.csv"

    df_subjects.to_csv(filename, index=False)


if __name__ == "__main__":
    simulate_and_save()

