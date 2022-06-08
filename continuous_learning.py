import numpy as np
import pandas as pd
import torch
import random
import copy
import sys
# progress bar
from time import sleep
from tqdm import tqdm

# load environment and models
from env import TwoStepEnv
from uvfa import UVFA
from sfgpi import SFGPI

#########################
# SIMULATION-PARAMETERS #
#########################
# note: N, B, T and M are short aliases, used mainly in the csv filename,
# but will be avoided in code for legibility
N = n_subjects = 100      # repetitions of simulation to average over
B = n_blocks = 50         # number of blocks
T = block_size = 100      # number of trials in a block
M = n_tasks_per_block = 2 # number of unique tasks per block ("multi-tasking")
p_task_change = 0         # probability of task change
p_feature_change = 0    # probability of feature change
p_transition_change = 0.5 # probability of transition change

#########################
# exploration           #
#########################
# dynamic exploration schedule
EPS_START = .5
EPS_DECAY = 20
EPS_END   = .05
def epsilon_greedy(Q, block, trial):
    """
    Returns an epsilon-greedy policy and epsilon for a given block and trial.
    The 0-th block of the simulation is free exploration with epsilon = 1.
    For all subsequent blocks, the first 10 trials are greedy (epsilon = 0).
    After that, epsilon follows an exponential decay starting at epsilon = 0.5.
    """
    if block == 0:
        eps = 1 # random in first block
    elif trial < 10:
        eps = 0 # greedy for the first steps in a block
    else: # normal decay
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-(trial-10)/ EPS_DECAY)

    pi = np.ones(TwoStepEnv.n_actions) * eps / TwoStepEnv.n_actions
    pi[np.argmax(Q)] += 1 - eps
    return (pi, eps)

#########################
# model hyperparams     #
#########################
# note: I haven't really optimised these, learning rates might not be balanced.
LEARNING_RATE_UVFA = 2
LEARNING_RATE_SFGPI = .75
GAMMA = .99           # discounting for TD
BUFFER_CAPACITY = 250 # how many transitions can be stored for offline learning
BATCH_SIZE = 50       # from how many transitions to learn at a time (UVFA only)

#########################
# Possible tasks        #
#########################
# Tomov & Schulz 2021
#TASKS = [[1,-1,0], [-1,1,0], [1,-2,0], [-2,1,0], [1,1,1]]

# all combinations of -1, 0, 1, except for [0,0,0]
TASKS = [[1,0,0],  [0,1,0],  [0,0,1],  [1,1,0],  [1,0,1], [0,1,1],  [1,1,1],
         [-1,0,0], [0,-1,0], [0,0,-1], [-1,1,0], [-1,0,1], [0,-1,1], [-1,1,1],
         [1,-1,0], [1,0,-1], [0,1,-1], [1,-1,1], [1,1,-1], [-1,-1,0],
         [-1,0,-1], [0,-1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]


def init_blocks_randomly(n_blocks, block_size, n_tasks_per_block,
                         p_task_change, p_feature_change, p_transition_change):
    """
    Inits an experiment in which tasks, features and transitions can change
    stochastically, p_*_change are the probabilities of the respective changes.
    Returns a list of blocks, i.e. (tasks, world) tuples,
    as well as the optimal rewards and leaves for each.
    changes is a boolean tuple of booleans indicating what changed.
    """
    blocks = [] # list of tuples, each of which of the form (tasks, world)
    tasks_of_block = []
    world_of_block = []
    optimal_reward = np.zeros((n_blocks, n_tasks_per_block), dtype=np.int16)
    optimal_leaves = np.zeros((n_blocks, n_tasks_per_block, TwoStepEnv.n_states), dtype=bool)
    changes = [] # list of tuples, each like (task_change, feature_change)

    # init environment and tasks
    env = TwoStepEnv()
    w_old = random.choices(TASKS, k=n_tasks_per_block)
    # w_old has dim n_tasks_per_block * D

    for b in range(n_blocks):
        task_change = random.random() < p_task_change # sample Bernoulli
        if task_change:
            w_old = w_old[:-1] # discard the oldest task
            w_new = random.choices(TASKS) # sample new task
            # stack new and old tasks, such that the new task is the first
            w_old = w_new + w_old
            # TODO make sure, the new task is truely distinct from all old ones
        tasks_of_block.append(w_old)

        feature_change = random.random() < p_feature_change # sample Bernoulli
        if feature_change and b > 0:
            env = copy.deepcopy(env)
            # change feature of the leaves that are optimal wrt. the new task
            env.swap_optimal_feature(optimal_leaves[b-1,0,:])

        transition_change = random.random() < p_transition_change
        if transition_change and b > 0:
            env = copy.deepcopy(env)
            # change transition of the path that is optimal wrt. the new task
            env.swap_transitions(optimal_leaves[b-1,0,:])

        world_of_block.append(env)

        for t, w in enumerate(w_old):
            x, y = world_of_block[b].optimal_trajectory(w)
            optimal_reward[b,t] = x
            optimal_leaves[b,t,:] = y

        # store changes
        changes.append((task_change, feature_change, transition_change))

    blocks = list(zip(tasks_of_block, world_of_block))
    return (blocks, optimal_reward, optimal_leaves, changes)


def init_blocks_smartly():
    """
    Design a curriculum of tasks somehow,
    e.g. such that tasks are maximally diverse in a block.
    """
    pass


def run_uvfa(blocks, optimal_reward, verbose=True):
    """
    Run UVFA algorithm, which is a network model (task, state) -> Q
    trained by backprop and offline learning from a replay buffer.
    Returns the regret and chosen leaves (= paths) for each block and trial.
    """
    print("UVFA")

    # init UVFA model
    input_size = TwoStepEnv.n_states + TwoStepEnv.n_features # UVFA gets the state + the task
    output_size = TwoStepEnv.n_actions # and predicts Q values for all actions
    uvfa = UVFA(input_size, output_size, LEARNING_RATE_UVFA,
                GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TARGET_UPDATE=2)
    uvfa_regret = [] # record of all regrets across trials and blocks
    uvfa_leaves = [] # record of all leaf nodes

    for block, (tasks, env) in tqdm(enumerate(blocks), total=n_blocks):

        if verbose:
            print("Tasks of this block: {}".format(tasks))
            #print("phi-6 of this block: {}".format(env.phi[5,:]))

        for trial in range(block_size):
            w = tasks[trial % n_tasks_per_block] # alternate tasks

            # simulate trajectory
            s = 0 # start at root
            trial_reward = 0
            while not env.terminal[s]:
                s_1hot = np.zeros(TwoStepEnv.n_states); s_1hot[s] = 1
                sw = torch.from_numpy(np.hstack([s_1hot, w])).float()
                Q = uvfa.predict_Q(sw)
                Q = Q.detach().numpy().astype(np.float64)
                pi,_ = epsilon_greedy(Q, block, trial)

                # sample next action / state
                a = np.nonzero(np.random.multinomial(1, pi))[0]
                a = torch.from_numpy(a)
                s_next = np.squeeze(env.P[s,a,:])
                s_next = np.nonzero(np.random.multinomial(1, s_next))[0][0]
                s_next_1hot = np.zeros(TwoStepEnv.n_states); s_next_1hot[s_next] = 1
                sw_next = torch.from_numpy(np.hstack([s_next_1hot, w])).float()

                # compute reward of next state
                # (NB: here, rewards are associated with states, not with actions)
                r = env.phi[s_next] @ w
                trial_reward += r

                # store transition in replay buffer and transit
                uvfa.replay_buffer.push(sw, a, sw_next, torch.tensor(r))
                s = s_next

            # update the neural network after each trial
            uvfa.train(trial)
            # fill record of trial
            regret = optimal_reward[block, trial % n_tasks_per_block] - \
                     trial_reward
            uvfa_regret.append(regret)
            uvfa_leaves.append(s)
            #print("trial {}: reward {}".format(trial, trial_reward))

    return (uvfa_regret, uvfa_leaves)


def run_sfgpi(blocks, optimal_reward, verbose=True):
    """
    Run SFGPI algorithm, which uses libraries of successor features for each task
    to compute the best policy for an unseen task by general. policy improvement.
    Returns the regret and chosen leaves (= paths) for each block and trial.
    """
    print("SFGPI")

    # init SFGPI
    input_size = TwoStepEnv.n_states # SFGPI only gets the state (not the task)
    output_size = TwoStepEnv.n_actions * TwoStepEnv.n_features # and predicts SF for all actions
    sfgpi = SFGPI(TwoStepEnv.n_states, TwoStepEnv.n_actions, TwoStepEnv.n_features,
                  LEARNING_RATE_SFGPI, GAMMA, BUFFER_CAPACITY, BATCH_SIZE)
    sfgpi_regret = [] # record of all regret across trials and blocks
    sfgpi_leaves = [] # record of all leaf nodes

    for block, (tasks, env) in tqdm(enumerate(blocks), total=n_blocks):

        if verbose:
            print("Tasks of this block: {}".format(tasks))
            #print("phi-6 of this block: {}".format(env.phi[5,:]))

        for trial in range(block_size):
            w = tasks[trial % n_tasks_per_block] # alternate tasks

            # simulate trajectory
            s = 0 # start at root
            trial_reward = 0
            while not env.terminal[s]:
                Q = sfgpi.predict_Q(s, w)
                pi,_ = epsilon_greedy(Q, block, trial)

                # sample next action / state
                a = np.nonzero(np.random.multinomial(1, pi))[0]
                s_next = np.squeeze(env.P[s,a,:])
                s_next = np.nonzero(np.random.multinomial(1, s_next))[0][0]

                # importantly, we don't store reward in transitions
                # but only the next state's features
                phi_next = env.phi[s_next]

                # anyways, compute reward of next state
                # (NB: here, rewards are received in states, not with actions)
                r = env.phi[s_next] @ w
                trial_reward += r

                # store transition in replay buffer and transit
                sfgpi.store_transition(w, s, a, s_next, phi_next)
                s = s_next

                # train online (last transition only)
                sfgpi.train_online(w)

            # update the neural network after each trial
            #sfgpi.train_offline(w, trial)

            # fill record of trial
            regret = optimal_reward[block, trial % n_tasks_per_block] - \
                     trial_reward
            sfgpi_regret.append(regret)
            sfgpi_leaves.append(s)
            #if s == 5: # test if psi ~ phi for an example state
            #    print("trial {}: psi5 {}".format(trial, psi[a,:]))

    return (sfgpi_regret, sfgpi_leaves)


def _min_distance(wold, wnew, metric="euclid"): # TODO rename vectors
    """
    Calculates the minimal distance between a set of vectors wold and wnew.
    """
    if metric == "euclid":
        d = min([np.sqrt((wnew-w) @ (wnew-w).T) for w in wold])
    elif metric == "angle":
        d = 1 - min([abs(wnew @ w.T / np.sqrt(wnew @ wnew.T) / np.sqrt(w @ w.T))
                     for w in wold])
    elif metric == "manhattan":
        d = min([np.sum(np.abs(wnew-w)) for w in wold])
    else:
        raise ValueError()
    return d

def _in_span(wall, wold):
    # 0 if wnew in span(wold), else 1
    return np.linalg.matrix_rank(wall) - np.linalg.matrix_rank(wold)


def collect_first_trials(blocks, n_trials, changes, uvfa_regret, uvfa_leaves,
                         sfgpi_regret, sfgpi_leaves, optimal_leaves):
    """
    Make a dataframe with the first n trials (n_trials) of each block.
    """
    df = pd.DataFrame(columns=["block", "trial", "algo", "task",
                               "leaf", "regret", "correct",
                               "task_change", "task_span", "task_angle",
                               "task_euclid", "task_manhattan",
                               "feature_change", "feature_angle",
                               "feature_euclid", "feature_manhattan",
                               "max_value_diff", "transition_change",
                               "possible_correct",
                               "mean_block_regret", "mean_block_correct"])
    # here we look only at the first trial of a task in a block
    for algo, leaves, regret in zip(["uvfa", "sfgpi"],
                                    [uvfa_leaves, sfgpi_leaves],
                                    [uvfa_regret, sfgpi_regret]):
        leaves = np.reshape(leaves, (n_blocks, block_size))
        regret = np.reshape(regret, (n_blocks, block_size))
        for b in range(1, n_blocks): # skip first block
            old_tasks, old_world = blocks[b-1]
            new_tasks, new_world = blocks[b]

            new_w = np.array(new_tasks) # n_task_per_block x dim
            old_w = np.array(old_tasks) # n_task_per_block x dim

            # changes?
            task_change, feature_change, transition_change = changes[b]

            # mean performance over whole block
            mean_block_regret = np.mean(regret[b,:])
            mean_block_correct = np.mean(regret[b,:] == 0)

            # distances are trial specific
            for t in range(n_trials):
                # task distances
                task_angle = _min_distance(old_w, new_w[t % M], metric="angle")
                task_euclid = _min_distance(old_w, new_w[t % M], metric="euclid")
                task_manhattan = _min_distance(old_w, new_w[t % M], metric="manhattan")

                # feature distances
                leaf = leaves[b,t]
                old_phi = old_world.phi[leaf,:]
                new_phi = new_world.phi[leaf,:]
                feature_angle  = 1 - abs(new_phi @ old_phi.T / np.sqrt(new_phi @ new_phi.T)
                                     / np.sqrt(old_phi @ old_phi.T))
                feature_euclid = np.sqrt((new_phi-old_phi) @ (new_phi-old_phi).T)
                feature_manhattan = np.sum(np.abs(new_phi-old_phi))

                # maximal value difference
                max_value_diff = np.max([np.max(np.abs(new_world.phi @ new_w[t % M].T
                                                       - old_world.phi @ w.T))
                                         for w in old_w])

                row = {
                    "block":          b,
                    "trial":          t,
                    "algo":           algo,
                    "task":           t % M,
                    "leaf":           leaves[b,t],
                    "regret":         regret[b,t],
                    "correct":        regret[b,t] == 0,
                    "task_change":    task_change and t % M == 0,
                    "task_span":      _in_span(new_tasks, old_w),
                    "task_angle":     task_angle,
                    "task_euclid":    task_euclid,
                    "task_manhattan": task_manhattan,
                    "feature_change": feature_change,
                    "feature_angle":  feature_angle,
                    "feature_euclid": feature_euclid,
                    "feature_manhattan": feature_manhattan,
                    "transition_change": transition_change,
                    "max_value_diff": max_value_diff,
                    "possible_correct": np.sum(optimal_leaves[b,t % M,:]),
                    "mean_block_regret": mean_block_regret,
                    "mean_block_correct": mean_block_correct,
                }
                df = pd.concat([df, pd.DataFrame(row, columns=df.columns, index=[b, t, algo])])

    # TODO check if df looks ok
    #print(df.head())
    return df


def simulate_and_save(subject_id="all"):
    """
    A subject would see all of the blocks (n_blocks) in a row,
    applying both algorithms.
    If subject_id == "all", simulate all subjects sequentially.
    Otherwise, simulate a single subject (for use on a cluster in parallel).
    Saves the resulting data frame in /sim/sim_N{N}_... if subject_id == "all"
    or /sim/sim_subject{subject_id}_... otherwise.
    """
    df_subjects = pd.DataFrame()

    if subject_id == "all":
        subject_id = range(n_subjects)

    for subject in subject_id:
        print(f"Subject {subject}")
        blocks, optimal_reward, optimal_leaves, changes = init_blocks_randomly(
            B, T, M, p_task_change, p_feature_change, p_transition_change)
        uvfa_regret,  uvfa_leaves  = run_uvfa(blocks,  optimal_reward, verbose=False)
        sfgpi_regret, sfgpi_leaves = run_sfgpi(blocks, optimal_reward, verbose=False)

        n_first_trials = 10
        df = collect_first_trials(blocks, n_first_trials, changes,
                                  uvfa_regret, uvfa_leaves,
                                  sfgpi_regret, sfgpi_leaves, optimal_leaves)
        df.insert(0, "subject", subject)
        df_subjects = pd.concat([df_subjects, df])

    if subject_id == "all":
        prefix = f"N{n_subjects}"
    else:
        prefix = f"subject{subject_id}"

    filename = f"./sim/sim_{prefix}_B{n_blocks}_T{block_size}_M{n_tasks_per_block}_ptask{p_task_change}_pfeature{p_feature_change}_ptransition{p_transition_change}.csv"

    df.to_csv(filename, index=False)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        # run for a specific subject
        subject_id = [int(sys.argv[1])]
        simulate_and_save(subject_id=subject_id)
    else:
        # run for all subjects
        simulate_and_save()


