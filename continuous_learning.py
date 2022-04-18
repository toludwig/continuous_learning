import numpy as np
import torch
import random
from matplotlib import pyplot as plt

from env import TwoStepEnv
from uvfa import UVFA
from sfgpi import SFGPI

# Implements the experiment logic
# (blockwise training on some tasks, swapping out a task after each block).
# Models are updated in an online fashion.

block_size = 300 # for now fixed, might be stochastic
n_blocks = 4
n_tasks_per_block = 2 # always learn some tasks concurrently

# hyperparams
# for simulated annealing lower temperature and epsilon
# softmax temperature for softmax policy
TAU_START = 10
TAU_DECAY = 20
TAU_END   = 1
# epsilon for epsilon-greedy policy
EPS_START = .05
EPS_DECAY = 20
EPS_END = .05

LEARNING_RATE = 1
GAMMA = .99 # discounting for TD
BUFFER_CAPACITY = 250 # how many transitions can be stored for offline learning
BATCH_SIZE = 150 # from how many transitions to learn at a time
TARGET_UPDATE = 2 # update target net every N trials

# init environment and tasks
env = TwoStepEnv()
tasks = [[1,0,0],  [0,1,0],  [0,0,1],  [1,1,0],  [1,0,1], [0,1,1],  [1,1,1],
         [-1,0,0], [0,-1,0], [0,0,-1], [-1,1,0], [-1,0,1], [0,-1,1], [-1,1,1],
         [1,-1,0], [1,0,-1], [0,1,-1], [1,-1,1], [1,1,-1], [-1,-1,0],
         [-1,0,-1], [0,-1,-1], [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]
random.shuffle(tasks) # shuffles inplace
tasks = [[1,-1,0], [-1,1,0], [1,-2,0], [-2,1,0], [1,1,1]]
tasks_of_block = {block: tasks[block:block+n_tasks_per_block]
                  for block in range(n_blocks)}
feature5_block = {0: env.phi[5,:], 1: [5,0,0], 2: [5,0,0], 3: [5,0,0]}
optimal_reward, optimal_leaves = zip(*[env.optimal_trajectory(w)
                                       for w in tasks])

# init UVFA model
input_size = env.n_states + env.n_features # UVFA gets the state + the task
output_size = env.n_actions # and predicts Q values for all actions
uvfa = UVFA(input_size, output_size,
            LEARNING_RATE, GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TARGET_UPDATE)
uvfa_regret = [] # record of all regrets across trials and blocks
uvfa_leaves = [] # record of all leaf nodes

# init SFGPI
input_size = env.n_states # SFGPI only gets the state (not the task!)
output_size = env.n_actions * env.n_features # and predicts SF for all actions
sfgpi = SFGPI(input_size, output_size, env.n_actions, env.n_features,
              LEARNING_RATE, GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TARGET_UPDATE)
sfgpi_regret = [] # record of all regrets across trials and blocks
sfgpi_leaves = [] # record of all leaf nodes



def run_uvfa():
    print("UVFA")
    for block in range(n_blocks):
        print("Tasks of this block: {}".format(tasks_of_block[block]))

        # TODO
        env.swap_feature(5, feature5_block[block])
        print("Feature 5 of this block: {}".format(feature5_block[block]))

        # TODO optimal trajectory of block taking feature change into account
        #optimal_reward[block], optimal_leaves[block] = zip([env.optimal_trajectory(w) for w in tasks_of_block[block]])

        for trial in range(block_size):
            w = tasks_of_block[block][trial % n_tasks_per_block] # alternate tasks

            # simulate trajectory
            s = 0 # start at root
            trial_reward = 0
            while not env.terminal[s]:
                s_1hot = np.zeros(env.n_states); s_1hot[s] = 1
                sw = torch.from_numpy(np.hstack([s_1hot, w])).float()
                Q = uvfa.predict(sw)
                Q = Q.detach().numpy().astype(np.float64)
                step = trial # TODO could be within or across blocks
                pi = policy(Q, step)

                # sample next action / state
                a = np.nonzero(np.random.multinomial(1, pi))[0]
                a = torch.from_numpy(a)
                s_next = np.squeeze(env.P[s,a,:])
                s_next = np.nonzero(np.random.multinomial(1, s_next))[0][0]
                s_next_1hot = np.zeros(env.n_states); s_next_1hot[s_next] = 1
                sw_next = torch.from_numpy(np.hstack([s_next_1hot, w])).float()

                # compute reward of next state
                # (NB: here, rewards are associated with states, not with actions)
                r = env.phi[s_next] @ w
                trial_reward += r

                # transit and store transition in replay buffer
                s = s_next
                uvfa.replay_buffer.push(sw, a, sw_next, torch.tensor(r))

            # update the neural network after each trial
            uvfa.train(trial)
            # fill record of trial
            regret = optimal_reward[block + trial % n_tasks_per_block] - \
                     trial_reward
            uvfa_regret.append(regret)
            uvfa_leaves.append(s)
            #print("trial {}: reward {}".format(trial, trial_reward))


def run_sfgpi():
    print("SFGPI")
    for block in range(n_blocks):
        print("Tasks of this block: {}".format(tasks_of_block[block]))

        # TODO
        env.swap_feature(5, feature5_block[block])
        print("Feature 5 of this block: {}".format(feature5_block[block]))

        for trial in range(block_size):
            w = tasks_of_block[block][trial % n_tasks_per_block] # alternate tasks

            # simulate trajectory
            s = 0 # start at root
            trial_reward = 0
            while not env.terminal[s]:
                s_1hot = np.zeros(env.n_states); s_1hot[s] = 1
                s_1hot = torch.from_numpy(s_1hot).float()
                psi = sfgpi.predict(s_1hot)
                Q   = psi @ w
                # TODO have to take max over Q wrt to the policies (1 per task)
                step = trial # TODO could be within or across blocks
                pi = policy(Q, step)

                # sample next action / state
                a = np.nonzero(np.random.multinomial(1, pi))[0]
                a = torch.from_numpy(a)
                s_next = np.squeeze(env.P[s,a,:])
                s_next = np.nonzero(np.random.multinomial(1, s_next))[0][0]
                s_next_1hot = np.zeros(env.n_states); s_next_1hot[s_next] = 1
                s_next_1hot = torch.from_numpy(s_next_1hot).float()

                # importantly, we don't store reward in transitions
                # but only the next state's features
                phi_next = torch.tensor(env.phi[s_next])

                # anyways, compute reward of next state
                # (NB: here, rewards are received in states, not with actions)
                r = env.phi[s_next] @ w
                trial_reward += r

                # transit and store transition in replay buffer
                s = s_next
                sfgpi.replay_buffer.push(s_1hot, a, s_next_1hot, phi_next)

            # update the neural network after each trial
            sfgpi.train(trial)
            # fill record of trial
            regret = optimal_reward[block + trial % n_tasks_per_block] - \
                     trial_reward
            sfgpi_regret.append(regret)
            sfgpi_leaves.append(s)
            # if s == 9: # test if psi ~ phi for an example state
            #     print("trial {}: psi9 {}".format(trial, psi[a,:]))

def epsilon_greedy(Q, step):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * step / EPS_DECAY)
    #print(eps)
    pi = np.ones(env.n_actions) * eps / env.n_actions
    pi[np.argmax(Q)] += 1 - eps
    return pi

def softmax(Q, step):
    tau = TAU_END + (TAU_START - TAU_END) * np.exp(-1. * step / TAU_DECAY)
    #print(tau)
    pi = np.exp(Q / tau)
    pi = pi / np.sum(pi)
    return pi

def plot_regret(single_tasks=False):
    plt.figure()

    for y in [uvfa_regret, sfgpi_regret]:
        if single_tasks:
            for p in range(n_tasks_per_block):
                x = range(n_blocks*block_size//n_tasks_per_block)
                y = y[p::n_tasks_per_block]
                y_smoothed = np.convolve(y, np.ones(10), 'same') / 10
                plt.plot(x, y_smoothed)
        else:
            x = range(n_blocks*block_size)
            y_smoothed = np.convolve(y, np.ones(10), 'same') / 10
            plt.plot(x, y_smoothed)

    # plot vertical bars between blocks
    plt.vlines(range(0, block_size*n_blocks, block_size), 0, max(y))

    # label which tasks were learned
    for block in range(n_blocks):
        s = "w1 = {}\nw2 = {}".format(*tasks_of_block[block])
        plt.annotate(s, [block * block_size + 20, max(y) - 1])

    plt.xlabel("trial")
    plt.ylabel("regret")
    plt.legend(["UVFA", "SFGPI"], loc="center right")
    plt.show()


def plot_leaves():
    plt.figure()

    for p, (leaves, regret) in enumerate([[uvfa_leaves, uvfa_regret],
                                          [sfgpi_leaves, sfgpi_regret]]):
        plt.subplot(3, 1, p+1)
        y = np.zeros((len(leaves), env.n_states))
        y[range(len(leaves)), leaves] = 1 # 1hot encoding

        # distinguish correct (=1) and incorrect choices (=-1)
        y = y - 2 * y * np.hstack([np.array(regret)[np.newaxis].T != 0
                                   for _ in range(env.n_states)])

        cmap = plt.get_cmap("PiYG", 3) # 3 values: -1 (pink), 0, 1 (green)
        plt.imshow(y.T, cmap=cmap, aspect="auto", interpolation="none")
        ylim = [3.5,12.5] # hardcoded TODO use env.terminal
        plt.ylim(ylim)
        plt.yticks(ticks=range(4,13), labels=range(5,14)) # hardcoded
        plt.title("UVFA" if p==0 else "SFGPI")

        # plot vertical bars between blocks
        plt.vlines(range(0, block_size*n_blocks, block_size), *ylim)

    plt.xlabel("trial")
    plt.ylabel("final node")

    # extra subplot for labels
    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.xlim([0, n_blocks])
    plt.ylim([0, 1.2])
    # label which tasks were learned
    for block in range(n_blocks):
        s = "w1 = {}\nw2 = {}".format(*tasks_of_block[block])
        s = s + "\nphi6 = {}".format(feature5_block[block])
        s = s + "\n\noptimal = {}".format( # +1 because of 0-indexing
            np.array(optimal_leaves[block:block+n_tasks_per_block]) + 1)
        plt.annotate(s, [block + .2, 1])

    cbar = plt.colorbar(orientation="horizontal", ticks=[-.66,0,.66])
    cbar.ax.set_xticklabels(["wrong", "", "correct"])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # policy = softmax
    policy = epsilon_greedy
    run_uvfa()
    run_sfgpi()
    plot_leaves()
    plot_regret()
