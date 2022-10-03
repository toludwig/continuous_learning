# GUI for human experiment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard
from time import sleep, time

from supermarket import Supermarket
from mb_shopping import MB_shopping

# constants
symbol = ['A', 'B‍', 'C', 'D',
          'E', 'F‍', 'G', 'H',
          'I', 'J‍', 'K', 'L',
          'M', 'N‍', 'O', 'P']

B = 2 # number of blocks
T = 15 # number of trials per block
M = 3 # number of concurrent tasks
G = 3 # number of goals per task
recipes = [(2,3,9), (3,12,15), (6,9,12)] # XXX triples are recipes
tags = "_fixedstart"


def _item2integer(item):
    return np.nonzero(item)[0][0]

def _task2binary(task):
    # recipe is an integer, task is a 3-hot vector
    binary = np.zeros(16) # XXX
    for t in task:
        binary[t] = 1
    return binary


class HumanShopping():

    def __init__(self):
        self.block = 0 # block index
        self.trial = 0 # trial index
        self.task = recipes[0] # current task = recipe

        self.env  = Supermarket()

        # state of trial
        self.start = 0 # TODO random per trial?
        task_bin = _task2binary(self.task)
        item_bin = self.env.reset(task_bin)
        self.item = _item2integer(item_bin)
        self.found = np.zeros(16, dtype=bool) # memory of each found item
        #self.path_length = 0 # counter for steps

        # records across blocks
        self.reward = np.zeros(B*T)
        self.regret = np.zeros(B*T)
        self.sum_RT = np.zeros(B*T)
        self.df = pd.DataFrame(data=np.zeros([B*T,5]),
                               columns=["block", "trial", "task",
                                        "reward", "regret", "sum_RT"])

        # init listener
        self.keylistener = keyboard.Listener(on_press=self.walk)
        self.keylistener.start()
        # init gui
        self.init_figure()
        self.time_start_trial = time() # TODO move this somewhere else?
        self.update_labels(self.item in self.task)
        plt.show() # just once

    def walk(self, k):
        # navigation using arrow keys
        # TODO actionkeys = {"h":3,"j":2,"k":0,"l":1}
        if k == keyboard.Key.up:
            a = 0
        elif k == keyboard.Key.right:
            a = 1
        elif k == keyboard.Key.down:
            a = 2
        elif k == keyboard.Key.left:
            a = 3
        else:
            return -1

        if self.env.cell[0] > 0:
            plt.scatter(0, .5, s=30, c='k', marker='^')
        if self.env.cell[1] < self.env.grid_size[1]-1:
            plt.scatter(0.5, 0, s=30, c='k', marker='>')
        if self.env.cell[0] < self.env.grid_size[0]-1:
            plt.scatter(0, -.5, s=30, c='k', marker='v')
        if self.env.cell[1] > 0:
            plt.scatter(-.5, 0, s=30, c='k', marker='<')

        task_bin = _task2binary(self.task)
        item_bin, reward, done = self.env.step(a, task_bin)
        self.item = _item2integer(item_bin)
        self.found[self.item] = True
        tt = self.block*T + self.trial # trial-index across blocks
        self.reward[tt] += reward
        goal_found = self.item in self.task
        self.update_labels(goal_found)
        if done: # if trial finished, procede in experiment
            self.procede()

    def update_labels(self, goal_found):
        self.fig.clear()
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        axis = plt.gca()
        # task label
        for i, t in enumerate(self.task):
            axis.text(-1+i/10, 1, symbol[t],
                      color=('red' if self.found[t] else 'black'))
        # item label
        axis.text(0, 0, symbol[self.item], ha='center', va='center',
                  #fontname='Arial',
                  color=('red' if goal_found else 'black'),
                  fontsize=50)

        # trial label
        axis.text(0, 1, str(self.trial+1) + "/15", ha='center')

        # score label
        axis.text(1, 1, str(self.reward[self.trial]), ha='right')

        # draw arrows
        if self.env.cell[0] > 0:
            plt.scatter(0, .5, s=30, c='k', marker='^')
        if self.env.cell[1] < self.env.grid_size[1]-1:
            plt.scatter(0.5, 0, s=30, c='k', marker='>')
        if self.env.cell[0] < self.env.grid_size[0]-1:
            plt.scatter(0, -.5, s=30, c='k', marker='v')
        if self.env.cell[1] > 0:
            plt.scatter(-.5, 0, s=30, c='k', marker='<')

        plt.draw()

    def init_figure(self):
        self.fig = plt.figure()
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.rc('font', family='sans-serif')

    def fixation_cross(self):
        self.fig.clear()
        plt.scatter(0, 0, s=30, c='k', marker='+')
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.draw()
        # wait for 1s
        sleep(2)

    def procede(self):
        # after a block, scramble items
        if self.trial == T-1: # if block finished
            if self.block < B: # but more blocks are left
                self.trial = 0 # reset trial
                self.block += 1
                self.env.scramble() # reorder environment
            else:
                print("Done with experiment.")
                self.fig.close()
                return

        # store task records
        tt = self.block*T + self.trial # trial-index across blocks
        task_bin = _task2binary(self.task)
        optimal_dist, _ = MB_shopping(self.env, task_bin)
        self.reward[tt] -= 6 # -6 because at goals we get +1 (two more than -1)
        self.regret[tt] = -self.reward[tt] - optimal_dist
        self.sum_RT[tt] = time() - self.time_start_trial
        self.time_start_trial = time() + 3 # because of 3s delay between trials
        self.df.loc[tt,:] = [self.block, self.trial, self.trial % M,
                             self.reward[tt], self.regret[tt], self.sum_RT[tt]]
        if tt % 1 == 0: # TODO every 10th trial
            self.dump_records()


        # init new trial
        self.trial += 1
        self.task = recipes[self.trial % M] # alternate tasks
        task_bin = _task2binary(self.task)
        item_bin = self.env.reset(task_bin)
        self.item = _item2integer(item_bin)
        self.found = np.zeros(16, dtype=bool)

        sleep(1)
        self.fixation_cross()
        self.update_labels(self.item in self.task)


    def dump_records(self):
        """
        Dump the records.
        """
        filename = f"./supermarket/tobi_B{B}_T{T}_M{M}_G{G}" + tags + ".csv"
        self.df.to_csv(filename, index=False)

def show_instructions():
    """
    You want to buy ...
    """


if __name__ == "__main__":
    #show_instructions()
    HumanShopping()
