import numpy as np
from matplotlib import pyplot as plt
from pynput import keyboard
import random
import pickle

from zoo import Zoo
from zoo_feeding import init_curriculum, run_training_test
from zoo_optimizer import hill_climbing, mb_sim_1st, mf_sim_1st, diverse_mb_paths_constraint

"""
Interactive zoo feeding game.
There are blocks with M training tasks, interleaved, followed by a test task.
After each block, the environment changes, so the subject has to relearn.
Each environment is optimized for distinct choices of our models in the test.
The subject only sees the current state (its symbol and features) and reward,
and can press left or right to get to the next state.
"""


B = 10
T = 20
M = 2

class ZooExperiment():

    def __init__(self):
        # init listener
        self.keylistener = keyboard.Listener(on_press=self.walk)
        self.keylistener.start()
        # init gui
        self.init_figure()

        self.block = -1
        self.trial = -1

        self.env = None
        self.cur = None
        self.init_block()
        self.init_trial()
        plt.show()


    def init_block(self):
        """
        Picks a curriculum of random kind and optimizes the environment.
        TODO counter-balance kinds
        """
        self.block += 1
        print(self.block)
        with open(f"./zoo/block_{self.block}.pickle", "rb") as f:
            self.env, self.cur, self.df = pickle.load(f)

    def draw_task(self, task):
        """
        Draws task / animal with its preferences.
        """
        self.fig.clear()
        img = plt.imread(f"./img/{Zoo.animals[tuple(task)]}.png")
        ib = OffsetImage(img, zoom=0.25)
        ab = AnnotationBbox(ib, [x[0], y[0]], frameon = False)
        plt.add_artist(ab) # task symbol
        plt.annotate(str(task), [0,1]) # features
        plt.draw()


    def draw_state(self, env, task):
        """
        Draw the state's symbol, its features and its reward given the task.
        """
        phi = env.phi[env.state]

        self.fig.clear()
        plt.annotate(Zoo.fruit[tuple(phi)], [2,2])
        plt.barh([0,1], phi)
        plt.yticks([0,1], ["sweet", "sour"])
        plt.axis("off")
        plt.annotate("Reward: " + str(phi @ task))
        plt.draw()


    def init_figure(self):
        self.fig = plt.figure()
        plt.axis('off')
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.rc('font', family='sans-serif')


    def init_trial(self):
        self.trial += 1
        self.task = self.cur[self.block][self.trial % M]
        self.draw_task(self.task)


    def walk(self, k):
        # navigation using arrow keys
        # TODO actionkeys = {"h":3,"j":2,"k":0,"l":1}
        if k == keyboard.Key.left:
            a = 0
        elif k == keyboard.Key.right:
            a = 1
        else:
            return -1

        _, reward, done = self.env.step(a, self.task)

        if done:
            init_trial()
        else:
            draw_state()


if __name__ == "__main__":
    exp = ZooExperiment()
