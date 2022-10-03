import numpy as np
import random

from env import Env

class Supermarket(Env):
    """
    A supermarket is a grid of items which can be collected.
    Each state is a tuple of a cell and a list of as-yet collected items.
    There are as many unique items (products) as there are cells,
    and each state has a one-hot feature vector coding the item at the cell.
    """

    def __init__(self, n_goals, n_grid, seed=5):
        self.REWARD_SCALE = 10 # amount of reward on any goal
        self.n_grid = n_grid
        # state vars
        self.item = np.zeros([n_grid, n_grid, n_grid**2]) # 1-hot encoded item for each cell in the grid
        self.cell = [0,0] # row/col coordinates, [0,0] is top left corner
        self.coll = np.zeros([n_grid, n_grid]) # binary grid of as yet collected items (not only goals)
        self.goal_state = np.zeros(n_goals) # 1 bit per goal, 1 if collected
        self.start = [0,0] # the cell where the trial starts (may change in reset function)
        # init
        random.seed(seed) # set seed for scrambling
        self.scramble()


    def scramble(self):
        """
        Randomly allocates items at each cell with one-hot features.
        Stores the permutation of items in self.cell2item and self.item2cell.
        """
        self.cell2item = np.arange(self.n_grid**2)
        self.item2cell = np.arange(self.n_grid**2)

        random.shuffle(self.cell2item) # permute
        # invert mapping
        for i in range(self.n_grid**2):
            self.item2cell[self.cell2item[i]] = i

        # TODO permute further until the start item is not a goal
        # while task[self.cell2item[start]] == 1:
        #     random.shuffle(self.cell2item) # inplace

        # assign items row-wise according to cell2item
        self.item = np.zeros((self.n_grid, self.n_grid, self.n_grid**2))
        p = 0
        for i in range(self.n_grid):
            for j in range(self.n_grid):
                self.item[i,j,self.cell2item[p]] = 1
                p += 1


    def reset(self, task, start_random=False):
        """
        Empties all collected items.
        Sets state back to a starting cell (random if start_random else [0,0]).
        Only constraint is that on the start node there should not be a goal item.
        Returns the observation at the starting state.
        """
        self.coll = np.zeros([self.n_grid, self.n_grid]) # binary grid of collected items
        self.item_sum = np.zeros(self.n_grid**2) # count of how often an item was visited

        i,j = [0,0]
        if start_random:
            # new random start state position that is not on a goal state
            ijgoal = True
            while ijgoal:
                i = random.choice(range(self.n_grid))
                j = random.choice(range(self.n_grid))

                for k,t in enumerate(task):
                    ijgoal = t==1 and self.item[i,j,k]==1
                    if ijgoal:
                        break

        # reset cell to start
        self.start = [i,j]
        self.cell = [i,j]

        return self.item[i,j,:]



    def step(self, action, task):
        """
        Takes the action, advancing the cell and observing/collecting an item.
        There are 4 actions (0-north, 1-east, 2-south, 3-west).
        The task is a 3-hot vector specifying goal items.
        There is a -1 reward for each step, and +1 on collecting a goal item.

        Returns a tuple of observation, reward, and whether the episode is done.
        The observation is a concatenated vector of the 1-hot encoded observed item
        and the goal state, i.e. 1 bit per goal indicating if it was collected.
        """
        reward = 0 # there is a penalty of -1 for an eligible action

        if action == 0: # north
            if self.cell[0] > 0:
                self.cell[0] -= 1
                reward -= 1
        elif action == 1: # east
            if self.cell[1] < self.n_grid-1:
                self.cell[1] += 1
                reward -= 1
        elif action == 2: # south
            if self.cell[0] < self.n_grid-1:
                self.cell[0] += 1
                reward -= 1
        elif action == 3: # west
            if self.cell[1] > 0:
                self.cell[1] -= 1
                reward -= 1
        else:
            raise Exception("not a valid action")

        # current item
        item = self.item[tuple(self.cell)]
        self.item_sum += item # keep track of sum of items

        # if not visited before:
        if self.coll[tuple(self.cell)] == 0:
            self.coll[tuple(self.cell)] = 1 # collect item at cell
            # reward is given for each collected goal item
            isgoal = task @ item # 1 if item is a goal, else 0
            reward += self.REWARD_SCALE * isgoal

        # update goal state
        goal_item = np.nonzero(task)[0]
        goal_cell = self.item2cell[goal_item]
        self.goal_state = np.reshape(self.coll, self.n_grid**2)[goal_cell]

        # observe a concatentation of item and goal state
        # TODO for the current observation, still use the old goal state
        # TODO because on finding the goal, this should still be counted
        observation = np.hstack([item, self.goal_state])

        # check if all items demanded by the task are collected (at least once)
        done = np.all(task <= self.item_sum)

        return (observation, reward, done)


