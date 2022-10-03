import numpy as np
import random

from env import Env

"""
A zoo is a 3-step environment with 10 states (0 is the starting/root node) and 8 paths.
Each state is marked with a unique symbol (fruit/vegetable) and features (amount of vitamins).
Reward is calculated as the dot product of these features times the task (demands of animals).
"""

class Zoo(Env):

    n_paths = 8
    n_states = 10
    n_actions = 2
    n_features = 2
    min_feature = 1
    max_feature = 3

    fruit = { # (sweet, sour)
        (1,1): "apple",
        (1,2): "blueberry",
        (1,3): "lemon",
        (2,1): "cherry",
        (2,2): "peach",
        (2,3): "grapes",
        (3,1): "melon",
        (3,2): "strawberry",
        (3,3): "pineapple"
    }

    animals = {
        (-1,-1): "mouse",
        (-1, 0): "turtle",
        (-1, 1): "penguin",
        (0 ,-1): "elephant",
        (0 , 0): "sloth",
        (0 , 1): "chicken",
        (1 ,-1): "tiger",
        (1 , 0): "crocodile",
        (1 , 1): "giraffe"
    }

    # Transitions are deterministic: T[s,a] = s'
    T = np.zeros([n_states, n_actions], dtype=int)
    T[0, 0] = 1
    T[0, 1] = 2
    T[1, 0] = 3
    T[1, 1] = 4
    T[2, 0] = 4
    T[2, 1] = 5
    T[3, 0] = 6
    T[3, 1] = 7
    T[4, 0] = 7
    T[4, 1] = 8
    T[5, 0] = 8
    T[5, 1] = 9

    # paths are most of the time just indexed (0-7)
    path_idx2items = [[1,3,6],[1,3,7],[1,4,7],[1,4,8],
                      [2,4,7],[2,4,8],[2,5,8],[2,5,9]]
    path_items2idx = lambda x: Zoo.path_idx2items.index(x)


    def __init__(self):
        self.state = 0
        self.phi = np.zeros([Zoo.n_states, Zoo.n_features], dtype=int)
        self.assign_features_randomly()


    def reset(self):
        """
        Resets state to starting state.
        """
        self.state = 0
        observation = (self.state, self.phi[self.state,:])
        return observation


    def step(self, action, task):
        """
        Observation includes the state index itself,
        because in the human task there is a unique symbol for each.
        """
        self.state = int(Zoo.T[self.state, action])
        observation = (self.state, self.phi[self.state])
        reward = self.phi[self.state] @ task
        done = self.state > 5
        return (observation, reward, done)


    ############################################################################
    # MB and MF solutions                                                      #
    ############################################################################

    def path_sum(self, task):
        """
        Calculates all the path sums.
        """
        f = self.phi
        pathsum = np.zeros(8) # there are 8 paths
        pathsum[0] = (f[1] + f[3] + f[6]) @ task
        pathsum[1] = (f[1] + f[3] + f[7]) @ task
        pathsum[2] = (f[1] + f[4] + f[7]) @ task
        pathsum[3] = (f[1] + f[4] + f[8]) @ task
        pathsum[4] = (f[2] + f[4] + f[7]) @ task
        pathsum[5] = (f[2] + f[4] + f[8]) @ task
        pathsum[6] = (f[2] + f[5] + f[8]) @ task
        pathsum[7] = (f[2] + f[5] + f[9]) @ task
        return pathsum


    def path_similarity(self, p1, p2):
        p1 = self.path_idx2items[p1]
        p2 = self.path_idx2items[p2]
        # how many elements of p1 are also in p2
        sim = 0
        for p in p1:
            sim += p in p2
        return sim


    def max_path_similarity(self, p, ps):
        """
        Compute the similarity of p to the closest path in ps.
        """
        maxsim = max([self.path_similarity(p, ps[k])
                      for k in range(len(ps))])
        return maxsim


    def path_composed(self, p, p1, p2):
        """
        True if p is a composition of p1 and p2,
        i.e. if it has elements from both p1 and p2
        """
        p  = set(self.path_idx2items[p])
        p1 = set(self.path_idx2items[p1])
        p2 = set(self.path_idx2items[p2])

        return len(p & p1) > 0 and len(p & p2) > 0


    def any_path_composed(self, p, ps1, ps2):
        """Check if p is a composition of any two p1 in ps1 and p2 in ps2."""
        return any(self.path_composed(p, p1, p2) for p1 in ps1 for p2 in ps2)


    def mb_paths(self, task):
        """
        Returns the MB (model based) paths as a binary vector
        of length 8, the k-th bit indicating if path k is optimal.
        """
        pathsum = self.path_sum(task)
        mb_reward = np.max(pathsum)
        mb_paths  = np.array(mb_reward == pathsum)
        return (mb_reward, list(map(int, mb_paths)))


    def mf_paths(self, previous_tasks):
        """
        Returns the MF (model free) paths which are the MB path
        with the highest reward, wrt. to the tasks of the last block.
        """
        max_mb_reward = -100 # dummy init
        max_mb_paths  = []
        for task in previous_tasks:
            mb_reward, mb_paths = self.mb_paths(task)
            if mb_reward > max_mb_reward:
                max_mb_reward = mb_reward
                max_mb_paths  = mb_paths
        return (max_mb_reward, list(map(int, max_mb_paths)))



    #############################################################################
    # CHANGE FEATURES                                                           #
    #############################################################################

    def assign_features_randomly(self, unique=True):
        """
        Samples random features from [min_feature, max_feature]^n_features.
        If unique == True, we sample without replacement.
        """
        rang = range(self.min_feature, self.max_feature+1)
        vecs = [[a, b] for a in rang for b in rang]
        if unique:
            self.phi[1:,:] = np.array(random.sample(vecs, k=Zoo.n_states-1))
        else:
            self.phi[1:,:] = np.array(random.choices(vecs, k=Zoo.n_states-1))


    def mutate_features_randomly(self):
        """
        Swap the features of two states (picked at random) in the environment.
        """
        # pick a random state to change the features of
        s1, s2 = random.sample(range(1, Zoo.n_states), k=2)
        self.phi[[s1,s2],:] = self.phi[[s2,s1],:]


if __name__ == '__main__':
    zoo = Zoo()

    #zoo.fill_features(tasks)
