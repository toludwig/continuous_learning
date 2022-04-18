import numpy as np

# define simple two-step decision process like in Tomov & Schulz 2021
class TwoStepEnv():
    def __init__(self):
        self.n_states = 13
        self.n_actions = 3
        self.n_features = 3
        self.S = range(self.n_states)
        self.A = range(self.n_actions)
        self.P = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.phi = np.zeros([self.n_states, self.n_features])
        self.terminal = np.zeros([self.n_states])
        self.terminal[4:13] = 1

        # fill transition probabilities
        self.P[0,0,1] = 1
        self.P[0,1,2] = 1
        self.P[0,2,3] = 1
        self.P[1,0,4] = 1
        self.P[1,1,5] = 1
        self.P[1,2,6] = 1
        self.P[2,0,7] = 1
        self.P[2,1,8] = 1
        self.P[2,2,9] = 1
        self.P[3,0,10] = 1
        self.P[3,1,11] = 1
        self.P[3,2,12] = 1

        # fill features
        self.phi[4,:] = [0,1,0]
        self.phi[5,:] = [10,0,0]
        self.phi[6,:] = [4,4,13]
        self.phi[7,:] = [9,0,0]
        self.phi[8,:] = [10,10,0]
        self.phi[9,:] = [0,9,0]
        self.phi[10,:] = [0,0,1]
        self.phi[11,:] = [0,10,6]
        self.phi[12,:] = [1,0,0]


    def optimal_trajectory(self, task):
        """ Computes the optimal trajectory and its reward given a task. """
        reward = self.phi @ task
        # TODO would have to check each path, but we take a shortcut
        # because we know there is only reward at the leaves
        optimal_leaf   = np.argmax(reward)
        optimal_reward = reward[optimal_leaf]
        return (optimal_reward, optimal_leaf)

    def swap_feature(self, index, new_phi):
        self.phi[index,:] = new_phi
