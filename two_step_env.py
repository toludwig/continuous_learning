import numpy as np
import random

from env import Env

# define simple two-step decision process like in Tomov & Schulz 2021
class TwoStepEnv(Env):

    # constants (things that stay fixed in all instances)
    n_states = 13
    n_actions = 3
    n_features = 3
    n_leaves = 9

    def __init__(self):
        self.S = range(self.n_states)
        self.A = range(self.n_actions)
        self.P = np.zeros([self.n_states, self.n_actions, self.n_states])
        self.phi = np.zeros([self.n_states, self.n_features])
        self.terminal = np.zeros([self.n_states], dtype=np.int16)
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


    def optimal_trajectory(self, task, nsims=100):
        """
        Computes the optimal trajectory and its reward given a task.
        Simulates nsims paths with a random policy.
        Returns optimal reward and a vector of length n_states
        indicating which leaf nodes are optimal.
        """
        # TODO change this to value iteration
        task = np.array(task)
        reward = np.zeros(nsims)
        leaves = np.zeros(nsims, dtype=np.int16)
        for sim in range(nsims):
            s = 0 # start at root
            r = self.phi[s,:] @ task
            while not self.terminal[s] == 1:
                a = np.random.randint(self.n_actions) # random policy
                s = self.take_action(s, a) # transition
                r += self.phi[s,:] @ task
            reward[sim] = r
            leaves[sim] = s

        optimal_reward = np.max(reward)
        optimal_leaves = np.zeros(TwoStepEnv.n_states, dtype=bool)
        for s in range(TwoStepEnv.n_states):
            if s in leaves:
                optimal_leaves[s] = np.max(reward[leaves == s]) == optimal_reward
        return (optimal_reward, optimal_leaves)


    def take_action(self, s, a):
        """ Samples next state given state and action. """
        P = np.squeeze(self.P[s,a,:])
        return np.nonzero(np.random.multinomial(1, P))[0][0]


    def redefine_feature(self, index, new_phi):
        """ Reassign the value of a specific feature. """
        self.phi[index,:] = new_phi


    def decrease_value_of_optimal_feature(self, task, optimal_leaf):
        """
        TODO deprecated: this causes a bias in regret!
        Lower the feature of an as yet optimal leaf (for a given task),
        just in so much as another one will be optimal.
        """
        phi = self.phi[optimal_leaf,:]
        # update phi by subtracting task,
        # so features with positive weights get smaller and vice versa
        # keep on doing this, till another one is optimal
        # NB: because we only change leaf nodes, we only need to check for their optimality
        Q      = phi @ task
        Q_leaf = [Q - 1] # dummy
        while not any(Q_leaf > Q): # as long as no other leaf is optimal
            phi = phi - task
            self.swap_feature(optimal_leaf, phi)
            Q_leaf = self.phi[self.terminal==1,:] @ task
            Q = phi @ task
        return phi #, optimal_leaf) # return new phi + index


    def swap_optimal_feature(self, optimal_leaves):
        """
        Swap features of all optimal leaves with those from suboptimal ones.
        This way, there will be an enforced feature prediction error,
        when using an old (converged) policy.
        """
        leaves = range(4,13) # TODO don't hardcode

        optimal = np.nonzero(optimal_leaves)[0].tolist() # 01 to list
        suboptimal = set(leaves) - set(optimal)

        for leaf in optimal:
            if suboptimal == set(): # if empty (this should not happen!)
                suboptimal = set(leaves) # but if if does, pick from all
            index = random.choices(list(suboptimal))
            tmp = self.phi[leaf,:].copy()
            self.phi[leaf,:] = self.phi[index,:]
            self.phi[index,:] = tmp
            # remove index from suboptimal, so we don't swap the same twice
            suboptimal = suboptimal - set(index)


    def swap_transitions(self, optimal_leaves):
        """
        Swap two transitions on the first stage.
        Always change such that the optimal path is affected.
        """
        optimal_leaf = np.nonzero(optimal_leaves)[0][0] # pick the 1st optimal
        optimal_s = (optimal_leaf-1) // self.n_actions # parent of optimal leaf
        other_s = list(set([1,2,3]) - set([optimal_s]))
        other_s = np.random.choice(other_s)

        # swap transitions from root node (=0) to first layer
        tmp = self.P[0, :, optimal_s].copy()
        self.P[0, :, optimal_s] = self.P[0, :, other_s]
        self.P[0, :, other_s] = tmp
