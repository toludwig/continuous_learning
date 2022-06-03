import numpy as np
import random
from collections import namedtuple, deque

# online learning of SFGPI
class SFGPI():
    def __init__(self, n_states, n_actions, n_features,
                 LEARNING_RATE, GAMMA, BUFFER_CAPACITY, BATCH_SIZE):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.n_features = n_features

        self.LEARNING_RATE = LEARNING_RATE
        self.GAMMA = GAMMA
        self.BUFFER_CAPACITY = BUFFER_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.tasks = [] # list of all encountered task vectors

        # for each task we learn separate SFs and seperate policies
        # each sf object is of size n_states x n_actions x n_features
        self.sf_by_task = []
        # each policy is of size n_states x n_actions
        self.pi_by_task = []
        # also store experiences by task
        self.replay_buffer_by_task = [] # one buffer for each task

    def predict_Q(self, state, task):
        # if task not known yet...
        known_task = task in self.tasks
        # known_task = self.tasks != [] and np.any(np.all(task == self.tasks, axis=1)) # XXX for numpy arrays

        if not known_task:
            self._add_new_task(task)
            # TODO do GPI only for new tasks or for all?
            # NB: have to do it also for every new state in that task

        # do GPI, i.e. Q_max = max_i Q^{\pi_i}(s,a)
        Q_task = np.empty([self.n_states, self.n_actions, len(self.tasks)])
        for (tid, sf) in enumerate(self.sf_by_task):
            Q_task[:,:,tid] = np.einsum("ijk,k", sf, task)

        tid = self.tasks.index(task)
        Q_max = np.max(Q_task[state,:,:], axis=1) # take max policy over tasks
        # XXX self.pi_by_task[tid][state,:] = self.policy(Q_max, step=step)
        return Q_max # self.pi_by_task[tid][state,:]

    def train_online(self, task, step):
        tid = self.tasks.index(task)

        # learn only from last transition
        (s, a, s_next, phi) = self.replay_buffer_by_task[tid].last()
        sf = self.sf_by_task[tid]

        # TEST print successor features to see convergence
        # if s == 1:
        #     print("psi2, action{}: {}".format(a, sf[s,a,:]))
        # if s == 2:
        #     print("psi3, action{}: {}".format(a, sf[s,a,:]))

        psi = sf[s,a,:].squeeze()
        a_next = np.argmax(np.squeeze(self.sf_by_task[tid][s_next,:,:]) @ task)
                           # XXX self.pi_by_task[tid][s_next,:])
        psi_next = sf[s_next, a_next, :].squeeze()
        delta = phi + self.GAMMA * psi_next - psi

        sf[s,a,:] += self.LEARNING_RATE * delta

        # update features
        self.sf_by_task[tid] = sf

        # TODO HER: update *all* policies in s
        #for (t,w) in enumerate(self.tasks):
        #    Q = sf[s,:,:].squeeze() @ w
        #    self.pi_by_task[t][s,:] = self.policy(Q, step=step)


    def train_offline(self, task, step):
        tid = self.tasks.index(task)

        # don't train offline with too few samples, lest we bias
        if len(self.replay_buffer_by_task[tid]) < self.BATCH_SIZE:
            return

        # TODO train all tasks?
        for (t,w) in enumerate(self.tasks):
            sf = self.sf_by_task[t]

            # sample some transitions to replay (only those of the same task)
            batch = self.replay_buffer_by_task[t].sample(self.BATCH_SIZE)
            for (s, a, s_next, phi) in batch:
                psi = sf[s,a,:].squeeze()
                a_next = np.argmax(self.pi_by_task[t][s_next,:])
                psi_next = sf[s_next, a_next, :].squeeze()
                delta = phi + self.GAMMA * psi_next - psi
                sf[s,a,:] += self.LEARNING_RATE * .05 * delta

            # update features
            self.sf_by_task[t] = sf
            # update policy
            for s in range(self.n_states):
                Q = sf[s,:,:].squeeze() @ w
                self.pi_by_task[t][s,:] = self.policy(Q, step=step)

    def _add_new_task(self, task):
        self.tasks.append(task)
        sf = np.zeros([self.n_states, self.n_actions, self.n_features]) # has to be 0 so that the psi_next = 0 for the leaf-nodes
        sf[0:3, :, :] = 15 # optimistic init
        self.sf_by_task.append(sf)
        self.pi_by_task.append(np.ones([self.n_states, self.n_actions]) / self.n_actions) # uniform policy
        self.replay_buffer_by_task.append(ReplayMemory(self.BUFFER_CAPACITY))

    def store_transition(self, task, s, a, s_next, phi_next):
        tid = self.tasks.index(task)
        self.replay_buffer_by_task[tid].push(s, a, s_next, phi_next)


#############################################################
# REPLAY
#############################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'features'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample with replacement (works even if batch_size > len(buffer))"""
        return random.choices(self.memory, k=batch_size)

    def last(self):
        """Returns last entry"""
        return self.memory[-1]

    def __len__(self):
        return len(self.memory)
