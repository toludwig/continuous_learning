import numpy as np
import random
from collections import namedtuple, deque

# online learning of SFGPI
class SFGPI():
    def __init__(self, n_states, n_actions, n_features,
                 ALPHA, GAMMA, BUFFER_CAPACITY, BATCH_SIZE, TASK_CAPACITY=100):
        self.n_states   = n_states
        self.n_actions  = n_actions
        self.n_features = n_features

        self.ALPHA = ALPHA
        self.GAMMA = GAMMA
        self.BUFFER_CAPACITY = BUFFER_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE
        self.TASK_CAPACITY = TASK_CAPACITY # default value very high
        # we entroduce a hard constraint on how many tasks can be learned.
        # If that capacity in task is exceeded, the oldest tasks / SFs
        # will be forgotten, i.e. deleted

        self.tasks = [] # list of all encountered task vectors

        # for each task we learn separate SFs and seperate policies
        # each sf object is of size n_states x n_actions x n_features
        self.sf_by_task = []
        # also store experiences by task
        self.replay_buffer_by_task = [] # one buffer for each task

    def predict_Q(self, state, task):
        # TODO maybe move this out to the caller, since it's a side effect
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

        Q_max = np.max(Q_task[state,:,:], axis=1) # take max policy over tasks

        return Q_max

    def train_online(self, task, her=False):
        tid = self.tasks.index(task)

        # learn only from last transition
        (s, a, s_next, phi_next) = self.replay_buffer_by_task[tid].last()
        sf = self.sf_by_task[tid]

        # TEST print successor features to see convergence
        # if s == 1:
        #     print("psi1, action{}: {}".format(a, sf[s,a,:]))
        # if s == 2:
        #     print("psi2, action{}: {}".format(a, sf[s,a,:]))

        psi = sf[s,a,:].squeeze()
        a_next = np.argmax(np.squeeze(self.sf_by_task[tid][s_next,:,:]) @ task)
        psi_next = sf[s_next, a_next, :].squeeze()
        delta = phi_next + self.GAMMA * psi_next - psi

        sf[s,a,:] += self.ALPHA * delta

        # update features
        self.sf_by_task[tid] = sf

        # HER: update *all* SFs in s
        if her:
            for (t,w) in enumerate(self.tasks):
                sf = self.sf_by_task[t]
                psi = sf[s,a,:].squeeze()
                a_next = np.argmax(np.squeeze(self.sf_by_task[t][s_next,:,:]) @ w)
                psi_next = sf[s_next, a_next, :].squeeze()
                delta = phi_next + self.GAMMA * psi_next - psi

                sf[s,a,:] += self.ALPHA * delta
                self.sf_by_task[t] = sf


    def train_offline(self, task, her=True):
        """
        Learn from of a sampled batch of transitions in the buffer.
        TODO if her=True, we learn for tasks that are not currently relevant (???)
        """
        tid = self.tasks.index(task)

        # don't train offline with too few samples, lest we bias
        if len(self.replay_buffer_by_task[tid]) < self.BATCH_SIZE:
            return

        #if her: # TODO train all tasks?
        for (t,w) in enumerate(self.tasks):
            sf = self.sf_by_task[t]

            # sample some transitions to replay (only those of the same task)
            batch = self.replay_buffer_by_task[t].sample(self.BATCH_SIZE)
            for (s, a, s_next, phi) in batch:
                psi = sf[s,a,:].squeeze()
                a_next = np.argmax(np.squeeze(self.sf_by_task[tid][s_next,:,:]) @ task)
                psi_next = sf[s_next, a_next, :].squeeze()
                delta = phi + self.GAMMA * psi_next - psi
                sf[s,a,:] += self.ALPHA * delta

            # update features
            self.sf_by_task[t] = sf


    def _add_new_task(self, task):
        # if task capacity reached, delete the oldest
        if len(self.tasks) == self.TASK_CAPACITY:
            self.tasks.pop(0)
            self.sf_by_task.pop(0)
            self.replay_buffer_by_task.pop(0)

        self.tasks.append(task)
        sf = np.zeros([self.n_states, self.n_actions, self.n_features]) # has to be 0 so that the psi_next = 0 for the leaf-nodes
        # HACK sf[0:3, :, :] = 15 # optimistic init
        self.sf_by_task.append(sf)
        self.replay_buffer_by_task.append(ReplayMemory(self.BUFFER_CAPACITY))


    def store_transition(self, task, s, a, s_next, phi_next):
        tid = self.tasks.index(task)
        self.replay_buffer_by_task[tid].push(s, a, s_next, phi_next)

    def reset_all_buffers(self):
        for tid in range(len(self.tasks)):
            self.replay_buffer_by_task[tid].memory.clear()

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
