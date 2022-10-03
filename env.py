from abc import ABC, abstractmethod, abstractproperty

class Env(ABC):
    """
    Env defines an abstract class interface for environments.
    An Env has methods for moving and providing reward.
    """

    # @abstractproperty
    # S = [] # state space (usually a list)

    # @abstractproperty
    # A = [] # action space (usually a list)

    # @abstractproperty
    # P = np.zeros([len(S), len(S), len(A)]) # probability for each transition

    # @abstractproperty
    # phi = np.zeros([len(S), len(n_features)]) # feature vectors of each state

    # @abstractproperty
    # terminal = np.zeros([len(S)]) # binary vector indicating terminal states

    # @abstractproperty
    # state = None # current state

    @abstractmethod
    def step(self, action):
        """
        Takes the action in the current state, changing the state.
        Returns a tuple with an observation and the received reward.
        """
        pass
