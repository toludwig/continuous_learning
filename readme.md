Continuous Multi Task Reinforcement learning
============================================

In this project we compare two algorithms for generalization across tasks,
Universal value function approximation (UVFA; [Schaul et al., 2015](https://www.semanticscholar.org/paper/Universal-Value-Function-Approximators-Schaul-Horgan/5dc2a215bd7cd5bdd3a0baa8c967575632696fac)) and
Successor feature generalized policy improvement (SFGPI; [Barreto et al. 2017](https://www.semanticscholar.org/paper/Successor-Features-for-Transfer-in-Reinforcement-Barreto-Dabney/d8686b657b61a37da351af2952aabd8b281de408))
where tasks are defined as different weightings $w$ of the features $\phi$
of the environment.
Both together form the reward for a state $s$: $$r(s) = w^T \phi.$$

Here we use a 2-step deterministic environment like in [Tomov & Schulz, 2021](https://www.semanticscholar.org/paper/Multi-Task-Reinforcement-Learning-in-Humans-Tomov-Schulz/50ee7d0767f79e35fb6d06f5d97f3440b6afcaf9).

`continous_learning.py` runs the simulation of both algorithms on our block paradigm.
This consists of $B$ blocks with $T$ trials each.
In each block $M$ tasks are learned concurrently.
From block to block, tasks can change (such that one of the old ones is droppped and a new one is introduced),
__or__ the features of the environment can change themselves (such that the features of two final nodes are swapped).
We are mainly interested in the actions of the algorithms in the first trial of a new block.

`sfgpi.py` is an implementation of SFGPI using libraries storing learned tasks and successor features,
and `uvfa.py` implements UVFA with a feed-forward neural network as a function approximator.

`continous_stats.py` plots different summary statistics (mainly 1st-block statistics) of the simulations.
