# Continuous Multi-task Reinforcement learning

In this project, we compare two algorithms for generalization across tasks,
Universal Value Function Approximation (UVFA; [Schaul et al., 2015](https://www.semanticscholar.org/paper/Universal-Value-Function-Approximators-Schaul-Horgan/5dc2a215bd7cd5bdd3a0baa8c967575632696fac)) and
Successor Feature Generalized Policy Improvement (SFGPI; [Barreto et al. 2017](https://www.semanticscholar.org/paper/Successor-Features-for-Transfer-in-Reinforcement-Barreto-Dabney/d8686b657b61a37da351af2952aabd8b281de408)).
Here, tasks are defined as different weightings $w$ of the features $\phi$
of the environment.
Both together form the reward for a state $s$: $$r(s) = w^T \phi(s).$$

Here, we use a block design to test the effect of changes in the task and the environment on the choices of the algorithms.
This design consists of $B$ blocks with $T$ trials each.
In each block $M$ tasks are learned concurrently, i.e. alternatingly, one per trial.
From block to block, 3 different types of changes can happen:
- tasks can change such that one of the old ones is droppped and a new one is introduced (tasks are drawn with replacement from vectors $w \in {-1,0,1}^3)$)
- features of two final states can be swapped (where one of them corresponds to a path which is optimal wrt. the test task, i.e. that of the 1st trial)
- transitions of two nodes on the first level of decision tree can be swapped (where one of them lies on a path that is optimal wrt. the test task)

We are mainly interested in the actions of the algorithms in the first trial of a new block ("test" trial), i.e. how they react on the different types of changes.
Another aspect of interest is how (quickly) they re-learn after a change.


## How is the code structured?

`ènv.py` defines the environment with its features and transitions.
Here, we use a 2-step deterministic environment like in [Tomov & Schulz, 2021](https://www.semanticscholar.org/paper/Multi-Task-Reinforcement-Learning-in-Humans-Tomov-Schulz/50ee7d0767f79e35fb6d06f5d97f3440b6afcaf9).

`uvfa.py` implements UVFA with a feed-forward neural network as an approximator for the value function $(w,s) \to Q$.

`sfgpi.py` is an implementation of SFGPI which learns successor features for each task and combines them to find policies for unseen tasks.

`continuous_learning.py` runs the simulation of both algorithms on our block paradigm with $N$ repetitions / "subjects" and outputs .csv files in `/sim`.

`plot_raw.py` generates plots of the raw outputs of the algorithms,
i.e. their chosen actions and the corresponding regret, over the whole block.

`plot_stats.py` plots different summary statistics (mainly 1st-block statistics) of the simulations.

## How to use the code?
1. To run the simulation, go to `continuous_learning.py` and edit the simulation parameters at the top of the file. *Careful*: $N=1$ subject takes ~45sec with $B=50$ and $T=300$.
2. Run `python continuous_learning.py` to run extensive simulations and store .csvs of the summary statistics.
3. Run `python plot_stats.py` to see these summary statistics. Pick any csv file, by editing the simulation parameters (again at the top of the file).

If you want to see raw output, it is better just to run a few blocks.
`plot_raw.py` is designed for that (again, edit parameters at the top of the file).
