"""
Plotting raw outputs, i.e. chosen leaves (= paths) and corresponding regrets.
Runs a small scale simulation with 1 subject and few blocks,
such that individual algorithm behaviour becomes clear.
"""

import numpy as np
from matplotlib import pyplot as plt
from continuous_learning import init_blocks_randomly, run_uvfa, run_sfgpi
from env import TwoStepEnv

#########################
# SIMULATION-PARAMETERS #
#########################
# run only one subject and few blocks
# for demo purposes, I use both task and feature changes here
N = n_subjects = 1        # repetitions of simulation to average over
B = n_blocks = 6          # number of blocks
T = block_size = 100      # number of trials in a block
M = n_tasks_per_block = 2 # number of unique tasks per block ("multi-tasking")
p_task_change = 0.5       # probability of task change
p_feature_change = 0.5    # probability of feature change
p_transition_change = 0   # TODO implement


def plot_regret(block_idx, uvfa_regret, sfgpi_regret,
                blocks, optimal_leaves, single_tasks=False):
    """
    Plots regret for each trial over blocks.
    block_idx = which blocks to show
    """
    # slice only these blocks
    blocks         = blocks[block_idx]
    uvfa_regret    = np.reshape(uvfa_regret, [n_blocks, block_size])[block_idx]
    sfgpi_regret   = np.reshape(sfgpi_regret, [n_blocks, block_size])[block_idx]
    optimal_leaves = optimal_leaves[block_idx,:,:]
    n_blocks_to_show = len(blocks)

    plt.figure()
    for algo, regret in enumerate([uvfa_regret, sfgpi_regret]):
        if single_tasks:
            linestyles = ["solid", "dotted", "dashed", "dashdot"]
            colors = ["blue", "orange"]
            for t in range(n_tasks_per_block):
                x = range(n_blocks_to_show*block_size//n_tasks_per_block)
                y = regret[:,t::n_tasks_per_block].flatten() # only the t-th task
                y_smoothed = np.convolve(y, np.ones(10), 'same') / 10
                plt.plot(x, y_smoothed, color=colors[algo], linestyle=linestyles[t])
        else:
            x = range(n_blocks_to_show*block_size)
            y = regret.flatten()
            y_smoothed = np.convolve(y, np.ones(10), 'same') / 10
            plt.plot(x, y_smoothed)

    # plot vertical bars between blocks
    single_task_correct = n_tasks_per_block if single_tasks else 1
    plt.vlines(np.arange(0, block_size*n_blocks_to_show, block_size)//single_task_correct, 0, max(y), 'k')

    # label which tasks were learned
    for b, block in enumerate(blocks):
        tasks, env = block
        s = "".join([f"w{m} = {w}\n" for (m,w) in enumerate(tasks)])
        phi_change = np.any(env.phi != blocks[b-1][1].phi)
        s = s + ("no " if not phi_change else "") + r"$\phi$ change"
        plt.annotate(s, [b * block_size // single_task_correct + block_size/10,
                         max(y) * .9])

    plt.xlabel("trial")
    plt.ylabel("regret")
    plt.xticks(np.arange(0, n_blocks_to_show*block_size, block_size)// single_task_correct)
    plt.legend([f"UVFA, w{m}" for m in range(M)] +
               [f"SFGPI, w{m}" for m in range(M)], loc="center right")
    plt.show()


def plot_leaves(block_idx, uvfa_regret, uvfa_leaves,
                sfgpi_regret, sfgpi_leaves, optimal_leaves):
    """
    Plots chosen leaves (= paths) for each trial over blocks.
    block_idx = which blocks to show
    """
    # slice only these blocks
    uvfa_regret    = np.reshape(uvfa_regret, [n_blocks, block_size])[block_idx]
    uvfa_leaves    = np.reshape(uvfa_leaves, [n_blocks, block_size])[block_idx]
    sfgpi_regret   = np.reshape(sfgpi_regret, [n_blocks, block_size])[block_idx]
    sfgpi_leaves   = np.reshape(sfgpi_leaves, [n_blocks, block_size])[block_idx]
    optimal_leaves = optimal_leaves[block_idx,:,:]
    n_blocks_to_show = uvfa_regret.shape[0]

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1,1,2]})

    for p, (leaves, regret) in enumerate([[uvfa_leaves, uvfa_regret],
                                          [sfgpi_leaves, sfgpi_regret]]):
        leaves = leaves.flatten()
        regret = regret.flatten()
        n_states = TwoStepEnv.n_states

        y = np.zeros((n_states, len(leaves)))
        y[leaves, range(len(leaves))] = 1 # 1hot encoding

        # distinguish correct (=1) and incorrect choices (=-1)
        y = y - 2 * y * np.vstack([regret != 0 for _ in range(n_states)])

        cmap = plt.get_cmap("RdYlBu", 3) # 3 values: -1 (red), 0, 1 (blue)
        tmp = ax[p].imshow(y, cmap=cmap, aspect="auto", interpolation="none")
        ylim = [3.5,12.5] # hardcoded TODO use env.terminal
        ax[p].set_ylim(ylim)
        ax[p].set_yticks(ticks=range(4,13), labels=range(5,14)) # TODO hardcoded
        ax[p].set_title("UVFA" if p==0 else "SFGPI")

        # plot vertical bars between blocks
        ax[p].vlines(range(0, block_size*n_blocks, block_size), *ylim, 'k')

        ax[p].set_xlabel("trial")
        ax[p].set_ylabel("final node")
        ax[p].set_xticks(range(0, n_blocks*block_size, block_size))

    # extra subplot for labels
    ax[2].set_axis_off()
    ax[2].set_xlim([0, n_blocks])
    ax[2].set_ylim([0, 1.2])

    # label which tasks were learned
    for b, block in enumerate(blocks):
        tasks, env = block
        s = "".join([f"w{m} = {w}\n" for (m,w) in enumerate(tasks)])
        phi_change = np.any(env.phi != blocks[b-1][1].phi)
        s = s + ("no " if not phi_change else "") + r"$\phi$ change"
        s = s + "\n\n"
        s = s + "\n".join(["optimal{} = {}".format(
            m, np.nonzero(optimal_leaves[b,m,:])[0] + 1) for m in range(M)])
        ax[2].annotate(s, [b + .2, .5])

    cbar = fig.colorbar(tmp, orientation="horizontal", ticks=[-.66,0,.66], ax=ax[2])
    cbar.ax.set_xticklabels(["wrong", "", "correct"])

    fig.tight_layout()
    fig.show()
    plt.show()



def plot_epsilon():
    """
    Plot epsilon over time (dynamic exploration schedule).
    """
    x = range(n_blocks*block_size)
    y = np.zeros(n_blocks*block_size)
    dummy = [1,0,0]
    for b in range(n_blocks):
        for t in range(block_size):
            y[b*block_size+t] = epsilon_greedy(dummy,b,t)[1]

    plt.figure()
    plt.plot(x, y)
    # plot vertical bars between blocks
    plt.vlines(range(0, block_size*n_blocks, block_size), 0, 1, 'k')
    plt.xlabel("trial")
    plt.ylabel("epsilon")
    plt.show()


if __name__ == "__main__":

    blocks, optimal_reward, optimal_leaves, changes = init_blocks_randomly(
        B, T, M, p_task_change, p_feature_change)
    uvfa_regret,  uvfa_leaves  = run_uvfa(blocks,  optimal_reward, verbose=False)
    sfgpi_regret, sfgpi_leaves = run_sfgpi(blocks, optimal_reward, verbose=False)

    block_idx = slice(0,6) # NB this has to be a slice
    plot_regret(block_idx, uvfa_regret, sfgpi_regret, blocks, optimal_leaves, single_tasks=True)
    plot_leaves(block_idx, uvfa_regret, uvfa_leaves, sfgpi_regret, sfgpi_leaves, optimal_leaves)

    #plot_epsilon()
