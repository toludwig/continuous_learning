import numpy as np
# plotting
from matplotlib import pyplot as plt
import seaborn as sb
# stats
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# helper
def _dual2dec(goal_state):
    # transform dual goal_state (least-significant bit first) to decimal
    dec = 0
    for i,s in enumerate(goal_state):
        dec += s*2**i
    return dec


#########################
# SIMULATION-PARAMETERS #
#########################
prefix = "sim" # "tobi" # 
N = 20 # repetitions / subjects
B = 3 # number of blocks
T = 200 # trials per block
M = 4 # number of concurrent tasks (learned alternatingly)
G = 1 # number of goals per task
#recipes = [(2,3,9), (3,12,15), (6,9,12), (2,6,15)] # XXX triples are recipes
gridsize = [4,4]
tags = "alpha0.5_gamma0.9"
#tags = "fixedstart"

def plot_regret(df):
    plt.figure()
    df["bt"] = df["block"] * T + df["trial"] # df["trial"] #
    df["task"] = df["trial"] % M
    df["task"] = df["task"].astype(int)
    df["regret"] = df["regret"] + 3
    sb.lineplot(data=df, x="bt", y="regret", hue="task")
    for b in range(1,B):
        plt.axvline(b*T)
    plt.xlabel("trial")
    plt.ylim([0,100])
    plt.show()


def plot_path(path, task, item2cell, plot_order=False, n_grid=4):
    """
    Plots the path in a grid.
    Path is a list of items.
    """
    path = item2cell[path]
    print(path) # TODO
    flat = np.zeros(n_grid**2)
    if plot_order:
        for k, cell in enumerate(path):
            flat[cell] = k
    else: # visit count for each cell
        for cell in path:
            flat[cell] += 1
    grid = np.reshape(flat, gridsize)
    plt.matshow(grid)
    plt.colorbar(label=("visit order" if plot_order else "visit count"))

    # plot goal positions
    goal_items = np.arange(n_grid**2)[np.ones(n_grid**2) == task]
    goal_cells = item2cell[goal_items]
    goal_j, goal_i = np.unravel_index(goal_cells, [n_grid, n_grid])
    plt.scatter(goal_i, goal_j, c="k", marker="x")

    plt.title("Path")
    plt.show()


def plot_value_map(sf, task, item2cell, goal_state=[0,0,0], n_grid=4):
    """
    Plots a map of the supermarket with color indicating state value.
    - SF is given as a 3d-array of shape n_states x n_actions x n_features
    - task is the k-hot vector indicating the goals (TODO for now only 1 goal)
    """
    Q = np.einsum("ijk,k", sf, task) # shape: n_states x n_actions
    V = np.max(Q, axis=1) # shape: n_states

    # states do not directly map on physical states (cells)
    # but a combination of cells and goal_state
    # hence, we pick only the values for the given goal state
    dec = _dual2dec(goal_state)
    n_features = n_grid**2
    V = V[dec*n_features:(dec+1)*n_features]

    # now map items back on cells
    V = np.reshape(V[item2cell], [n_grid, n_grid])

    print(V)

    # plot V
    plt.matshow(V)
    plt.colorbar(label="value")

    # plot goal positions
    goal_items = np.arange(n_grid**2)[np.ones(n_grid**2) == task]
    goal_cells = item2cell[goal_items]
    goal_j, goal_i = np.unravel_index(goal_cells, [n_grid, n_grid])
    plt.scatter(goal_i, goal_j, c="k", marker="x")

    plt.title("Value")
    plt.show()


if __name__ == "__main__":
    if prefix == "sim":
        df = pd.read_csv(f"./supermarket/sim_N{N}_B{B}_T{T}_M{M}_G{G}_{tags}.csv")
    else:
        df = pd.read_csv(f"./supermarket/{prefix}_B{B}_T{T}_M{M}_G{G}_{tags}.csv")
    plot_regret(df)
