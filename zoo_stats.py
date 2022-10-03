"""
For trying out certain figures, stats, ...
"""
import numpy as np
# plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sb
# stats
import pandas as pd

from zoo import Zoo

#########################
# SIMULATION-PARAMETERS #
#########################
prefix = "sim" # "tobi" # 
N = 10 # repetitions / subjects
B = 10 # number of blocks
T = 50 # trials per block
M = 2 # number of concurrent tasks (learned alternatingly)

tags = ""


def plot_mb_sim_over_trials(df):
    plt.figure()
    sb.lineplot(data=df, x="trialblock", y="mb_sim", hue="task")
    plt.show()


def plot_most_common_paths(df, block=2):
    """
    Plot the most common paths for a block and its previous block.
    """
    # select block and the block before
    df = df.loc[np.isin(df["block"],[block-1,block]),:]
    df = _most_common_paths_per_block(df)

    # now plot the most common paths of the previous
    plot_paths_in_tree(df.mc_path)


def _most_common_paths_per_block(df):
    # compute most common path per block
    df = df.groupby(["block", "task", "sf_path"]).size()
    df = df.reset_index()
    df = df.groupby(["block", "task"]).apply(lambda x: x.loc[x.iloc[:,3] == x.iloc[:,3].max(), "sf_path"])
    df = df.reset_index()
    del df["level_2"]
    df = df.rename(columns={"sf_path": "mc_path"}) # mc = most common
    return df
# # TODO most common path
# for t in range(M):
#     counts = np.unique(path[b,range(t,T,M)], return_counts=True)
# p_mc = 



if __name__ == "__main__":
    if prefix == "sim":
        df = pd.read_csv(f"./zoo/sim_N{N}_B{B}_T{T}_M{M}_{tags}.csv")
    else:
        df = pd.read_csv(f"./zoo/{prefix}_B{B}_T{T}_M{M}_{tags}.csv")

    # pick one subject only
    #df = df.loc[df["subject"] == 0,:]

    #plot_raster_sim(df, sim_to_what="mb", first_trial=False)
    #plot_most_common_paths(df)
    plot_mb_sim_over_trials(df)
