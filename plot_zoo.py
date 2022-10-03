import numpy as np
import pickle
# plotting
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
# stats
import pandas as pd
import seaborn as sb

from zoo import Zoo
from zoo_feeding import run_training_test

#########################
# SIMULATION-PARAMETERS #
#########################
prefix = "sim" # "tobi" # 
N = 1 # repetitions / subjects
B = 100 # number of blocks
T = 50 # trials per block
M = 2 # number of concurrent tasks (learned alternatingly)

tags = ""
#tags = "alpha0.5_gamma0.9"
#tags = "fixedstart"

def plot_regret(df):
    plt.figure()
    df["bt"] = df["block"] * T + df["trial"]
    df["task"] = df["trial"] % M
    df["task"] = df["task"].astype(int)
    df["regret"] = df["regret"] + 3
    sb.lineplot(data=df, x="bt", y="regret", hue="task")
    for b in range(1,B):
        plt.axvline(b*T)
    plt.xlabel("trial")
    plt.show()


def plot_zoo_paths(env, paths, tasks, labels=[], colors=[], titles=[],
                   n_row=1, n_col=1, show_phi=True, show_symbols=True):
    """
    Plot zoo environment with paths in it (given by index).
    Paths is a list of lists where the outer list creates a new zoo subplot
    and the every inner list has the paths that are plotted in each zoo.
    Labels is a list of names for the paths, again a list of lists.
    n_row and n_col determine the subplot layout.
    """
    fig, ax = plt.subplots(nrows=n_row, ncols=n_col)
    if n_row == 1:
        ax = ax[np.newaxis] # to be able to index with row,col

    # state center coordinates
    x = np.array([0,-1,1,-2,0,2,-3,-1,1,3])
    y = np.array([0,-1,-1,-2,-2,-2,-3,-3,-3,-3])

    # define some colors (max 8 paths per zoo)
    cs = [[1,0,0], [0,1,0], [0,0,1], [0,0,0],
          [1,1,0], [1,0,1], [0,1,1], [.5,.5,.5]]
    if colors == []:
        colors = [cs]*(n_row*n_col)

    sub = 0 # subplot counter
    for row in range(n_row):
        for col in range(n_col):
            # draw a new zoo (white squares)
            ax[row, col].scatter(x, y, s=[700]*10, c="w", marker="s",
                                 linewidths=1, edgecolors="k")
            if show_phi:
                for i in range(1, env.n_states):
                    ax[row, col].text(x[i], y[i]-.5, env.phi[i], ha="center")
            if show_symbols:
                # animal at root node
                task = _parse_list(tasks[sub], ",")
                img = plt.imread(f"./img/{Zoo.animals[tuple(task)]}.png")
                ib = OffsetImage(img, zoom=0.25)
                ab = AnnotationBbox(ib, [x[0], y[0]], frameon = False)
                ax[row, col].add_artist(ab)

                # fruits
                for i in range(1, env.n_states):
                    img = plt.imread(f"./img/{Zoo.fruit[tuple(env.phi[i])]}.png")
                    ib = OffsetImage(img, zoom=0.25)
                    ab = AnnotationBbox(ib, [x[i], y[i]], frameon = False)
                    ax[row, col].add_artist(ab)

            ps_sub = paths[sub]
            print(ps_sub)
            paths_items = []
            for p in ps_sub:
                paths_items += [[0] + env.path_idx2items[p]]

            ls = []
            for i,p in enumerate(paths_items):
                x_jitter = np.random.rand() * .2 - .1 # range [-.1,.1]
                y_jitter = np.random.rand() * .2 - .1 # range [-.1,.1]
                l, = ax[row, col].plot(x[p]+x_jitter,
                                       y[p]+y_jitter,
                                       c=colors[sub][i])
                ls += [l]

                if(titles != []):
                    ax[row, col].set_title(titles[sub])
                if(labels != []):
                    ax[row, col].legend(ls, labels[sub])
                ax[row, col].set_axis_off()
                ax[row, col].set_xlim(-3.5,3.5)
                ax[row, col].set_ylim(-3.5,0.5)
            sub += 1
    plt.tight_layout(pad=2.5)
    plt.show()


def plot_zoo_mb_paths(env, tasks):
    ps = [] # paths
    ls = [] # labels
    for w in tasks:
        _, p_mb = env.mb_paths(w)
        p_mb = np.nonzero(p_mb)[0]
        ps += [p_mb]
        ls += [list(range(len(p_mb)))]
    plot_zoo_paths(env, ps, labels=[], titles=list(map(str, tasks)), n_row=1, n_col=len(tasks))


def _parse_list(l, delimiter=" "):
    return list(map(int, l.strip('[]').split(delimiter)))


def plot_training_test(df, combine_train_paths=False):
    """
    Plots one tree for the MB paths of the training tasks (one for each)
    and one for the test task, including MB, MF and SF paths.
    """
    # training
    p_train = []
    if combine_train_paths:
        for m in range(M):
            p_train += [df.loc[df["trial"] == -M+m, "mb_path"].iat[0]]
    else:
        for m in range(M):
            p_train += [[df.loc[df["trial"] == -M+m, "mb_path"].iat[0]]]


    # test
    p_test = []
    p_test += [df.loc[df["trial"] == 0, "mb_path"].iat[0]]
    p_test += [int(df.loc[df["trial"] == 0, "mf_path"].iat[0])] # XXX weird int bug
    p_test += [df.loc[df["trial"] == 0, "sf_path"].iat[0]]

    tasks = df.loc[:,"task"].to_list()

    if combine_train_paths:
        ps = [p_train, p_test]
        ls = [[f"MB {w}" for w in tasks], [f"MB {tasks[M]}", "MF", "SFGPI"]]
        ts = ["training", "test"]
        cs = [] # [["r"]*M, ["r", "g", "b"]]
    else:
        ps = [*p_train, p_test]
        ls = [*[[f"MB {w}"] for w in tasks], [f"MB {tasks[M]}", "MF", "SFGPI"]]
        ts = ["training 1", "training 2", "test"]
        cs = []
    plot_zoo_paths(env, ps, tasks, labels=ls, colors=cs, titles=ts, n_row=1, n_col=3)



def plot_raster_sim(df, sim_to_what="mf", nth_trial=1, first_trial=False):
    """
    Plots a raster of paths (8 different ones) against trials,
    indicating the taken path and its similarity with the other paths.
    The other paths (sim_to_what) can be "mb", "mf", "p0", "p1", "p01"
    where the pX similarity compares to the paths taken in the previous block.
    If first_trial is true, only plot the first trial per block.
    """
    plt.figure()
    n_paths = 8

    if first_trial: # pick only first trial per block
        df = df.loc[df["trial"] == 0, :]
    else: # pick every n_th trial per block
        df = df.loc[df["trial"] % nth_trial == 0, :]
    df = df.reset_index(drop=True)

    sim  = np.zeros((n_paths, len(df)))
    paths_1hot = np.zeros((n_paths, len(df)))
    for t in range(len(df)):
        sf_path = df.at[t, "sf_path"]
        sim[sf_path, t] = 1 + df.at[t, (sim_to_what + "_sim")]
        # paths = df.at[t, (sim_to_what + "_paths")]
        # paths = _parse_list(paths)
        # paths = np.array(paths, dtype=int)
        # paths_1hot[paths, t] = 1 # mark the paths to which we compare

    # taken paths in color coding similarity
    cs = plt.cm.gist_heat_r([0,.2,.4,.6,.8]) # 4 values for similarity
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', cs)
    plt.imshow(sim, cmap=cmap, aspect="auto", interpolation="none")

    # colorbar
    cb = plt.colorbar(ticks=[0,1,2,3,4])
    cb.ax.set_yticklabels(["", "sim=0", "sim=1", "sim=2", sim_to_what])

    # plot vertical bars between blocks
    ylim = [-0.5,7.5]
    plt.ylim(ylim)

    plt.ylabel("#path")

    if first_trial:
        plt.xlabel("block")
    else:
        plt.xticks(range(0, B*T//nth_trial, T//nth_trial))
        plt.vlines(range(0, B*T//nth_trial, T//nth_trial), *ylim, 'b')
        plt.xlabel("trial")

    plt.title(f"Taken paths and similarity with {sim_to_what}-paths")
    plt.show()


def plot_epsilon():
    """
    Plot epsilon over time (dynamic exploration schedule).
    """
    from zoo_feeding import epsilon_greedy
    x = range(B*T)
    y = np.zeros(B*T)
    dummy = [1,0,0]
    for b in range(B):
        for t in range(T):
            y[b*T+t] = epsilon_greedy(dummy,t)[1]

    plt.figure()
    plt.plot(x, y)
    # plot vertical bars between blocks
    plt.vlines(range(0, T*B, T), 0, 1, 'k')
    plt.xlabel("trial")
    plt.ylabel("epsilon")
    plt.show()


if __name__ == "__main__":
    # if prefix == "sim":
    #     df = pd.read_csv(f"./zoo/sim_N{N}_B{B}_T{T}_M{M}_{tags}.csv")
    # else:
    #     df = pd.read_csv(f"./zoo/{prefix}_B{B}_T{T}_M{M}_{tags}.csv")

    #plot_regret(df)
    #plot_epsilon()

    # pick one subject only
    #df = df.loc[df["subject"] == 0,:]

    #plot_raster_sim(df, sim_to_what="mb", first_trial=False)
    #plot_most_common_paths(df)

    #plot_zoo_mb_paths(env, [[0,1], [1,0]])
    # curriculum = ([[0,1], [1,0]], [1,1])
    # curriculum = ([[0,1], [-1,0]], [-1,1])
    # curriculum = ([[0,-1], [1,0]], [1,-1])
    # curriculum = ([[0,-1], [-1,0]], [-1,-1])

    # load an optimized block
    for b in range(4):
        with open(f"./zoo/block_{b}.pickle", "rb") as f:
            env, cur, df = pickle.load(f)
            df = run_training_test(env, cur)
            plot_training_test(df)
