import numpy as np
# plotting
from matplotlib import pyplot as plt
import seaborn as sb
# stats
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

#########################
# SIMULATION-PARAMETERS #
#########################
N = n_subjects = 100      # repetitions of simulation to average over
B = n_blocks = 50         # number of blocks
T = block_size = 100      # number of trials in a block
M = n_tasks_per_block = 2 # number of unique tasks per block ("multi-tasking")
p_task_change = 0       # probability of task change
p_feature_change = 0    # probability of feature change
p_transition_change = 0.5 # probability of transition change


def plot_distance(df, metric="task_euclid", ymeasure="regret"):
    """
    Plot scatter of x = task/feature/value distance and y = ymeasure.
    metric can be one of {task, feature} x {euclid, angle, manhattan}
    or max_value_diff.
    """
    # only 1st trial in block
    df = df[df["trial"] == 0]

    # XXX exclude cases where both task and features change
    df = df[np.logical_not(np.logical_and(
        df["task_change"], df["feature_change"]))]

    # mean over subjects (to make "correct" continous)
    df = df.groupby(by=["block", "algo"]).agg({ymeasure: "mean", f"{metric}": "mean"})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0)

    df_uvfa = df[df["algo"] == "uvfa"]
    df_sfgpi = df[df["algo"] == "sfgpi"]

    # anova based on the conditions
    model_uvfa = ols(f"{ymeasure} ~ {metric}", data=df_uvfa).fit()
    model_sfgpi = ols(f"{ymeasure} ~ {metric}", data=df_sfgpi).fit()
    # parameters
    b_uvfa = model_uvfa.params["Intercept"]
    m_uvfa = model_uvfa.params[metric]
    b_sfgpi = model_sfgpi.params["Intercept"]
    m_sfgpi = model_sfgpi.params[metric]

    # correlation
    r_uvfa = np.corrcoef(df_uvfa[metric], df_uvfa[ymeasure])[0,1]
    r_sfgpi = np.corrcoef(df_sfgpi[metric], df_sfgpi[ymeasure])[0,1]

    # TODO add jitter for scatter, built-in jitter non-functional
    # sigma = .2
    # df[f"{change}_{metric}"] += np.random.randn(len(df)) * sigma
    # df[ymeasure] += np.random.randn(len(df)) * sigma

    plt.figure()
    sb.scatterplot(x=metric, y=ymeasure, hue="algo", alpha=.5,
                   hue_order=["uvfa", "sfgpi"], data=df)
    #cmap = plt.get_cmap() # TODO same colors
    plt.axline(xy1=(0, b_uvfa), slope=m_uvfa, color="blue", label=f"r={r_uvfa:.3f}") # uvfa
    plt.axline(xy1=(0, b_sfgpi), slope=m_sfgpi, color="orange", label=f"r={r_sfgpi:.3f}") # sfgpi
    plt.title(f"{ymeasure} by distance ({metric})")
    plt.xlabel(f"distance ({metric})")
    plt.xlabel(metric)
    plt.ylabel(ymeasure)
    plt.legend()
    plt.show()


def plot_changes(df, change="task", ymeasure="regret"):
    """
    Plot barplot with x=change and y=ymeasure.
    change can be one of {task, feature, transition}.
    """
    # only 1st trial in block
    df = df[df["trial"] == 0]

    # XXX exclude cases where both task and features change
    df = df[np.logical_not(np.logical_and(df["task_change"], df["feature_change"]))]

    # mean over blocks for continous ymeasure
    df = df.groupby(by=["subject", "algo", f"{change}_change"]).agg(
        {ymeasure: "mean", f"{ymeasure}": "mean"})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0)

    model = ols(f"{ymeasure} ~ C(algo) + C({change}_change) + C({change}_change):C(algo)",
                data=df).fit()
    print(model.params)
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    plt.figure()
    sb.barplot(x=f"{change}_change", y=ymeasure, hue="algo", data=df)
    plt.title(f"{ymeasure} by {change} change")
    plt.xticks(ticks=[0,1], labels=["no", "yes"])
    plt.xlabel(f"{change} change")
    plt.ylabel(ymeasure)
    plt.show()


def plot_blocks(df, change="task", ymeasure="regret"):
    """
    Plot 1st trial over blocks as lines with y=ymeasure and split by change.
    change can be one of {task, feature, transition}.
    """
    # only 1st trial in block
    df = df[df["trial"] == 0]

    # XXX exclude cases where both task and features change
    df = df[np.logical_not(np.logical_and(df["task_change"], df["feature_change"]))]

    # mean over subjects for continous ymeasure
    df = df.groupby(by=["block", "algo", f"{change}_change"]).agg(
        {ymeasure: "mean", f"{ymeasure}": "mean"})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0)

    model = ols(f"{ymeasure} ~ block + block:C(algo)", df).fit()
    print(model.params)
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    plt.figure()
    sb.lineplot(data=df, x="block", y=ymeasure, hue="algo", style=f"{change}_change",
                hue_order=["uvfa", "sfgpi"])
    plt.show()


def plot_first_n_trials(df, offset=0, change="task", ymeasure="regret"):
    """
    Plot first n occurrences of the m-th task (m = offset), # TODO param for n?
    as lines of with y=ymeasure and split by change.
    change can be one of {task, feature, transition}.
    """
    df = df.groupby(["algo", f"{change}_change", "trial", "task"]).agg(
        {ymeasure: ["mean"]})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0)

    plt.figure()
    sb.lineplot(data=df[df["task"] == offset], x="trial", y=f"{ymeasure}",
                hue="algo", style=f"{change}_change", hue_order=["uvfa", "sfgpi"])
    plt.title(f"{ymeasure} by {change} change")
    plt.xlabel("trial")
    plt.ylabel(ymeasure)
    plt.show()


def filter_for_one_change(df, change="task", history=0):
    """
    Filter such that only one kind of change happens, and the other not.
    History is an integer determining how many steps in the past to filter.
    change can be one of {task, feature, transition}.
    """
    change_not = "feature" if change == "task" else "task"
    for h in range(history+1):
        df = df.loc[np.roll(df[f"{change_not}_change"] == False, h)]

    # filter such that in the block before there was NO change at all
    # df = df.loc[np.logical_not(np.logical_or(np.roll(df["feature_change"], 1),
    #                                          np.roll(df["task_change"], 1))),:]

    # TEST
    #print("#Rows remaining: " + str(len(df)))
    return df


def plot_possible_correct(df, change="task"):
    """
    Lineplot over blocks (1st trials) of how many correct leaves there are.
    """
    # only 1st trial in block
    df = df[df["trial"] == 0]

    df = df.groupby(["algo", "block", f"{change}_change"]).agg(
        {"possible_correct": ["mean"]})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0)

    sb.lineplot(data=df, x="block", y="possible_correct", hue="algo", style=f"{change}_change")
    plt.show()


if __name__ == "__main__":
    # load
    df = pd.read_csv(f"./sim/sim_N{N}_B{B}_T{T}_M{M}_ptask{p_task_change}_pfeature{p_feature_change}_ptransition{p_transition_change}.csv")

    # TODO filter only first 50 blocks
    #df = df[df["block"] < 50]

    #plot_first_n_trials(df, offset=0, change="feature", ymeasure="correct")

    #plot_distance(df, metric="task_euclid", ymeasure="correct")
    #plot_distance(df, metric="max_value_diff", ymeasure="correct")

    #plot_changes(df, change="feature", ymeasure="correct")

    plot_blocks(df, change="feature", ymeasure="correct") # "mean_block_regret")

    #plot_possible_correct(df, change="feature")
