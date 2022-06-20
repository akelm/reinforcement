import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


def plot_episode_stats_multiple(*all_stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    for idx, stats in enumerate(all_stats):
        lengths_smoothed = pd.Series(stats.episode_lengths).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
        plt.plot(lengths_smoothed, label=str(idx), alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time (Smoothed over window size {})".format(smoothing_window))
    plt.yscale('log')
    plt.legend()
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    for stats in all_stats:
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window,
                                                                    min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, alpha=0.8)

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    for stats in all_stats:
        plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")

    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


def plot_max_Q(res_avg):
    if not res_avg: return

    num_states = len(next(iter(res_avg.values()))[0])
    plt.figure(figsize=( min(20, num_states + 2), 5))
    for idx, val in enumerate(res_avg.values()):
        plt.plot(np.max(list(val[0].values()), axis=1), '.-', alpha=1, label=str(idx))
    plt.xlabel('state')
    plt.ylabel('max Q')
    plt.legend()
    plt.show()


def plot_mean_Q(res_avg):
    if not res_avg: return

    num_states = len(next(iter(res_avg.values()))[0])
    plt.figure(figsize=( min(20, num_states + 2), 5))
    for idx, val in enumerate(res_avg.values()):
        plt.plot(np.mean(list(val[0].values()), axis=1), '.-', alpha=1, label=str(idx))
    plt.xlabel('state')
    plt.ylabel('max Q')
    plt.legend()
    plt.show()