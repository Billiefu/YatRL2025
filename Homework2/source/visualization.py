"""
Copyright (C) 2025 Fu Tszkok

:module: visualization
:function: Provides functions to visualize the results of reinforcement learning algorithms.
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
import matplotlib.pyplot as plt


def q_to_v_table(q_table):
    """Converts a Q-table to a state-value table (V-table)."""
    v_table = {}
    for state, action_values in q_table.items():
        # The value of a state is the max Q-value among all actions from that state
        if action_values:
            v_table[state] = max(action_values.values())
        else:
            v_table[state] = 0.0
    return v_table


def plot_learning_curves(histories_dict, window_size=25):
    """Plots the learning curves (sum of rewards per episode) for multiple TD algorithms.
    :param histories_dict: A dictionary where keys are algorithm names (str) and values are lists of total rewards per episode.
    :param window_size: The size of the moving average window for smoothing the curve.
    """
    plt.figure(figsize=(12, 6))

    for name, history in histories_dict.items():
        # Calculate moving average to smooth the curve and show the trend
        if len(history) >= window_size:
            smoothed_rewards = np.convolve(history, np.ones(window_size) / window_size, mode='valid')
            plt.plot(smoothed_rewards, label=f'{name} (Smoothed)')
        else:
            # Plot raw data if not enough points for smoothing
            plt.plot(history, label=f'{name} (Raw)')

    plt.title('Learning Curve: Rewards per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_value_function(env, v, title="State-Value Function (V*)"):
    """Creates a heatmap to visualize the state-value function (V).
    :param env: The maze environment instance.
    :param v: A dictionary mapping states (tuples) to their values.
    :param title: The title for the plot.
    """
    value_grid = np.full(env.maze.shape, np.nan)
    for state, value in v.items():
        value_grid[state] = value

    # Handle cases where not all states were visited
    if not v:
        vmin, vmax = -1, 0
    else:
        vmin = np.nanmin(value_grid)
        vmax = np.nanmax(value_grid)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')

    for r in range(env.height):
        for c in range(env.width):
            if not np.isnan(value_grid[r, c]):
                val = value_grid[r, c]
                text_color = "w" if val < (vmin + vmax) / 2 else "k"
                ax.text(c, r, f'{val:.1f}', ha='center', va='center', color=text_color)

    plt.colorbar(im, ax=ax, label='State Value')
    ax.set_title(title)
    ax.set_xticks(np.arange(env.width))
    ax.set_yticks(np.arange(env.height))
    # Set custom labels for cliff walk environment for better readability
    ax.set_xticklabels(np.arange(1, env.width + 1))
    ax.set_yticklabels(np.arange(1, env.height + 1))
    plt.show()
