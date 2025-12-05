"""
Copyright (C) 2025 Fu Tszkok

:module: main
:function: Executes and demonstrates TD learning algorithms on the Cliff Walk environment.
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

from algorithm import *
from cliffwalk import CliffWalkEnv
from layout import *
from visualization import *


# --- 1. Environment Setup ---

# Initialize the Cliff Walk environment.
cliffwalk_env = CliffWalkEnv()
# cliffwalk_env = CliffWalkEnv(cliffwalk=cliffwalk1)
# cliffwalk_env = CliffWalkEnv(cliffwalk=cliffwalk2)
# cliffwalk_env = CliffWalkEnv(cliffwalk=cliffwalk4)

# Hyperparameters for the TD learning algorithms
EPISODES = 500
ALPHA = 0.5
GAMMA = 1.0
EPSILON = 0.1

# Render the cliffwalk structure to the console for visual inspection.
cliffwalk_env.render()

# Dictionaries to store results for combined plot
histories_for_plot = {}


# --- 2. SARSA ---

# Run the SARSA algorithm to find the optimal value function and policy.
sarsa_q_table, sarsa_policy, sarsa_history = sarsa(cliffwalk_env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
histories_for_plot['SARSA'] = sarsa_history

# Display the resulting policy visually on the grid.
# Note SARSA's "safe" path far from the cliff.
print("Final Policy Found by SARSA:")
cliffwalk_env.print_policy(sarsa_policy)

# Convert the learned Q-table to a V-table for visualization
sarsa_v_table = q_to_v_table(sarsa_q_table)
# Visualize the state-value function learned by SARSA
visualize_value_function(cliffwalk_env, sarsa_v_table, "State-Value Function (V) for SARSA")


# --- 3. Expected SARSA ---

# Run the Expected SARSA algorithm to find the optimal value function and policy.
sarsa_q_table, sarsa_policy, sarsa_history = expected_sarsa(cliffwalk_env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
histories_for_plot['Expected SARSA'] = sarsa_history

# Display the resulting policy visually on the grid.
# Note Expected SARSA's "safe" path far from the cliff.
print("Final Policy Found by Expected SARSA:")
cliffwalk_env.print_policy(sarsa_policy)

# Convert the learned Q-table to a V-table for visualization
sarsa_v_table = q_to_v_table(sarsa_q_table)
# Visualize the state-value function learned by Expected SARSA
visualize_value_function(cliffwalk_env, sarsa_v_table, "State-Value Function (V) for Expected SARSA")


# --- 4. N-step SARSA ---

# Run the N-step SARSA algorithm to find the optimal value function and policy.
sarsa_q_table, sarsa_policy, sarsa_history = n_step_sarsa(cliffwalk_env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
histories_for_plot['N-step SARSA'] = sarsa_history

# Display the resulting policy visually on the grid.
# Note N-step SARSA's "safe" path far from the cliff.
print("Final Policy Found by N-step SARSA:")
cliffwalk_env.print_policy(sarsa_policy)

# Convert the learned Q-table to a V-table for visualization
sarsa_v_table = q_to_v_table(sarsa_q_table)
# Visualize the state-value function learned by N-step SARSA
visualize_value_function(cliffwalk_env, sarsa_v_table, "State-Value Function (V) for N-step SARSA")


# --- 5. Q-learning ---

# Run the Q-learning algorithm to find the optimal value function and policy.
q_q_table, q_policy, q_history = q_learning(cliffwalk_env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON)
histories_for_plot['Q-learning'] = q_history

# Display the resulting policy.
# Note Q-learning's "optimal" but risky path along the cliff edge.
print("Final Policy Found by Q-learning:")
cliffwalk_env.print_policy(q_policy)

# Convert the learned Q-table to a V-table
q_v_table = q_to_v_table(q_q_table)
# Visualize the state-value function learned by Q-learning
visualize_value_function(cliffwalk_env, q_v_table, "State-Value Function (V) for Q-learning")


# --- 6. Final Comparison Visualization ---

# Plot the learning curves of both algorithms on the same graph to compare performance.
plot_learning_curves(histories_for_plot)
