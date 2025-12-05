"""
Copyright (C) 2025 Fu Tszkok

:module: algorithm
:function: Implements model-free temporal difference learning algorithms (SARSA, Q-learning, etc.).
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import random
from collections import defaultdict, deque
from tqdm import tqdm


def choose_action_e_greedy(q_table, state, actions, epsilon):
    """Chooses an action using an epsilon-greedy policy.
    :param q_table: The Q-table mapping states to action-values.
    :param state: The current state.
    :param actions: A list of possible actions from the state.
    :param epsilon: The probability of choosing a random action (exploration rate).
    :return: The chosen action.
    """
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        return random.choice(actions)
    else:
        # Exploit: choose the best known action
        q_values = q_table[state]
        # Find the maximum Q-value
        max_q = max(q_values.values())
        # Get all actions that have the maximum Q-value (to handle ties)
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)


def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Solves the environment using the Q-learning algorithm (off-policy TD control).
    :param env: An instance of the environment.
    :param episodes: The total number of episodes to train for.
    :param alpha: The learning rate (float).
    :param gamma: The discount factor for future rewards (float).
    :param epsilon: The exploration rate for the epsilon-greedy policy (float).
    :return: A tuple (q_table, policy, history) where:
             - q_table is the learned Q-value function (dict).
             - policy is the optimal policy derived from the Q-table (dict).
             - history is a list of total rewards for each episode.
    """
    # Initialize Q-table with zeros for all state-action pairs
    q_table = defaultdict(lambda: {action: 0.0 for action in env.action_keys})
    history = []

    for _ in tqdm(range(episodes), desc="Q-learning"):
        state = env.start_pos
        done = False
        total_reward = 0

        while not done:
            # Choose action using the epsilon-greedy policy based on the current Q-table
            action = choose_action_e_greedy(q_table, state, env.action_keys, epsilon)

            # Take the action and observe the outcome
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Q-learning update rule
            old_value = q_table[state][action]
            # Find the best Q-value for the next state (greedy part)
            next_max = max(q_table[next_state].values())

            # Update Q-value: Q(s, a) <- Q(s, a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s, a)]
            td_target = reward + gamma * next_max
            td_error = td_target - old_value
            q_table[state][action] = old_value + alpha * td_error

            state = next_state

        history.append(total_reward)

    # Derive the final policy from the learned Q-table
    policy = {s: max(q_table[s], key=q_table[s].get) for s in q_table}
    return q_table, policy, history


def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Solves the environment using the SARSA algorithm (on-policy TD control).
    :param env: An instance of the environment.
    :param episodes: The total number of episodes to train for.
    :param alpha: The learning rate (float).
    :param gamma: The discount factor for future rewards (float).
    :param epsilon: The exploration rate for the epsilon-greedy policy (float).
    :return: A tuple (q_table, policy, history)
    """
    q_table = defaultdict(lambda: {action: 0.0 for action in env.action_keys})
    history = []

    for _ in tqdm(range(episodes), desc="SARSA"):
        state = env.start_pos
        done = False
        total_reward = 0

        # Choose the first action based on the policy
        action = choose_action_e_greedy(q_table, state, env.action_keys, epsilon)

        while not done:
            # Take action and observe outcome
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Choose the *next* action based on the policy for the *next* state
            next_action = choose_action_e_greedy(q_table, next_state, env.action_keys, epsilon)

            # SARSA update rule
            old_value = q_table[state][action]
            # Get the Q-value for the next state and the action chosen for it
            next_value = q_table[next_state][next_action]

            # Update Q-value: Q(s, a) <- Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]
            td_target = reward + gamma * next_value
            td_error = td_target - old_value
            q_table[state][action] = old_value + alpha * td_error

            # Move to the next state and action
            state = next_state
            action = next_action

        history.append(total_reward)

    policy = {s: max(q_table[s], key=q_table[s].get) for s in q_table}
    return q_table, policy, history


def expected_sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Solves the environment using the Expected SARSA algorithm.
    :param env: An instance of the environment.
    :param episodes: The total number of episodes to train for.
    :param alpha: The learning rate (float).
    :param gamma: The discount factor for future rewards (float).
    :param epsilon: The exploration rate for the epsilon-greedy policy (float).
    :return: A tuple (q_table, policy, history)
    """
    q_table = defaultdict(lambda: {action: 0.0 for action in env.action_keys})
    history = []

    for _ in tqdm(range(episodes), desc="Expected SARSA"):
        state = env.start_pos
        done = False
        total_reward = 0

        while not done:
            action = choose_action_e_greedy(q_table, state, env.action_keys, epsilon)
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Expected SARSA update rule
            old_value = q_table[state][action]

            # Calculate the expected Q-value of the next state
            next_q_values = q_table[next_state]
            best_action = max(next_q_values, key=next_q_values.get)
            num_actions = len(env.action_keys)

            expected_q = 0.0
            for a in env.action_keys:
                if a == best_action:
                    prob = 1 - epsilon + (epsilon / num_actions)
                else:
                    prob = epsilon / num_actions
                expected_q += prob * next_q_values[a]

            td_target = reward + gamma * expected_q
            td_error = td_target - old_value
            q_table[state][action] = old_value + alpha * td_error

            state = next_state

        history.append(total_reward)

    policy = {s: max(q_table[s], key=q_table[s].get) for s in q_table}
    return q_table, policy, history


def n_step_sarsa(env, n_steps=5, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1, max_steps_per_episode=1000):
    """Solves the environment using n-step SARSA, strictly following the TD update rule.
    :param env: An instance of the environment.
    :param n_steps: The number of steps 'n' for the lookahead return.
    :param episodes: The total number of episodes to train for.
    :param alpha: The learning rate (float).
    :param gamma: The discount factor for future rewards (float).
    :param epsilon: The exploration rate for the epsilon-greedy policy (float).
    :param max_steps_per_episode: A safeguard to prevent infinitely long episodes in complex mazes.
    :return: A tuple (q_table, policy, history)
    """
    q_table = defaultdict(lambda: {action: 0.0 for action in env.action_keys})
    history = []

    for _ in tqdm(range(episodes), desc=f"{n_steps}-step SARSA"):
        state = env.start_pos
        done = False
        total_reward = 0

        # Use a deque for efficient storage of the recent trajectory
        trajectory = deque()

        # Choose initial action A_0 from S_0
        action = choose_action_e_greedy(q_table, state, env.action_keys, epsilon)

        step_counter = 0
        while not done and step_counter < max_steps_per_episode:
            # Take action A_t, observe R_t+1 and S_t+1
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            # Store the experience (S_t, A_t, R_t+1)
            trajectory.append((state, action, reward))

            # Choose next action A_t+1 from S_t+1
            if not done:
                next_action = choose_action_e_greedy(q_table, next_state, env.action_keys, epsilon)
            else:
                # No next action if the episode terminates
                next_action = None

                # If we have collected at least n steps, we can update the Q-value for the state that was n steps ago.
            if len(trajectory) >= n_steps:
                # The state to update is S_tau, which is the oldest in our trajectory buffer
                s_tau, a_tau, _ = trajectory[0]

                # G = R_t+1 + gamma*R_t+2 + ...
                G = 0
                for i in range(n_steps):
                    # Get reward R from (S, A, R) tuple
                    r = trajectory[i][2]
                    G += (gamma ** i) * r

                # Add the bootstrap value if the episode didn't end within n steps
                # G = G + gamma^n * q(S_t+n, A_t+n)
                if not done:
                    # S_t+n is the current `next_state`
                    # A_t+n is the current `next_action`
                    G += (gamma ** n_steps) * q_table[next_state][next_action]

                # N-step SARSA update rule
                old_value = q_table[s_tau][a_tau]
                td_error = G - old_value
                q_table[s_tau][a_tau] = old_value + alpha * td_error

                # Remove the oldest experience as it has now been used for an update
                trajectory.popleft()

            # Update state and action for the next iteration (Line 8 of pseudocode)
            state = next_state
            action = next_action
            step_counter += 1

        # After the episode ends, there are still (n-1) states in the trajectory
        # that need to be updated. Their returns are calculated until the end of the episode.
        while len(trajectory) > 0:
            s_tau, a_tau, _ = trajectory[0]

            # Calculate the return G for the remaining items. Since the episode is over,
            # there is no bootstrapping term.
            G = 0
            for i in range(len(trajectory)):
                r = trajectory[i][2]
                G += (gamma ** i) * r

            # Perform the update
            old_value = q_table[s_tau][a_tau]
            td_error = G - old_value
            q_table[s_tau][a_tau] = old_value + alpha * td_error

            # Remove the experience and continue until the buffer is empty
            trajectory.popleft()

        history.append(total_reward)

    policy = {s: max(q_table[s], key=q_table[s].get) for s in q_table}
    return q_table, policy, history
