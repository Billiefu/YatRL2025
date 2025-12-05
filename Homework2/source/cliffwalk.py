"""
Copyright (C) 2025 Fu Tszkok

:module: cliffwalk
:function: Defines the Cliff Walk environment by inheriting from the base MazeEnvironment.
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np

from Homework1.source.maze import MazeEnvironment
from layout import cliffwalk3


class CliffWalkEnv(MazeEnvironment):
    """Implements the Cliff Walk environment from Sutton and Barto's 'Reinforcement Learning'."""

    def __init__(self, cliffwalk=None, reward_step=-1.0, reward_cliff=-100.0, reward_goal=0.0):
        """Initializes the Cliff Walk environment.
        :param cliffwalk: A 2D NumPy array or list of lists defining the cliffwalk layout.
                          (0: path, 1: wall, 2: start, 3: goal, 4:cliff). If None, a default cliffwalk is used.
        :param reward_step: Reward for any step that is not into the cliff or goal. Default is -1.
        :param reward_cliff: Reward for stepping into the cliff. Default is -100.
        :param reward_goal: Reward for reaching the goal. Default is 0.
        """
        # If no custom cliffwalk is provided, use a default layout.
        if cliffwalk is None:
            cliffwalk = cliffwalk3

        # Store the specific reward for falling off the cliff.
        self.reward_cliff = reward_cliff

        # Initialize the parent class with the provided cliffwalk and standard rewards.
        # The parent will handle walls, boundaries, and goal rewards.
        super().__init__(maze=cliffwalk, reward_step=reward_step, reward_goal=reward_goal)

    def step(self, current_pos, action):
        """Executes an action and overrides the parent's step method to handle the cliff.
        :param current_pos: The agent's current state as a tuple (row, col).
        :param action: The action to be taken (e.g., 'N', 'S', 'W', 'E').
        :return: A tuple (next_pos, reward, done).
        """
        # If the agent is already at the goal, it stays there with zero reward.
        if current_pos == self.goal_pos:
            return self.goal_pos, 0, True

        move = self.actions.get(action)
        if move is None:
            raise ValueError(f"Invalid action: {action}")

        # Calculate the potential next position.
        next_pos_theoretical = (current_pos[0] + move[0], current_pos[1] + move[1])

        # Check if the theoretical next position is valid and within bounds first.
        if 0 <= next_pos_theoretical[0] < self.height and 0 <= next_pos_theoretical[1] < self.width:
            # Now check if the valid position is a cliff tile.
            if self.maze[next_pos_theoretical] == 4:
                # Agent fell off the cliff!
                reward = self.reward_cliff
                next_pos = self.start_pos  # Reset to start
                done = False
                return next_pos, reward, done

        # If not a cliff, use the parent's logic
        return super().step(current_pos, action)

    def render(self):
        """Overrides the parent's render method to display cliff tiles."""
        symbols = {0: '⬜', 1: '⬛', 2: '✳️', 3: '✅', 4: '🔥'}
        for r in range(self.height):
            for c in range(self.width):
                print(symbols[self.maze[r, c]], end="")
            print()
        print()

    def print_policy(self, policy):
        """Overrides the parent's policy printing to also show the cliff.
        :param policy: A dictionary mapping states (tuples) to actions (strings).
        """
        policy_symbols = {'N': '🔼', 'S': '🔽', 'W': '◀️', 'E': '▶️'}
        # Use dtype=object to prevent NumPy from truncating emoji characters.
        policy_grid = np.full(self.maze.shape, ' ', dtype=object)

        # Populate the grid with policy actions
        for state, action in policy.items():
            if self.maze[state] != 1 and self.maze[state] != 3:
                policy_grid[state] = policy_symbols[action]

        # Overlay symbols for walls, cliff, and start position
        for r in range(self.height):
            for c in range(self.width):
                if self.maze[r, c] == 1:
                    policy_grid[r, c] = '⬛'
                elif self.maze[r, c] == 2:
                    policy_grid[r, c] = '✳️'
                elif self.maze[r, c] == 3:
                    policy_grid[r, c] = '✅'
                # This handles the cliff in the child class
                elif hasattr(self, 'reward_cliff') and self.maze[r, c] == 4:
                    policy_grid[r, c] = '🔥'

        # Print the final policy grid row by row.
        for r in range(self.height):
            for c in range(self.width):
                print(policy_grid[r, c], end="")
            print()
        print()
