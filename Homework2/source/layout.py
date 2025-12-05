"""
Copyright (C) 2025 Fu Tszkok

:module: layout
:function: Defines a collection of predefined cliffwalk layouts for the reinforcement learning environment.
:author: Fu Tszkok
:date: 2025-11-05
:license: AGPLv3 + Additional Restrictions (Non-Commercial Use)

This code is licensed under GNU Affero General Public License v3 (AGPLv3) with additional terms.
- Commercial use prohibited (including but not limited to sale, integration into commercial products)
- Academic use requires clear attribution in code comments or documentation

Full AGPLv3 text available in LICENSE file or at <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np

# A simple 3x3 cliff walk layout for basic testing and debugging.
cliffwalk1 = np.array([
    [2, 0, 4],
    [4, 0, 0],
    [0, 0, 3]
])

# A 5x5 layout that includes both walls (1) and cliffs (4).
# The goal is partially enclosed, requiring the agent to navigate around obstacles.
cliffwalk2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 4, 1, 1, 0],
    [0, 4, 3, 1, 0],
    [0, 4, 0, 1, 0],
    [2, 4, 0, 0, 0]
])

# The classic 4x12 Cliff Walk environment as described in Sutton and Barto's textbook.
# The entire bottom row between the start and goal is a cliff.
cliffwalk3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]
])

# A large and complex 21x21 layout combining a maze structure with numerous cliff tiles.
# This layout is suitable for testing the scalability and robustness of the learning algorithms in a more challenging and dangerous environment.
cliffwalk4 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 4, 0, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1],
    [1, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 4, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 4, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 4, 0, 1],
    [1, 4, 4, 0, 1, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 1, 1, 1, 0, 4, 0, 1, 1, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 1, 1, 1, 1, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 4, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1, 0, 4, 0, 1],
    [1, 4, 4, 0, 1, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1, 0, 4, 0, 1],
    [1, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 1],
    [1, 0, 4, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 4, 4, 1],
    [1, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 1],
    [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# A simple 4x11 maze layout for initialing cliffwalk environment.
maze3 = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]
])
