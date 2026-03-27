# agents/heuristic_agents.py

import numpy as np
import random

from utils.sudoku_utils import get_valid_actions, is_valid

class HeuristicAgent:

    def get_valid_numbers(self, board, row, col):
        possible = set(range(1, 10))

        # Remove row values
        possible -= set(board[row])

        # Remove column values
        possible -= set(board[:, col])

        # Remove subgrid
        r, c = (row // 3) * 3, (col // 3) * 3
        possible -= set(board[r:r+3, c:c+3].flatten())

        return list(possible)

    def act(self, state):
        valid_actions = get_valid_actions(state)

        if not valid_actions:
            # fallback (rare case)
            return (
                random.randint(0, 8),
                random.randint(0, 8),
                random.randint(1, 9)
            )

        # Smart choice: pick action from most constrained cell
        best_action = None
        min_options = float('inf')

        for (r, c, v) in valid_actions:
            # count how many options this cell has
            options = [
                val for val in range(1, 10)
                if is_valid(state, r, c, val)
            ]

            if len(options) < min_options:
                min_options = len(options)
                best_action = (r, c, v)

        return best_action