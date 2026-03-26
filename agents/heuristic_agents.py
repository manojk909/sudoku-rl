# agents/heuristic_agents.py

import numpy as np
import random


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
        best_cell = None
        best_options = None

        for row in range(9):
            for col in range(9):
                if state[row][col] == 0:
                    options = self.get_valid_numbers(state, row, col)

                    if not options:
                        continue

                    if best_options is None or len(options) < len(best_options):
                        best_options = options
                        best_cell = (row, col)

        if best_cell:
            row, col = best_cell
            val = random.choice(best_options)
            return (row, col, val)

        # fallback
        return (
            random.randint(0, 8),
            random.randint(0, 8),
            random.randint(1, 9)
        )