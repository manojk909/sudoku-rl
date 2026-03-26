# env/sudoku_env.py

import numpy as np
from configs.config import *
from utils.sudoku_utils import is_valid, is_solved


class SudokuEnv:
    def __init__(self, puzzle):
        self.original = np.array(puzzle)
        self.board = self.original.copy()

        self.steps = 0
        self.invalid_moves = 0

    def reset(self):
        self.board = self.original.copy()
        self.steps = 0
        self.invalid_moves = 0
        return self.board

    def step(self, action):
        row, col, val = action
        reward = STEP_PENALTY
        done = False

        self.steps += 1

        # Already filled cell
        if self.board[row][col] != 0:
            self.invalid_moves += 1
            return self.board, REWARD_INVALID, False

        # Valid move
        if is_valid(self.board, row, col, val):
            self.board[row][col] = val
            reward += REWARD_VALID
        else:
            self.invalid_moves += 1
            reward += REWARD_CONSTRAINT_VIOLATION

        # Check solved
        if is_solved(self.board):
            reward += REWARD_COMPLETE
            done = True

        # Termination
        if self.steps >= MAX_STEPS or self.invalid_moves >= MAX_INVALID_MOVES:
            done = True

        return self.board, reward, done