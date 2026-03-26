# utils/sudoku_utils.py

import numpy as np


def is_valid(board, row, col, val):
    # Row
    if val in board[row]:
        return False

    # Column
    if val in board[:, col]:
        return False

    # Subgrid
    r, c = (row // 3) * 3, (col // 3) * 3
    if val in board[r:r+3, c:c+3]:
        return False

    return True


def is_solved(board):
    return np.all(board != 0)


def get_empty_cells(board):
    empty = []
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                empty.append((i, j))
    return empty


def print_board(board):
    for i in range(9):
        if i % 3 == 0:
            print("-" * 25)

        for j in range(9):
            if j % 3 == 0:
                print("|", end=" ")

            val = board[i][j]
            print(val if val != 0 else ".", end=" ")

        print("|")

    print("-" * 25)

def get_valid_actions(board):
    actions = []

    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for val in range(1, 10):
                    if is_valid(board, row, col, val):
                        actions.append((row, col, val))

    return actions