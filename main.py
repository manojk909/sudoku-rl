# main.py

from runner.simulate import run_episode
from evaluation.grader import evaluate_episode, compute_score
from utils.sudoku_utils import print_board


# Example puzzle (medium)
puzzle = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],

    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],

    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]


env, total_reward = run_episode(puzzle)

metrics = evaluate_episode(env, total_reward)
score = compute_score(metrics)

print("\nFinal Board:")
print_board(env.board)

print("\nMetrics:", metrics)
print("Score:", score)