# evaluation/grader.py

import numpy as np


def evaluate_episode(env, total_reward):
    filled = np.count_nonzero(env.board)

    return {
        "filled_cells": filled,
        "completion_ratio": filled / 81,
        "invalid_moves": env.invalid_moves,
        "steps": env.steps,
        "total_reward": total_reward,
        "solved": filled == 81
    }


def compute_score(metrics):
    score = metrics["total_reward"]

    if metrics["solved"]:
        score += 100

    score += metrics["completion_ratio"] * 50
    score -= metrics["invalid_moves"] * 3
    score -= metrics["steps"] * 0.2

    return score