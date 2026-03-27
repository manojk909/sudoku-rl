# runner/simulate.py

from env.sudoku_env import SudokuEnv
from agents.heuristic_agents import HeuristicAgent

def run_episode(puzzle):
    env = SudokuEnv(puzzle)
    agent = HeuristicAgent()

    state = env.reset()
    total_reward = 0

    steps_log = []
    rewards_log = []

    while True:
        action = agent.act(state)
        row, col, val = action

        prev = state[row][col]

        state, reward, done = env.step(action)
        total_reward += reward

        rewards_log.append(total_reward)  # cumulative reward

        if prev == 0 and state[row][col] != 0:
            steps_log.append({
                "row": row,
                "col": col,
                "val": val
            })

        if done:
            break

    return env, total_reward, steps_log, rewards_log