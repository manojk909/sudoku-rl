# runner/simulate.py

from env.sudoku_env import SudokuEnv
from agents.heuristic_agents import HeuristicAgent


def run_episode(puzzle):
    env = SudokuEnv(puzzle)
    agent = HeuristicAgent()

    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)
        state, reward, done = env.step(action)

        total_reward += reward

        if done:
            break
        print(f"Step {env.steps}, Reward: {reward}")
    return env, total_reward
    