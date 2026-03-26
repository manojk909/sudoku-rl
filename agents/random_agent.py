# agents/random_agent.py

import random


class RandomAgent:
    def act(self, state):
        return (
            random.randint(0, 8),
            random.randint(0, 8),
            random.randint(1, 9)
        )