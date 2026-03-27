from flask import Flask, render_template, jsonify
from runner.simulate import run_episode
from utils.sudoku_utils import print_board
import numpy as np

app = Flask(__name__)

# Example puzzle
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run")
def run_simulation():
    try:
        env, total_reward, steps = run_episode(puzzle)
        return jsonify({
            "steps": steps,
            "reward": float(total_reward),
            "reward_history": rewards,
            "steps_count": int(env.steps),
            "invalid_moves": int(env.invalid_moves)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)