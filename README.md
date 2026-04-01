# 🌧️ NADI Lite — Urban Sewer Overflow Management

> **OpenEnv-compliant Reinforcement Learning Environment**
> Built for the Meta × Hugging Face OpenEnv Hackathon 2026

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.1-blue)](https://meta-pytorch.org/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Motivation

Urban India faces a growing crisis: every monsoon season, aging drainage infrastructure is overwhelmed by rainfall, causing sewer overflows that flood streets, contaminate water supplies, and disrupt millions of lives. Municipal corporations must decide — with limited budgets and crews — which drains to clean, when to deploy extra capacity, and when to issue public alerts.

**NADI Lite** (Named for the Hindi word for river/drain) simulates this real-world challenge as a Reinforcement Learning environment. An AI agent learns to monitor dozens of city wards, predict overflow risk from rainfall and blockage data, and take targeted interventions to prevent overflow cascades.

This environment is grounded in real-world patterns from Indian meteorological data and urban census statistics, making it a meaningful testbed for agentic decision-making under resource and time constraints.

---

## 🏙️ Environment Description

The agent manages a city divided into **N wards** (1 to 10 depending on task difficulty). Each ward has:

- A drainage network with a fixed capacity (L/s)
- A current blockage level (0–100%)
- Live rainfall data (affected by seasonal monsoon patterns)
- A maintenance history and public complaint frequency

Each timestep, the agent observes conditions across all wards and takes one action. The simulation advances: rainfall increases blockage, blockage raises overflow risk, and overflow triggers penalties. The agent's goal is to **minimize overflow events** over the episode.

### Physics Model

```
effective_capacity = drain_capacity × (1 - blockage/100)
overflow_risk      = (rainfall × area) / effective_capacity
overflow           = True if risk ≥ 0.75 AND blockage > 50
```

Blockage evolves each step:
```
blockage[t+1] = blockage[t] + rainfall × 0.04 − natural_decay
```

---

## 📊 Observation Space

Each observation contains a list of per-ward snapshots plus global episode state.

### Per-Ward Fields

| Field | Type | Range | Description |
|---|---|---|---|
| `ward_id` | int | 0–N | Ward identifier |
| `ward_name` | str | — | Human-readable name |
| `rainfall` | float | 0–200 mm | Current rainfall |
| `population_density` | float | 1000–50000 | Persons per km² |
| `drain_capacity` | float | 50–500 L/s | Maximum drain throughput |
| `blockage_level` | float | 0–100% | Current blockage percentage |
| `last_cleaned_days` | int | ≥0 | Days since last cleaning |
| `overflow_risk` | float | 0–1 | Computed risk score |
| `is_overflowing` | bool | — | Active overflow flag |
| `complaint_frequency` | float | 0–15 | Avg complaints/week |

### Global Fields

| Field | Type | Description |
|---|---|---|
| `step` | int | Current timestep |
| `total_overflow_events` | int | Cumulative overflows this episode |
| `budget_remaining` | float\|null | Remaining budget (hard task only) |
| `done` | bool | Episode complete flag |
| `reward` | float | Reward received this step |

---

## ⚡ Action Space

Each step the agent issues **one action** targeting a specific ward:

| Action | Effect | Cost (budget) |
|---|---|---|
| `clean_drain` | Reduce blockage by 30 × intensity | 10 units |
| `do_nothing` | No change | 0 units |
| `deploy_extra_capacity` | Boost drain throughput ×1.5 for 3 steps | 8 units |
| `issue_alert` | Halve complaint escalation rate | 2 units |

**Action parameters:**
```json
{
  "action_type": "clean_drain",
  "ward_id": 3,
  "intensity": 1.0
}
```
The `intensity` field (0.1–2.0) scales both the effect and the cost of the action.

---

## 🏆 Reward Design

NADI uses a **dense, multi-signal reward** to give the agent clear learning signals at every step:

| Signal | Value | Condition |
|---|---|---|
| Overflow penalty | **−10** | Per ward overflowing this step |
| Prevention bonus | **+3** | Per ward at risk but not overflowing |
| Blockage reduction | **+1** | Per ward with decreased blockage |
| Unnecessary action | **−2** | Cleaning a ward with <15% blockage |
| Delay penalty | **−1** | High-risk ward ignored 2+ consecutive steps |
| Budget exceeded | **−5** | Budget goes negative |
| Partial progress | **+0.5×Δrisk** | Total city-wide risk reduction |

The reward breakdown is available in every observation for interpretability.

---

## 📋 Task Descriptions

### 🟢 Task 1 — Easy
- **1 ward**, static rainfall, unlimited budget
- **30 steps**
- Learn to keep a single drain clear against steady rainfall
- **Score:** `1 − (overflow_events / max_steps)`

### 🟡 Task 2 — Medium
- **5 wards**, limited to 2 actions per step, static rainfall
- **40 steps**
- Prioritize resource allocation across multiple wards
- **Score:** `0.7 × prevention_rate + 0.3 × blockage_score`

### 🔴 Task 3 — Hard
- **10 wards**, dynamic seasonal rainfall, budget = 300 units
- **50 steps**, delayed capacity deployment effects (3-step duration)
- Balance budget, timing, and triage under monsoon pressure
- **Score:** `0.5 × prevention + 0.25 × budget_efficiency + 0.25 × blockage_score`

All scores are in **[0.0, 1.0]** and are **deterministic** for a given seed.

---

## 🗂️ Project Structure

```
nadi-openenv/
├── inference.py             # Baseline LLM agent script (MANDATORY)
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # Container for HF Spaces
├── requirements.txt
├── README.md
│
├── env/
│   ├── __init__.py
│   ├── models.py            # Pydantic: NadiObservation, NadiAction, NadiState
│   ├── environment.py       # Core: reset(), step(), state
│   └── reward.py            # Dense reward function
│
├── tasks/
│   ├── __init__.py
│   └── tasks.py             # Task configs + graders (easy/medium/hard)
│
├── data/
│   ├── __init__.py
│   ├── generator.py         # Synthetic ward data pipeline
│   └── wards.json           # Pre-generated ward profiles (seed=42)
│
├── server/
│   ├── __init__.py
│   └── app.py               # FastAPI HTTP server
│
├── graders/
│   └── __init__.py
│
└── simulation/
    └── __init__.py
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- Docker (for deployment)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/nadi-openenv
cd nadi-openenv

# Install dependencies
pip install -r requirements.txt

# Pre-generate ward data
python -c "from data.generator import generate_wards; generate_wards(20, 42, 'data/wards.json')"

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Verify it's running
curl http://localhost:7860/health
```

### Run the Baseline Inference

```bash
# With an LLM backend
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
python inference.py

# Heuristic baseline (no API key needed)
python inference.py
```

### Docker Build & Run

```bash
# Build image
docker build -t nadi-lite .

# Run locally
docker run -p 7860:7860 \
  -e OPENAI_API_KEY="sk-..." \
  nadi-lite

# Test endpoints
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'
```

---

## 🌐 API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| GET | `/health` | — | Health check (returns 200) |
| GET | `/` | — | Environment info |
| POST | `/reset` | `{"task": "easy", "seed": 42}` | Reset episode |
| POST | `/step` | `{"action": {...}}` | Take an action |
| GET | `/state` | — | Internal state |
| GET | `/tasks` | — | List all tasks |
| POST | `/grade/{task}` | — | Get episode score |

### Example Workflow

```python
import httpx

BASE = "http://localhost:7860"

# 1. Reset
obs = httpx.post(f"{BASE}/reset", json={"task": "medium", "seed": 42}).json()

# 2. Step loop
while not obs["done"]:
    action = {
        "action_type": "clean_drain",
        "ward_id": obs["wards"][0]["ward_id"],
        "intensity": 1.0
    }
    obs = httpx.post(f"{BASE}/step", json={"action": action}).json()

# 3. Grade
score = httpx.post(f"{BASE}/grade/medium").json()
print(f"Score: {score['score']}")
```

---

## 🤖 Hugging Face Spaces Deployment

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space (Docker SDK)
huggingface-cli repo create nadi-lite --type space --space-sdk docker

# Push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/nadi-lite
git push hf main
```

**Space settings required:**
- SDK: Docker
- Hardware: CPU Basic (2 vCPU, 16 GB RAM)
- Port: 7860

**Environment variables to set in Space settings:**
```
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=<your-key>
HF_TOKEN=<your-hf-token>
```

---

## 📈 Baseline Results

Results from the heuristic baseline agent (no LLM, rule-based fallback):

| Task | Score | Total Reward | Overflow Events | Steps |
|---|---|---|---|---|
| Easy | ~0.87 | ~+45 | ~4 | 30 |
| Medium | ~0.72 | ~+110 | ~28 | 40 |
| Hard | ~0.58 | ~+180 | ~105 | 50 |

*Results are reproducible with `seed=42`. A well-trained RL agent or capable LLM is expected to achieve 0.90+ on Easy and 0.75+ on Medium.*

---

## 🔬 Data Design

Ward profiles are generated with `seed=42` for full reproducibility:

| Parameter | Range | Distribution |
|---|---|---|
| Rainfall (coastal) | 40–200 mm | Normal(120, 40) + seasonal |
| Rainfall (inland) | 10–110 mm | Normal(60, 25) + seasonal |
| Population density | 1,000–50,000 /km² | Uniform by category |
| Drain capacity | 50–500 L/s | Normal, inversely correlated with density |
| Baseline blockage | 0–100% | Derived from age + density |
| Last cleaned | 0–90 days | Uniform |

Seasonal rainfall follows a sinusoidal pattern peaking at step 90 (monsoon peak), giving the hard task meaningful temporal dynamics.

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

- Meta × Hugging Face OpenEnv team for the framework and hackathon
- India Meteorological Department for rainfall distribution inspiration
- Census of India for population density statistics
