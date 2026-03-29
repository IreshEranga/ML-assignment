# 🔋 Smart Grid Energy Management via Reinforcement Learning

> Machine Learning Module Assignment — Group Project  
> **Algorithm:** Q-Learning + SARSA (Tabular RL, No Deep Learning)  
> **Dataset:** [Energy Consumption, Generation, Prices & Weather — Spain (Kaggle)](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Team Members](#team-members)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Dataset Download](#dataset-download)
6. [How to Run](#how-to-run)
7. [Algorithms](#algorithms)
8. [Results Summary](#results-summary)
9. [Submission Checklist](#submission-checklist)

---

## 🎯 Project Overview

We train a **Reinforcement Learning agent** to operate a smart grid energy storage system. The agent observes real electricity market conditions from Spain (2015–2018) and decides **when to buy, sell, or store energy** to minimise cost while maintaining grid stability.

| Component | Detail |
|---|---|
| **Problem** | Energy arbitrage + grid stability in smart grid |
| **Dataset** | Spain electricity market — 35,064 hourly records |
| **Language** | Python 3.8+ |
| **Environment** | Jupyter Notebook |
| **Algorithm 1** | Q-Learning (off-policy TD control) |
| **Algorithm 2** | SARSA (on-policy TD control) |
| **No deep learning** | Pure tabular RL only |

---

## 👥 Team Members

See `members.txt` for student IDs and emails.

---

## 📁 Repository Structure

```
smartgrid_rl/
│
├── 1_qlearning_smartgrid.ipynb    ← Q-Learning: full implementation
├── 2_sarsa_smartgrid.ipynb        ← SARSA: full implementation + comparison
│
├── energy_dataset.csv             ← [DOWNLOAD FROM KAGGLE — see below]
├── weather_features.csv           ← [DOWNLOAD FROM KAGGLE — see below]
│
├── members.txt                    ← Student IDs and emails
├── submission.txt                 ← Dataset + GitHub + YouTube links
├── README.md                      ← This file
└── report.pdf                     ← Full PDF report
```

> **Generated outputs** (created when notebooks are run):
> - `eda_plots.png` — Exploratory data analysis
> - `ql_training.png` — Q-Learning training curves
> - `ql_evaluation.png` — Q-Learning test evaluation
> - `ql_qtable.png` — Q-table heatmap
> - `sarsa_training.png` — SARSA training curves
> - `comparison_ql_vs_sarsa.png` — Side-by-side comparison
> - `comparison_metrics.csv` — Results table

---

## ⚙️ Environment Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/smartgrid-rl.git
cd smartgrid-rl
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# OR using conda
conda create -n smartgrid python=3.9
conda activate smartgrid
```

### Step 3 — Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn jupyter notebook
```

**Full dependency list:**

| Package | Version (tested) | Purpose |
|---|---|---|
| `numpy` | ≥ 1.21 | Array operations, Q-table |
| `pandas` | ≥ 1.3 | Dataset loading and preprocessing |
| `matplotlib` | ≥ 3.4 | Plots and charts |
| `seaborn` | ≥ 0.11 | Heatmaps and styled plots |
| `jupyter` | ≥ 1.0 | Notebook environment |

---

## 📥 Dataset Download

The notebooks require two CSV files from Kaggle.

### Instructions

1. Go to: **https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather**
2. Click **Download** (you need a free Kaggle account)
3. Unzip the downloaded archive
4. Copy these two files into the **same folder** as the notebooks:
   - `energy_dataset.csv`
   - `weather_features.csv`

### Dataset Description

| File | Rows | Key Columns |
|---|---|---|
| `energy_dataset.csv` | 35,064 | `price actual`, `total load actual`, `generation solar`, `generation wind onshore` |
| `weather_features.csv` | ~175,000 | `temp`, `wind_speed`, `clouds_all` (5 Spanish cities) |

**Time range:** Jan 2015 – Dec 2018  
**Frequency:** Hourly

---

## ▶️ How to Run

### Launch Jupyter Notebook

```bash
jupyter notebook
```

This opens a browser. Navigate to the project folder.

### Run Order

```
Step 1 → Open: 1_qlearning_smartgrid.ipynb
         Run all cells: Kernel → Restart & Run All

Step 2 → Open: 2_sarsa_smartgrid.ipynb
         Run all cells: Kernel → Restart & Run All
```

> **Note:** Each notebook is **self-contained** and can be run independently. Notebook 2 re-trains Q-Learning internally for a fair comparison.

### Expected Runtime

| Notebook | Approx. Runtime |
|---|---|
| Q-Learning (200 episodes) | 3–8 minutes |
| SARSA (200 episodes) | 3–8 minutes |

Runtime depends on CPU speed. Both run entirely on CPU — no GPU required.

### Adjustable Hyperparameters

Find these near the top of the training section in each notebook:

```python
N_EPISODES    = 200     # Number of training episodes (increase for better convergence)
alpha         = 0.10    # Learning rate
gamma         = 0.95    # Discount factor
epsilon       = 1.0     # Initial exploration rate
epsilon_min   = 0.05    # Minimum exploration rate
epsilon_decay = 0.995   # Decay per episode
```

---

## 🤖 Algorithms

### RL Environment Design

**State Space** — Discretised into 180 states (4 × 3 × 5 × 3):

| Dimension | Values | Description |
|---|---|---|
| Hour Bin | 4 | Night / Morning / Afternoon / Evening |
| Price Bin | 3 | Low / Medium / High |
| Battery Bin | 5 | 0–20% / 20–40% / 40–60% / 60–80% / 80–100% |
| Net Load Bin | 3 | Low / Medium / High |

**Action Space:**

| Action | Code | Effect |
|---|---|---|
| Buy | 0 | Purchase 10 MWh from grid; charge battery |
| Sell | 1 | Discharge 10 MWh from battery; sell to grid |
| Idle | 2 | No energy transaction |

**Reward Function:**
```
reward = net_profit − stability_penalty
       = (sell_revenue − buy_cost) − 0.05 × net_load
```

---

### Algorithm 1: Q-Learning (Off-Policy)

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s',a') − Q(s,a) ]
```

- **Off-policy**: target uses the *best* possible next action
- Learns the optimal policy regardless of exploration behaviour
- Tends to be more aggressive in energy trading

---

### Algorithm 2: SARSA (On-Policy)

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α [ r + γ · Q(s', a') − Q(s,a) ]
```
where `a'` is the **actual action** taken by the ε-greedy policy.

- **On-policy**: target reflects what the agent actually does
- Safer, penalises risky exploration paths during learning
- Better suited for real-world live deployment

---

## 📊 Results Summary

*(Exact values generated when notebooks are run)*

| Metric | Q-Learning | SARSA |
|---|---|---|
| Total Reward (test) | See notebook | See notebook |
| Mean Reward/step | See notebook | See notebook |
| Training Episodes | 200 | 200 |
| State Space Size | 180 states | 180 states |
| Q-table Entries | 540 (180×3) | 540 (180×3) |

**Key insight:** Q-Learning maximises expected reward aggressively; SARSA is more conservative and safer for critical infrastructure — important for real smart grid deployments.

---

## ✅ Submission Checklist

- [x] `members.txt` — Student IDs and emails
- [x] `submission.txt` — Dataset link, GitHub link, YouTube link
- [x] `1_qlearning_smartgrid.ipynb` — Q-Learning notebook with results
- [x] `2_sarsa_smartgrid.ipynb` — SARSA notebook with comparison
- [x] `report.pdf` — Full report with appendix (source code as text)
- [x] GitHub repository with detailed commit history
- [x] YouTube demo video (≤ 20 minutes, ≤ 4 min per member)

---

## 📚 References

1. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Jha, N. (2019). Energy Consumption, Generation, Prices and Weather Dataset. Kaggle. https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
3. Watkins, C.J.C.H. & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279–292.
4. Rummery, G.A. & Niranjan, M. (1994). On-line Q-learning using connectionist systems. *Technical Report CUED/F-INFENG/TR 166*, Cambridge University.

---

*Submitted for Machine Learning module — Group Assignment*
