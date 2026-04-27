# Serverless Prewarming with Contextual Bandits

Adaptive instance prewarming for serverless platforms using contextual bandit algorithms. The project simulates a serverless environment with cold-start latency, SLA constraints, and infrastructure cost, then compares learning-based prewarming policies (LinUCB, Linear Thompson Sampling, Epsilon-Greedy Linear) against static baselines across five synthetic workload patterns.

## Overview

In serverless computing, requests that arrive when no warm instance is available trigger a *cold start* — a significant latency penalty. Prewarming instances mitigates this but incurs cost. This project frames the prewarming decision as a contextual bandit problem: at each time step, an agent observes recent workload context and selects how many instances to keep warm (0, 1, 2, or 4), receiving a reward that balances latency, SLA violations, cold starts, and cost.

## Project Structure

```
├── main.py                          # Entry point — runs experiments and saves results
├── serverless_bandits/
│   ├── config.py                    # Environment and experiment configuration
│   ├── environment.py               # Serverless simulator (latency, cold starts, SLA, cost)
│   ├── policies.py                  # Bandit policies (LinUCB, LTS, Eps-Greedy) and baselines
│   ├── workloads.py                 # Synthetic workload generators
│   ├── runner.py                    # Experiment loop, aggregation, and summary formatting
│   ├── visualize.py                 # Plotting utilities
│   └── proposal_figures.py          # Figure generation for the report
├── results/
│   ├── summary.csv                  # Aggregated experiment results
│   └── proposal_figures/            # Generated figures (reward, latency, cost)
├── report.tex                       # Project report (LaTeX source)
└── report.pdf                       # Compiled report
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

Run the full experiment with default settings (240 time steps, 3 seeds, all 5 workloads):

```bash
python main.py
```

Customize the experiment:

```bash
# Specific workloads and longer horizon
python main.py --horizon 500 --workloads bursty abrupt_shift periodic_spikes

# Custom seeds and output path
python main.py --seeds 42 99 123 --output results/my_experiment.csv
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--horizon` | 240 | Number of time steps per run |
| `--seeds` | 7 11 19 | Random seeds for repeated experiments |
| `--workloads` | all five | Workload scenarios to evaluate |
| `--output` | `results/summary.csv` | Output path for the results CSV |

## Results

After running, a summary table is printed to the console and saved to CSV. The key metrics reported for each policy–workload pair are: cumulative reward, average latency, P95 latency, SLA violation rate, total cold starts, total cost, and cumulative regret.

To regenerate the report figures:

```bash
python -c "from serverless_bandits.proposal_figures import generate_proposal_figures; generate_proposal_figures()"
```

## Authors

Sumaiya Azad (u1494915), Rabeya Hossain (u1591808)
