# Serverless Prewarming with Contextual Bandits

This project simulates a serverless computing environment and uses AI-based decision policies to choose how many warm instances should be kept ready in advance. The main objective is to reduce cold starts and latency while still keeping infrastructure cost under control.

The project is built as an experiment framework. It generates request workloads, simulates serverless behavior under those workloads, runs multiple policies on the same scenarios, and compares their performance using latency, SLA, cold-start, reward, regret, and cost metrics.

## Project Goal

In serverless systems, requests may arrive when no warm instance is ready. This causes a cold start, which increases response time and can lead to SLA violations. Prewarming extra instances helps reduce this problem, but it also increases cost. This project studies that tradeoff and asks:

"Can an AI-based policy learn how many instances to prewarm based on recent traffic patterns?"

## How the Project Works

The full workflow is:

1. Generate a workload trace representing incoming requests over time.
2. Build a context vector from recent system history.
3. Let a policy choose a prewarming action.
4. Simulate the environment for that action and compute latency, cold starts, SLA violations, and cost.
5. Convert the outcome into a reward signal.
6. Update the learning policy using the observed reward.
7. Repeat for the full horizon and compare all policies across workloads and seeds.

Each time step represents one decision window. In the current setup, the action space is:

- `0` warm instances
- `1` warm instance
- `2` warm instances
- `4` warm instances

## Project Structure

### `main.py`

This is the entry point of the project.

It:

- Parses command-line arguments such as horizon, seeds, workloads, and output path
- Builds the experiment and environment configurations
- Loads the default policies
- Runs all experiments
- Prints a formatted summary
- Saves the aggregated output to `results/summary.csv`

### `serverless_bandits/config.py`

This file contains the main configuration classes:

- `EnvironmentConfig` defines system-level settings such as:
  - action space
  - requests per instance
  - base latency
  - cold-start latency
  - SLA threshold
  - warm-instance cost
  - reward weights
- `ExperimentConfig` defines:
  - simulation horizon
  - random seeds
  - workloads to evaluate

These settings make the simulator easy to tune without changing the rest of the code.

### `serverless_bandits/workloads.py`

This module generates synthetic traffic patterns used for testing.

Implemented workloads:

- `steady_low`: low and stable request traffic
- `steady_medium`: medium and stable traffic
- `bursty`: mostly moderate demand with random bursts
- `periodic_spikes`: repeating traffic spikes with periodic behavior
- `abrupt_shift`: sudden change from low demand to high demand

These workloads are important because they test whether a policy can adapt to both simple and highly dynamic conditions.

### `serverless_bandits/environment.py`

This module simulates the serverless platform itself.

For each time step, the environment:

- Receives the incoming request demand
- Receives the selected prewarming action
- Computes available warm capacity
- Calculates overflow demand that must be handled as cold starts
- Computes queue penalty, mean latency, and p95 latency
- Calculates SLA violations
- Calculates warm-instance cost
- Produces a reward value

The environment also tracks recent history and exposes it as a context vector for the policy.

### `serverless_bandits/policies.py`

This file defines all decision policies used in the experiment.

Baseline policies:

- `always_0`: never prewarm
- `always_1`: always keep 1 warm instance
- `always_2`: always keep 2 warm instances
- `threshold`: rule-based logic using recent demand and burstiness

AI / learning-based policies:

- `linucb`: contextual bandit using upper confidence bounds
- `linear_thompson_sampling`: contextual bandit using probabilistic sampling
- `epsilon_greedy_linear`: contextual bandit with exploration through epsilon-greedy action selection

These policies use the same environment but make different action decisions, allowing fair comparison.

### `serverless_bandits/runner.py`

This module manages the experiment loop.

It:

- Runs one full episode for a chosen workload, seed, and policy
- Repeats experiments across all workloads and seeds
- Collects metrics from every run
- Aggregates results using mean and standard deviation
- Formats the final summary for printing

### `results/summary.csv`

This file stores the aggregated experiment results. It is the main output artifact of the project and can be used for analysis, charts, or presentation slides.

## Context Used by the AI Policies

The learning policies do not act randomly. They use a context vector built from recent system behavior. The context includes:

- latest request count
- moving average of requests
- moving standard deviation of requests
- burstiness
- demand trend
- latest latency
- average recent latency
- latest cold starts
- average recent cold starts
- previous SLA violations
- time-phase features using sine and cosine

This helps the model understand both short-term workload behavior and time-based patterns.

## Reward Design

The environment converts system performance into a single reward value. The reward is negative when performance is poor or cost is high.

The reward penalizes:

- higher latency
- SLA violations
- cold starts
- warm-instance cost

This makes the learning problem practical: the policy is encouraged to find a balance between performance and cost rather than optimizing only one metric.

## Metrics Used for Evaluation

Each policy is evaluated using:

- average latency
- p95 latency
- SLA violation rate
- total cold starts
- total cost
- cumulative reward
- cumulative regret

These metrics show both system quality and the efficiency of the decision policy.

## End-to-End Execution Flow

When the project runs:

1. `main.py` creates configs and loads policies.
2. For each workload and random seed, the runner creates a workload trace.
3. The environment starts with empty history.
4. At every time step, the environment builds the context.
5. The policy selects an action from `{0, 1, 2, 4}`.
6. The environment simulates the outcome and returns a reward.
7. The learning policy updates itself using the observed reward.
8. After the horizon ends, the run is summarized.
9. After all runs complete, results are aggregated and written to CSV.

## Current Outcome of the Project

The project is functionally complete as an experimental simulator. It already includes:

- configurable environment settings
- multiple synthetic workloads
- baseline decision policies
- AI-based contextual bandit policies
- repeated experiments over multiple seeds
- automatic aggregation and CSV export

From the current results, the strongest adaptive policies are generally:

- `linear_thompson_sampling`
- `linucb`

These perform especially well on dynamic workloads such as bursty traffic, periodic spikes, and abrupt workload shifts. Simpler fixed policies can still do well on very stable workloads, but they are less robust when demand changes suddenly.

## How to Run

Run the default experiment:

```bash
python3 main.py
```

Run with custom settings:

```bash
python3 main.py --horizon 300 --seeds 7 11 19 --workloads bursty abrupt_shift
```

Write results to a custom output path:

```bash
python3 main.py --output results/custom_summary.csv
```

## Short Summary

This project is a serverless prewarming simulator powered by contextual bandit methods. It models incoming traffic, system response, and cost, then compares different decision strategies for choosing warm instances. The core idea is to use recent workload context to make smarter prewarming decisions that reduce cold starts and latency while keeping costs reasonable.
