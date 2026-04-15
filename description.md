# Project Proposal: Contextual Bandits for Per-Function Autoscaling in Serverless


Team members:

Our project idea is to study whether a contextual bandit can make better autoscaling decisions for serverless functions than simple fixed heuristics. In serverless systems, functions are invoked on demand, and there is a constant tradeoff between latency and cost: keeping instances warm reduces cold starts and improves response time, but it also wastes resources when traffic is low. We want to test the hypothesis that an adaptive AI system can use recent workload context to choose better prewarming levels than rules like always keeping a fixed number of instances warm.

To accomplish this, we plan to build a simplified serverless simulator with changing traffic patterns such as steady, bursty, and periodic demand, then compare several contextual bandit methods like LinUCB or Thompson Sampling against non-learning baselines. The agent will observe features such as recent request volume, burstiness, latency, and cold starts, choose a prewarm level for the next time window, and receive reward based on latency, SLA violations, cold starts, and operational cost. We hope the main contribution will be a clear case study showing when adaptive bandit-based decision making improves the latency-cost tradeoff in cloud resource management, along with an empirical comparison of learning-based and heuristic autoscaling strategies.

## Overview

This project studies whether a contextual bandit can make better autoscaling decisions for serverless functions than fixed heuristic policies. In serverless systems, functions are invoked on demand, and response latency often increases when requests arrive while no warm instances are available. At the same time, keeping function instances warm all the time increases operational cost. The core problem is therefore a cost-latency tradeoff under changing workloads.

We propose to model autoscaling as an online decision problem. At each time step, the system observes recent workload statistics for a function, chooses a small autoscaling action such as a prewarm level, then receives feedback in the form of latency, cold starts, SLA violations, and cost. A contextual bandit uses this feedback to improve future decisions based on the observed context.

## Motivation

Serverless workloads are highly dynamic. A function may receive very low traffic during some periods and sudden bursts during others. Static policies such as always keeping one instance warm or using a simple threshold often waste resources or fail during bursts. A contextual bandit is attractive because it can adapt decisions to workload conditions without requiring a full long-horizon reinforcement learning setup.

This project fits course topics in:
- sequential decision making
- learning from evaluative feedback
- contextual bandits
- online decision making under uncertainty

## Problem Statement

Given recent workload context for a serverless function, choose an autoscaling action that balances:
- response latency
- cold-start frequency
- SLA violations
- infrastructure cost

The main research question is:

Can contextual bandits learn better per-function autoscaling decisions than fixed heuristic policies under changing serverless workloads?

## Scope

To keep the project manageable, we will optimize only one control variable:

- **Prewarm count**: the number of warm function instances kept ready for the next time window

We will not optimize multiple knobs such as memory tier, reserved concurrency, and prewarming simultaneously. Restricting the action space makes the project easier to implement and easier to evaluate.

## System Design

### 1. Workload Environment

We will build a simplified serverless simulator that operates in discrete time windows. For each window, the simulator will model:
- number of incoming requests
- available warm instances
- cold-start penalty when demand exceeds warm capacity
- per-window cost of keeping instances warm
- latency and SLA outcomes

The simulator will support multiple traffic patterns such as:
- steady low traffic
- steady medium traffic
- bursty traffic
- periodic traffic with spikes

If feasible, we may also incorporate trace-driven workloads from published serverless traces, but the baseline version will use synthetic traffic generation.

### 2. Context Features

At each decision step, the bandit observes context features derived from recent workload behavior. Candidate features include:
- request count in the previous window
- moving average of requests over the last 3 or 5 windows
- recent variance or burstiness of traffic
- recent average latency
- recent number of cold starts
- recent SLA violation rate
- optional time-of-day bucket for periodic workloads

These features form the input to the contextual bandit.

### 3. Action Space

The action is the prewarm level for the next time window. A compact action space will be used, such as:
- prewarm 0 instances
- prewarm 1 instance
- prewarm 2 instances
- prewarm 4 instances

This keeps the learning problem simple while still allowing meaningful tradeoffs.

### 4. Reward Function

The reward will combine service quality and cost. A candidate reward is:

reward = -(latency penalty + SLA violation penalty + cost penalty)

More concretely, the reward may depend on:
- average latency or p95 latency
- number of SLA violations
- number of cold starts
- number of warm instances kept active

The exact weights will be tuned so that neither cost nor latency dominates unrealistically.

## Learning Methods

We will compare several contextual bandit methods, such as:
- LinUCB
- Thompson Sampling
- epsilon-greedy with contextual estimates

These methods will be evaluated against non-learning baselines.

## Baselines

To show that learning helps, we will compare against simple heuristic policies such as:
- always prewarm 0
- always prewarm 1
- always prewarm 2
- threshold-based rule:
  - if recent traffic exceeds a threshold, increase prewarming
  - otherwise keep low prewarming

These baselines are important because they represent what a practitioner might deploy without learning.

## Experimental Plan

### Step 1: Build the simulator
Implement a serverless environment with time-windowed traffic, warm capacity, cold-start penalties, latency, and cost.

### Step 2: Generate workload patterns
Create synthetic traces representing stable, bursty, and non-stationary traffic.

### Step 3: Define context and reward
Select the context features and finalize the reward function.

### Step 4: Implement baseline policies
Add fixed and threshold-based heuristics.

### Step 5: Implement contextual bandit algorithms
Implement and validate LinUCB, Thompson Sampling, and epsilon-greedy.

### Step 6: Run experiments
Evaluate all methods across different traffic regimes and multiple random seeds.

### Step 7: Analyze performance
Measure tradeoffs among latency, cold starts, SLA violations, and cost.

## Evaluation Metrics

The project will evaluate:
- average latency
- p95 latency
- total cold starts
- SLA violation rate
- total cost
- cumulative reward
- regret over time

The strongest result would show that contextual bandits outperform static heuristics especially under bursty or changing workloads.

## Expected Contributions

This project aims to contribute:
- a simple but credible simulator for per-function autoscaling decisions
- an empirical comparison of contextual bandits and heuristic policies
- an analysis of when adaptive decision-making helps most in serverless systems
- a clear case study connecting contextual bandits to practical cloud resource management

## Risks and Limitations

Potential risks include:
- the simulator may be too simplistic if not designed carefully
- the project may resemble prior serverless RL work if the framing is too broad
- reward design may strongly affect results

To reduce these risks, we will:
- focus on contextual bandits rather than full RL
- optimize only prewarm count
- compare against transparent heuristics
- test across multiple workload patterns

## End Goal

The final goal is to show that a contextual bandit can use recent workload context to choose better prewarm levels for a serverless function than fixed heuristic policies, improving the latency-cost tradeoff under dynamic traffic conditions.
