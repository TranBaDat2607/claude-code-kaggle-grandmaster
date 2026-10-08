---
name: simulation-and-optimization
description: Playbook for Kaggle simulation competitions (agents playing games in an environment against other teams' bots, e.g. Lux AI, Kore, ConnectX, Halite) and combinatorial optimisation competitions (Santa / traveling-salesman / scheduling / packing puzzles). Use when the submission is an agent or a solution file scored by an objective rather than predictions on a test set.
---

# Simulation & Optimisation Playbook

## A. Simulation (agent) competitions

Scoring is a skill rating (TrueSkill/Elo-like) from matches against other submissions; the LB is
noisy and keeps moving until well after the deadline.

1. **Set up the local environment** (`kaggle-environments` or the competition kit) and a
   **local arena**: run your agent versions against each other and against public bots, track
   win rates with enough games for statistical significance (hundreds).
2. **Rule-based agent first.** A strong heuristic bot with good game understanding is the
   baseline that many ML approaches fail to beat early. Encode strategy explicitly, profile the
   per-step time limit.
3. **Imitation learning** from top-ranked agents' episodes (download replays via the Kaggle
   episodes API): train a policy network (CNN/transformer over the game map, unit-wise action
   heads) to predict top players' actions. Very effective mid-competition.
4. **Reinforcement learning** (PPO/IMPALA with self-play, league of past versions) when compute
   allows; initialise from the imitation policy; reward shaping early, sparse win reward later.
5. **Search**: MCTS / beam search / minimax with a learned value or heuristic evaluation when the
   game is tractable (ConnectX-style).
6. **Engineering**: respect time per turn (watch the overage budget), deterministic seeds for
   debugging, a replay viewer, unit tests for game-rule edge cases, crash-free submissions (a
   crash = loss).
7. Submit improved versions early: ratings need many games to converge. Keep multiple
   diverse agents active if allowed.

## B. Combinatorial optimisation (Santa-style)

The score is a deterministic objective of your solution file; the public LB *is* the private
LB, so this is pure optimisation.

1. **Understand the objective exactly**; write a fast scorer (numba / C++ / Rust) and verify it
   against the official one on the sample submission.
2. **Construct** a good initial solution: greedy, problem-specific heuristics, LKH/Concorde for
   TSP-like structures, OR-Tools CP-SAT / MIP (Gurobi/HiGHS) for assignment/scheduling.
3. **Improve** with local search: 2-opt/3-opt/Or-opt, swap/insert moves, k-exchange, simulated
   annealing with tuned temperature schedule, tabu search, large neighbourhood search (destroy &
   repair with an exact solver on subproblems).
4. **Exploit structure**: decompose into independent subproblems, solve subproblems exactly with
   MIP/CP, use delta-evaluation of moves (O(1) score updates), precompute lookups.
5. **Compute**: multi-core, run many restarts with different seeds overnight; keep the best
   solution file versioned. Small improvements late are common — keep optimisers running.
6. For LLM-themed optimisation (e.g. permutation puzzles scored by a language model's
   perplexity), cache scores, batch evaluations on GPU, and treat it as black-box local search
   with smart move proposals.
