"""Reinforcement-learning subsystem for SimuMax.

Split into two sibling submodules:

- :mod:`simumax.rl.env` — the Gymnasium environment and its backend.
- :mod:`simumax.rl.agents` — agents that consume the environment
  (static-schedule baselines today; learned policies later).
"""
