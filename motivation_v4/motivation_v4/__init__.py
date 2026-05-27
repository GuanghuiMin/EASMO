"""motivation_v4 — Decision-state sensitivity for context compression.

Tests the claim that context importance for an LLM agent is best
measured by whether removing a history span changes the downstream
agent's *decision state*, not by structural / coverage metrics.

Builds on motivation_v3 (which showed structural-vs-behavioral
ranking-mismatch). Reuses v3's 30 successful dev trajectories and the
same downstream agent prompt.
"""
