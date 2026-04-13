# Research Findings

## Carry Signal Contemporaneous Contamination (2026-04-13)

### Finding
The carry factor (F1M - F2M) / F2M * 12 has near-zero predictive IC at lag=1 in this 13-commodity universe over 2005-2024.

### Mechanism
F1M is mechanically responsive to same-day spot price moves. When CL's spot price rises, F1M rises more than F2M, increasing carry on the same day the return is positive. This produces:
- Lag=0 IC: 0.048 (contaminated by concurrent price moves)
- Lag=1 IC: 0.006 (after removing contamination — no signal)
- Corr(delta_carry_z, concurrent_CL_return): 0.70

### Grinold-Kahn Analysis
With IC=0.006 and breadth=6 independent bets/day:
- Expected SR = IC * sqrt(Breadth) = 0.006 * sqrt(6 * 252) = 0.23
- After daily rebalancing costs (~1% annual drag): net Sharpe ≈ 0
- After weekly rebalancing costs (~0.25% annual drag): net Sharpe ≈ +0.1

### Academic Context
Academic carry Sharpe of 0.5-0.9 (Koijen et al. 2018) uses:
- 20-40 commodities (vs 13 here)
- 30+ year samples (vs 20 years here)
- RAW carry levels for ranking (vs z-scored here)
- Monthly rebalancing (vs daily/weekly here)

This finding is genuine, not a code bug. The infrastructure correctly implements and detects carry's weakness in this setting.
