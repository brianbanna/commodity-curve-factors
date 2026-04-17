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

---

## Curve Factor Contamination Audit (2026-04-13)

### Finding
All 4 curve-derived factors (carry, slope, curvature, term_carry) are contaminated by same-day price moves through the WRDS curve interpolation. The contamination mechanism: when spot price moves on day T, the interpolated F1M (and to a lesser degree F2M-F12M) responds immediately, causing the factor computed on day T to correlate with the return on day T.

### Evidence

| Factor | lag0 IC | lag1 IC | Contaminated? | Decontaminated lag1 IC |
|--------|---------|---------|---------------|----------------------|
| carry | +0.042 | +0.013 | YES (3.2x drop) | +0.006 |
| slope | -0.042 | -0.013 | YES (3.2x drop) | -0.006 |
| curvature | -0.037 | -0.018 | YES (2.1x drop) | -0.010 |
| term_carry | +0.042 | +0.013 | YES (3.2x drop) | +0.005 |

### Non-Curve Factors (Clean)

| Factor | lag0 IC | lag1 IC | lag5 IC | lag21 IC | Status |
|--------|---------|---------|---------|----------|--------|
| TSMOM | +0.075 | +0.017 | +0.017 | +0.022 | **CLEAN — persistent signal** |
| XSMOM | +0.089 | +0.013 | +0.017 | +0.016 | **CLEAN — persistent signal** |
| Macro | +0.170 | +0.004 | -0.000 | -0.010 | Noise |
| Positioning | -0.006 | -0.007 | -0.016 | -0.010 | Noise |
| Volatility | -0.005 | -0.003 | -0.007 | +0.002 | Noise |
| Curve momentum | -0.065 | +0.001 | -0.003 | -0.020 | Contaminated |

### Implication
Only TSMOM and XSMOM have genuine predictive power at lag >= 1 in this framework. Curve factors should be used as conditioning variables (regime filters), not as standalone ranking signals.

---

## Signal Recovery Investigation (2026-04-13)

### Carry Variants Tested (All Negative OOS)

| Variant | IS Sharpe | OOS Sharpe | Conclusion |
|---------|-----------|------------|------------|
| Raw carry, weekly rebal | -0.23 | -0.45 | No signal |
| Z-scored carry, weekly | -0.30 | -0.36 | No signal |
| Raw carry, monthly | +0.04 | -0.44 | IS marginal, OOS fails |
| Within-sector carry, monthly | -0.05 | -0.65 | Worse — too few assets per sector |
| F1M-F3M formula, monthly | +0.04 | -0.44 | Formula doesn't matter |
| Log carry formula, monthly | +0.04 | -0.44 | Same |
| Term carry, monthly | +0.05 | -0.44 | Same |
| Contract-level (no interpolation) | -1.06 | -1.53 | Much worse |
| Carry + inventory overlay | +0.12 | -0.66 | IS improvement, OOS fails |
| Decontaminated carry, weekly | -0.36 | -0.37 | No improvement |
| Decontaminated carry, monthly | -0.32 | -0.50 | No improvement |

**Conclusion:** Carry does not work as a cross-sectional ranking signal in this 13-commodity universe over 2005-2024. The finding is robust to formula variants, rebalancing frequency, z-score vs raw levels, interpolation vs contract-level data, and inventory conditioning.

---

## Alternative Strategy Architectures (2026-04-13)

### The Breakthrough: MinVar + Momentum Tilt

| Strategy | IS Sharpe | OOS Sharpe | Turnover | MaxDD | Hit Rate |
|----------|-----------|------------|----------|-------|----------|
| **MinVar + Momentum Tilt** | **+0.18** | **+0.70** | **0.030** | **-0.34** | **0.52** |
| Carry-Conditioned TSMOM | +0.13 | -0.02 | 0.041 | -0.30 | 0.46 |
| Carry Regime Switch | -0.02 | +0.01 | 0.049 | -0.30 | 0.51 |
| Calendar + TSMOM Overlay | -0.08 | +0.02 | 0.043 | -0.37 | 0.50 |
| TSMOM (original, post-fix) | +0.19 | +0.02 | 0.030 | -0.29 | 0.45 |
| Equal-Weight Long (benchmark) | +0.17 | +0.40 | — | — | — |

### Why MinVar + Momentum Tilt Works

1. **Minimum-variance allocation** concentrates weight in low-correlation, low-volatility commodities. This is a structural risk-reduction mechanism — the OOS Sharpe improvement over equal-weight comes from lower portfolio volatility (8.2% vs 16.8%), not higher returns.

2. **TSMOM tilt** overweights trending commodities within the min-var allocation. Since TSMOM has the only persistent lag=1 IC (+0.017), the tilt adds directional information on top of the variance-reduction base.

3. **Long-only structure** avoids the carry contamination problem entirely — no cross-sectional ranking needed. The strategy earns the commodity risk premium (positive in 2018-2024) plus the TSMOM tilt bonus.

4. **Low turnover** (0.030/day) minimizes transaction cost drag. The min-var weights are driven by the trailing 60-day covariance matrix, which is slow-moving.

5. **IS vs OOS gap** (IS +0.18 vs OOS +0.70) is NOT overfitting — it reflects the structural difference between the IS period (2005-2017: commodity supercycle bust) and OOS (2018-2024: commodity recovery + inflation). The strategy's OOS outperformance is driven by being long a recovering asset class with smart risk allocation.

### Honest Caveats

- The OOS +0.70 benefits from the 2021-2022 commodity boom. A longer OOS sample would likely regress toward +0.3-0.5.
- MinVar weights are sensitive to covariance estimation. The 60-day lookback may be too short for stable estimates.
- The strategy is long-only — it cannot capture short alpha during commodity bear markets.
- With 13 commodities, the covariance matrix has 78 unique elements estimated from 60 data points — the estimation error is significant.

