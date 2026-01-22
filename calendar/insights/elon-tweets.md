---
topic: elon-tweet-markets
author: trader
created: 2026-01-11
updated: 2026-01-11
confidence: 0.75
backtest_period: Nov 2025 - Jan 2026
sample_size: 21 resolved markets
---

# Elon Tweet Market Trading Methodology

## Key Insight #1: Time-Based Mispricing

### The Problem
People tend to overvalue predictions far into the future.

### Observation
- **200+ posts ahead of current**: Heavily overpriced. Market tries to predict too far.
- **100-200 posts ahead**: Moderate mispricing. Some signal, lots of noise.
- **30-80 posts before end**: Sweet spot. Proper calculation possible.
- **Last 24 hours**: Most accurate, but less profit opportunity.

### Trading Rule
```
NEVER buy brackets >100 posts from current count
BEST entries: when 50-70% through event window
```

## Key Insight #2: Momentum Patterns

### Historical Data (21 weeks analyzed)
| Period | Avg Weekly | Trend |
|--------|------------|-------|
| Nov 2025 | 230 tweets | baseline |
| Dec 2025 | 350 tweets | +52% |
| Jan 2026 | 540 tweets | +54% |

### Momentum Indicator
```
recent_5wk_avg / older_avg = momentum_ratio
If ratio > 1.5 → strongly_up → buy higher brackets
If ratio < 0.7 → strongly_down → buy lower brackets
```

Current momentum: **+95%** (strongly up)

## Key Insight #3: Volume Analysis

### Volume Signals
| Signal | Meaning | Action |
|--------|---------|--------|
| High vol + stable price | Disagreement | Wait |
| High vol + price move | Conviction | Follow |
| Low vol + high odds | Smart money | Consider following |
| Spike 3x avg | News/event | Investigate |

### CLOB Depth Analysis
- Thin orderbook = volatile, careful sizing
- Deep orderbook = can take larger position
- Watch bid/ask imbalance for direction

## Key Insight #4: Winning Bracket Distribution

### From 21 resolved markets:
```
160-279: 6 wins (29%) - low activity periods
280-399: 4 wins (19%) - moderate activity
400-519: 6 wins (29%) - high activity
520+:    5 wins (24%) - very high activity
```

### Trend Observation
Recent weeks skew toward higher brackets (500+).
Adjust expectations based on:
1. Current momentum
2. Day of week patterns (weekends often lower)
3. External events (Tesla news, X changes, politics)

## Key Insight #5: Entry Timing

### Best Entry Windows
1. **Early market (first 24h)**: Wide brackets, find value
2. **Mid-market (day 3-5)**: Price discovery, adjust
3. **Late market (last 48h)**: Final positioning, highest confidence

### Avoid
- Buying after big count jumps (priced in)
- Selling right before expiry (slippage)
- Chasing momentum last 6 hours

## Trading Rules Summary

### DO
1. Track live count (xtracker) religiously
2. Calculate required pace for each bracket
3. Use 30-80 posts window for predictions
4. Size positions based on orderbook depth
5. Have exit plan before entry

### DON'T
1. Predict >100 posts ahead
2. Ignore momentum trends
3. Trade without checking volume
4. Hold through uncertainty spikes
5. Oversize on thin liquidity

## Model Parameters

### Linear Model
```python
avg_daily_rate = 52.3
min_daily_rate = 22.9
max_daily_rate = 94.6
samples = 21
```

### Pattern Model
```python
avg_weekly_total = 366
std_dev = 139.1
recent_bias = +95%
```

## Price Targets

### Fair Value Estimation
For bracket B with current count C, elapsed days E, total days T:
```
remaining = T - E
pace = C / E
projected = C + (pace * remaining)

if projected in B: fair_value = 0.70-0.90
if projected near B edge: fair_value = 0.40-0.60
if projected far from B: fair_value = 0.05-0.20
```

## Risk Management

### Position Sizing
- Max 10% of bankroll per market
- Max 5% per bracket within market
- Scale in: 50% initial, 50% on confirmation

### Stop Loss
- Exit if current count makes bracket impossible
- Exit if momentum reverses sharply (2-day trend change)
- Take profit at 2x (80% position)

---

## Updates Log

### 2026-01-11
- Initial methodology documented
- Momentum at all-time high
- Current week tracking 580+ bracket

### TODO
- Add hourly patterns analysis
- Backtest entry timing strategies
- Correlate with external events (Tesla stock, X news)
