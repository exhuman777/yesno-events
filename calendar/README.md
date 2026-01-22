# YESNO.EVENTS Calendar System

## Purpose
Central calendar for prediction market traders. Track events that move markets.

## Directory Structure

```
calendar/
├── events/           # Scheduled events (FED, earnings, elections, etc.)
│   ├── 2026-01.md   # Monthly event files
│   └── recurring.md  # Recurring events (FOMC meetings, etc.)
├── news/             # News items with market impact
│   └── YYYY-MM-DD.md # Daily news files
├── insights/         # Trading observations & methodology
│   └── elon-tweets.md
├── markets/          # Active market tracking
│   └── watched.md    # Markets you're watching
└── feeds/            # RSS feed configs
    └── sources.md    # News sources configuration
```

## File Formats

### Event File (events/YYYY-MM.md)
```yaml
---
month: 2026-01
updated: 2026-01-11T18:00:00Z
---

# January 2026 Events

## 2026-01-15
- type: FED
  time: "14:00 ET"
  title: FOMC Minutes Release
  impact: HIGH
  markets: [interest-rates, spy, btc]
  notes: |
    Watch for rate guidance

## 2026-01-20
- type: POLITICAL
  title: Inauguration Day
  impact: HIGH
  markets: [trump-*, political-*]
```

### News File (news/YYYY-MM-DD.md)
```yaml
---
date: 2026-01-11
sources: [reuters, bloomberg, polymarket]
---

# News 2026-01-11

## 09:15 - Fed Governor Comments
- source: Reuters
- impact: MEDIUM
- sentiment: bearish
- markets: [rates]
- summary: |
    Fed governor hints at slower rate cuts

## 14:30 - Elon Tweet Storm
- source: X/Twitter
- impact: HIGH
- count: 45 tweets in 2h
- markets: [elon-tweets-*]
```

### Insight File (insights/*.md)
```yaml
---
topic: elon-tweets
created: 2026-01-11
updated: 2026-01-11
confidence: 0.8
---

# Elon Tweet Market Insights

## Key Observations

### Pricing Inefficiency
People tend to overvalue far-future predictions.
- 200+ posts ahead: heavily overpriced
- 30-80 before end: proper calculation possible
- Last 24h: most accurate pricing

### Volume Patterns
- High volume = uncertainty
- Low volume + high odds = smart money
- Spike detection: 3x avg = significant

## Trading Rules
1. Never buy brackets >100 posts from current
2. Best entries: 50-70% through event window
3. Watch for late surge patterns (last 2 days)
```

### Market Watch File (markets/watched.md)
```yaml
---
updated: 2026-01-11T18:00:00Z
---

# Watched Markets

## Active Positions

### elon-tweets-jan6-13
- position: YES 580+
- entry: 0.12
- current: 0.78
- size: $500
- thesis: |
    Momentum trend suggests 560-600 range
- exit_plan: |
    Sell at 0.90 or 24h before expiry

## Watchlist

### fed-rate-jan
- status: WATCHING
- trigger: Buy YES if drops below 0.60
- notes: Market overreacting to dovish signals
```

## API Integration (Tomorrow)

### Elon Posts (Live)
- endpoint: /api/elon/live-count
- refresh: 60s
- auto-calendar: YES (significant activity logged)

### News RSS
- sources defined in feeds/sources.md
- auto-categorize by impact
- link to relevant markets

## Calendar Event Types

| Type | Color | Description |
|------|-------|-------------|
| FED | Red | Federal Reserve events |
| EARNINGS | Blue | Company earnings |
| POLITICAL | Orange | Elections, policy |
| CRYPTO | Yellow | Crypto events |
| MARKET | Green | Market expiry dates |
| GEOPOLITICAL | Purple | Wars, treaties |
| ELON | Pink | Elon-related |
| NEWS | Gray | General news |
