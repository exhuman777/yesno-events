---
updated: 2026-01-11
active: true
---

# News Feed Sources

## Prediction Markets

### Polymarket
- type: API
- endpoint: polyrouter
- key: pk_244ce3d27ae836fde5b99b279330737c5df33f5aa7e37f2e7b1962b8b1db6aa9
- refresh: 60s
- auto_calendar: true
- categories: [markets, resolutions]

### Kalshi
- type: API
- status: TODO
- categories: [markets, events]

### Metaculus
- type: RSS
- url: https://www.metaculus.com/questions/feed/
- categories: [forecasts]

## Financial News

### Reuters Business
- type: RSS
- url: https://www.reuters.com/business/rss
- impact_keywords: [fed, inflation, rates, gdp, jobs]
- auto_categorize: true

### Bloomberg Markets
- type: RSS
- url: https://www.bloomberg.com/markets/rss
- impact_keywords: [fomc, treasury, earnings]

### CNBC
- type: RSS
- url: https://www.cnbc.com/id/100003114/device/rss/rss.html
- filter: breaking only

## Crypto

### CoinDesk
- type: RSS
- url: https://www.coindesk.com/arc/outboundfeeds/rss/
- categories: [btc, eth, regulation]

### The Block
- type: RSS
- url: https://www.theblock.co/rss.xml
- categories: [defi, exchanges]

## Social/Elon

### Elon Twitter/X
- type: API
- source: xtracker
- endpoint: TODO (tomorrow)
- metrics: [count, rate, content_analysis]
- auto_calendar: true
- triggers:
    - count_spike: 20 tweets/hour
    - keyword: [Tesla, X, SpaceX, DOGE]

### Twitter Trending
- type: scrape
- status: TODO
- relevance_filter: prediction_markets

## Geopolitical

### Reuters World
- type: RSS
- url: https://www.reuters.com/world/rss
- keywords: [war, conflict, sanctions, treaty]
- impact: HIGH

### Defense News
- type: RSS
- url: https://www.defensenews.com/rss/
- categories: [military, defense]

## Economic Calendar

### Forex Factory
- type: scrape
- url: https://www.forexfactory.com/calendar
- events: [NFP, CPI, FOMC, GDP]
- auto_import: weekly

### Investing.com
- type: API
- status: TODO
- events: [earnings, economic]

---

## Feed Processing Rules

### Impact Classification
```yaml
VERY_HIGH:
  - FOMC decisions
  - CPI/PCE releases
  - Major political events
  - War declarations

HIGH:
  - Fed speeches
  - Jobs reports
  - GDP releases
  - Major earnings

MEDIUM:
  - Fed minutes
  - Housing data
  - Trade data
  - Elon tweet spikes

LOW:
  - Regional fed data
  - Minor economic releases
```

### Auto-Calendar Rules
```yaml
# Add to calendar if:
- impact >= MEDIUM
- mentions watched market
- keyword match score > 0.7
- source reliability > 0.8
```

### Deduplication
- Same story from multiple sources: keep highest reliability
- Time window: 30 minutes
- Similarity threshold: 0.85
