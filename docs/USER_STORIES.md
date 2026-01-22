# YES/NO.EVENTS - User Stories & Usage Examples

## Table of Contents
- [User Personas](#user-personas)
- [User Stories](#user-stories)
- [Usage Examples](#usage-examples)
- [Workflow Scenarios](#workflow-scenarios)

---

## User Personas

### 1. Alex - The Casual Trader
- **Background**: Works in tech, trades prediction markets as a hobby
- **Goals**: Quick market browsing, occasional trades, track P&L
- **Pain Points**: Complex interfaces, too much data noise
- **Needs**: Simple navigation, clear price display, mobile-friendly

### 2. Maya - The Quantitative Analyst
- **Background**: Data scientist, builds trading models
- **Goals**: API access, historical data, pattern analysis
- **Pain Points**: Rate limits, incomplete APIs, no raw data export
- **Needs**: All endpoints accessible, bulk data, analytics tools

### 3. Jordan - The Copy Trader
- **Background**: New to prediction markets, wants to follow experts
- **Goals**: Find top traders, mirror their positions
- **Pain Points**: Hard to identify good traders, no position visibility
- **Needs**: Leaderboards, trader profiles, position tracking

### 4. Sam - The Market Maker
- **Background**: Professional trader, provides liquidity
- **Goals**: Monitor spreads, manage orders, maximize rebates
- **Pain Points**: Slow order management, poor orderbook visibility
- **Needs**: Real-time orderbook, quick order placement, spread analysis

### 5. Riley - The Researcher
- **Background**: Journalist/Analyst, writes about prediction markets
- **Goals**: Market trends, volume analysis, event correlations
- **Pain Points**: Data scattered across platforms
- **Needs**: Comprehensive data, visual analytics, export capabilities

---

## User Stories

### Market Discovery

#### US-001: Browse Trending Markets
**As** Alex (casual trader)
**I want** to see what markets are hot right now
**So that** I can find interesting opportunities quickly

**Acceptance Criteria:**
- Markets sorted by 24h volume
- Price and volume displayed clearly
- One-click to view market details

**Example:**
```
Press 1 (Markets) → See trending markets
    Trump wins 2024          52.3%   $4.5M 24h
    Bitcoin > $100k March    34.5%   $2.1M 24h
    Fed cuts rates Jan       78.2%   $987k 24h
```

---

#### US-002: Search Markets by Keyword
**As** Maya (quant analyst)
**I want** to search for specific markets
**So that** I can find markets related to my research

**Acceptance Criteria:**
- Search by keyword/phrase
- Filter by category (Politics, Crypto, Sports)
- Sort results by volume, date, or price

**Example:**
```
Press 1 (Markets) → Type "Elon" → Enter
    Elon tweets >100 next week    45.0%   $765k
    Elon buys another company     12.3%   $234k
    SpaceX Starship success       67.8%   $876k
```

---

### Trading

#### US-003: Place a Buy Order
**As** Alex (casual trader)
**I want** to buy shares in a market
**So that** I can profit if my prediction is correct

**Acceptance Criteria:**
- Select market from list
- Enter price and quantity
- Confirm order before submission
- See order confirmation

**Example:**
```
1. Navigate: Press 1 → Select market → Enter
2. View market details with orderbook
3. Enter: Size: 100, Price: 0.52
4. Press "Buy YES" button
5. See: "Order placed: BUY 100 @ $0.52"
```

---

#### US-004: View Open Orders
**As** Sam (market maker)
**I want** to see all my pending orders
**So that** I can manage my liquidity

**Acceptance Criteria:**
- List all open orders
- Show fill status
- Allow order cancellation

**Example:**
```
Press 3 (Portfolio) → Press 4 (Orders)
    MARKET                           SIDE   PRICE  SIZE   FILLED  STATUS
    Trump wins 2024                  BUY    51.0¢  5000   2000    OPEN
    Bitcoin > $100k                  SELL   36.0¢  3000   0       OPEN
```

---

### Portfolio Management

#### US-005: Track Portfolio Performance
**As** Jordan (copy trader)
**I want** to see my total P&L
**So that** I know if my strategy is working

**Acceptance Criteria:**
- Total portfolio value displayed
- Per-position P&L shown
- Color-coded gains/losses

**Example:**
```
Press 3 (Portfolio)
  USDC: $12,450.00  │  Portfolio: $8,234.50  │  Positions: 12

  ▶ OPEN POSITIONS
    MARKET                           SIDE   SHARES  AVG     VALUE    P&L
    Trump wins 2024                  YES    1000    52.0¢   $580.00  +$60.00
    Bitcoin > $100k                  NO     500     35.0¢   $160.00  -$15.00
```

---

#### US-006: View Trade History
**As** Maya (quant analyst)
**I want** to see my past trades
**So that** I can analyze my performance

**Acceptance Criteria:**
- Chronological trade list
- All trade details shown
- Filterable by date/market

**Example:**
```
Press 3 (Portfolio) → Press 2 (Trades)
    TIME         SIDE   MARKET                      PRICE   SIZE    VALUE
    2025-01-18   BUY    Trump wins 2024             51.0¢   1000    $510.00
    2025-01-17   SELL   Fed cuts rates              82.0¢   500     $410.00
    2025-01-17   BUY    Bitcoin > $100k             32.0¢   750     $240.00
```

---

### Research & Analytics

#### US-007: Find Top Traders
**As** Jordan (copy trader)
**I want** to see the leaderboard
**So that** I can find successful traders to follow

**Acceptance Criteria:**
- Traders ranked by P&L
- Volume and position count shown
- Wallet address for tracking

**Example:**
```
Press 5 (Research) → Press 1 (Leaderboard)
    ▶ TOP TRADERS (LIVE)  │  OVERALL  │  DAY
    #    ADDRESS               P&L           VOLUME      POSITIONS
    1    0xABC1...23EF    +$2.85M      $15.4M      45
    2    0xDEF4...56GH    +$1.92M      $12.3M      32
    3    0x789A...BCDE    +$1.46M      $9.8M       28
```

---

#### US-008: Analyze Market Correlations
**As** Maya (quant analyst)
**I want** to see how markets correlate
**So that** I can build multi-market strategies

**Acceptance Criteria:**
- Correlation matrix displayed
- Positive/negative correlations highlighted
- Actionable trading pairs suggested

**Example:**
```
Press 5 (Research) → Press 4 (Correlations)
    ◆ CORRELATION MATRIX (30 Day)
                  Trump   BTC    Fed    Elon   SpaceX
    Trump          1.00  +0.15  +0.42  +0.38  +0.25
    BTC           +0.15   1.00  -0.22  +0.65  +0.30
    Elon          +0.38  +0.65  -0.05   1.00  +0.72

    ◆ ACTIONABLE PAIRS
    Elon + SpaceX (r=0.72): Trade together, high correlation
    Elon + BTC (r=0.65): Elon tweets move crypto markets
```

---

#### US-009: Analyze Elon Tweet Patterns
**As** Riley (researcher)
**I want** to see Elon's tweeting patterns
**So that** I can predict tweet-related markets

**Acceptance Criteria:**
- Hourly heatmap displayed
- Daily patterns shown
- Peak hours identified

**Example:**
```
Press 4 (Elon Lab) → Press 2 (Hourly)
    ◆ HOURLY TWEET DISTRIBUTION (UTC)
    00:  ████████████████ 156
    01:  ███████████████ 145
    02:  ████████████ 98
    ...
    14:  ██████████████████████ 198  ← PEAK
```

---

### API Exploration

#### US-010: Test API Endpoints
**As** Maya (quant analyst)
**I want** to explore available APIs
**So that** I can integrate data into my models

**Acceptance Criteria:**
- All endpoints listed
- Parameter documentation
- Live testing capability

**Example:**
```
Press 7 (API Explorer)
    ◆ AVAILABLE APIs

    GAMMA API (8 endpoints)
    ├─ gamma_search(query) - Search markets
    ├─ gamma_get_market(id) - Get market details
    └─ gamma_get_comments(id) - Get market comments

    DATA API v2 (6 endpoints)
    ├─ dataapi_get_positions(wallet) - User positions
    ├─ dataapi_get_trades(wallet) - Trade history
    └─ dataapi_get_leaderboard(category) - Top traders
```

---

## Workflow Scenarios

### Scenario 1: Morning Market Check
**User**: Alex (casual trader)
**Time**: 5 minutes
**Goal**: Quick overview of markets and positions

```
1. Launch terminal: python app.py
2. Check portfolio: Press 3
   - See: USDC balance, total P&L, position count
3. Check positions: Already on Positions tab
   - Review overnight P&L changes
4. Check trending markets: Press ESC → Press 1
   - See what's moving today
5. Exit or continue trading
```

---

### Scenario 2: Research-Driven Trading
**User**: Maya (quant analyst)
**Time**: 30 minutes
**Goal**: Find mispriced market based on data

```
1. Launch terminal: python app.py
2. Check leaderboard: Press 5 → Press 1
   - Note: Top traders are buying "Fed cuts rates"
3. Analyze correlations: Press 4
   - See: Fed markets inversely correlate with BTC
4. Check tag volume: Press 2
   - See: "Fed" tag up 45% in 24h
5. Search specific market: Press ESC → Press 1 → "Fed"
   - View orderbook depth
   - Check spread quality
6. Place trade if opportunity exists
```

---

### Scenario 3: Copy Trading Setup
**User**: Jordan (copy trader)
**Time**: 15 minutes
**Goal**: Find and track a successful trader

```
1. Launch terminal: python app.py
2. Go to leaderboard: Press 5 → Press 1
3. Identify top trader: 0xABC1...23EF with +$2.85M P&L
4. Note wallet address
5. Check their positions via API Explorer:
   - Press 7 → Data API → Positions
   - Enter wallet address
6. See their current positions
7. Mirror positions based on conviction
```

---

### Scenario 4: Market Maker Workflow
**User**: Sam (market maker)
**Time**: Ongoing
**Goal**: Provide liquidity and earn rebates

```
1. Launch terminal: python app.py
2. Go to Analytics: Press 6 → Press 1 (Spreads)
   - Identify markets with wide spreads
3. Select target market: Press ESC → Press 1 → Select market
4. View orderbook depth:
   - Check current bid/ask
   - Calculate optimal spread
5. Place orders on both sides:
   - Buy YES at 50¢
   - Sell YES at 52¢
6. Monitor orders: Press 3 → Press 4 (Orders)
7. Adjust positions as market moves
```

---

### Scenario 5: Event Research
**User**: Riley (researcher)
**Time**: 45 minutes
**Goal**: Analyze Elon tweet patterns for article

```
1. Launch terminal: python app.py
2. Go to Elon Lab: Press 4
3. Check live stats: Press 1 (Live)
   - Recent tweet count and trends
4. Analyze hourly patterns: Press 2 (Hourly)
   - Note peak hours (14:00-18:00 UTC)
5. Analyze daily patterns: Press 3 (Daily)
   - Note most active days
6. View pattern insights: Press 4 (Patterns)
   - Day-of-week analysis
   - Key statistics
7. Export notes for article
```

---

## Quick Reference

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| 1-8 | Navigate to screen |
| ESC | Go back |
| r | Refresh current screen |
| q | Quit application |
| Tab | Switch input fields |
| Enter | Confirm/Select |

### Screen Numbers
| # | Screen | Description |
|---|--------|-------------|
| 1 | Markets | Browse and search markets |
| 2 | Trading | Place and manage orders |
| 3 | Portfolio | Positions, trades, activity |
| 4 | Elon Lab | Tweet analytics |
| 5 | Research | Leaderboard, tags, patterns |
| 6 | Analytics | Spreads, momentum, volume |
| 7 | API Explorer | All 50+ endpoints |
| 8 | Settings | Wallet, theme, preferences |

### Data Sources
| API | Use Case |
|-----|----------|
| Gamma | Markets, events, search |
| CLOB | Orderbooks, trades, spreads |
| Data v2 | Positions, leaderboard, activity |
| XTracker | User tracking, metrics |
| RTDS | Real-time prices, comments |
