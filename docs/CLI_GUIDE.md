# YES/NO.EVENTS CLI Complete Guide

Comprehensive reference for all CLI commands with real examples.

---

## Security Note

**Your API keys and wallet credentials are 100% LOCAL and SAFE.**
- Stored in `data/.trading_config.json` (gitignored)
- Never tracked by git, never pushed to remote
- Only used locally to sign transactions

---

## Quick Reference

| Command | Description | Example |
|---------|-------------|---------|
| `scan` | Scan Elon tweet markets | `./trade.sh scan` |
| `ev` | Find EV opportunities | `./trade.sh ev --edge 0.05` |
| `find` | Search all markets | `./trade.sh find "bitcoin"` |
| `price` | Get current price | `./trade.sh price 1172399` |
| `book` | View orderbook | `./trade.sh book 1172399` |
| `depth` | Orderbook depth analysis | `./trade.sh depth 1172399` |
| `profile` | User profile lookup | `./trade.sh profile 0x...` |
| `positions` | User positions | `./trade.sh positions 0x...` |
| `tags` | List market tags | `./trade.sh tags` |
| `market` | Market details | `./trade.sh market 1172399` |

---

## Market Discovery

### `scan` - Scan Elon Tweet Markets

Scans all active Elon Musk tweet prediction markets.

```bash
# Scan all events
./trade.sh scan

# Scan specific event
./trade.sh scan --event jan16_23

# Output as JSON
./trade.sh scan --json
```

**Real Output:**
```
=== Jan 16-23 (32 brackets) ===
  .    120-139    0.1%  $   387k  [1172375]
  .    140-159    0.1%  $   368k  [1172376]
  +    400-419    1.5%  $    82k  [1172393]
  +    420-439    2.1%  $    78k  [1172394]
  *    520-539   11.0%  $   116k  [1172399]   <- Peak probability
```

**Indicators:**
- `.` = probability < 1%
- `+` = probability 1-10%
- `*` = probability > 10%

---

### `find` - Search All Markets

Search Polymarket using Polyrouter API.

```bash
# Search by keyword
./trade.sh find "bitcoin"

# Limit results
./trade.sh find "trump" --limit 5
```

**Real Output:**
```
=== Search: 'bitcoin' ===

Title                                                        YES     Volume
---------------------------------------------------------------------------
Will Bitcoin reach $150,000 in January?                     0.4%     $8750k
Will Bitcoin reach $100,000 in January?                    41.5%     $4776k
Will Bitcoin dip to $85,000 in January?                    14.0%     $2645k
```

---

### `events` - List Active Events

```bash
# List top events by volume
./trade.sh events --limit 10

# JSON output
./trade.sh events --json
```

---

### `trending` - Trending Markets

```bash
./trade.sh trending --limit 10
```

---

## Analysis Commands

### `ev` - Expected Value Opportunities

Finds mispriced markets with edge > threshold.

```bash
# Default 3% edge threshold
./trade.sh ev

# Custom threshold (5% edge)
./trade.sh ev --edge 0.05
```

**Real Output:**
```
=== EV Opportunities (edge > 3%) ===

Event        Bracket         Price   Expected     Edge Signal
------------------------------------------------------------
jan17_19     165-189         29.5%      21.2%    +8.2% SHORT
jan17_19     140-164         28.5%      22.7%    +5.8% SHORT
jan13_20     560-579         20.0%      16.4%    +3.6% SHORT
jan17_19     190-214         14.0%      17.5%    -3.5% LONG
```

**Signals:**
- `SHORT` = Market overpriced, sell YES
- `LONG` = Market underpriced, buy YES

---

### `dist` - Probability Distribution

Analyze probability distribution across brackets.

```bash
./trade.sh dist --event jan16_23
```

---

### `depth` - Orderbook Depth Analysis

```bash
./trade.sh depth 1172399
```

**Output:**
```
=== Orderbook Depth: 1172399 ===

Best Bid: $0.105  |  Best Ask: $0.115
Spread: $0.010 (8.7%)
Bid Depth: $1,250.00  |  Ask Depth: $980.00
Imbalance: +0.12 (bid heavy)

Top 5 Bids:
  $0.105 x 500.0
  $0.100 x 300.0
  ...
```

---

## Price & Trading

### `price` - Current Price

```bash
./trade.sh price 1172399
```

**Output:**
```
YES: 0.1100  NO: 0.8900
CLOB: bid=0.105 ask=0.115
```

---

### `book` - Orderbook

```bash
./trade.sh book 1172399
```

---

### `history` - Price History

```bash
# Default fidelity
./trade.sh history 1172399

# Custom fidelity (minutes)
./trade.sh history 1172399 --fidelity 60
```

---

### `trades` - Recent Trades

```bash
./trade.sh trades 1172399 --limit 20
```

---

## User Research

### `profile` - User Profile

Look up any trader's profile by wallet address.

```bash
./trade.sh profile 0x1234567890abcdef...
```

**Output:**
```
=== Profile: 0x123456...cdef ===

Username:     trader123
PnL:          $12,345.67
Volume:       $456,789.00
Positions:    8,234.50
Markets:      156
Joined:       2024-03-15
```

---

### `positions` - User Positions

```bash
# Open positions
./trade.sh positions 0x1234...

# Include closed
./trade.sh positions 0x1234... --closed
```

---

### `activity` - Trading Activity

```bash
./trade.sh activity 0x1234... --limit 25
```

---

### `leaderboard` - Top Traders

```bash
# Daily leaderboard
./trade.sh leaderboard

# Weekly
./trade.sh leaderboard --period weekly
```

---

## Market Details

### `market` - Market Info

```bash
./trade.sh market 1172399
```

**Output:**
```
=== Market: 1172399 ===

Question:   Will Elon Musk post 520-539 tweets from January 16 to January 23, 2026?
Event:      N/A
Volume:     $115,862
Liquidity:  $18,442
Created:    2026-01-15
End Date:   2026-01-23

Outcomes: Yes, No
```

---

### `event` - Event Info

```bash
./trade.sh event 12345
```

---

### `tags` - Market Tags

List all available market tags for filtering.

```bash
./trade.sh tags
```

**Output:**
```
=== Market Tags (100) ===

  NEH                 Slovak Republic     robert Fico         madison sheahan
  Ankara              Auckland            fico                Nancy Pelosi
  SBF                 haley               istanbul            Trade War
  Bitcoin             Trump               Crypto              Election
```

---

### `gsearch` - Gamma Search

Search markets, events, and profiles via Gamma API.

```bash
./trade.sh gsearch "crypto"
```

---

### `spreads` - Bid-Ask Spreads

```bash
./trade.sh spreads 1172399 1172400 1172401
```

---

## Strategy Management

### `strat list` - List Strategies

```bash
./trade.sh strat list
./trade.sh strat list --type signals
```

---

### `strat new` - Create Strategy

```bash
./trade.sh strat new signal "Momentum Entry"
./trade.sh strat new backtest "Historical Analysis"
```

---

### `strat show` - View Strategy

```bash
./trade.sh strat show signals/momentum_entry.md
```

---

### `strat edit` - Edit Strategy

```bash
./trade.sh strat edit signals/momentum_entry.md
```

---

## Trading Commands

### Setup Wallet

```bash
# Set funder address (your wallet)
./trade.sh setup --funder 0xYOUR_ADDRESS

# Set signature type
# 0 = EOA (direct wallet)
# 1 = POLY_PROXY (Polymarket login)
# 2 = GNOSIS_SAFE (browser wallet)
./trade.sh setup --funder 0x... --sig-type 1

# Derive API credentials
export POLYMARKET_PRIVATE_KEY='your_key'
./trade.sh setup --derive
```

---

### `buy` - Place Buy Order

```bash
# Buy YES at $0.15, 100 shares
./trade.sh buy 1172399 0.15 100

# Buy NO
./trade.sh buy 1172399 0.85 100 --no

# Fill-or-Kill order
./trade.sh buy 1172399 0.15 100 --fok
```

---

### `sell` - Place Sell Order

```bash
./trade.sh sell 1172399 0.20 50
```

---

### `orders` - View Open Orders

```bash
./trade.sh orders
```

---

### `cancel` - Cancel Orders

```bash
# Cancel specific order
./trade.sh cancel ORDER_ID

# Cancel all
./trade.sh cancel --all
```

---

## Volume & Open Interest

### `volume` - Live Volume

```bash
./trade.sh volume
```

---

### `oi` - Open Interest

```bash
./trade.sh oi
./trade.sh oi 1172399
```

---

## TUI Terminal (yesno.sh)

Interactive terminal UI with keyboard navigation.

```bash
./yesno.sh
```

### Key Bindings

| Key | Action |
|-----|--------|
| `d` | Dashboard (overview) |
| `u` | User profile |
| `R` | Research (leaderboard, tags) |
| `A` | Analytics (EV scanner) |
| `W` | Wallet tracker |
| `f` | Toggle favorite |
| `p` | Portfolio view |
| `S` | Strategies |
| `/` | Search (fish-style live) |
| `1-5` | Switch event tabs |
| `Enter` | View market details |
| `b` | Buy |
| `s` | Sell |
| `c` | Chart |
| `?` | Help |
| `q` | Quit |

---

## Data Storage

All data is stored locally in `data/`:

| File | Description |
|------|-------------|
| `.trading_config.json` | Credentials (GITIGNORED) |
| `.yesno_config.json` | TUI preferences |
| `elon_tweets.json` | Tweet history |
| `cache/markets/` | Market data cache |
| `cache/prices/` | Price history cache |

**Cache auto-refreshes** - stale data is automatically updated from API.

---

## API Endpoints Used

| API | Base URL | Data |
|-----|----------|------|
| Polyrouter | api-v2.polyrouter.io | Aggregated search |
| Gamma | gamma-api.polymarket.com | Markets, events, profiles |
| CLOB | clob.polymarket.com | Orderbook, trading |
| Data | data-api.polymarket.com | Positions, leaderboard |

---

## Troubleshooting

### "No markets found"
- Check internet connection
- API may be rate-limited, wait and retry

### "Invalid signature"
- Verify private key is correct
- Check signature type matches wallet type
- Re-derive credentials with `./trade.sh setup --derive`

### "Unauthorized"
- Some Data API endpoints require authentication
- Set up credentials first

---

## Examples Workflow

### Morning Research Routine

```bash
# 1. Check active events
./trade.sh scan

# 2. Find EV opportunities
./trade.sh ev

# 3. Research top traders
./trade.sh leaderboard

# 4. Deep dive into interesting market
./trade.sh depth 1172399
./trade.sh history 1172399
```

### Trade Execution

```bash
# 1. Check price and spread
./trade.sh price 1172399
./trade.sh depth 1172399

# 2. Place order
./trade.sh buy 1172399 0.12 100

# 3. Monitor
./trade.sh orders
```

---

*Built for traders who think in probabilities.*
