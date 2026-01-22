# YES/NO.EVENTS - CLI Manual

## Quick Start

```bash
# 1. Clone and enter directory
git clone https://github.com/your-repo/yesno-events.git
cd yesno-events

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install textual rich requests flask pyyaml numpy

# 4. Run the terminal app
python app.py

# 5. Or run the web dashboard
python web_app.py
# Then open http://localhost:5001
```

---

## Terminal App Commands

### Global Keys (work everywhere)

| Key | Action |
|-----|--------|
| `ESC` | Go back / Close modal |
| `q` | Quit application |
| `?` | Open Help Guide |

---

### Main Menu (Home Screen)

| Key | Screen | Description |
|-----|--------|-------------|
| `1` | Markets | Browse & search markets |
| `2` | Trading | Place orders |
| `3` | Portfolio | Positions & P&L |
| `4` | Elon Lab | Tweet analytics |
| `5` | Research | Leaderboard & patterns |
| `6` | Analytics | Spreads & momentum |
| `7` | API Explorer | All 50+ endpoints |
| `8` | Settings | Wallet & theme |
| `9` | World Map | Global market view |
| `?` | Help | In-app guide |
| `q` | Quit | Exit application |

---

### Screen 1: Markets

| Key | Action |
|-----|--------|
| `F1` | Show trending markets (by volume) |
| `F2` | Enter search mode |
| `F3` | Show Elon-related markets only |
| `r` | Refresh data |
| `TAB` | Focus table for navigation |
| `Enter` | View market details |
| `j` | Move cursor down |
| `k` | Move cursor up |
| `ESC` | Go back to main menu |

**Search Mode (F2):**
1. Press `F2` to enter search mode
2. Type your query (e.g., "bitcoin", "trump")
3. Press `Enter` to search
4. Press `TAB` to focus results table
5. Use `j/k` or arrow keys to navigate
6. Press `Enter` to view market details

---

### Screen 2: Trading

| Key | Action |
|-----|--------|
| `1` | Limit order form |
| `2` | Market order form |
| `r` | Refresh orders |
| `ESC` | Go back |

**Placing an Order:**
1. Select a market from Markets screen first
2. Press `2` for Trading
3. Choose order type (1=Limit, 2=Market)
4. Enter: Token ID, Side (BUY/SELL), Price, Size
5. Confirm order

---

### Screen 3: Portfolio

| Key | Action |
|-----|--------|
| `1` | Positions tab |
| `2` | Trades tab |
| `3` | Activity tab |
| `4` | Orders tab |
| `r` | Refresh data |
| `ESC` | Go back |

**Tabs:**
- **Positions**: Current holdings with P&L
- **Trades**: Trade history
- **Activity**: Account activity log
- **Orders**: Open/pending orders

---

### Screen 4: Elon Lab

| Key | Action |
|-----|--------|
| `1` | Live overview |
| `2` | Hourly heatmap |
| `3` | Daily patterns |
| `4` | Behavior analysis |
| `r` | Refresh data |
| `ESC` | Go back |

**Tabs:**
- **Live**: Current tweet stats, recent activity
- **Hourly**: 24-hour heatmap by hour (UTC)
- **Daily**: Day-by-day tweet counts with sparklines
- **Patterns**: Day-of-week analysis, peak hours

---

### Screen 5: Research

| Key | Action |
|-----|--------|
| `1` | Leaderboard |
| `2` | Tag analytics |
| `3` | Pattern discovery |
| `4` | Correlations |
| `r` | Refresh data |
| `ESC` | Go back |

**Tabs:**
- **Leaderboard**: Top traders by P&L
- **Tags**: Market tags by volume
- **Patterns**: Price patterns & anomalies
- **Correlations**: Cross-market correlations

---

### Screen 6: Analytics

| Key | Action |
|-----|--------|
| `1` | Spread analysis |
| `2` | Momentum scanner |
| `3` | Volume leaders |
| `4` | Liquidity ratings |
| `r` | Refresh data |
| `ESC` | Go back |

**Tabs:**
- **Spreads**: Bid-ask spread analysis
- **Momentum**: Price momentum indicators
- **Volume**: Top volume markets
- **Liquidity**: Liquidity depth ratings

---

### Screen 7: API Explorer

| Key | Action |
|-----|--------|
| `1` | Gamma API |
| `2` | CLOB API |
| `3` | Data API |
| `4` | XTracker API |
| `5` | Analysis API |
| `6` | Account API |
| `r` | Refresh |
| `ESC` | Go back |

**API Categories:**
- **Gamma**: Markets, events, search, comments
- **CLOB**: Orderbook, trades, spreads
- **Data**: Positions, leaderboard, activity
- **XTracker**: User tracking, metrics
- **Analysis**: EV, price history, volume
- **Account**: Balances, orders, trades

---

### Screen 8: Settings

| Key | Action |
|-----|--------|
| `1` | Wallet tab |
| `2` | Theme tab |
| `3` | Display tab |
| `n` | Set Neon theme |
| `d` | Set Dark theme |
| `h` | Set Hacker theme |
| `o` | Set Ocean theme |
| `c` | Toggle compact mode |
| `s` | Toggle sparklines |
| `a` | Aligned number format |
| `k` | Compact number format |
| `l` | Logout (when logged in) |
| `ESC` | Go back |

**Themes:**
- **Neon**: Orange accent, dark bg
- **Dark**: Blue accent, navy bg
- **Hacker**: Green accent, black bg
- **Ocean**: Teal accent, deep blue bg

---

### Screen 9: World Map

| Key | Action |
|-----|--------|
| `1` | View US markets |
| `2` | View Europe markets |
| `3` | View Asia markets |
| `4` | View Crypto markets |
| `s` | Search (redirects to Markets) |
| `r` | Refresh data |
| `ESC` | Go back |

**Regions:**
- **US**: Politics, elections, Fed
- **Europe**: UK, Germany, France, EU
- **Asia**: China, Japan, India, Korea
- **Crypto**: Bitcoin, Ethereum, Solana (global)
- **Latin America**: Brazil, Argentina, Mexico
- **Africa**: Nigeria, South Africa, Egypt

---

### Help Screen (?)

| Key | Action |
|-----|--------|
| `1` | Overview |
| `2` | Navigation |
| `3` | Search guide |
| `4` | Trading guide |
| `5` | API reference |
| `6` | Pro tips |
| `ESC` | Go back |

---

### Market Detail Modal

| Key | Action |
|-----|--------|
| `b` | Buy YES |
| `s` | Sell YES |
| `c` | Toggle comments |
| `r` | Refresh data |
| `ESC` | Close modal |

---

## CLI Scripts

### trade.sh - Market Operations

```bash
# Scan Elon tweet markets
./trade.sh scan

# Find EV opportunities
./trade.sh ev --edge 0.05

# Get orderbook depth
./trade.sh book <market_id>

# Search markets
./trade.sh find "bitcoin"

# View price distribution
./trade.sh dist --event jan13_20

# Get spread analysis
./trade.sh spread <market_id>
```

### tracker.sh - Tweet Analytics

```bash
# View tweet calendar
./tracker.sh cal --days 7

# Full dashboard
./tracker.sh dash

# Export data
./tracker.sh export --format json

# Add tweet count manually
./tracker.sh add 2026-01-18 --hour 14 --count 5

# Show hourly breakdown
./tracker.sh hourly
```

### quant.sh - Quantitative Models

```bash
# Monte Carlo projection
./quant.sh mc <current_tweets> <hours_elapsed> --sims 50000

# Kelly Criterion calculator
./quant.sh kelly <market_id> <your_probability> --bankroll 1000

# Expected Value calculation
./quant.sh ev <market_id> --edge 0.05

# Correlation analysis
./quant.sh corr <market_id_1> <market_id_2>
```

---

## Web Dashboard

```bash
# Start web server
python web_app.py

# Access in browser
open http://localhost:5001
```

**Web Features:**
- Interactive charts (Chart.js)
- Real-time price updates
- Portfolio visualization
- Mobile-friendly design

---

## Environment Variables

```bash
# Optional: Set wallet for trading
export POLYMARKET_WALLET="0x..."
export POLYMARKET_PRIVATE_KEY="..."

# Optional: API endpoints (defaults work)
export GAMMA_API="https://gamma-api.polymarket.com"
export CLOB_API="https://clob.polymarket.com"
export DATA_API="https://data-api.polymarket.com"
```

---

## File Structure

```
yesno-events/
├── app.py              # Main TUI application
├── trading.py          # API functions
├── web_app.py          # Flask web dashboard
├── trade.sh            # Trading CLI
├── tracker.sh          # Tweet tracking CLI
├── quant.sh            # Quant models CLI
├── data/
│   ├── cache/          # API response cache
│   └── elon/           # Pre-computed tweet data
├── docs/
│   ├── CLI_MANUAL.md   # This file
│   ├── ARCHITECTURE.md # System design
│   └── USER_STORIES.md # Usage examples
├── tests/
│   ├── test_app_ui.py  # UI tests
│   └── test_trading_api.py # API tests
└── templates/          # Web templates
```

---

## Troubleshooting

### App won't start
```bash
# Check Python version (need 3.10+)
python3 --version

# Reinstall dependencies
pip install --upgrade textual rich requests
```

### No markets loading
```bash
# Check internet connection
curl -I https://gamma-api.polymarket.com

# Check cache
ls -la data/cache/markets/
```

### Tests failing
```bash
# Run with verbose output
python -m pytest tests/ -v --tb=long

# Run specific test
python -m pytest tests/test_app_ui.py::TestHelpScreen -v
```

### Theme not changing
```bash
# Reset state by restarting app
# Or press the theme key twice (n, n)
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  YES/NO.EVENTS - Quick Reference                        │
├─────────────────────────────────────────────────────────┤
│  NAVIGATION                                             │
│    1-9     Jump to screen                               │
│    ?       Help guide                                   │
│    ESC     Go back                                      │
│    q       Quit                                         │
├─────────────────────────────────────────────────────────┤
│  MARKETS (1)                                            │
│    F1      Trending    F2  Search    F3  Elon           │
│    TAB     Focus table                                  │
│    Enter   View detail                                  │
│    j/k     Navigate                                     │
├─────────────────────────────────────────────────────────┤
│  TABS (most screens)                                    │
│    1-4     Switch tabs                                  │
│    r       Refresh                                      │
├─────────────────────────────────────────────────────────┤
│  SETTINGS (8)                                           │
│    n/d/h/o Theme: Neon/Dark/Hacker/Ocean               │
│    c       Compact mode                                 │
│    s       Sparklines                                   │
├─────────────────────────────────────────────────────────┤
│  RUN                                                    │
│    python app.py        Terminal UI                     │
│    python web_app.py    Web dashboard (:5001)           │
└─────────────────────────────────────────────────────────┘
```
