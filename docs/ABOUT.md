# About YES/NO.EVENTS

## What is YES/NO.EVENTS?

**YES/NO.EVENTS** is a professional-grade trading terminal for Polymarket prediction markets. It combines:

- **Bloomberg Terminal-style interface** - Dense, information-rich displays
- **Keyboard-first operation** - Trade without touching the mouse
- **Claude AI integration** - Natural language trading commands
- **Multi-platform** - Terminal TUI + Web dashboard + CLI tools

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            YES/NO.EVENTS TRADING PLATFORM                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐          │
│  │  TERMINAL TUI      │   │  WEB DASHBOARD     │   │  CLAUDE AI         │          │
│  │  app.py            │   │  dashboard4all.py  │   │  claude_trader.py  │          │
│  │  13+ screens       │   │  Port 8888         │   │  Natural language  │          │
│  │  Full kbd control  │   │  Real-time UI      │   │  trading assistant │          │
│  └─────────┬──────────┘   └─────────┬──────────┘   └─────────┬──────────┘          │
│            │                        │                        │                      │
│            └────────────────────────┼────────────────────────┘                      │
│                                     │                                               │
│                                     ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────┐ │
│  │                         TRADING API LAYER - trading.py                         │ │
│  │                         (3,700+ lines, 100+ functions)                         │ │
│  └───────────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                               │
│  ┌──────────────┬──────────────┬────┴───────────┬───────────────┬───────────────┐  │
│  │  POLYROUTER  │   GAMMA      │    CLOB        │   DATA API    │   ANTHROPIC   │  │
│  │  Search      │   Markets    │    Orderbook   │   Positions   │   Claude API  │  │
│  └──────────────┴──────────────┴────────────────┴───────────────┴───────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **TUI App** | app.py | 6,400+ | Terminal interface, 13 screens |
| **API Layer** | trading.py | 3,700+ | 100+ functions, all Polymarket APIs |
| **Web Dashboard** | dashboard4all.py | 12,000+ | Browser interface, REST endpoints |
| **Claude AI** | claude_trader.py | 500+ | Natural language trading |

---

## Key Features

### 1. Claude AI Trading (NEW)

Trade using natural language commands:

```
"Buy 10 YES shares of Bitcoin above 100k at 35 cents"
"Sell my Trump positions at 85 cents"
"Find markets about AI with high volume"
"Build a ladder from 60 to 70 cents, 5 shares each"
```

**Features:**
- Parses commands into executable trades
- Confirmation before execution
- Market database with 1400+ indexed markets
- Research functions for quick lookups

### 2. Terminal TUI (13 Screens)

| Key | Screen | Description |
|-----|--------|-------------|
| `1` | Markets | Browse & search all markets |
| `2` | Trading | Buy/sell with orderbook |
| `3` | Portfolio | Positions, P&L tracking |
| `4` | Elon Lab | Tweet analytics |
| `5` | Research | Market discovery |
| `6` | Analytics | Spread, momentum |
| `7` | API Explorer | Test 50+ endpoints |
| `8` | Settings | Wallet, API keys |
| `9` | Edge Scanner | Find mispricing |
| `0` | Automation | Spike detection |
| `-` | Quant | Monte Carlo, backtest |
| `C` | Claude AI | Natural language |
| `?` | Help | Keyboard shortcuts |

### 3. Web Dashboard

- **Real-time market grid** with card-style display
- **Claude AI chat interface** built-in
- **Event calendar** with FOMC, earnings
- **Elon tweet tracker** with analytics
- **Market detail** with orderbook, trades

### 4. CLI Tools

```bash
./trade.sh scan              # Scan markets
./trade.sh ev                # Find +EV opportunities
./trade.sh buy 1148943 0.15 100  # Buy shares
./quant.sh mc 450 72         # Monte Carlo projection
./quant.sh kelly 1148943 0.20    # Kelly criterion
```

---

## Quick Start

### 1. Setup

```bash
# Clone and setup
git clone <repo-url>
cd dashboard4all
python3.12 -m venv .venv
source .venv/bin/activate
pip install py-clob-client textual rich pyyaml numpy scikit-learn flask anthropic
```

### 2. Configure

```bash
# Create config
cat > data/.trading_config.json << 'EOF'
{
  "funder": "0xYourWallet",
  "private_key": "your_private_key",
  "ANTHROPIC_API_KEY": "sk-ant-api03-..."
}
EOF
```

### 3. Run

```bash
# Terminal TUI
./yesno.sh

# Web Dashboard (localhost:8888)
./run.sh
```

---

## API Integrations

| API | Endpoint | Functions | Purpose |
|-----|----------|-----------|---------|
| **CLOB** | clob.polymarket.com | 15 | Trading, orderbook |
| **Gamma** | gamma-api.polymarket.com | 12 | Markets, events |
| **Data** | data-api.polymarket.com | 10 | Positions, leaderboard |
| **Polyrouter** | api-v2.polyrouter.io | 2 | Fast search |
| **Anthropic** | api.anthropic.com | 4 | Claude AI |

---

## Trading Examples

### Natural Language (Claude AI)

```python
# In TUI: Press 'C' for Claude AI screen
# In Web: Click Claude AI tab or press 'C'

"Buy 5 YES of oil tanker market"
→ Parses: {action: "BUY", side: "YES", quantity: 5, market_query: "oil tanker"}
→ Shows confirmation with price
→ Execute or cancel
```

### CLI Trading

```bash
# Buy YES shares
./trade.sh buy 1148943 0.15 100

# Sell with limit price
./trade.sh sell 1148943 0.85 50

# Ladder orders (multiple at different prices)
./trade.sh ladder 1148943 0.60 0.70 10
# Places 10 orders from 60¢ to 70¢
```

### API Direct

```python
from trading import place_order, get_orderbook

# Check orderbook first
ob = get_orderbook("12345")
print(f"Best bid: {ob['bids'][0]}, Best ask: {ob['asks'][0]}")

# Place order
result = place_order(
    token_id="12345",
    side="BUY",
    price=0.35,
    size=10
)
print(f"Order ID: {result['id']}")
```

---

## Design Philosophy

### Lo-Fi Aesthetic
- Muted colors (`#0a0a12`, `#12121a`)
- Soft accent (`#e94560` pink)
- Monospace fonts (JetBrains Mono)
- ASCII-style borders (`┌─ ─┐ └─ ─┘`)

### Keyboard First
- Single-key navigation (1-9, 0, -)
- Vim-style movement (j/k, h/l)
- Quick actions (B=buy, S=sell, W=watch)
- Search focus with `/`

### Information Dense
- Bloomberg Terminal inspiration
- Multiple panels visible
- Real-time data updates
- Minimal chrome, maximum data

---

## Data Sources

### Polymarket APIs
- **Gamma**: Markets, events, tags, profiles
- **CLOB**: Orderbooks, trades, prices
- **Data**: Positions, activity, leaderboard

### Pre-computed Data
- Elon tweets (31 days, 2,198 tweets)
- Market database (1,400+ markets indexed)
- Tag analytics, volume patterns

---

## Limitations

1. **Rate Limiting**: Polymarket APIs may rate-limit
2. **Data Delay**: Market data may be delayed
3. **No Guarantees**: Not financial advice
4. **Self-hosted**: Runs locally only

---

## License

MIT License - Free for personal and commercial use.

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run tests: `python -m pytest tests/ -v`
5. Submit pull request

---

*Built with Python, Textual, Anthropic Claude, and Polymarket APIs*

*For serious prediction market traders who demand professional-grade tools.*
