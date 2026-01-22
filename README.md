# YES/NO.EVENTS

**A terminal for people who trade prediction markets and want to stop clicking around.**

I built this because Polymarket's web UI is slow when you're trying to research multiple markets, compare prices, and place orders quickly. I wanted something that feels like a Bloomberg terminal but for prediction markets.

## What this is

A keyboard-driven trading terminal for [Polymarket](https://polymarket.com). You can:

- Browse and search markets without leaving your terminal
- See orderbooks, spreads, and prices at a glance
- Place orders, manage positions, track P&L
- Run quant models (Monte Carlo, Kelly criterion)
- Track Elon's tweet patterns (yes, really - there are markets for that)

## What this is NOT

- **Not a bot** - You make the decisions, this just makes research faster
- **Not financial advice** - I'm a developer, not a trader. Use at your own risk
- **Not for beginners** - You should already understand prediction markets
- **Not polished** - This is a personal tool I'm sharing, not a product

## Quick look

```
╔══════════════════════════════════════════════════════════════════════╗
║  YES/NO.EVENTS                                        ● CONNECTED    ║
╚══════════════════════════════════════════════════════════════════════╝

  1 MARKETS          2 TRADING          3 PORTFOLIO
  > Trending         > Place orders     > Positions
  > Search           > Open orders      > P&L

  4 ELON LAB         5 RESEARCH         6 ANALYTICS
  > Tweet heatmap    > Top traders      > Spreads
  > Patterns         > Tag volume       > Momentum

Press 1-6 to navigate  |  ? help  |  q quit
```

## Install

```bash
git clone https://github.com/exhuman777/yesno-events.git
cd yesno-events

python3.12 -m venv .venv
source .venv/bin/activate
pip install py-clob-client textual rich pyyaml numpy scikit-learn

chmod +x *.sh
```

## Run

```bash
# Terminal UI (17 screens, vim-like navigation)
./yesno.sh

# Web dashboard (localhost:8888)
./run.sh

# Trading cockpit with live feed (localhost:8891)
python cockpit_web.py
```

## Three interfaces

| Interface | Port | Best for |
|-----------|------|----------|
| **Terminal TUI** | - | Power users, SSH, keyboard warriors |
| **Web Dashboard** | 8888 | Browsing, charts, casual research |
| **Trading Cockpit** | 8891 | Active trading, live orderbook |

## Trading

To actually trade (not just browse), you need Polymarket API credentials:

```bash
# Create config file
cat > data/.trading_config.json << 'EOF'
{
  "private_key": "0x...",
  "api_key": "...",
  "api_secret": "...",
  "passphrase": "..."
}
EOF
```

Get these from your Polymarket account. Never commit this file.

## CLI tools

```bash
./trade.sh scan              # Scan markets
./trade.sh ev                # Find +EV opportunities
./trade.sh book 1148943      # View orderbook
./quant.sh mc 450 72         # Monte Carlo: 450 tweets in 72h
./quant.sh kelly 1148943 0.20  # Kelly sizing
./search.sh markets "bitcoin"  # Search
./tracker.sh cal             # Tweet calendar
```

## The Elon thing

There are Polymarket markets on "How many times will Elon tweet this week?"

I track his tweeting patterns (31 days of data, hourly heatmaps) to find edges. Sounds ridiculous but the markets have real volume. Screen 4 (Elon Lab) shows:
- Hourly activity heatmap
- Day-of-week patterns
- Peak detection
- Historical comparison

## Project structure

```
yesno-events/
├── app.py              # Terminal UI (Textual) - 6,752 lines
├── dashboard4all.py    # Web dashboard - 12,779 lines
├── cockpit_web.py      # Trading cockpit with live feed
├── quant.py            # Monte Carlo, Kelly criterion
├── search.py           # TF-IDF vector search
├── tracker.py          # Tweet calendar
├── rtds_client.py      # WebSocket streaming
├── *.sh                # CLI wrappers
├── tests/              # Test suite
├── docs/               # Documentation
├── data/               # Cache, config (gitignored secrets)
└── calendar/           # Event tracking
```

## APIs integrated

| API | Purpose |
|-----|---------|
| **CLOB** | Trading, orderbook, prices |
| **Gamma** | Markets, events, profiles |
| **Data** | Positions, leaderboard |
| **Polyrouter** | Fast search, trending |
| **WebSocket** | Real-time price streaming |

## Keyboard shortcuts (TUI)

| Key | Action |
|-----|--------|
| `1-9, 0` | Switch screens |
| `j/k` | Navigate down/up |
| `Enter` | Select/view details |
| `b` | Buy |
| `s` | Sell |
| `/` | Search |
| `q` | Quit |

## Limitations

- **Polymarket only** - No other prediction markets
- **API rate limits** - Don't spam requests
- **Pre-computed Elon data** - Tweet data is static (Dec 19 - Jan 18, 2025)
- **Solo project** - I fix bugs when I have time

## Related projects

- [claude-trader](https://github.com/exhuman777/claude-trader) - Natural language trading via Claude Code
- [polymarket-python](https://github.com/exhuman777/polymarket-python) - Clean Python API wrapper

## License

MIT - Do whatever you want with it.

---

*Built for traders who think in probabilities.*
