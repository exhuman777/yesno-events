# CLAUDE.md

> Instructions for AI agents (Claude Code, Cursor, Copilot, etc.) working with this codebase.

## What is this?

YES/NO.EVENTS is a Polymarket trading terminal with three interfaces:
- **TUI** (app.py) - Terminal UI built with Textual
- **Web** (dashboard4all.py) - Browser dashboard on port 8888
- **Cockpit** (cockpit_web.py) - Trading interface on port 8891

## Quick commands

```bash
# Setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install py-clob-client textual rich pyyaml numpy scikit-learn

# Run
./yesno.sh          # TUI
./run.sh            # Web dashboard (8888)
python cockpit_web.py  # Trading cockpit (8891)

# Test
python -m pytest tests/ -v

# Kill servers
lsof -ti:8888 | xargs kill -9
lsof -ti:8891 | xargs kill -9
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YES/NO.EVENTS                           │
├─────────────────────────────────────────────────────────────┤
│  app.py          │  dashboard4all.py  │  cockpit_web.py    │
│  TUI (Textual)   │  Web (HTTP)        │  Trading (HTTP+WS) │
│  6,752 lines     │  12,779 lines      │  1,745 lines       │
├─────────────────────────────────────────────────────────────┤
│                     SHARED MODULES                          │
│  quant.py   search.py   tracker.py   rtds_client.py        │
├─────────────────────────────────────────────────────────────┤
│                   POLYMARKET APIs                           │
│  CLOB │ Gamma │ Data │ Polyrouter │ WebSocket              │
└─────────────────────────────────────────────────────────────┘
```

## File map

| File | Lines | Purpose | Entry point? |
|------|-------|---------|--------------|
| `app.py` | 6,752 | TUI terminal (Textual framework) | Yes: `./yesno.sh` |
| `dashboard4all.py` | 12,779 | Web dashboard, embedded HTML/CSS/JS | Yes: `./run.sh` |
| `cockpit_web.py` | 1,745 | Trading cockpit with WebSocket | Yes: `python cockpit_web.py` |
| `quant.py` | 521 | Monte Carlo, Kelly criterion | No |
| `search.py` | 571 | TF-IDF vector search | No |
| `tracker.py` | 570 | Tweet calendar aggregation | No |
| `rtds_client.py` | 611 | WebSocket real-time client | No |

## Key patterns

### 1. Venv path injection
Both `app.py` and `dashboard4all.py` manually inject venv site-packages for portability:
```python
import sys
from pathlib import Path
venv_path = Path(__file__).parent / ".venv/lib/python3.12/site-packages"
if venv_path.exists():
    sys.path.insert(0, str(venv_path))
```

### 2. Shell wrappers
All `.sh` scripts activate venv before running Python:
```bash
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python3 app.py "$@"
```

### 3. API endpoints
```python
# Polyrouter (fast search)
"https://api-v2.polyrouter.io/markets/search"
"https://api-v2.polyrouter.io/markets/trending"

# Gamma (market data)
"https://gamma-api.polymarket.com/markets"
"https://gamma-api.polymarket.com/events"

# CLOB (trading)
"https://clob.polymarket.com/book"
"https://clob.polymarket.com/price"

# Data (positions)
"https://data-api.polymarket.com/positions"
```

### 4. Price format
Prices are 0.00-1.00 internally, displayed as cents:
```python
# Internal: 0.35
# Display: 35¢ or $0.35
f"{price*100:.0f}¢"
```

## Common tasks

### Add a new screen to TUI
1. Create screen class in `app.py` extending `Screen`
2. Add to `SCREENS` dict
3. Add keybinding in `on_key()`

### Add API endpoint
1. Add function to relevant section in `app.py` or create new module
2. Handle rate limits (add small delays)
3. Cache responses in `data/cache/`

### Modify web dashboard
Dashboard is single-file with embedded HTML/CSS/JS. Search for:
- `HTML_TEMPLATE =` for HTML
- `<style>` for CSS
- `<script>` for JavaScript

## Data files

| Path | Content | Gitignored? |
|------|---------|-------------|
| `data/.trading_config.json` | API credentials | Yes |
| `data/elon_tweets.json` | Pre-computed tweets | No |
| `data/cache/` | API response cache | Yes |
| `data/.yesno_config.json` | TUI settings | No |

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Specific
python -m pytest tests/test_trading_api.py -v
python -m pytest tests/test_quant_models.py -v
```

## Don't

- Don't commit `data/.trading_config.json` (secrets)
- Don't spam Polymarket APIs (rate limits)
- Don't assume prices are percentages (they're 0.00-1.00)
- Don't edit shell scripts without testing venv activation

## Do

- Do read existing code before adding features
- Do use the existing API patterns
- Do test with `python -m pytest`
- Do check `git status` before committing
