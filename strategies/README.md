# Polymarket Quant Strategies

Trading strategies, signals, and automations for Elon tweet markets and beyond.

## Directory Structure

```
strategies/
├── signals/        # Entry/exit signal definitions
├── automations/    # If-this-then-that rules
├── backtests/      # Historical analysis & backtests
├── models/         # Prediction models & EV calculators
└── README.md       # This file
```

## Strategy File Format

Each `.md` file follows this structure:

```markdown
---
name: Strategy Name
type: signal|automation|backtest|model
markets: elon_tweets|all|specific_ids
status: draft|testing|live|archived
version: 1.0
---

# Strategy Name

## Overview
Brief description of what this strategy does.

## Logic
Detailed explanation of the trading logic.

## Code
\`\`\`python
# Executable code block
def signal():
    pass
\`\`\`

## Parameters
| Param | Default | Description |
|-------|---------|-------------|

## Backtest Results
Historical performance data.

## Notes
Additional observations.
```

## Quick Start

```bash
# List all strategies
./trade.sh strat list

# Run a strategy
./trade.sh strat run signals/momentum.md

# Backtest a strategy
./trade.sh strat backtest signals/momentum.md --from 2026-01-01

# Create new strategy from template
./trade.sh strat new signal my_signal
```

## CLI Integration

Strategies integrate with the trading CLI:

```bash
# Run automation in background
./trade.sh auto start automations/bracket_ladder.md

# Check running automations
./trade.sh auto status

# Stop automation
./trade.sh auto stop bracket_ladder
```

## Contributing

1. Fork the repo
2. Create your strategy in `strategies/`
3. Test with paper trading
4. Submit PR with backtest results

## Vision

This will become a shareable library of:
- Trading strategies
- Signal generators
- Risk management rules
- Backtesting frameworks
- Claude-assisted analysis prompts

All in simple `.md` format for easy reading, editing, and version control.
