# FEATURES - Reusable Components

This document catalogs all features with code examples for reuse in other projects.

---

## Table of Contents

1. [Market Data](#1-market-data)
2. [Trading Execution](#2-trading-execution)
3. [Portfolio Management](#3-portfolio-management)
4. [Quantitative Models](#4-quantitative-models)
5. [Analytics](#5-analytics)
6. [Automation](#6-automation)
7. [Social/Research](#7-socialresearch)
8. [TUI Components](#8-tui-components)

---

## 1. Market Data

### 1.1 Search Markets

```python
from trading import search_markets, polyrouter_search, polyrouter_trending

# Fast search via Polyrouter (aggregated, fast)
results = polyrouter_search("bitcoin", limit=20)

# Official search via Gamma API
results = search_markets("bitcoin", limit=20)

# Get trending markets
trending = polyrouter_trending(limit=20)
for m in trending:
    print(f"{m['question']}: {m.get('yes_price', 0)}")
```

### 1.2 Get Market Details

```python
from trading import gamma_get_market, get_market_info, get_price

# Full market data
market = gamma_get_market("market_id")
# Returns: {id, question, description, endDate, outcomes, volume, ...}

# Price data
price = get_price("market_id")
# Returns: {'yes': 0.65, 'no': 0.35, 'mid': 0.65, 'source': 'clob'}

# Combined market info
info = get_market_info("market_id")
# Returns: {id, title, yes_price, no_price, volume, spread, ...}
```

### 1.3 Orderbook

```python
from trading import get_orderbook, get_orderbook_depth, get_spread

# Full orderbook
book = get_orderbook("market_id", outcome="yes")
# Returns: OrderBookSummary with bids[], asks[]

# Depth analysis
depth = get_orderbook_depth("market_id")
# Returns: {bid_depth, ask_depth, imbalance, levels: [...]}

# Spread
spread = get_spread("market_id")
# Returns: {'spread': 0.02, 'spread_pct': 3.1, 'bid': 0.63, 'ask': 0.65}
```

### 1.4 Batch Operations

```python
from trading import clob_get_prices_batch, clob_get_spreads

# Batch price fetch (efficient for multiple markets)
token_ids = ["token1", "token2", "token3"]
prices = clob_get_prices_batch(token_ids)
# Returns: {token_id: {price, side}, ...}

# Batch spreads
spreads = clob_get_spreads(token_ids)
# Returns: {token_id: {bid, ask, spread}, ...}
```

---

## 2. Trading Execution

### 2.1 Configuration

```python
from trading import load_config, save_config, derive_user_creds

# Load credentials
config = load_config()
# From: data/.trading_config.json

# Required fields:
# {
#   "host": "https://clob.polymarket.com",
#   "chain_id": 137,
#   "signature_type": 2,  # GNOSIS_SAFE for Polymarket proxy
#   "funder": "0xYourPolymarketWallet",
#   "private_key": "0xYourPrivateKey",
#   "user_api_creds": {
#     "api_key": "...",
#     "secret": "...",
#     "passphrase": "..."
#   }
# }

# Derive API credentials from private key
creds = derive_user_creds()
# Saves to config automatically
```

### 2.2 Place Orders

```python
from trading import place_order, place_bracket_orders, place_ladder_orders

# Single limit order
result = place_order(
    market_id="market_id",
    side="BUY",           # BUY or SELL
    price=0.50,           # 0.01 to 0.99
    size=10,              # Minimum 5 shares
    outcome="yes",        # yes or no
    order_type="GTC"      # GTC (Good Till Cancelled)
)
# Returns: {'orderID': '0x...', 'status': 'live', 'success': True}

# Bracket orders (both sides of spread)
result = place_bracket_orders(
    market_id="market_id",
    center_price=0.50,    # Midpoint
    spread=0.02,          # Distance from center
    size=10,
    outcome="yes"
)
# Places: BUY @ 0.48, SELL @ 0.52

# Ladder orders (multiple price levels)
result = place_ladder_orders(
    market_id="market_id",
    start_price=0.40,
    end_price=0.50,
    steps=5,              # 5 orders
    total_size=100,       # Split across orders
    side="BUY",
    outcome="yes"
)
```

### 2.3 Order Management

```python
from trading import get_open_orders, cancel_order, cancel_all_orders

# Get all open orders
orders = get_open_orders()
for o in orders:
    print(f"{o['id']}: {o['side']} @ {o['price']}")

# Cancel single order
result = cancel_order("order_id")
# Returns: {'canceled': ['order_id']}

# Cancel all orders
result = cancel_all_orders()
# Returns: {'canceled': [...]}
```

### 2.4 Mass Orders

```python
from trading import place_mass_orders

# Place multiple orders with delay
orders = [
    {"market_id": "m1", "side": "BUY", "price": 0.40, "size": 10, "outcome": "yes"},
    {"market_id": "m1", "side": "SELL", "price": 0.60, "size": 10, "outcome": "yes"},
    {"market_id": "m2", "side": "BUY", "price": 0.30, "size": 20, "outcome": "no"},
]
results = place_mass_orders(orders, delay_ms=100)
```

---

## 3. Portfolio Management

### 3.1 Positions

```python
from trading import get_positions, dataapi_get_positions, data_get_positions

# Via CLOB (authenticated)
positions = get_positions()

# Via Data API (by address)
positions = dataapi_get_positions(
    user_address="0x...",
    limit=100,
    sort_by="TOKENS"  # TOKENS, VALUE, PNL
)

# Alternative
positions = data_get_positions("0x...", closed=False)
```

### 3.2 Trades History

```python
from trading import get_user_trades, dataapi_get_trades, get_market_trades

# User's trades
trades = get_user_trades(limit=50)

# Trades by address or market
trades = dataapi_get_trades(
    user_address="0x...",  # Optional
    market="market_id",    # Optional
    limit=100,
    side="BUY"             # Optional filter
)

# Market's recent trades
trades = get_market_trades("market_id", limit=50)
```

### 3.3 Activity Feed

```python
from trading import dataapi_get_activity, dataapi_get_value

# Activity stream
activity = dataapi_get_activity(
    user_address="0x...",
    limit=100,
    activity_type=None  # TRADE, DEPOSIT, etc.
)

# Portfolio value
value = dataapi_get_value("0x...")
# Returns: {total_value, positions_value, cash_balance}
```

---

## 4. Quantitative Models

### 4.1 Technical Indicators

```python
from trading import simple_moving_average, exponential_moving_average

prices = [0.50, 0.52, 0.51, 0.53, 0.55, 0.54, 0.56, 0.58, 0.57, 0.59]

# Simple Moving Average
sma = simple_moving_average(prices, window=5)
# Returns: [None, None, None, None, 0.522, 0.530, 0.538, 0.552, 0.560, 0.568]

# Exponential Moving Average
ema = exponential_moving_average(prices, window=5)
# Returns: EMA values with more weight on recent prices
```

### 4.2 Forecasting Models

```python
from trading import (
    linear_regression_forecast,
    mean_reversion_model,
    momentum_model,
    run_all_models
)

# Linear regression
forecast = linear_regression_forecast(prices, forecast_periods=5)
# Returns: [predicted_1, predicted_2, ..., predicted_5]

# Mean reversion (Ornstein-Uhlenbeck)
forecast = mean_reversion_model(
    prices,
    equilibrium=0.5,   # Long-term mean
    speed=0.1,         # Reversion speed
    volatility=0.02,   # Random noise
    periods=10
)

# Momentum
signal = momentum_model(prices, lookback=10, threshold=0.02)
# Returns: {'signal': 'BULLISH'/'BEARISH'/'NEUTRAL', 'strength': 0.0-1.0}

# Ensemble (all models)
results = run_all_models("market_id", forecast_periods=5)
# Returns: {sma, ema, linear, mean_reversion, momentum, ensemble}
```

### 4.3 Backtesting

```python
from trading import backtest_model, optimize_model_params

# Backtest a model
results = backtest_model(
    market_id="market_id",
    model_type='linear',  # linear, mean_reversion, momentum
    lookback=50,
    test_periods=10
)
# Returns: {accuracy, mae, rmse, predictions, actuals}

# Optimize parameters
best_params = optimize_model_params(
    market_id="market_id",
    model_type='mean_reversion'
)
# Returns: {equilibrium, speed, volatility, score}
```

### 4.4 Price History

```python
from trading import get_price_history, store_price_snapshot, get_price_history_local

# From API (limited by rate limits)
history = get_price_history(
    market_id="market_id",
    interval="1h",    # 1m, 5m, 1h, 1d
    fidelity=60       # Points
)

# Store snapshot locally
store_price_snapshot("market_id", {"yes": 0.65, "no": 0.35, "timestamp": "..."})

# Retrieve local history
history = get_price_history_local("market_id", limit=100)
```

---

## 5. Analytics

### 5.1 Expected Value

```python
from trading import find_ev_opportunities, analyze_distribution

# Find EV opportunities
opportunities = find_ev_opportunities(min_edge=0.05)
# Returns: [{market_id, market_price, fair_value, edge, direction}, ...]

# Distribution analysis for an event
distribution = analyze_distribution(event="event_id")
# Returns: {brackets: [...], total_volume, peak_bracket}
```

### 5.2 Leaderboard

```python
from trading import dataapi_get_leaderboard, data_get_leaderboard

# Top traders
leaders = dataapi_get_leaderboard(
    category="OVERALL",     # OVERALL, POLITICS, CRYPTO, etc.
    time_period="DAY",      # DAY, WEEK, MONTH, ALL
    order_by="PNL",         # PNL, VOLUME, WIN_RATE
    limit=25
)

# Simplified version
leaders = data_get_leaderboard(period="daily", limit=25)
```

### 5.3 Holders Analysis

```python
from trading import dataapi_get_holders, get_top_holders

# Top holders for a market
holders = dataapi_get_holders(
    market_id="market_id",
    limit=20,
    min_balance=1,
    include_both_sides=True
)
# Returns: [{address, yes_balance, no_balance, cost_basis, pnl}, ...]

# Simplified
holders = get_top_holders("market_id", limit=10)
```

---

## 6. Automation

### 6.1 Spike Detection

```python
from trading import detect_volume_spikes, detect_liquidity_spikes, scan_all_spikes

# Volume spikes
spikes = detect_volume_spikes(
    markets=None,           # None = all markets
    threshold=2.0,          # 2x normal volume
    min_volume=5000
)
# Returns: [{market_id, current_volume, avg_volume, spike_ratio}, ...]

# Liquidity spikes
spikes = detect_liquidity_spikes(
    markets=None,
    threshold=1.5,
    min_liquidity=1000
)

# Combined scan
all_spikes = scan_all_spikes(threshold_vol=2.0, threshold_liq=1.5)
# Returns: {volume_spikes: [...], liquidity_spikes: [...]}
```

### 6.2 Auto-Trading

```python
from trading import (
    load_automation_config,
    save_automation_config,
    auto_trade_on_spike,
    run_automation_scan,
    log_automation_event,
    get_automation_log
)

# Configure automation
config = {
    "enabled": True,
    "dry_run": True,        # Simulate only
    "volume_threshold": 2.0,
    "max_position_size": 100,
    "allowed_markets": ["market1", "market2"],
}
save_automation_config(config)

# Manual trade on spike
spike = {"market_id": "...", "spike_ratio": 2.5, "direction": "BUY"}
result = auto_trade_on_spike(spike, config)

# Run full automation scan
results = run_automation_scan()
# Detects spikes and optionally trades (if not dry_run)

# Log events
log_automation_event("SPIKE_DETECTED", {"market_id": "...", "ratio": 2.5})

# View log
log = get_automation_log(limit=50)
```

---

## 7. Social/Research

### 7.1 User Profiles

```python
from trading import gamma_get_profile, gamma_get_public_profile

# Full profile (authenticated)
profile = gamma_get_profile("0x...")
# Returns: {address, username, avatar, bio, stats, ...}

# Public profile
profile = gamma_get_public_profile("0x...")
```

### 7.2 Comments

```python
from trading import gamma_get_comments

# Event comments
comments = gamma_get_comments(
    entity_type="Event",
    entity_id="event_id",
    limit=50
)

# Market comments
comments = gamma_get_comments(
    entity_type="Market",
    entity_id="market_id",
    limit=50
)
```

### 7.3 XTracker (Twitter Analytics)

```python
from trading import (
    xtracker_get_user,
    xtracker_get_trackings,
    xtracker_get_metrics,
    load_elon_historic_tweets,
    analyze_elon_patterns
)

# Get user data
user = xtracker_get_user(handle="elonmusk")

# Get trackings
trackings = xtracker_get_trackings(handle="elonmusk", active_only=True)

# Get metrics
metrics = xtracker_get_metrics(
    user_id="user_id",
    metric_type="daily",
    start_date="2026-01-01",
    end_date="2026-01-21"
)

# Pre-computed Elon data
tweets = load_elon_historic_tweets()
# Returns: 31 days of hourly tweet counts

# Analyze patterns
patterns = analyze_elon_patterns(tweets)
# Returns: {hourly_avg, daily_avg, peak_hours, patterns}
```

### 7.4 Tags

```python
from trading import gamma_list_tags

# All tags
tags = gamma_list_tags()
# Returns: [{tag, market_count, volume}, ...]
```

---

## 8. TUI Components

### 8.1 Screen Template

```python
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Container
from textual.binding import Binding

class MyScreen(Screen):
    """Template for new screens"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._render(), id="content"),
            id="container"
        )

    def _render(self) -> str:
        lines = []
        lines.append("[bold cyan]MY SCREEN[/]")
        lines.append("─" * 40)
        lines.append("Content here...")
        return "\n".join(lines)

    def action_back(self):
        self.app.pop_screen()

    def action_refresh(self):
        self.query_one("#content", Static).update(self._render())
```

### 8.2 Modal Screen (Trading)

```python
from textual.screen import ModalScreen
from textual.widgets import Button, Input

class TradingModal(ModalScreen):
    """Modal for order placement"""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("enter", "confirm", "Confirm"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Place Order", id="title"),
            Input(placeholder="Size", id="size-input"),
            Input(placeholder="Price", id="price-input"),
            Button("BUY", id="buy-btn", variant="success"),
            Button("SELL", id="sell-btn", variant="error"),
            id="modal-container"
        )

    def on_button_pressed(self, event):
        if event.button.id == "buy-btn":
            self._place_order("BUY")
        elif event.button.id == "sell-btn":
            self._place_order("SELL")

    def _place_order(self, side):
        size = self.query_one("#size-input", Input).value
        price = self.query_one("#price-input", Input).value
        # Execute order...
        self.dismiss()
```

### 8.3 Background Data Loading

```python
from textual.app import ComposeResult
from textual.screen import Screen
from textual.work import work
from textual.widgets import Static

class DataScreen(Screen):
    def __init__(self):
        super().__init__()
        self.data = []

    def on_mount(self):
        self.load_data()  # Start background load

    @work(thread=True)
    def load_data(self):
        """Load data in background thread"""
        from trading import polyrouter_trending
        self.data = polyrouter_trending() or []
        # Update UI from main thread
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        content = self.query_one("#content", Static)
        content.update(self._render())

    def _render(self) -> str:
        if not self.data:
            return "Loading..."
        lines = [f"Found {len(self.data)} markets"]
        for m in self.data[:10]:
            lines.append(f"  {m.get('question', '?')[:50]}")
        return "\n".join(lines)
```

### 8.4 State Management

```python
class AppState:
    """Global application state"""

    logged_in: bool = False
    wallet: str = ""
    private_key: str = ""
    theme: str = "neon"

    THEMES = {
        "neon": {"accent": "#ff8c00", "bg": "#1a1a2e"},
        "matrix": {"accent": "#00ff00", "bg": "#0a0a0a"},
        "ocean": {"accent": "#00d4aa", "bg": "#0d1b2a"},
    }

    @classmethod
    def login(cls, wallet: str, pk: str) -> bool:
        if wallet.startswith("0x") and len(wallet) == 42:
            cls.wallet = wallet
            cls.private_key = pk
            cls.logged_in = True
            return True
        return False

    @classmethod
    def logout(cls):
        cls.logged_in = False
        cls.wallet = ""
        cls.private_key = ""

    @classmethod
    def get_colors(cls):
        return cls.THEMES.get(cls.theme, cls.THEMES["neon"])

STATE = AppState()
```

---

## Quick Reference

### Import All Trading Functions

```python
from trading import (
    # Config
    load_config, save_config, get_client, derive_user_creds,

    # Market Data
    search_markets, get_price, get_orderbook, get_spread,
    polyrouter_search, polyrouter_trending,
    gamma_get_market, gamma_get_event, gamma_search,

    # Trading
    place_order, cancel_order, cancel_all_orders,
    place_bracket_orders, place_ladder_orders,
    get_open_orders,

    # Portfolio
    get_positions, get_balances, get_user_trades,
    dataapi_get_positions, dataapi_get_trades,

    # Analytics
    find_ev_opportunities, dataapi_get_leaderboard,
    dataapi_get_holders,

    # Quant
    run_all_models, backtest_model,
    simple_moving_average, exponential_moving_average,

    # Automation
    detect_volume_spikes, detect_liquidity_spikes,
    run_automation_scan,

    # Social
    gamma_get_profile, gamma_get_comments,
    load_elon_historic_tweets, analyze_elon_patterns,
)
```

### Common Patterns

```python
# Pattern 1: Safe API call with fallback
def get_data_safe():
    try:
        return api_call()
    except Exception:
        return FALLBACK_DATA

# Pattern 2: Cached data
CACHE = {}
def get_cached(key, fetch_func, ttl=300):
    import time
    if key in CACHE and time.time() - CACHE[key]['time'] < ttl:
        return CACHE[key]['data']
    data = fetch_func()
    CACHE[key] = {'data': data, 'time': time.time()}
    return data

# Pattern 3: Rate limit handling
import time
def rate_limited_call(func, *args, delay=0.1):
    time.sleep(delay)
    return func(*args)
```
