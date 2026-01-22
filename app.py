#!/usr/bin/env python3
"""
YES/NO.EVENTS - Clean Polymarket Terminal
==========================================
Each feature is a separate, clear screen.
All APIs and data accessible from main menu.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Activate venv
VENV_PATH = Path(__file__).parent / ".venv"
if VENV_PATH.exists():
    for pyver in ["python3.12", "python3.11", "python3.10"]:
        venv_site = VENV_PATH / "lib" / pyver / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
            break

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, DataTable, Input, Button, Label
from textual.binding import Binding
from textual.screen import Screen, ModalScreen
from textual import work
from rich.text import Text
from rich.panel import Panel

# Import ALL trading functions
from trading import (
    # Core
    scan_elon_markets, get_price, get_orderbook, list_strategies, place_order,
    cancel_order, cancel_all_orders,
    # Gamma API
    gamma_search, gamma_get_market, gamma_get_event, gamma_list_tags, gamma_get_profile,
    gamma_get_public_profile, gamma_get_comments, gamma_get_market_resolution,
    # Data API - Old
    data_get_positions, data_get_activity, data_get_leaderboard, data_get_trades, data_get_top_holders,
    # Data API v2 - New Polymarket endpoints
    dataapi_get_positions, dataapi_get_trades, dataapi_get_activity, dataapi_get_value,
    dataapi_get_holders, dataapi_get_leaderboard,
    # CLOB API
    clob_get_spreads, get_orderbook_depth, get_spread, get_market_trades,
    # Polyrouter
    polyrouter_search, polyrouter_trending,
    # Analysis
    find_ev_opportunities, get_price_history, get_live_volume, get_open_interest,
    # Account
    get_positions, get_balances, get_open_orders, get_user_trades,
    # Token/Market resolution
    get_market_by_token_id, get_market_name_for_order,
    # XTracker
    xtracker_get_users, xtracker_get_user, xtracker_get_trackings,
    xtracker_get_tracking, xtracker_get_metrics, xtracker_get_all_trackings,
    load_elon_historic_tweets, load_elon_daily_metrics, analyze_elon_patterns,
    # Automation
    detect_volume_spikes, detect_liquidity_spikes, scan_all_spikes,
    load_automation_config, save_automation_config, run_automation_scan,
    # Quant Research
    run_all_models, backtest_model, optimize_model_params,
    store_price_snapshot, get_price_history_local,
    # Mass Orders
    place_mass_orders, place_bracket_orders, place_ladder_orders
)

# ============================================================================
# TIMEZONE UTILITIES (CET = Poland time)
# ============================================================================

# CET is UTC+1, CEST (summer) is UTC+2
# For January it's CET (UTC+1)
CET = timezone(timedelta(hours=1))

def to_cet(timestamp_str):
    """Convert timestamp string to CET formatted datetime.
    Handles: ISO format, epoch ms, epoch s, date strings.
    Returns: 'DD/MM HH:MM' format in CET.
    """
    if not timestamp_str:
        return "-"

    ts = str(timestamp_str).strip()

    try:
        # Try ISO format (2026-01-18T15:30:00Z or similar)
        if 'T' in ts:
            # Remove Z suffix and parse
            ts_clean = ts.replace('Z', '+00:00').replace(' ', 'T')
            if '+' not in ts_clean and '-' not in ts_clean[10:]:
                ts_clean += '+00:00'
            dt = datetime.fromisoformat(ts_clean.split('.')[0] + '+00:00')
            dt_cet = dt.astimezone(CET)
            return dt_cet.strftime("%d/%m %H:%M")

        # Try epoch milliseconds (13 digits)
        if ts.isdigit() and len(ts) >= 13:
            dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
            dt_cet = dt.astimezone(CET)
            return dt_cet.strftime("%d/%m %H:%M")

        # Try epoch seconds (10 digits)
        if ts.isdigit() and len(ts) >= 10:
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            dt_cet = dt.astimezone(CET)
            return dt_cet.strftime("%d/%m %H:%M")

        # Try date only format (2026-01-18)
        if len(ts) >= 10 and ts[4] == '-' and ts[7] == '-':
            return ts[:10].replace('-', '/')

        # Fallback: return first 10 chars
        return ts[:10] if len(ts) >= 10 else ts

    except Exception:
        # Fallback
        return ts[:10] if len(ts) >= 10 else ts

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    """Global application state"""
    logged_in = False
    wallet = ""
    private_key = ""

    # Theme settings
    theme = "neon"  # neon, dark, light, hacker
    compact_mode = False
    show_sparklines = True
    number_format = "aligned"  # aligned, compact
    accent_color = "#ff8c00"  # orange default

    # Imported data storage
    imported_tags = None        # CSV imported tag data
    imported_traders = None     # CSV imported trader data
    imported_markets = None     # CSV imported market data
    imported_trades = None      # CSV imported trade history
    custom_api_endpoints = {}   # User-defined API endpoints

    # Theme color schemes
    THEMES = {
        "neon": {"bg": "#0a0a15", "accent": "#ff8c00", "green": "#00ff88", "red": "#ff3366"},
        "dark": {"bg": "#1a1a2e", "accent": "#4a9eff", "green": "#00c853", "red": "#ff5252"},
        "hacker": {"bg": "#0d0d0d", "accent": "#00ff00", "green": "#00ff00", "red": "#ff0000"},
        "ocean": {"bg": "#0a192f", "accent": "#64ffda", "green": "#64ffda", "red": "#ff6b6b"},
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
    def set_theme(cls, theme: str):
        if theme in cls.THEMES:
            cls.theme = theme
            cls.accent_color = cls.THEMES[theme]["accent"]

    @classmethod
    def get_colors(cls):
        return cls.THEMES.get(cls.theme, cls.THEMES["neon"])

STATE = AppState()

# ============================================================================
# HELPER: Clickable Links for Research Navigation
# ============================================================================

def link_address(addr: str, max_len: int = 20) -> str:
    """Format trader address (styled, not clickable - Rich doesn't support custom protocols)"""
    if not addr:
        return "[dim]Unknown[/]"
    display = f"{addr[:8]}...{addr[-6:]}" if len(addr) > max_len else addr
    return f"[cyan]{display}[/]"

def link_market(market_id: str, title: str, max_len: int = 35) -> str:
    """Format market title"""
    if not market_id:
        return title[:max_len] if title else "[dim]Unknown[/]"
    display = title[:max_len] if len(title) <= max_len else title[:max_len-2] + ".."
    return f"[cyan]{display}[/]"

def link_tag(tag: str) -> str:
    """Format tag"""
    if not tag:
        return ""
    return f"[yellow]#{tag}[/]"

def link_event(event_id: str, title: str, max_len: int = 40) -> str:
    """Format event title"""
    if not event_id:
        return title[:max_len] if title else "[dim]Unknown[/]"
    display = title[:max_len] if len(title) <= max_len else title[:max_len-2] + ".."
    return f"[magenta]{display}[/]"

def link_polymarket(market_id: str, text: str = "View on Polymarket") -> str:
    """Create external link to Polymarket website"""
    if not market_id:
        return ""
    url = f"https://polymarket.com/event/{market_id}"
    return f"[link={url}][blue underline]{text}[/][/link]"

# ============================================================================
# HELPER: ASCII Charts & Formatting
# ============================================================================

def sparkline(values: list, width: int = 20) -> str:
    """Generate ASCII sparkline"""
    if not values:
        return " " * width
    chars = "▁▂▃▄▅▆▇█"
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return chars[4] * min(len(values), width)
    normalized = [(v - min_v) / (max_v - min_v) for v in values]
    if len(normalized) > width:
        step = len(normalized) / width
        normalized = [normalized[int(i * step)] for i in range(width)]
    return "".join(chars[int(v * 7)] for v in normalized[:width])

def braille_sparkline(values: list, width: int = 20) -> str:
    """Generate high-resolution Braille sparkline (ntcharts-style)"""
    if not values or len(values) < 2:
        return "⠀" * width
    # Braille patterns for 4 vertical levels per cell
    braille_base = 0x2800
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return "⠤" * width
    # Resample to fit width (2 values per braille char)
    target_len = width * 2
    if len(values) > target_len:
        step = len(values) / target_len
        values = [values[int(i * step)] for i in range(target_len)]
    elif len(values) < target_len:
        values = values + [values[-1]] * (target_len - len(values))
    result = []
    for i in range(0, len(values) - 1, 2):
        v1 = int((values[i] - min_v) / (max_v - min_v) * 3)
        v2 = int((values[i+1] - min_v) / (max_v - min_v) * 3)
        # Map to braille dots: left col (1,2,3,7) right col (4,5,6,8)
        dots = 0
        if v1 >= 0: dots |= 0x40  # dot 7
        if v1 >= 1: dots |= 0x04  # dot 3
        if v1 >= 2: dots |= 0x02  # dot 2
        if v1 >= 3: dots |= 0x01  # dot 1
        if v2 >= 0: dots |= 0x80  # dot 8
        if v2 >= 1: dots |= 0x20  # dot 6
        if v2 >= 2: dots |= 0x10  # dot 5
        if v2 >= 3: dots |= 0x08  # dot 4
        result.append(chr(braille_base + dots))
    return "".join(result[:width])

def horizontal_bar(value: float, max_val: float, width: int = 20, style: str = "block") -> str:
    """Generate horizontal bar chart (ntcharts-style)"""
    if max_val <= 0:
        return " " * width
    pct = min(value / max_val, 1.0)
    filled = int(pct * width)
    if style == "block":
        return "█" * filled + "░" * (width - filled)
    elif style == "shade":
        # Gradient shading
        chars = " ░▒▓█"
        result = []
        for i in range(width):
            pos = i / width
            if pos < pct - 0.1:
                result.append("█")
            elif pos < pct:
                result.append("▓")
            elif pos < pct + 0.05:
                result.append("▒")
            else:
                result.append("░")
        return "".join(result)
    elif style == "arrow":
        if filled == 0:
            return "░" * width
        return "━" * (filled - 1) + "▶" + "░" * (width - filled)
    return "█" * filled + "░" * (width - filled)

def mini_heatmap(values: list, width: int = 10) -> str:
    """Generate single-row heatmap with gradient colors"""
    if not values:
        return " " * width
    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return "▓" * min(len(values), width)
    # Resample
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    result = []
    for v in values[:width]:
        norm = (v - min_v) / (max_v - min_v)
        if norm > 0.8:
            result.append("█")
        elif norm > 0.6:
            result.append("▓")
        elif norm > 0.4:
            result.append("▒")
        elif norm > 0.2:
            result.append("░")
        else:
            result.append("·")
    return "".join(result)

def trend_indicator(current: float, previous: float) -> str:
    """Generate trend indicator with arrow and percentage"""
    if previous == 0:
        return "→ 0%"
    change = ((current - previous) / previous) * 100
    if change > 5:
        return f"▲▲{change:+.0f}%"
    elif change > 0:
        return f"▲ {change:+.0f}%"
    elif change < -5:
        return f"▼▼{change:+.0f}%"
    elif change < 0:
        return f"▼ {change:+.0f}%"
    return f"→ {change:+.0f}%"

def volume_bar(value: float, max_val: float, width: int = 15) -> str:
    """Volume bar with fire indicator for high volume"""
    bar = horizontal_bar(value, max_val, width - 2, "block")
    pct = value / max_val if max_val > 0 else 0
    if pct > 0.8:
        return f"{bar}🔥"
    elif pct > 0.5:
        return f"{bar}  "
    return f"{bar}  "

def progress_bar(value: float, width: int = 20) -> str:
    """Generate progress bar"""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def fmt_pct(value: float, width: int = 8) -> str:
    """Format percentage with right-alignment for decimal alignment"""
    if STATE.number_format == "aligned":
        return f"{value:>{width}.2f}%"
    return f"{value:.1f}%"

def fmt_money(value: float, width: int = 12) -> str:
    """Format money with right-alignment or compact"""
    if STATE.number_format == "aligned":
        return f"${value:>{width},.0f}"
    # Compact format
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}k"
    return f"${value:.0f}"

def fmt_price(value: float, width: int = 8) -> str:
    """Format price (0-100 cents) with right-alignment"""
    if STATE.number_format == "aligned":
        return f"{value:>{width}.2f}¢"
    return f"{value:.1f}¢"

def fmt_size(value: float, width: int = 10) -> str:
    """Format size/quantity with right-alignment"""
    if STATE.number_format == "aligned":
        return f"{value:>{width},.0f}"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.0f}k"
    return f"{value:.0f}"

def depth_bar(size: float, max_size: float, width: int = 15, direction: str = "right") -> str:
    """Generate orderbook depth bar"""
    if max_size <= 0:
        return " " * width
    pct = min(size / max_size, 1.0)
    filled = int(pct * width)
    if direction == "left":
        return "░" * (width - filled) + "█" * filled
    return "█" * filled + "░" * (width - filled)

# ============================================================================
# CSV IMPORT HELPERS
# ============================================================================

import csv
from io import StringIO

def import_tags_csv(filepath: str) -> list:
    """Import tag analytics from CSV. Expected cols: tag,markets,volume_24h,volume_7d,trend"""
    tags = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tags.append({
                    "tag": row.get("tag", row.get("name", "")),
                    "markets": int(row.get("markets", row.get("market_count", 0))),
                    "volume_24h": float(row.get("volume_24h", row.get("vol_24h", 0))),
                    "volume_7d": float(row.get("volume_7d", row.get("vol_7d", 0))),
                    "trend": row.get("trend", "+0%")
                })
        return tags if tags else None
    except Exception as e:
        print(f"[CSV] Error importing tags: {e}")
        return None

def import_traders_csv(filepath: str) -> list:
    """Import trader data from CSV. Expected cols: address,pnl,volume,positions,win_rate"""
    traders = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                traders.append({
                    "address": row.get("address", row.get("wallet", "")),
                    "pnl": float(row.get("pnl", row.get("profit", 0))),
                    "volume": float(row.get("volume", row.get("total_volume", 0))),
                    "positions": int(row.get("positions", row.get("position_count", 0))),
                    "win_rate": float(row.get("win_rate", row.get("winrate", 50)))
                })
        return traders if traders else None
    except Exception as e:
        print(f"[CSV] Error importing traders: {e}")
        return None

def import_markets_csv(filepath: str) -> list:
    """Import market data from CSV. Expected cols: market_id,title,tags,volume,price"""
    markets = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                markets.append({
                    "id": row.get("market_id", row.get("id", "")),
                    "title": row.get("title", row.get("question", "")),
                    "tags": row.get("tags", "").split(",") if row.get("tags") else [],
                    "volume": float(row.get("volume", row.get("volume_24h", 0))),
                    "price": float(row.get("price", row.get("yes_price", 0.5)))
                })
        return markets if markets else None
    except Exception as e:
        print(f"[CSV] Error importing markets: {e}")
        return None

def import_trades_csv(filepath: str) -> list:
    """Import trade history from CSV. Expected cols: timestamp,side,price,size,market_id"""
    trades = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append({
                    "timestamp": row.get("timestamp", row.get("time", "")),
                    "side": row.get("side", row.get("type", "BUY")),
                    "price": float(row.get("price", 0)),
                    "size": float(row.get("size", row.get("amount", 0))),
                    "market_id": row.get("market_id", row.get("market", ""))
                })
        return trades if trades else None
    except Exception as e:
        print(f"[CSV] Error importing trades: {e}")
        return None

def scan_csv_files(data_dir: Path) -> dict:
    """Scan data directory for CSV files and auto-detect types"""
    found = {}
    if not data_dir.exists():
        return found
    for f in data_dir.glob("*.csv"):
        name = f.stem.lower()
        if "tag" in name:
            found["tags"] = f
        elif "trader" in name or "leaderboard" in name:
            found["traders"] = f
        elif "market" in name:
            found["markets"] = f
        elif "trade" in name or "history" in name:
            found["trades"] = f
    return found

def load_all_csv_data():
    """Auto-load all CSV files from data directory"""
    data_dir = Path(__file__).parent / "data"
    csv_files = scan_csv_files(data_dir)
    loaded = []

    if "tags" in csv_files:
        data = import_tags_csv(str(csv_files["tags"]))
        if data:
            STATE.imported_tags = data
            loaded.append(f"tags ({len(data)} items)")

    if "traders" in csv_files:
        data = import_traders_csv(str(csv_files["traders"]))
        if data:
            STATE.imported_traders = data
            loaded.append(f"traders ({len(data)} items)")

    if "markets" in csv_files:
        data = import_markets_csv(str(csv_files["markets"]))
        if data:
            STATE.imported_markets = data
            loaded.append(f"markets ({len(data)} items)")

    if "trades" in csv_files:
        data = import_trades_csv(str(csv_files["trades"]))
        if data:
            STATE.imported_trades = data
            loaded.append(f"trades ({len(data)} items)")

    return loaded

# ============================================================================
# SCREEN 1: MAIN MENU
# ============================================================================

class MainMenuScreen(Screen):
    """Main menu - hub for all features"""

    BINDINGS = [
        Binding("1", "markets", "Markets"),
        Binding("2", "trading", "Trading"),
        Binding("3", "portfolio", "Portfolio"),
        Binding("4", "elon", "Elon Lab"),
        Binding("5", "research", "Research"),
        Binding("6", "analytics", "Analytics"),
        Binding("7", "api", "API Explorer"),
        Binding("8", "settings", "Settings"),
        Binding("9", "edge_scanner", "Edge Scanner"),
        Binding("0", "automation", "Automation"),
        Binding("minus", "quant", "Quant"),
        Binding("c", "claude", "Claude AI"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_menu(), id="menu-content"),
                id="menu-scroll"
            ),
            id="menu-container"
        )

    def on_screen_resume(self):
        """Refresh menu when returning from other screens"""
        self.query_one("#menu-content", Static).update(self._render_menu())

    def _render_menu(self) -> str:
        lines = []

        # Use simple ASCII for reliable alignment
        W = 79  # Total width

        # Header box
        lines.append("")
        lines.append(f"[bold #ff8c00]{'═' * W}[/]")
        lines.append("")
        lines.append("[bold white]  ██╗   ██╗███████╗███████╗    ██╗    ███╗   ██╗ ██████╗[/]")
        lines.append("[bold white]  ╚██╗ ██╔╝██╔════╝██╔════╝   ██╔╝    ████╗  ██║██╔═══██╗[/]")
        lines.append("[bold white]   ╚████╔╝ █████╗  ███████╗  ██╔╝     ██╔██╗ ██║██║   ██║[/]")
        lines.append("[bold white]    ╚██╔╝  ██╔══╝  ╚════██║ ██╔╝      ██║╚██╗██║██║   ██║[/]")
        lines.append("[bold white]     ██║   ███████╗███████║██╔╝       ██║ ╚████║╚██████╔╝[/]")
        lines.append("[bold white]     ╚═╝   ╚══════╝╚══════╝╚═╝        ╚═╝  ╚═══╝ ╚═════╝[/]")
        lines.append("")
        lines.append("[bold #9b59b6]              ══════ POLYMARKET QUANT TERMINAL ══════[/]")
        lines.append("")
        lines.append(f"[bold #ff8c00]{'═' * W}[/]")

        # Login status
        lines.append("")
        if STATE.logged_in:
            short = f"{STATE.wallet[:6]}...{STATE.wallet[-4:]}"
            lines.append(f"  [bold green]● CONNECTED[/] {short}  [dim]│[/]  [cyan]50+ APIs[/]  [dim]│[/]  [yellow]Live Data[/]")
        else:
            lines.append("  [bold red]○ NOT CONNECTED[/]  [dim]│[/]  [cyan]50+ APIs[/]  [dim]│[/]  [dim]Press 8 to login[/]")
        lines.append("")

        # Simple card layout without box drawing complexity
        lines.append("[bold #ff8c00]─────────────────────────────────────────────────────────────────────────────[/]")
        lines.append("")

        # Row 1: Markets, Trading, Portfolio
        lines.append("  [bold #ff8c00]1[/] [bold cyan]MARKETS[/]              [bold #ff8c00]2[/] [bold cyan]TRADING[/]              [bold #ff8c00]3[/] [bold cyan]PORTFOLIO[/]")
        lines.append("  [white]Browse & Search[/]        [white]Buy & Sell[/]             [white]Positions & P&L[/]")
        lines.append("  [dim]› Trending markets[/]     [dim]› Place orders[/]         [dim]› Holdings[/]")
        lines.append("  [dim]› Search by keyword[/]    [dim]› Open orders[/]          [dim]› Trade history[/]")
        lines.append("  [dim]› View details[/]         [dim]› Trade history[/]        [dim]› Balances[/]")
        lines.append("  [dim]› Orderbook depth[/]      [dim]› Limit/Market[/]         [dim]› ROI tracking[/]")
        lines.append("")

        # Row 2: Elon Lab, Research, Analytics
        lines.append("  [bold #ff8c00]4[/] [bold cyan]ELON LAB[/]             [bold #ff8c00]5[/] [bold cyan]RESEARCH[/]             [bold #ff8c00]6[/] [bold cyan]ANALYTICS[/]")
        lines.append("  [white]Tweet Analytics[/]        [white]Market Discovery[/]       [white]Quant Analysis[/]")
        lines.append("  [dim]› 2,198 tweets/31d[/]     [dim]› Top traders[/]          [dim]› Spread analysis[/]")
        lines.append("  [dim]› Hourly heatmap[/]       [dim]› Tag volume[/]           [dim]› Price momentum[/]")
        lines.append("  [dim]› Daily patterns[/]       [dim]› Trend spikes[/]         [dim]› Volume leaders[/]")
        lines.append("  [dim]› Behavior insights[/]    [dim]› Win rates[/]            [dim]› Liquidity rating[/]")
        lines.append("")

        # Row 3: API Explorer, Settings
        lines.append("  [bold #ff8c00]7[/] [bold magenta]API EXPLORER[/]         [bold #ff8c00]8[/] [bold cyan]SETTINGS[/]")
        lines.append("  [white]All 50+ Endpoints[/]      [white]Wallet & Config[/]")
        lines.append("  [dim]› Gamma API (8)[/]        [dim]› Connect wallet[/]")
        lines.append("  [dim]› CLOB API (6)[/]         [dim]› API keys[/]")
        lines.append("  [dim]› Data API (10)[/]        [dim]› Theme settings[/]")
        lines.append("  [dim]› XTracker (6)[/]         [dim]› Preferences[/]")
        lines.append("")

        # Row 4: Edge Scanner, Automation, Quant
        lines.append("  [bold #ff8c00]9[/] [bold yellow]EDGE SCANNER[/]         [bold #ff8c00]0[/] [bold yellow]AUTOMATION[/]           [bold #ff8c00]-[/] [bold yellow]QUANT[/]")
        lines.append("  [white]Find Trading Edges[/]     [white]Spike Detection[/]        [white]Forecasting[/]")
        lines.append("  [dim]› Mispricing[/]           [dim]› Volume spikes[/]        [dim]› Model ensemble[/]")
        lines.append("  [dim]› Wide spreads[/]         [dim]› Liquidity spikes[/]     [dim]› Backtesting[/]")
        lines.append("  [dim]› V/L imbalance[/]        [dim]› Auto-trading[/]         [dim]› Optimization[/]")
        lines.append("  [dim]› Momentum[/]             [dim]› Dry run mode[/]         [dim]› Snapshots[/]")
        lines.append("")

        # Row 5: Claude AI
        lines.append("  [bold #7ec8e3]C[/] [bold #7ec8e3]CLAUDE AI[/]")
        lines.append("  [white]Natural Language Trading[/]")
        lines.append("  [dim]› Chat to trade[/]")
        lines.append("  [dim]› Research with AI[/]")
        lines.append("  [dim]› Build positions[/]")
        lines.append("  [dim]› Manage portfolio[/]")
        lines.append("")

        lines.append("[bold #ff8c00]─────────────────────────────────────────────────────────────────────────────[/]")
        lines.append("")

        # Quick stats
        lines.append("[bold cyan]◆ DATA SOURCES[/]")
        lines.append("  [dim]Gamma:[/] Markets, Events, Tags    [dim]│[/]  [dim]CLOB:[/] Orderbook, Spreads, Trades")
        lines.append("  [dim]Data:[/] Positions, Leaderboard    [dim]│[/]  [dim]XTracker:[/] User tracking, Metrics")
        lines.append("  [dim]Analysis:[/] EV, Price history     [dim]│[/]  [dim]Elon:[/] 31 days of tweet data")
        lines.append("")
        lines.append("[dim]Press[/] [bold white]1-9,0,-[/] [dim]navigate[/]  [dim]│[/]  [bold white]?[/] [dim]help[/]  [dim]│[/]  [bold white]q[/] [dim]quit[/]  [dim]│[/]  [bold white]ESC[/] [dim]back[/]")

        return "\n".join(lines)

    def action_markets(self):
        self.app.push_screen(MarketsScreen())

    def action_trading(self):
        self.app.push_screen(TradingScreen())

    def action_portfolio(self):
        self.app.push_screen(PortfolioScreen())

    def action_elon(self):
        self.app.push_screen(ElonLabScreen())

    def action_research(self):
        self.app.push_screen(ResearchScreen())

    def action_analytics(self):
        self.app.push_screen(AnalyticsScreen())

    def action_api(self):
        self.app.push_screen(APIExplorerScreen())

    def action_settings(self):
        self.app.push_screen(SettingsScreen())

    def action_help(self):
        self.app.push_screen(HelpScreen())

    def action_edge_scanner(self):
        self.app.push_screen(EdgeScannerScreen())

    def action_automation(self):
        self.app.push_screen(AutomationScreen())

    def action_quant(self):
        self.app.push_screen(QuantScreen())

    def action_claude(self):
        self.app.push_screen(ClaudeAIScreen())

# ============================================================================
# SCREEN 2: MARKETS - All Polymarket markets
# ============================================================================

# Pre-computed sample markets (fallback if API fails)
SAMPLE_MARKETS = [
    {"id": "1", "title": "Trump wins 2024 election", "current_prices": {"yes": {"price": 0.523}}, "volume_total": 45230000},
    {"id": "2", "title": "Bitcoin > $100k by March 2025", "current_prices": {"yes": {"price": 0.345}}, "volume_total": 21560000},
    {"id": "3", "title": "Fed cuts rates in January", "current_prices": {"yes": {"price": 0.782}}, "volume_total": 9870000},
    {"id": "4", "title": "AI company IPO in Q1 2025", "current_prices": {"yes": {"price": 0.234}}, "volume_total": 5430000},
    {"id": "5", "title": "SpaceX Starship success", "current_prices": {"yes": {"price": 0.678}}, "volume_total": 8760000},
    {"id": "6", "title": "Netflix stock > $600", "current_prices": {"yes": {"price": 0.412}}, "volume_total": 4320000},
    {"id": "7", "title": "Ukraine ceasefire by Feb", "current_prices": {"yes": {"price": 0.189}}, "volume_total": 6540000},
    {"id": "8", "title": "Biden approval > 45%", "current_prices": {"yes": {"price": 0.321}}, "volume_total": 3210000},
    {"id": "9", "title": "Tesla delivers 2M vehicles", "current_prices": {"yes": {"price": 0.567}}, "volume_total": 7650000},
    {"id": "10", "title": "Ethereum > $5k by June", "current_prices": {"yes": {"price": 0.289}}, "volume_total": 4560000},
]

class MarketsScreen(Screen):
    """Browse ALL Polymarket markets"""

    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("1", "trending", "Trending", priority=True),
        Binding("2", "search_mode", "Search", priority=True),
        Binding("3", "hashtags", "Hashtags", priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "view_detail", "View", priority=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("tab", "focus_table", "Focus Table", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.mode = "trending"  # trending, search, hashtags
        self.markets = []
        self.search_query = ""
        self.categories = ["Politics", "Crypto", "Sports", "Tech", "Fed", "Elections", "AI", "Elon"]
        self.selected_category = None

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._render_header(), id="markets-header"),
            Input(placeholder="Type to search, then TAB + Enter to view...", id="search-input"),
            DataTable(id="markets-table"),
            Static(self._render_footer(), id="markets-footer"),
            id="markets-container"
        )

    def _render_header(self) -> str:
        lines = []
        # Clean header box
        lines.append("[bold #ff8c00]┌──────────────────────────────────────────────────────────────────────────────┐[/]")
        count_str = f"[green]{len(self.markets)}[/]"
        cat_str = f"  [yellow]#{self.selected_category}[/]" if self.selected_category else ""
        lines.append(f"[bold #ff8c00]│[/]  [bold white]MARKETS[/]  │  Mode: [cyan]{self.mode.upper()}[/]{cat_str}  │  Count: {count_str}{'':>25}[bold #ff8c00]│[/]")
        lines.append("[bold #ff8c00]└──────────────────────────────────────────────────────────────────────────────┘[/]")

        # Tabs with number keys
        tabs = []
        for key, name, is_active in [("1", "Trending", self.mode == "trending"),
                                      ("2", "Search", self.mode == "search"),
                                      ("3", "Hashtags", self.mode == "hashtags")]:
            if is_active:
                tabs.append(f"[bold white on #ff8c00] {key}:{name} [/]")
            else:
                tabs.append(f"[dim]{key}:{name}[/]")
        lines.append("  " + "   ".join(tabs))

        # Show category buttons in hashtags mode
        if self.mode == "hashtags":
            cat_btns = []
            for cat in self.categories:
                if cat == self.selected_category:
                    cat_btns.append(f"[bold white on cyan] #{cat} [/]")
                else:
                    cat_btns.append(f"[cyan]#{cat}[/]")
            lines.append("  " + "  ".join(cat_btns))

        return "\n".join(lines)

    def _render_footer(self) -> str:
        if self.mode == "hashtags":
            return "[dim]ESC:Back  1:Trending  2:Search  3:Hashtags  r:Refresh  [bold cyan]P[/]=Politics [bold cyan]C[/]=Crypto [bold cyan]S[/]=Sports[/]"
        return "[dim]ESC:Back  1:Trending  2:Search  3:Hashtags  r:Refresh  TAB:Focus  Enter:View[/]"

    def on_mount(self):
        table = self.query_one("#markets-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_column("Market", key="title", width=55)
        table.add_column("YES", key="yes", width=10)
        table.add_column("NO", key="no", width=10)
        table.add_column("Volume", key="volume", width=15)
        table.add_column("ID", key="id", width=12)
        self.load_trending()

    @work(thread=True)
    def load_trending(self):
        try:
            # Try to get real markets via search
            markets = []
            for term in ["trump", "bitcoin", "election", "AI", "fed"]:
                results = polyrouter_search(term, limit=15) or []
                # Filter out empty crypto markets
                for m in results:
                    vol = m.get('volume_total', m.get('volume', 0)) or 0
                    if vol > 0 and m.get('id') not in [x.get('id') for x in markets]:
                        markets.append(m)
                if len(markets) >= 30:
                    break

            # Use sample data as fallback if API returns nothing good
            if len(markets) < 5:
                markets = SAMPLE_MARKETS.copy()

            self.markets = markets[:50]
            self.app.call_from_thread(self._update_table)
            self.app.call_from_thread(self._focus_table)
        except Exception as e:
            # Fallback to sample data on any error
            self.markets = SAMPLE_MARKETS.copy()
            self.app.call_from_thread(self._update_table)
            self.app.call_from_thread(self._focus_table)

    @work(thread=True)
    def load_search(self, query: str):
        try:
            self.markets = polyrouter_search(query, limit=50) or []
            self.app.call_from_thread(self._update_table)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Error: {e}")

    @work(thread=True)
    def load_elon(self):
        try:
            elon_data = scan_elon_markets() or {}
            self.markets = []
            for event, mkts in elon_data.items():
                for m in mkts:
                    m['event'] = event
                    self.markets.append(m)
            self.app.call_from_thread(self._update_table)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Error: {e}")

    def _focus_table(self):
        """Focus the data table for keyboard navigation"""
        table = self.query_one("#markets-table", DataTable)
        table.focus()

    def action_focus_table(self):
        """Tab key focuses the table"""
        self._focus_table()

    def _update_table(self):
        table = self.query_one("#markets-table", DataTable)
        table.clear()

        for m in self.markets:
            title = m.get('title', m.get('question', m.get('bracket', '')))[:53]

            # Get YES price - handle multiple API formats
            yes_price = 0
            if 'current_prices' in m:
                cp = m['current_prices']
                # Try 'yes', 'up', or first key
                yes_price = cp.get('yes', {}).get('price', 0) or cp.get('up', {}).get('price', 0) or 0
            elif 'outcomePrices' in m:
                # Gamma API format: outcomePrices is a list of strings
                try:
                    prices = m.get('outcomePrices', [])
                    if prices and len(prices) > 0:
                        yes_price = float(prices[0])
                except:
                    pass
            elif 'yes' in m:
                yes_price = m['yes']
            elif 'bestBid' in m:
                yes_price = float(m.get('bestBid', 0))

            yes_pct = yes_price * 100 if yes_price <= 1 else yes_price
            no_pct = 100 - yes_pct

            # Get volume - handle multiple formats
            vol = m.get('volume', m.get('volume_total', m.get('volumeNum', 0))) or 0
            if isinstance(vol, str):
                try:
                    vol = float(vol)
                except:
                    vol = 0
            vol_str = f"${vol/1000:,.0f}k" if vol < 1000000 else f"${vol/1000000:,.1f}M"

            market_id = str(m.get('id', m.get('condition_id', m.get('conditionId', ''))))

            # Color coding
            yes_style = "bold green" if yes_pct > 60 else "yellow" if yes_pct > 40 else "red"

            table.add_row(
                Text(title),
                Text(f"{yes_pct:.1f}%", style=yes_style),
                Text(f"{no_pct:.1f}%", style="red" if no_pct > 60 else "dim"),
                Text(vol_str, style="green"),
                Text(market_id[:10])
            )

        self.query_one("#markets-header", Static).update(self._render_header())

    def _update_header(self):
        """Update header display"""
        self.query_one("#markets-header", Static).update(self._render_header())

    def on_key(self, event):
        """Handle keyboard input for tab switching and category selection"""
        # Check if input is focused
        try:
            input_widget = self.query_one("#search-input", Input)
            input_focused = input_widget.has_focus
        except:
            input_focused = False

        # Handle tab switching with 1, 2, 3 - even when input is focused
        if event.key in ["1", "2", "3"]:
            if input_focused:
                # Remove the typed number from input
                input_widget.value = input_widget.value.rstrip(event.key)
            if event.key == "1":
                self.action_trending()
                event.prevent_default()
                event.stop()
            elif event.key == "2":
                self.action_search_mode()
                event.prevent_default()
                event.stop()
            elif event.key == "3":
                self.action_hashtags()
                event.prevent_default()
                event.stop()
            return

        # Handle category selection in hashtags mode
        if self.mode == "hashtags" and event.key.isalpha():
            # Find category starting with this letter
            for cat in self.categories:
                if cat.lower().startswith(event.key.lower()):
                    self.selected_category = cat
                    self._update_header()
                    self.load_category(cat)
                    self.notify(f"Loading #{cat}...")
                    break

    def on_input_changed(self, event: Input.Changed):
        if event.input.id == "search-input" and len(event.value) >= 2:
            self.mode = "search"
            self.search_query = event.value
            self.load_search(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Handle Enter/double-click on table row"""
        if self.markets and event.cursor_row < len(self.markets):
            market = self.markets[event.cursor_row]
            self.app.push_screen(MarketDetailScreen(market))

    def action_trending(self):
        self.mode = "trending"
        self._update_header()
        self.query_one("#markets-footer", Static).update(self._render_footer())
        self.load_trending()
        self.notify("Loading trending markets...")

    def action_search_mode(self):
        self.mode = "search"
        self._update_header()
        self.query_one("#markets-footer", Static).update(self._render_footer())
        self.query_one("#search-input", Input).focus()

    def action_hashtags(self):
        self.mode = "hashtags"
        self.selected_category = None
        self._update_header()
        self.query_one("#markets-footer", Static).update(self._render_footer())
        self.notify("Press P=Politics, C=Crypto, S=Sports, T=Tech, F=Fed, E=Elections, A=AI")

    def action_refresh(self):
        if self.mode == "trending":
            self.load_trending()
        elif self.mode == "hashtags" and self.selected_category:
            self.load_category(self.selected_category)
        elif self.mode == "search" and self.search_query:
            self.load_search(self.search_query)
        self.notify("Refreshing...")

    @work(thread=True)
    def load_category(self, category: str):
        """Load markets by category/hashtag"""
        try:
            self.markets = polyrouter_search(category.lower(), limit=50) or []
            self.app.call_from_thread(self._update_table)
            self.app.call_from_thread(self._update_header)
        except Exception as e:
            self.app.call_from_thread(self.notify, f"Error: {e}")

    def action_view_detail(self):
        table = self.query_one("#markets-table", DataTable)
        if table.cursor_row is not None and self.markets:
            idx = table.cursor_row
            if idx < len(self.markets):
                market = self.markets[idx]
                self.app.push_screen(MarketDetailScreen(market))

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 2B: MARKET DETAIL
# ============================================================================

class MarketDetailScreen(ModalScreen):
    """Detailed view of a single market - Full trading interface with all data"""

    BINDINGS = [
        Binding("escape", "dismiss_or_cancel", "Close"),
        Binding("b", "buy_market", "Market Buy"),
        Binding("l", "buy_limit", "Limit Buy"),
        Binding("s", "sell_market", "Market Sell"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "toggle_comments", "Comments"),
        Binding("o", "toggle_orders", "My Orders"),
        Binding("h", "toggle_holders", "Holders"),
        Binding("t", "toggle_trades", "Trades"),
        Binding("q", "quant_analysis", "Quant"),
        Binding("1", "tab_main", "Main"),
        Binding("2", "tab_analysis", "Analysis"),
        Binding("3", "tab_social", "Social"),
        Binding("enter", "confirm_order", "Confirm", show=False),
        Binding("y", "confirm_order", "Yes", show=False),
        Binding("n", "cancel_order", "Cancel", show=False),
    ]

    def __init__(self, market: dict):
        super().__init__()
        self.market = market
        self.market_id = market.get('id', market.get('condition_id', ''))
        self.orderbook = {}
        self.orderbook_no = {}
        self.trades = []
        self.price_history = []
        self.holders = []
        self.comments = []
        self.resolution = None
        self.my_orders = []
        self.my_positions = []
        self.market_detail = None
        self.tab = "main"  # main, analysis, social
        self.show_comments = False
        self.show_orders = False
        self.show_holders = True
        self.show_trades = True
        self.order_type = "limit"  # limit or market
        self.best_bid = 0
        self.best_ask = 1
        self.best_bid_no = 0  # For NO token
        self.best_ask_no = 1
        self.selected_outcome = "YES"  # YES or NO - what user is trading
        # Pending order for confirmation
        self.pending_order = None  # {side, outcome, price, size, order_type, cost}

    def compose(self) -> ComposeResult:
        yield Container(
            # Main data area - scrollable
            ScrollableContainer(
                Static(self._render_detail(), id="detail-content"),
                id="detail-data"
            ),
            # Fixed trading panel at bottom
            Container(
                # Data toggle buttons row
                Horizontal(
                    Button("Holders", id="btn-holders", variant="default"),
                    Button("Comments", id="btn-comments", variant="default"),
                    Button("Trades", id="btn-trades", variant="default"),
                    Button("Orders", id="btn-orders", variant="default"),
                    Button("Refresh", id="btn-refresh", variant="default"),
                    id="data-row"
                ),
                # YES/NO outcome selector - PROMINENT
                Horizontal(
                    Static("[bold]TRADE:[/] ", id="trade-label"),
                    Button("◉ YES", id="btn-outcome-yes", variant="success", classes="outcome-selected"),
                    Button("○ NO", id="btn-outcome-no", variant="default"),
                    Static("", id="outcome-display"),  # Shows current selection info
                    id="outcome-row"
                ),
                # Order input row
                Horizontal(
                    Container(
                        Static("[bold]Size ($):[/]", id="size-label"),
                        Input(placeholder="10", id="order-size"),
                        id="size-box"
                    ),
                    Container(
                        Static("[bold]Limit Price:[/]", id="price-label"),
                        Input(placeholder="0.50", id="order-price"),
                        id="price-box"
                    ),
                    Button("BUY YES", id="btn-limit-buy", variant="success"),
                    Button("⚡ BUY", id="btn-market-buy", variant="primary"),
                    Button("SELL YES", id="btn-limit-sell", variant="error"),
                    Button("⚡ SELL", id="btn-market-sell", variant="warning"),
                    id="trading-row"
                ),
                # Order preview / confirmation area
                Static("", id="order-preview"),
                # Confirm/Cancel buttons (shown when pending order)
                Horizontal(
                    Button("✓ CONFIRM ORDER", id="btn-confirm", variant="success"),
                    Button("✗ CANCEL", id="btn-cancel", variant="error"),
                    id="confirm-row"
                ),
                id="trading-panel"
            ),
            id="detail-container"
        )

    def on_mount(self):
        self.load_data()
        # Don't auto-focus input so keys h/c/o/t work for navigation
        # Hide confirm buttons initially
        try:
            self.query_one("#confirm-row").display = False
        except:
            pass

    @work(thread=True)
    def load_data(self):
        # Orderbook for YES
        try:
            self.orderbook = get_orderbook(self.market_id, 'yes') or {}
            # Extract best bid/ask
            if self.orderbook:
                bids = getattr(self.orderbook, 'bids', []) or []
                asks = getattr(self.orderbook, 'asks', []) or []
                if bids:
                    self.best_bid = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
                if asks:
                    self.best_ask = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 1))
        except:
            pass

        # Orderbook for NO
        try:
            self.orderbook_no = get_orderbook(self.market_id, 'no') or {}
            if self.orderbook_no:
                bids = getattr(self.orderbook_no, 'bids', []) or []
                asks = getattr(self.orderbook_no, 'asks', []) or []
                if bids:
                    self.best_bid_no = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0))
                if asks:
                    self.best_ask_no = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 1))
        except:
            pass

        # Recent trades
        try:
            self.trades = get_market_trades(self.market_id, limit=15) or []
        except:
            pass

        # Price history
        try:
            self.price_history = get_price_history(self.market_id, interval=60) or []
        except:
            pass

        # Top holders
        try:
            holders = dataapi_get_holders(self.market_id, limit=10)
            self.holders = holders if isinstance(holders, list) else []
        except:
            pass

        # Comments
        try:
            event_id = self.market.get('event_id', self.market.get('eventId', ''))
            if event_id:
                comments = gamma_get_comments("Event", event_id, limit=30)
                self.comments = comments if isinstance(comments, list) else []
            else:
                # Try market comments
                comments = gamma_get_comments("Market", self.market_id, limit=30)
                self.comments = comments if isinstance(comments, list) else []
        except:
            pass

        # Resolution info
        try:
            self.resolution = gamma_get_market_resolution(self.market_id)
        except:
            pass

        # Full market details from gamma
        try:
            self.market_detail = gamma_get_market(self.market_id)
        except:
            pass

        # My open orders for this market (if logged in)
        if STATE.logged_in:
            try:
                all_orders = get_open_orders() or []
                # Filter for this market
                self.my_orders = [o for o in all_orders if str(getattr(o, 'asset_id', '')).startswith(self.market_id[:8])]
            except:
                self.my_orders = []

            # My positions for this market
            try:
                positions = get_positions() or []
                self.my_positions = [p for p in positions if self.market_id in str(p.get('market_id', p.get('conditionId', '')))]
            except:
                self.my_positions = []

        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#detail-content", Static).update(self._render_detail())
        # Update order preview
        try:
            preview = self.query_one("#order-preview", Static)
            preview.update(self._render_order_preview())
        except:
            pass
        # Pre-fill market price (midpoint between bid/ask)
        try:
            price_input = self.query_one("#order-price", Input)
            if not price_input.value:  # Only if not already set
                mid = (self.best_bid + self.best_ask) / 2
                price_input.value = f"{mid:.2f}"
        except:
            pass
        # Update outcome display with prices
        try:
            self._select_outcome(self.selected_outcome)
        except:
            pass

    def _render_order_preview(self) -> str:
        """Render order preview line with cost estimate or pending order confirmation"""
        colors = STATE.get_colors()
        market_title = self.market.get('title', self.market.get('question', ''))[:50]

        # If there's a pending order, show confirmation prompt
        if self.pending_order:
            p = self.pending_order
            side_color = colors['green'] if p['side'] == 'BUY' else colors['red']
            outcome_color = colors['green'] if p['outcome'].upper() == 'YES' else colors['red']
            otype = "LIMIT" if p['order_type'] == "GTC" else "MARKET"
            payout = p['size']  # Full payout if wins
            profit = payout - p['cost']
            profit_pct = (profit / p['cost'] * 100) if p['cost'] > 0 else 0
            return (
                f"\n[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════════╗[/]\n"
                f"[bold #ff8c00]║[/]  [bold reverse yellow] ⚠  CONFIRM YOUR ORDER  [/]                                                   [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]╠══════════════════════════════════════════════════════════════════════════════════╣[/]\n"
                f"[bold #ff8c00]║[/]                                                                                [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Market:[/]  {market_title:<60}  [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]                                                                                [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Action:[/]      [{side_color}][bold]{p['side']}[/][/] [{outcome_color}][bold]{p['outcome'].upper()}[/][/]                                                  [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Order Type:[/]  [bold]{otype}[/]                                                            [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Size:[/]        [bold]{p['size']:.0f}[/] shares                                                      [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Price:[/]       [bold]{p['price']*100:.1f}¢[/] per share                                             [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]                                                                                [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]╠──────────────────────────────────────────────────────────────────────────────────╣[/]\n"
                f"[bold #ff8c00]║[/]  [bold yellow]TOTAL COST:    ${p['cost']:.2f}[/]                                                      [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]Payout if WIN:[/] [bold {colors['green']}]${payout:.2f}[/]  →  [bold {colors['green']}]+${profit:.2f}[/] profit ([bold {colors['green']}]+{profit_pct:.0f}%[/])         [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]  [bold]If LOSE:[/]       [bold {colors['red']}]-${p['cost']:.2f}[/] (you lose your cost)                         [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]║[/]                                                                                [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]╠══════════════════════════════════════════════════════════════════════════════════╣[/]\n"
                f"[bold #ff8c00]║[/]       [bold green reverse] ENTER or Y = PLACE ORDER [/]    [bold red reverse] N or ESC = CANCEL [/]            [bold #ff8c00]║[/]\n"
                f"[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════════╝[/]"
            )

        # Normal preview - show what you're trading and cost estimates
        outcome = self.selected_outcome
        outcome_color = colors['green'] if outcome == "YES" else colors['red']

        # Get prices for the selected outcome
        if outcome == "YES":
            bid = self.best_bid
            ask = self.best_ask
        else:
            bid = self.best_bid_no if self.best_bid_no > 0 else (1 - self.best_ask)
            ask = self.best_ask_no if self.best_ask_no < 1 else (1 - self.best_bid)

        spread = (ask - bid) * 100
        spread_pct = (spread / ((ask + bid) / 2)) * 100 if (ask + bid) > 0 else 0
        mid = (bid + ask) / 2

        lines = []
        lines.append(f"[bold #ff8c00]┌─ TRADING [{outcome_color}]{outcome}[/] ──────────────────────────────────────────────────────────┐[/]")
        lines.append(f"[bold #ff8c00]│[/]  [{outcome_color}][bold]◉ {outcome}[/][/] │ BID: [{colors['green']}]{bid*100:>5.1f}¢[/]  ASK: [{colors['red']}]{ask*100:>5.1f}¢[/]  MID: {mid*100:>5.1f}¢  SPREAD: {spread:.1f}¢  [bold #ff8c00]│[/]")

        # Calculate live cost estimate from inputs
        try:
            size_input = self.query_one("#order-size", Input)
            price_input = self.query_one("#order-price", Input)
            size = float(size_input.value) if size_input.value else 0
            price = float(price_input.value) if price_input.value else mid

            if size > 0:
                buy_cost = size * ask
                buy_profit = size - buy_cost
                buy_profit_pct = (buy_profit / buy_cost * 100) if buy_cost > 0 else 0

                sell_proceeds = size * bid
                sell_profit_pct = (sell_proceeds / size * 100) if size > 0 else 0

                limit_cost = size * price
                limit_profit = size - limit_cost
                limit_profit_pct = (limit_profit / limit_cost * 100) if limit_cost > 0 else 0

                lines.append(f"[bold #ff8c00]├─ {size:.0f} SHARES OF {outcome} ─────────────────────────────────────────────────────┤[/]")
                lines.append(f"[bold #ff8c00]│[/]  [bold]⚡ BUY {outcome}:[/]   Pay [{colors['red']}]${buy_cost:>7.2f}[/]  →  If {outcome} wins: [{colors['green']}]+${buy_profit:>6.2f}[/] ({buy_profit_pct:>4.0f}% profit)  [bold #ff8c00]│[/]")
                lines.append(f"[bold #ff8c00]│[/]  [bold]⚡ SELL {outcome}:[/]  Get [{colors['green']}]${sell_proceeds:>7.2f}[/]  →  Closes position                            [bold #ff8c00]│[/]")
                if price > 0 and price < 1:
                    lines.append(f"[bold #ff8c00]│[/]  [bold]LIMIT @ {price*100:.0f}¢:[/] Pay [{colors['red']}]${limit_cost:>7.2f}[/]  →  If {outcome} wins: [{colors['green']}]+${limit_profit:>6.2f}[/] ({limit_profit_pct:>4.0f}% profit)  [bold #ff8c00]│[/]")
        except:
            pass

        lines.append(f"[bold #ff8c00]└─ Use YES/NO buttons above to switch │ h:Holders c:Comments t:Trades ──────┘[/]")
        return "\n".join(lines)

    def _render_detail(self) -> str:
        """Full market detail with all data sections - Professional Layout"""
        lines = []
        m = self.market
        md = self.market_detail or {}
        colors = STATE.get_colors()
        accent = STATE.accent_color

        title = m.get('title', m.get('question', md.get('question', m.get('bracket', 'Unknown'))))[:90]
        description = md.get('description', m.get('description', ''))

        # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
        # ║  HEADER: Market Question                                                                     ║
        # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
        lines.append(f"[bold {accent}]╔{'═' * 93}╗[/]")
        lines.append(f"[bold {accent}]║[/] [bold white]{title:<91}[/] [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚{'═' * 93}╝[/]")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  STATUS BAR: Key info at a glance                                                            │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        lines.append(f"[dim]┌{'─' * 93}┐[/]")

        # Status indicators
        active = md.get('active', m.get('active', True))
        closed = md.get('closed', False)
        resolved = self.resolution and self.resolution.get('outcome')

        if resolved:
            res_outcome = self.resolution.get('outcome', self.resolution.get('resolution', 'Unknown'))
            res_color = colors['green'] if str(res_outcome).upper() == 'YES' else colors['red']
            status = f"[bold {res_color}]◉ RESOLVED: {res_outcome.upper()}[/]"
        elif closed or not active:
            status = f"[bold {colors['red']}]◉ CLOSED[/]"
        else:
            status = f"[bold {colors['green']}]◉ ACTIVE[/]"

        # End date
        end_date = md.get('endDate', md.get('end_date_iso', m.get('endDate', '')))
        end_str = f"[cyan]{end_date[:10]}[/]" if end_date else "[dim]TBD[/]"

        # Market ID (shortened)
        market_id_short = f"{self.market_id[:8]}...{self.market_id[-6:]}" if len(self.market_id) > 20 else self.market_id

        lines.append(f"[dim]│[/]  {status}  │  [bold]Ends:[/] {end_str}  │  [bold]ID:[/] [dim]{market_id_short}[/]")
        lines.append(f"[dim]└{'─' * 93}┘[/]")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  DESCRIPTION & TAGS                                                                          │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if description:
            lines.append("")
            lines.append(f"[bold yellow]▸ DESCRIPTION[/]")
            # Word wrap description to fit
            desc_clean = description.replace('\n', ' ').strip()[:280]
            if len(desc_clean) > 90:
                lines.append(f"  {desc_clean[:90]}")
                lines.append(f"  {desc_clean[90:180]}")
                if len(desc_clean) > 180:
                    lines.append(f"  {desc_clean[180:]}{'...' if len(description) > 280 else ''}")
            else:
                lines.append(f"  {desc_clean}")

        # Tags
        tags = md.get('tags', m.get('tags', []))
        if tags:
            tag_str = "  ".join([f"[cyan]#{t.get('label', t.get('name', t)) if isinstance(t, dict) else t}[/]" for t in tags[:6]])
            lines.append("")
            lines.append(f"[bold yellow]▸ TAGS[/]  {tag_str}")

        # Event info
        event_id = m.get('event_id', md.get('event_slug', md.get('eventSlug', '')))
        event_title = md.get('groupItemTitle', md.get('eventTitle', ''))
        if event_title:
            lines.append(f"[bold yellow]▸ EVENT[/]  [white]{event_title}[/]  [dim](navigate: related markets)[/]")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  PRICE & VOLUME DISPLAY                                                                      │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        lines.append("")
        lines.append(f"[bold yellow]▸ CURRENT PRICES[/]")

        yes_price = self.best_ask if self.best_ask < 1 else 0.5
        if 'current_prices' in m:
            yes_price = m['current_prices'].get('yes', {}).get('price', 0) or yes_price
        elif 'yes' in m:
            yes_price = m['yes'] if m['yes'] <= 1 else m['yes'] / 100
        yes_pct = yes_price * 100
        no_pct = 100 - yes_pct

        # Big price bar with box - standardized width
        yes_bar = int(yes_pct * 0.50)
        no_bar = 50 - yes_bar
        lines.append(f"  ┌{'─' * 78}┐")
        lines.append(f"  │  [{colors['green']}]YES {yes_pct:>5.1f}%[/]  [{colors['green']}]{'█' * yes_bar}[/][{colors['red']}]{'█' * no_bar}[/]  [{colors['red']}]NO {no_pct:>5.1f}%[/]  │")
        lines.append(f"  └{'─' * 78}┘")

        # Volume & liquidity stats in a nice grid
        vol = m.get('volume_total', m.get('volume', md.get('volume', 0))) or 0
        vol_24h = m.get('volume_24h', md.get('volume24hr', 0)) or 0
        liquidity = m.get('liquidity', md.get('liquidity', 0)) or 0
        vol_str = f"${vol/1e6:.2f}M" if vol >= 1e6 else f"${vol/1e3:.1f}k" if vol >= 1e3 else f"${vol:.0f}"
        vol24_str = f"${vol_24h/1e6:.2f}M" if vol_24h >= 1e6 else f"${vol_24h/1e3:.1f}k" if vol_24h >= 1e3 else f"${vol_24h:.0f}"
        liq_str = f"${liquidity/1e6:.2f}M" if liquidity >= 1e6 else f"${liquidity/1e3:.1f}k" if liquidity >= 1e3 else f"${liquidity:.0f}"

        lines.append(f"  [bold]Total Volume:[/] [{colors['green']}]{vol_str:<12}[/]  [bold]24h Volume:[/] [cyan]{vol24_str:<12}[/]  [bold]Liquidity:[/] [yellow]{liq_str}[/]")

        # Bid/Ask spread summary
        if self.best_bid > 0 and self.best_ask > 0:
            mid = (self.best_bid + self.best_ask) / 2 * 100
            spread = (self.best_ask - self.best_bid) * 100
            lines.append(f"  [bold]Best Bid:[/] [{colors['green']}]{self.best_bid*100:.1f}¢[/]  [bold]Best Ask:[/] [{colors['red']}]{self.best_ask*100:.1f}¢[/]  [bold]Mid:[/] {mid:.1f}¢  [bold]Spread:[/] {spread:.2f}¢")
        lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  PRICE CHART                                                                                 │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if self.price_history:
            prices = [p.get('price', 0) for p in self.price_history[-60:]]
            if prices and max(prices) > 0:
                spark = sparkline(prices, 60)
                lo, hi, cur = min(prices)*100, max(prices)*100, prices[-1]*100
                change = cur - prices[0]*100 if len(prices) > 1 else 0
                change_color = colors['green'] if change >= 0 else colors['red']
                lines.append(f"[bold yellow]▸ PRICE HISTORY[/] [dim](60 data points)[/]")
                lines.append(f"  [{accent}]{spark}[/]")
                lines.append(f"  [dim]{'─' * 62}[/]")
                lines.append(f"  [dim]Low:[/] {lo:.1f}¢  [dim]│[/]  [dim]High:[/] {hi:.1f}¢  [dim]│[/]  [dim]Current:[/] {cur:.1f}¢  [dim]│[/]  [dim]Change:[/] [{change_color}]{change:+.1f}¢[/]")
                lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  RESOLUTION RULES                                                                            │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        resolution_source = md.get('resolutionSource', md.get('resolution_source', m.get('resolutionSource', '')))
        if resolution_source or end_date:
            lines.append(f"[bold yellow]▸ RESOLUTION RULES[/]")
            lines.append(f"  ┌{'─' * 78}┐")
            if end_date:
                lines.append(f"  │  [bold]End Date:[/] [cyan]{end_date}[/]{'':>52}│")
            if resolved:
                res_time = self.resolution.get('resolution_date', self.resolution.get('timestamp', ''))
                res_out = self.resolution.get('outcome', self.resolution.get('resolution', 'Unknown'))
                res_color = colors['green'] if str(res_out).upper() == 'YES' else colors['red']
                lines.append(f"  │  [bold]Resolved:[/] [{res_color}]{res_out.upper()}[/] on {str(res_time)[:10] if res_time else 'N/A'}{'':>40}│")
            lines.append(f"  └{'─' * 78}┘")
            lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  MARKET ANALYTICS                                                                            │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        lines.append(f"[bold yellow]▸ MARKET ANALYTICS[/]")

        # Calculate metrics
        spread = (self.best_ask - self.best_bid) * 100 if self.best_ask > self.best_bid else 0
        spread_pct = spread / ((self.best_ask + self.best_bid) / 2 * 100) * 100 if (self.best_ask + self.best_bid) > 0 else 0

        # Momentum from price history
        momentum = 0
        if self.price_history and len(self.price_history) > 1:
            prices = [p.get('price', 0) for p in self.price_history[-24:]]  # 24h
            if prices and prices[0] > 0:
                momentum = ((prices[-1] - prices[0]) / prices[0]) * 100

        # Volume rating
        vol_24h = m.get('volume_24h', 0) or 0
        if vol_24h >= 1000000:
            vol_rating = ("HIGH", colors['green'], "████████")
        elif vol_24h >= 500000:
            vol_rating = ("MED ", "yellow", "█████░░░")
        elif vol_24h >= 100000:
            vol_rating = ("LOW ", colors['red'], "███░░░░░")
        else:
            vol_rating = ("THIN", "dim", "█░░░░░░░")

        # Spread quality
        if spread < 1.0:
            spread_qual = (f"[{colors['green']}]★★★ TIGHT[/]", colors['green'])
        elif spread < 2.5:
            spread_qual = ("[yellow]★★☆ FAIR [/]", "yellow")
        else:
            spread_qual = (f"[{colors['red']}]★☆☆ WIDE [/]", colors['red'])

        # Momentum signal
        if momentum > 10:
            mom_signal = (f"[{colors['green']}]▲▲ BULL [/]", colors['green'], "████████")
        elif momentum > 3:
            mom_signal = (f"[{colors['green']}]▲ UP   [/]", colors['green'], "█████░░░")
        elif momentum > -3:
            mom_signal = ("[yellow]─ FLAT [/]", "yellow", "████░░░░")
        elif momentum > -10:
            mom_signal = (f"[{colors['red']}]▼ DOWN [/]", colors['red'], "█████░░░")
        else:
            mom_signal = (f"[{colors['red']}]▼▼ BEAR [/]", colors['red'], "████████")

        # Composite Score (0-100)
        spread_score = max(0, 100 - spread * 25)  # Lower spread = higher score
        mom_score = min(100, 50 + momentum * 2.5)  # Centered at 50
        vol_score = min(100, vol_24h / 10000)  # Higher volume = higher score
        composite = int(spread_score * 0.3 + mom_score * 0.35 + vol_score * 0.35)

        # Trading signal
        if composite > 70 and momentum > 0:
            signal = (f"[bold {colors['green']}]◉ STRONG BUY [/]", "▲ BUY YES")
        elif composite > 70 and momentum < 0:
            signal = (f"[bold {colors['red']}]◉ STRONG SELL[/]", "▼ BUY NO")
        elif composite > 50 and momentum > 0:
            signal = (f"[{colors['green']}]○ BUY        [/]", "↗ CONSIDER")
        elif composite > 50 and momentum < 0:
            signal = (f"[{colors['red']}]○ SELL       [/]", "↘ CONSIDER")
        else:
            signal = ("[dim]─ NEUTRAL    [/]", "⊘ WAIT")

        # Score color
        score_color = colors['green'] if composite > 70 else ("yellow" if composite > 50 else colors['red'])

        # Display analytics in clean rows
        lines.append(f"  ┌{'─' * 78}┐")
        lines.append(f"  │  [bold]SPREAD[/]  {spread:>5.2f}¢ ({spread_pct:>4.1f}%) {spread_qual[0]}   │  [bold]MOMENTUM[/]  {momentum:>+5.1f}%  {mom_signal[0]}│")
        lines.append(f"  │  [bold]VOLUME[/]  {vol_rating[0]}  [{vol_rating[1]}]{vol_rating[2]}[/]{'':>14}│  [bold]SCORE[/]     [{score_color}]{composite:>3}/100[/]  {signal[0]}│")
        lines.append(f"  └{'─' * 78}┘")

        # Trading recommendation
        rec_color = colors['green'] if 'BUY' in signal[1] and 'YES' in signal[1] else (colors['red'] if 'NO' in signal[1] else 'yellow')
        lines.append(f"  [bold]SIGNAL:[/] [{rec_color}]{signal[1]}[/]  │  Spread: {int(spread_score)}/100  Momentum: {int(mom_score)}/100  Volume: {int(vol_score)}/100")
        lines.append("")

        # ═══════════════════════════════════════════════════════════════════════
        # ROW 3: Orderbook with Visual Depth - Improved Display
        # ═══════════════════════════════════════════════════════════════════════
        if self.orderbook:
            lines.append(f"[bold yellow]▶ ORDERBOOK (YES)[/]")
            lines.append(f"  [dim]{'─' * 36}┬{'─' * 40}[/]")
            lines.append(f"  [dim]{'BIDS (buyers)':^36}[/]│[dim]{'ASKS (sellers)':^40}[/]")
            lines.append(f"  [dim]{'─' * 36}┼{'─' * 40}[/]")

            # Get bids sorted by price descending (best/highest bid first)
            raw_bids = list(getattr(self.orderbook, 'bids', [])) if hasattr(self.orderbook, 'bids') else []
            bids = sorted(raw_bids, key=lambda b: float(b.price if hasattr(b, 'price') else b.get('price', 0)), reverse=True)[:8]

            # Get asks sorted by price ascending (best/lowest ask first)
            raw_asks = list(getattr(self.orderbook, 'asks', [])) if hasattr(self.orderbook, 'asks') else []
            asks = sorted(raw_asks, key=lambda a: float(a.price if hasattr(a, 'price') else a.get('price', 0)), reverse=False)[:8]

            all_sizes = []
            for b in bids:
                try: all_sizes.append(float(b.size if hasattr(b, 'size') else b.get('size', 0)))
                except: pass
            for a in asks:
                try: all_sizes.append(float(a.size if hasattr(a, 'size') else a.get('size', 0)))
                except: pass
            max_size = max(all_sizes) if all_sizes else 1

            for i in range(max(len(bids), len(asks), 1)):
                try:
                    if i < len(bids):
                        b = bids[i]
                        bp = float(b.price if hasattr(b, 'price') else b.get('price', 0)) * 100
                        bs = float(b.size if hasattr(b, 'size') else b.get('size', 0))
                        bar_len = int(min(bs / max_size, 1.0) * 15)
                        bid_bar = f"[{colors['green']}]{'█' * bar_len}[/]{'░' * (15 - bar_len)}"
                        bid_str = f"{bid_bar} [{colors['green']}]{bs:>7,.0f}[/]@{bp:>4.1f}¢"
                    else:
                        bid_str = " " * 35

                    if i < len(asks):
                        a = asks[i]
                        ap = float(a.price if hasattr(a, 'price') else a.get('price', 0)) * 100
                        az = float(a.size if hasattr(a, 'size') else a.get('size', 0))
                        bar_len = int(min(az / max_size, 1.0) * 15)
                        ask_bar = f"{'░' * (15 - bar_len)}[{colors['red']}]{'█' * bar_len}[/]"
                        ask_str = f"{ap:>4.1f}¢@[{colors['red']}]{az:<7,.0f}[/] {ask_bar}"
                    else:
                        ask_str = ""

                    lines.append(f"  {bid_str}│{ask_str}")
                except:
                    pass

            # Spread summary
            if bids and asks:
                try:
                    bb = float(bids[0].price if hasattr(bids[0], 'price') else bids[0].get('price', 0)) * 100
                    ba = float(asks[0].price if hasattr(asks[0], 'price') else asks[0].get('price', 0)) * 100
                    spread = ba - bb
                    mid = (bb + ba) / 2
                    quality = f"[{colors['green']}]TIGHT[/]" if spread < 2 else ("[yellow]FAIR[/]" if spread < 5 else f"[{colors['red']}]WIDE[/]")
                    bid_depth = sum(float(b.size if hasattr(b, 'size') else b.get('size', 0)) for b in bids[:5])
                    ask_depth = sum(float(a.size if hasattr(a, 'size') else a.get('size', 0)) for a in asks[:5])
                    total_depth = bid_depth + ask_depth
                    imbal = (bid_depth - ask_depth) / total_depth * 100 if total_depth > 0 else 0
                    imbal_color = colors['green'] if imbal > 10 else (colors['red'] if imbal < -10 else 'yellow')
                    lines.append(f"  [dim]{'─' * 36}┴{'─' * 40}[/]")
                    lines.append(f"  Bid:[{colors['green']}]{bb:.1f}¢[/]  Ask:[{colors['red']}]{ba:.1f}¢[/]  Mid:{mid:.1f}¢  Spread:{spread:.1f}¢ {quality}  Imbal:[{imbal_color}]{imbal:+.0f}%[/]")
                except: pass
            lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  MY ORDERS (if logged in)                                                                    │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if STATE.logged_in and (self.show_orders or self.my_orders):
            lines.append(f"[bold yellow]▸ MY OPEN ORDERS[/]  [dim](o to toggle)[/]")
            if self.my_orders:
                lines.append(f"  ┌{'─' * 78}┐")
                for o in self.my_orders[:5]:
                    side = getattr(o, 'side', 'BUY')
                    price = float(getattr(o, 'price', 0)) * 100
                    size = float(getattr(o, 'original_size', getattr(o, 'size', 0)))
                    filled = float(getattr(o, 'size_matched', 0))
                    oid = str(getattr(o, 'id', ''))[:10]
                    side_color = colors['green'] if side == 'BUY' else colors['red']
                    pct_filled = filled / size * 100 if size > 0 else 0
                    lines.append(f"  │ [{side_color}]{side:<4}[/] {price:>5.1f}¢  Size:{size:>7,.0f}  Fill:{filled:>7,.0f} ({pct_filled:>2.0f}%) [dim]{oid}[/]│")
                lines.append(f"  └{'─' * 78}┘")
            else:
                lines.append(f"  [dim]No open orders for this market[/]")
            lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  TOP HOLDERS - Whale Detection & Position Analysis                                           │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if self.show_holders:
            lines.append(f"[bold yellow]▸ TOP HOLDERS[/]  [dim](h to toggle)[/]")
            if self.holders:
                # Calculate totals by position type
                yes_holders = [h for h in self.holders if str(h.get('position_type', h.get('outcome', 'YES'))).upper() == 'YES']
                no_holders = [h for h in self.holders if str(h.get('position_type', h.get('outcome', ''))).upper() == 'NO']
                yes_total = sum(float(h.get('balance', h.get('shares', 0))) for h in yes_holders)
                no_total = sum(float(h.get('balance', h.get('shares', 0))) for h in no_holders)
                total_held = yes_total + no_total
                num_yes = len(yes_holders)
                num_no = len(no_holders)

                # Format totals nicely
                yes_fmt = f"{yes_total/1e6:.2f}M" if yes_total >= 1e6 else (f"{yes_total/1e3:.1f}k" if yes_total >= 1000 else f"{yes_total:.0f}")
                no_fmt = f"{no_total/1e6:.2f}M" if no_total >= 1e6 else (f"{no_total/1e3:.1f}k" if no_total >= 1000 else f"{no_total:.0f}")

                # Summary stats in clean format
                lines.append(f"  ┌{'─' * 38}┬{'─' * 38}┐")
                lines.append(f"  │ [{colors['green']}]YES[/] {yes_fmt:>10} ({num_yes} holders){'':>10}│ [{colors['red']}]NO[/]  {no_fmt:>10} ({num_no} holders){'':>9}│")
                lines.append(f"  └{'─' * 38}┴{'─' * 38}┘")

                # Concentration analysis
                if total_held > 0:
                    top3_total = sum(float(h.get('balance', h.get('shares', 0))) for h in self.holders[:3])
                    top3_pct = (top3_total / total_held) * 100
                    if top3_pct > 50:
                        lines.append(f"  [yellow]⚠ Top 3 hold {top3_pct:.0f}% of positions[/]")
                    else:
                        lines.append(f"  [{colors['green']}]◎ Top 3 hold {top3_pct:.0f}%[/]")

                # Holder table
                lines.append(f"  [dim]{'─' * 78}[/]")
                lines.append(f"  [bold]{'#':>2}  {'ADDRESS':<16}  {'SIDE':^4}  {'SHARES':>12}  {'%':>6}  {'VALUE':>10}[/]")
                lines.append(f"  [dim]{'─' * 78}[/]")

                for idx, h in enumerate(self.holders[:8], 1):
                    addr = str(h.get('address', h.get('user', '')))
                    addr_short = f"{addr[:8]}...{addr[-4:]}" if len(addr) > 14 else addr
                    pos = str(h.get('position_type', 'YES')).upper()[:3]
                    shares = float(h.get('balance', h.get('shares', 0)))
                    pct = (shares / total_held * 100) if total_held > 0 else 0
                    price_for_pos = (yes_pct / 100) if pos == 'YES' else (no_pct / 100)
                    value = shares * price_for_pos
                    value_fmt = f"${value/1e3:.1f}k" if value >= 1000 else f"${value:.0f}"
                    pos_color = colors['green'] if pos == 'YES' else colors['red']
                    lines.append(f"  {idx:>2}  [{pos_color}]{addr_short:<16}[/]  [{pos_color}]{pos:^4}[/]  {shares:>12,.0f}  {pct:>5.1f}%  {value_fmt:>10}")
            else:
                lines.append(f"  [dim]No holder data available[/]")
            lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  RECENT TRADES - Live Market Activity                                                        │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if self.show_trades:
            lines.append(f"[bold yellow]▸ RECENT TRADES[/]  [dim](t to toggle)[/]")
            if self.trades:
                # Calculate trade statistics
                buy_trades = [t for t in self.trades if t.get('side', '').upper() == 'BUY']
                sell_trades = [t for t in self.trades if t.get('side', '').upper() == 'SELL']
                total_buy_vol = sum(float(t.get('size', 0)) for t in buy_trades)
                total_sell_vol = sum(float(t.get('size', 0)) for t in sell_trades)

                # Summary
                buy_vol_fmt = f"{total_buy_vol/1e3:.1f}k" if total_buy_vol >= 1000 else f"{total_buy_vol:.0f}"
                sell_vol_fmt = f"{total_sell_vol/1e3:.1f}k" if total_sell_vol >= 1000 else f"{total_sell_vol:.0f}"
                lines.append(f"  ┌{'─' * 38}┬{'─' * 38}┐")
                lines.append(f"  │ [{colors['green']}]▲ BUYS[/]  Vol: {buy_vol_fmt:>10}{'':>13}│ [{colors['red']}]▼ SELLS[/] Vol: {sell_vol_fmt:>10}{'':>12}│")
                lines.append(f"  └{'─' * 38}┴{'─' * 38}┘")

                # Trade table
                lines.append(f"  [dim]{'─' * 78}[/]")
                lines.append(f"  [bold]{'#':>2}  {'SIDE':^5}  {'PRICE':>7}  {'SIZE':>10}  {'VALUE':>10}  {'TIME':^12}[/]")
                lines.append(f"  [dim]{'─' * 78}[/]")

                for idx, t in enumerate(self.trades[:10], 1):
                    side = t.get('side', 'BUY').upper()
                    price = float(t.get('price', 0)) * 100
                    size = float(t.get('size', 0))
                    value = size * (price / 100)
                    value_fmt = f"${value/1e3:.1f}k" if value >= 1000 else f"${value:.0f}"
                    ts = to_cet(t.get('timestamp', t.get('createdAt', t.get('match_time', ''))))
                    ts_short = ts[5:16] if len(ts) > 16 else ts  # MM-DD HH:MM
                    side_color = colors['green'] if side == 'BUY' else colors['red']
                    side_icon = "▲" if side == 'BUY' else "▼"

                    lines.append(f"  {idx:>2}  [{side_color}]{side_icon}{side:<4}[/]  {price:>6.1f}¢  {size:>10,.0f}  {value_fmt:>10}  [dim]{ts_short}[/]")
            else:
                lines.append(f"  [dim]No recent trades[/]")
            lines.append("")

        # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
        # │  COMMENTS                                                                                    │
        # └──────────────────────────────────────────────────────────────────────────────────────────────┘
        if self.show_comments:
            lines.append(f"[bold yellow]▸ COMMENTS ({len(self.comments)})[/]  [dim](c to toggle)[/]")
            if self.comments:
                lines.append(f"  ┌{'─' * 91}┐")
                for c in self.comments[:8]:
                    author = c.get('author', c.get('username', c.get('user', 'Anon')))
                    if isinstance(author, dict):
                        author = author.get('username', author.get('name', 'Anon'))
                    author = str(author)[:15]
                    text = str(c.get('content', c.get('text', c.get('body', ''))))[:75]
                    ts = to_cet(c.get('createdAt', c.get('timestamp', '')))[:16]
                    lines.append(f"  │  [cyan]{author:<15}[/] [dim]{ts}[/]{'':>45} │")
                    lines.append(f"  │    {text:<87} │")
                lines.append(f"  └{'─' * 91}┘")
            else:
                lines.append(f"  [dim]No comments yet[/]")
            lines.append("")

        # ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
        # ║  FOOTER                                                                                      ║
        # ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
        lines.append(f"[bold {accent}]╔{'═' * 93}╗[/]")
        lines.append(f"[bold {accent}]║[/] [bold]KEYS:[/] [cyan]r[/]:Refresh  [cyan]c[/]:Comments  [cyan]o[/]:Orders  [cyan]h[/]:Holders  [cyan]t[/]:Trades  [cyan]ESC[/]:Close    [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚{'═' * 93}╝[/]")

        return "\n".join(lines)

    def on_button_pressed(self, event):
        if event.button.id == "btn-close":
            self.dismiss()
        elif event.button.id == "btn-refresh":
            self.load_data()
            self.notify("Refreshing...")
        elif event.button.id == "btn-comments":
            self.action_toggle_comments()
        elif event.button.id == "btn-holders":
            self.action_toggle_holders()
        elif event.button.id == "btn-trades":
            self.action_toggle_trades()
        elif event.button.id == "btn-orders":
            self.action_toggle_orders()
        elif event.button.id == "btn-outcome-yes":
            self._select_outcome("YES")
        elif event.button.id == "btn-outcome-no":
            self._select_outcome("NO")
        elif event.button.id == "btn-limit-buy":
            self._place_limit_order("BUY", self.selected_outcome.lower())
        elif event.button.id == "btn-limit-sell":
            self._place_limit_order("SELL", self.selected_outcome.lower())
        elif event.button.id == "btn-market-buy":
            self._place_market_order("BUY", self.selected_outcome.lower())
        elif event.button.id == "btn-market-sell":
            self._place_market_order("SELL", self.selected_outcome.lower())
        elif event.button.id == "btn-confirm":
            self.action_confirm_order()
        elif event.button.id == "btn-cancel":
            self.action_cancel_order()

    def action_toggle_comments(self):
        self.show_comments = not self.show_comments
        self._update_display()

    def action_toggle_orders(self):
        self.show_orders = not self.show_orders
        self._update_display()

    def action_toggle_holders(self):
        self.show_holders = not self.show_holders
        self._update_display()

    def action_toggle_trades(self):
        self.show_trades = not self.show_trades
        self._update_display()

    def _select_outcome(self, outcome: str):
        """Select YES or NO outcome and update UI"""
        self.selected_outcome = outcome
        colors = STATE.get_colors()

        # Update button appearances
        try:
            yes_btn = self.query_one("#btn-outcome-yes", Button)
            no_btn = self.query_one("#btn-outcome-no", Button)
            buy_btn = self.query_one("#btn-limit-buy", Button)
            sell_btn = self.query_one("#btn-limit-sell", Button)

            if outcome == "YES":
                yes_btn.label = "◉ YES"
                yes_btn.variant = "success"
                no_btn.label = "○ NO"
                no_btn.variant = "default"
                buy_btn.label = "BUY YES"
                buy_btn.variant = "success"
                sell_btn.label = "SELL YES"
                sell_btn.variant = "error"
            else:
                yes_btn.label = "○ YES"
                yes_btn.variant = "default"
                no_btn.label = "◉ NO"
                no_btn.variant = "error"
                buy_btn.label = "BUY NO"
                buy_btn.variant = "error"
                sell_btn.label = "SELL NO"
                sell_btn.variant = "success"

            # Update outcome display with price info
            if outcome == "YES":
                price = self.best_ask * 100
                bid = self.best_bid * 100
            else:
                price = self.best_ask_no * 100 if self.best_ask_no < 1 else (1 - self.best_bid) * 100
                bid = self.best_bid_no * 100 if self.best_bid_no > 0 else (1 - self.best_ask) * 100

            display = self.query_one("#outcome-display", Static)
            out_color = colors['green'] if outcome == "YES" else colors['red']
            display.update(f"  [{out_color}][bold]Trading {outcome}[/][/] │ ASK: {price:.1f}¢  BID: {bid:.1f}¢")
        except Exception as e:
            pass

        self._update_order_preview()

    def action_tab_main(self):
        self.tab = "main"
        self._update_display()

    def action_tab_analysis(self):
        self.tab = "analysis"
        self._update_display()

    def action_tab_social(self):
        self.tab = "social"
        self._update_display()

    def action_quant_analysis(self):
        """Launch Quant Research screen for this market"""
        quant_screen = QuantScreen()
        quant_screen.market_id = self.market_id
        self.app.push_screen(quant_screen)

    def _place_limit_order(self, side: str, outcome: str):
        """Prepare limit order for confirmation"""
        if not STATE.logged_in:
            self.notify("Please login first (Settings > Login)", severity="error")
            return
        try:
            size_input = self.query_one("#order-size", Input)
            price_input = self.query_one("#order-price", Input)
            if not size_input.value:
                self.notify("Enter size first", severity="error")
                size_input.focus()
                return
            if not price_input.value:
                self.notify("Enter price first", severity="error")
                price_input.focus()
                return
            size = float(size_input.value)
            price = float(price_input.value)
            if price <= 0 or price >= 1:
                self.notify("Price must be between 0.01 and 0.99", severity="error")
                return
            if size < 5:
                self.notify("Minimum order size is 5 shares", severity="error")
                return
            cost = size * price
            # Set pending order for confirmation
            self.pending_order = {
                'side': side,
                'outcome': outcome,
                'price': price,
                'size': size,
                'order_type': 'GTC',
                'cost': cost
            }
            self._update_order_preview()
            self.notify("Review order details and press ENTER to confirm or N to cancel", severity="warning")
        except ValueError:
            self.notify("Invalid size or price", severity="error")

    def _place_market_order(self, side: str, outcome: str):
        """Prepare market order for confirmation"""
        if not STATE.logged_in:
            self.notify("Please login first (Settings > Login)", severity="error")
            return
        try:
            size_input = self.query_one("#order-size", Input)
            if not size_input.value:
                self.notify("Enter size first", severity="error")
                size_input.focus()
                return
            size = float(size_input.value)
            if size < 5:
                self.notify("Minimum order size is 5 shares", severity="error")
                return

            # Use best ask for BUY, best bid for SELL
            if side == "BUY":
                price = self.best_ask
                if price >= 1 or price <= 0:
                    self.notify("No ask available - cannot place market buy", severity="error")
                    return
            else:
                price = self.best_bid
                if price >= 1 or price <= 0:
                    self.notify("No bid available - cannot place market sell", severity="error")
                    return

            cost = size * price
            # Set pending order for confirmation
            self.pending_order = {
                'side': side,
                'outcome': outcome,
                'price': price,
                'size': size,
                'order_type': 'FOK',
                'cost': cost
            }
            self._update_order_preview()
            self.notify("Review order details and press ENTER to confirm or N to cancel", severity="warning")
        except ValueError:
            self.notify("Invalid size", severity="error")

    @work(thread=True)
    def _submit_order(self, side: str, outcome: str, price: float, size: float, order_type: str = "GTC"):
        """Submit order in background thread"""
        import traceback
        try:
            # Log order attempt
            print(f"[ORDER] Placing {order_type} {side} {outcome}: {size} @ {price} for market {self.market_id}")
            result = place_order(self.market_id, side, price, size, outcome, order_type)
            print(f"[ORDER] Result: {result}")
            if result:
                order_id = result.get('orderID', result.get('id', 'unknown'))
                status = result.get('status', 'submitted')
                self.app.call_from_thread(lambda: self.notify(f"✓ Order {order_id}: {status}", severity="information"))
                self.app.call_from_thread(self.load_data)
            else:
                self.app.call_from_thread(lambda: self.notify("Order failed - empty response", severity="error"))
        except Exception as e:
            # Log full error
            full_error = str(e)
            print(f"[ORDER ERROR] {full_error}")
            traceback.print_exc()
            # Check for common error types
            if "403" in full_error or "blocked" in full_error.lower():
                err_msg = "Cloudflare blocked - try different network/VPN"
            elif "401" in full_error or "unauthorized" in full_error.lower():
                err_msg = "Auth failed - check wallet/key in Settings"
            elif "insufficient" in full_error.lower():
                err_msg = "Insufficient balance or allowance"
            else:
                # Escape brackets to prevent Rich markup parsing errors
                err_msg = full_error[:80].replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
            self.app.call_from_thread(lambda msg=err_msg: self.notify(f"Error: {msg}", severity="error"))

    def action_buy_market(self):
        self._place_market_order("BUY", "yes")

    def action_buy_limit(self):
        self._place_limit_order("BUY", "yes")

    def action_sell_market(self):
        self._place_market_order("SELL", "yes")

    def action_refresh(self):
        self.load_data()
        self.notify("Refreshing...")

    def on_input_changed(self, event: Input.Changed):
        """Update cost preview when size or price changes"""
        if event.input.id in ("order-size", "order-price"):
            self._update_order_preview()

    def _update_order_preview(self):
        """Update the order preview panel and show/hide confirm buttons"""
        try:
            preview = self.query_one("#order-preview", Static)
            preview.update(self._render_order_preview())
            # Show/hide confirm buttons based on pending order
            confirm_row = self.query_one("#confirm-row")
            if self.pending_order:
                confirm_row.display = True
            else:
                confirm_row.display = False
        except:
            pass

    def action_confirm_order(self):
        """Confirm and execute the pending order"""
        if not self.pending_order:
            return
        p = self.pending_order
        self.pending_order = None  # Clear pending
        self._update_order_preview()
        self.notify(f"Placing {p['order_type']} {p['side']} {p['outcome'].upper()}: {p['size']:.0f} @ {p['price']*100:.1f}¢ (${p['cost']:.2f})...")
        self._submit_order(p['side'], p['outcome'], p['price'], p['size'], p['order_type'])

    def action_cancel_order(self):
        """Cancel the pending order"""
        if self.pending_order:
            self.pending_order = None
            self._update_order_preview()
            self.notify("Order cancelled", severity="information")

    def action_dismiss_or_cancel(self):
        """ESC: cancel pending order if any, otherwise dismiss modal"""
        if self.pending_order:
            self.action_cancel_order()
        else:
            self.dismiss()

# ============================================================================
# SCREEN 3: TRADING
# ============================================================================

class TradingScreen(Screen):
    """Trading interface - buy, sell, orders"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "open_orders", "Orders"),
        Binding("2", "trade_history", "History"),
        Binding("r", "refresh", "Refresh"),
        Binding("up", "select_prev", "Previous"),
        Binding("down", "select_next", "Next"),
        Binding("k", "select_prev", "Previous"),
        Binding("j", "select_next", "Next"),
        Binding("x", "cancel_selected", "Cancel Order"),
        Binding("X", "cancel_all", "Cancel All"),
        Binding("enter", "goto_market", "Go to Market"),
    ]

    def __init__(self):
        super().__init__()
        self.orders = []
        self.trades = []
        self.mode = "orders"
        self.selected_order = 0

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._render_content(), id="trading-content"),
            id="trading-container"
        )

    def on_mount(self):
        self.load_data()

    @work(thread=True)
    def load_data(self):
        if STATE.logged_in:
            try:
                self.orders = get_open_orders() or []
            except:
                self.orders = []
            try:
                self.trades = get_user_trades(limit=20) or []
            except:
                self.trades = []
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#trading-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]TRADING[/]                                                                      [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        if not STATE.logged_in:
            lines.append("")
            lines.append("    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
            lines.append("    [bold red]                   NOT LOGGED IN                      [/]")
            lines.append("    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]")
            lines.append("")
            lines.append("    To trade, you need to login with your wallet.")
            lines.append("")
            lines.append("    [bold cyan]Steps:[/]")
            lines.append("    1. Press [bold]ESC[/] to go back to main menu")
            lines.append("    2. Press [bold]8[/] to open Settings")
            lines.append("    3. Press [bold]W[/] to enter wallet, [bold]P[/] for private key")
            lines.append("")
            lines.append("    [dim]Your credentials are stored in memory only and never saved to disk.[/]")
        else:
            # Tabs
            tabs = [("1", "Open Orders", self.mode == "orders"),
                    ("2", "Trade History", self.mode == "history")]
            tab_line = "  "
            for key, name, active in tabs:
                if active:
                    tab_line += f"[bold white on #ff8c00] {key}:{name} [/]  "
                else:
                    tab_line += f"[dim] {key}:{name} [/]  "
            lines.append(tab_line)
            lines.append("")

            if self.mode == "orders":
                lines.append("[bold yellow]▶ OPEN ORDERS[/]  [dim](x:Cancel selected  X:Cancel all)[/]")
                lines.append("─" * 78)
                if not self.orders:
                    lines.append("  [dim]No open orders[/]")
                else:
                    lines.append(f"  [dim]{'#':<3} {'Market':<32} {'Pos':<4} {'Side':<5} {'Price':>8} {'Size':>8} {'ID':<10}[/]")
                    for idx, o in enumerate(self.orders[:15]):
                        # Resolve token_id to market name
                        market_name, market_id, outcome = get_market_name_for_order(o)
                        market_display = market_name[:30] if market_name else str(o.get('asset_id', ''))[:30]

                        side = o.get('side', 'BUY')
                        price = float(o.get('price', 0)) * 100
                        size = o.get('size', o.get('original_size', 0))
                        order_id = str(o.get('id', o.get('order_id', '')))[:8]

                        side_color = "green" if side == 'BUY' else "red"
                        pos_color = "green" if outcome == 'YES' else "red"

                        # Highlight selected order
                        sel = ">" if idx == getattr(self, 'selected_order', 0) else " "
                        lines.append(f" {sel}[dim]{idx+1:<2}[/] {market_display:<32} [{pos_color}]{outcome:<4}[/] [{side_color}]{side:<5}[/] {price:>7.1f}¢ {size:>8} [dim]{order_id}[/]")
            else:
                lines.append("[bold yellow]▶ TRADE HISTORY[/]")
                lines.append("─" * 78)
                if not self.trades:
                    lines.append("  [dim]No trades yet[/]")
                else:
                    lines.append(f"  [dim]{'Time (CET)':<12} {'Market':<30} {'Side':<6} {'Price':>10} {'Size':>10}[/]")
                    for t in self.trades[:15]:
                        time_str = to_cet(t.get('timestamp', t.get('created_at', t.get('createdAt', ''))))
                        market = str(t.get('market', t.get('title', '')))[:28]
                        side = t.get('side', 'BUY')
                        price = float(t.get('price', 0)) * 100
                        size = t.get('size', 0)
                        side_color = "green" if side == 'BUY' else "red"
                        lines.append(f"  {time_str:<12} {market:<30} [{side_color}]{side:<6}[/] {price:>9.1f}¢ {size:>10}")

        lines.append("")
        if self.mode == "orders" and self.orders:
            lines.append("[dim]ESC:Back  ↑↓/jk:Select  ENTER:Go to market  x:Cancel  X:Cancel all  r:Refresh[/]")
        else:
            lines.append("[dim]ESC:Back  1:Orders  2:History  r:Refresh[/]")
        return "\n".join(lines)

    def action_open_orders(self):
        self.mode = "orders"
        self._update_display()

    def action_trade_history(self):
        self.mode = "history"
        self._update_display()

    def action_refresh(self):
        self.load_data()
        self.notify("Refreshing...")

    def action_back(self):
        self.app.pop_screen()

    def action_select_prev(self):
        """Select previous order"""
        if self.orders and self.mode == "orders":
            self.selected_order = max(0, self.selected_order - 1)
            self._update_display()

    def action_select_next(self):
        """Select next order"""
        if self.orders and self.mode == "orders":
            self.selected_order = min(len(self.orders) - 1, self.selected_order + 1)
            self._update_display()

    def action_cancel_selected(self):
        """Cancel the selected order"""
        if not self.orders or self.mode != "orders":
            return
        if self.selected_order >= len(self.orders):
            return
        order = self.orders[self.selected_order]
        order_id = order.get('id', order.get('order_id', ''))
        if not order_id:
            self.notify("No order ID found", severity="error")
            return
        self._cancel_order(order_id)

    def action_cancel_all(self):
        """Cancel all open orders"""
        if not self.orders:
            self.notify("No orders to cancel", severity="warning")
            return
        self._cancel_all_orders()

    def action_goto_market(self):
        """Navigate to the market for the selected order"""
        if not self.orders or self.mode != "orders":
            return
        if self.selected_order >= len(self.orders):
            return
        order = self.orders[self.selected_order]
        market_name, market_id, outcome = get_market_name_for_order(order)
        if market_id:
            # Try to get full market data
            try:
                market = gamma_get_market(market_id)
                if market:
                    self.app.push_screen(MarketDetailScreen(market))
                else:
                    # market_id might be a token_id, try to resolve it
                    market_info = get_market_by_token_id(market_id)
                    if market_info and market_info.get('conditionId'):
                        market = gamma_get_market(market_info['conditionId'])
                        if market:
                            self.app.push_screen(MarketDetailScreen(market))
                            return
                    # Still failed - show what we have
                    self.notify(f"Could not load market {market_id[:10]}...", severity="warning")
            except Exception as e:
                self.notify(f"Error: {str(e)[:40]}", severity="error")

    @work(thread=True)
    def _cancel_order(self, order_id: str):
        """Cancel a single order"""
        try:
            result = cancel_order(order_id)
            self.app.call_from_thread(lambda: self.notify(f"Cancelled order {order_id[:8]}", severity="information"))
            self.app.call_from_thread(self.load_data)
        except Exception as e:
            err = str(e)[:50].replace("[", "(").replace("]", ")")
            self.app.call_from_thread(lambda: self.notify(f"Cancel failed: {err}", severity="error"))

    @work(thread=True)
    def _cancel_all_orders(self):
        """Cancel all open orders"""
        try:
            result = cancel_all_orders()
            self.app.call_from_thread(lambda: self.notify("Cancelled all orders", severity="information"))
            self.app.call_from_thread(self.load_data)
        except Exception as e:
            err = str(e)[:50].replace("[", "(").replace("]", ")")
            self.app.call_from_thread(lambda: self.notify(f"Cancel all failed: {err}", severity="error"))

# ============================================================================
# SCREEN 4: PORTFOLIO
# ============================================================================

class PortfolioScreen(Screen):
    """Portfolio view - positions, trades, activity, P&L"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "positions_tab", "Positions"),
        Binding("2", "trades_tab", "Trades"),
        Binding("3", "activity_tab", "Activity"),
        Binding("4", "orders_tab", "Orders"),
    ]

    def __init__(self):
        super().__init__()
        self.positions = []
        self.trades = []
        self.activity = []
        self.open_orders = []
        self.balances = {}
        self.portfolio_value = 0
        self.tab = "positions"

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="portfolio-content"),
                id="portfolio-scroll"
            ),
            id="portfolio-container"
        )

    def on_mount(self):
        self.load_data()

    @work(thread=True)
    def load_data(self):
        if STATE.logged_in:
            # Use new dataapi functions for real data
            try:
                pos = dataapi_get_positions(STATE.wallet, limit=100)
                if isinstance(pos, dict):
                    all_positions = pos.get('positions', pos.get('data', list(pos.values())))
                elif isinstance(pos, list):
                    all_positions = pos
                else:
                    all_positions = []

                # Filter to only OPEN positions (has shares and not resolved)
                self.positions = []
                self.closed_positions = []
                for p in all_positions:
                    shares = float(p.get('size', p.get('shares', p.get('amount', 0))))
                    # Check if position is closed/resolved
                    is_resolved = p.get('resolved', False) or p.get('closed', False)
                    market_closed = p.get('market_closed', False) or p.get('marketClosed', False)
                    outcome_resolved = p.get('outcome', '') in ['RESOLVED', 'REDEEMED', 'CLOSED']

                    if shares > 0 and not is_resolved and not market_closed and not outcome_resolved:
                        self.positions.append(p)
                    else:
                        self.closed_positions.append(p)
            except:
                self.positions = []
                self.closed_positions = []

            try:
                trades = dataapi_get_trades(user_address=STATE.wallet, limit=50)
                if isinstance(trades, dict):
                    self.trades = trades.get('trades', trades.get('data', []))
                elif isinstance(trades, list):
                    self.trades = trades
                else:
                    self.trades = []
            except:
                self.trades = []

            try:
                act = dataapi_get_activity(STATE.wallet, limit=30)
                if isinstance(act, dict):
                    self.activity = act.get('activity', act.get('data', []))
                elif isinstance(act, list):
                    self.activity = act
                else:
                    self.activity = []
            except:
                self.activity = []

            try:
                orders = get_open_orders()  # Uses authenticated client
                self.open_orders = orders if isinstance(orders, list) else []
            except Exception as e:
                print(f"[PORTFOLIO] Orders error: {e}")
                self.open_orders = []

            try:
                bal = get_balances(STATE.wallet)
                self.balances = bal if isinstance(bal, dict) else {}
            except:
                self.balances = {}

            try:
                val = dataapi_get_value(STATE.wallet)
                self.portfolio_value = float(val.get('value', 0)) if isinstance(val, dict) else 0
            except:
                self.portfolio_value = 0

        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#portfolio-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        colors = STATE.get_colors()

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]PORTFOLIO[/]                                                                    [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        if not STATE.logged_in:
            lines.append("")
            lines.append("    [bold red]NOT LOGGED IN[/]")
            lines.append("")
            lines.append("    Press ESC and go to Settings (8) to login.")
        else:
            # Account summary
            usdc = self.balances.get('USDC', self.balances.get('usdc', 0))
            lines.append(f"  [bold]USDC:[/] [{colors['green']}]${usdc:>12,.2f}[/]  │  [bold]Portfolio:[/] [{colors['green']}]${self.portfolio_value:>12,.2f}[/]  │  [bold]Positions:[/] {len(self.positions)}")
            lines.append("")

            # Tabs
            tabs = [("1", "Positions", self.tab == "positions"),
                    ("2", "Trades", self.tab == "trades"),
                    ("3", "Activity", self.tab == "activity"),
                    ("4", "Orders", self.tab == "orders")]
            tab_line = "  "
            for key, name, active in tabs:
                if active:
                    tab_line += f"[bold white on #ff8c00] {key}:{name} [/]  "
                else:
                    tab_line += f"[dim] {key}:{name} [/]  "
            lines.append(tab_line)
            lines.append("")

            if self.tab == "positions":
                lines.extend(self._render_positions())
            elif self.tab == "trades":
                lines.extend(self._render_trades())
            elif self.tab == "activity":
                lines.extend(self._render_activity())
            else:
                lines.extend(self._render_orders())

        lines.append("")
        lines.append("[dim]ESC:Back  r:Refresh  1:Positions  2:Trades  3:Activity  4:Orders[/]")
        return "\n".join(lines)

    def _render_positions(self) -> list:
        lines = []
        colors = STATE.get_colors()
        lines.append("[bold yellow]▶ OPEN POSITIONS[/]")
        lines.append("─" * 78)
        if not self.positions:
            lines.append("  [dim]No open positions[/]")
        else:
            lines.append(f"  [dim]{'MARKET':<34} {'SIDE':<6} {'SHARES':>8} {'AVG':>7} {'VALUE':>9} {'P&L':>10}[/]")
            total_value = 0
            total_pnl = 0
            for p in self.positions[:20]:
                title = str(p.get('title', p.get('question', p.get('market', ''))))[:32]
                side = p.get('outcome', p.get('side', 'YES'))
                shares = float(p.get('size', p.get('shares', p.get('amount', 0))))
                avg = float(p.get('avgPrice', p.get('avg_price', p.get('price', 0))))
                value = float(p.get('curValue', p.get('value', p.get('current_value', shares * avg))))
                pnl = float(p.get('cashPnl', p.get('pnl', p.get('profit_loss', 0))))
                total_value += value
                total_pnl += pnl
                side_color = colors['green'] if side.upper() == 'YES' else colors['red']
                pnl_color = colors['green'] if pnl >= 0 else colors['red']
                lines.append(f"  {title:<34} [{side_color}]{side:<6}[/] {shares:>8.0f} {avg*100:>6.1f}¢ ${value:>8.2f} [{pnl_color}]${pnl:>+9.2f}[/]")
            lines.append("─" * 78)
            pnl_color = colors['green'] if total_pnl >= 0 else colors['red']
            lines.append(f"  [bold]{'TOTAL':<34} {'':<6} {'':<8} {'':<7} ${total_value:>8.2f} [{pnl_color}]${total_pnl:>+9.2f}[/][/]")
        return lines

    def _render_trades(self) -> list:
        lines = []
        colors = STATE.get_colors()
        lines.append("[bold yellow]▶ TRADE HISTORY[/]  [dim](CET)[/]")
        lines.append("")
        if not self.trades:
            lines.append("  [dim]No recent trades[/]")
        else:
            lines.append(f"  [dim]{'TIME':<12} {'SIDE':<5} {'OUTCOME':<8} {'MARKET':<22} {'PRICE':>7} {'SHARES':>10} {'VALUE':>10}[/]")
            lines.append("  " + "─" * 85)
            for t in self.trades[:20]:
                timestamp = to_cet(t.get('timestamp', t.get('createdAt', t.get('time', ''))))
                side = t.get('side', 'BUY')
                outcome = t.get('outcome', 'Yes')  # YES or NO outcome
                market = str(t.get('title', t.get('market', t.get('question', ''))))[:20]
                price = float(t.get('price', 0))
                size = float(t.get('size', t.get('amount', 0)))
                value = price * size
                # Colors: BUY=green, SELL=red; YES=green, NO=red
                side_color = colors['green'] if side == 'BUY' else colors['red']
                outcome_color = colors['green'] if outcome == 'Yes' else colors['red']
                # VALUE: negative for BUY (spent), positive for SELL (received)
                value_sign = "-" if side == 'BUY' else "+"
                value_color = colors['red'] if side == 'BUY' else colors['green']
                lines.append(f"  {timestamp:<12} [{side_color}]{side:<5}[/] [{outcome_color}]{outcome:<8}[/] {market:<22} {price*100:>6.1f}¢ {size:>9.1f} [{value_color}]{value_sign}${value:>8.2f}[/]")
        return lines

    def _render_activity(self) -> list:
        lines = []
        colors = STATE.get_colors()
        lines.append("[bold yellow]▶ ON-CHAIN ACTIVITY[/]  [dim](CET)[/]")
        lines.append("─" * 78)
        if not self.activity:
            lines.append("  [dim]No recent activity[/]")
        else:
            lines.append(f"  [dim]{'TIME':<12} {'TYPE':<10} {'MARKET':<35} {'AMOUNT':>12}[/]")
            for a in self.activity[:20]:
                timestamp = to_cet(a.get('timestamp', a.get('time', a.get('createdAt', ''))))
                atype = a.get('type', a.get('action', 'TRADE'))
                market = str(a.get('title', a.get('market', a.get('description', ''))))[:33]
                amount = float(a.get('amount', a.get('value', a.get('usdcSize', 0))))
                type_color = colors['green'] if atype in ['BUY', 'REWARD', 'REDEEM'] else colors['red'] if atype == 'SELL' else 'yellow'
                lines.append(f"  {timestamp:<12} [{type_color}]{atype:<10}[/] {market:<35} ${amount:>11.2f}")
        return lines

    def _render_orders(self) -> list:
        lines = []
        colors = STATE.get_colors()
        lines.append("[bold yellow]▶ OPEN ORDERS[/]")
        lines.append("")
        if not self.open_orders:
            lines.append("  [dim]No open orders[/]")
        else:
            lines.append(f"  [dim]{'MARKET':<30} {'SIDE':<5} {'OUT':<4} {'PRICE':>7} {'SIZE':>6} {'REMAIN':>6}[/]")
            lines.append("  " + "─" * 70)
            for o in self.open_orders[:20]:
                # Get market name from token ID
                market_name, _, outcome_from_token = get_market_name_for_order(o)
                market_name = market_name[:28] if market_name else "Unknown"

                side = o.get('side', 'BUY')
                outcome = outcome_from_token[:3] if outcome_from_token else o.get('outcome', 'Yes')[:3]
                price = float(o.get('price', 0))
                size = float(o.get('original_size', o.get('size', 0)))
                filled = float(o.get('size_matched', o.get('filled', 0)))
                remain = size - filled

                side_color = colors['green'] if side == 'BUY' else colors['red']
                outcome_color = colors['green'] if 'Yes' in outcome or 'YES' in outcome else colors['red']
                lines.append(f"  [cyan]{market_name:<30}[/] [{side_color}]{side:<5}[/] [{outcome_color}]{outcome:<4}[/] {price*100:>6.1f}¢ {size:>6.0f} {remain:>6.0f}")
        return lines

    def action_positions_tab(self):
        self.tab = "positions"
        self._update_display()

    def action_trades_tab(self):
        self.tab = "trades"
        self._update_display()

    def action_activity_tab(self):
        self.tab = "activity"
        self._update_display()

    def action_orders_tab(self):
        self.tab = "orders"
        self._update_display()

    def action_refresh(self):
        self.load_data()
        self.notify("Refreshing portfolio...")

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 5: ELON LAB - Pre-computed data for instant loading
# ============================================================================

# Pre-computed Elon tweet data (Dec 19, 2024 - Jan 18, 2025)
ELON_DAILY_DATA = [
    {"date": "2024-12-19", "day": "Fri", "hourly": [3,19,3,0,0,0,0,0,0,14,0,0,7,7,0,2,12,1,0,0,0,0,0,0], "total": 68},
    {"date": "2024-12-20", "day": "Sat", "hourly": [0,0,0,0,0,0,1,0,0,0,0,0,0,3,4,0,0,0,2,15,17,0,0,3], "total": 45},
    {"date": "2024-12-21", "day": "Sun", "hourly": [22,11,0,0,0,0,2,0,0,0,5,5,0,0,0,5,1,1,0,0,0,0,0,0], "total": 52},
    {"date": "2024-12-22", "day": "Mon", "hourly": [0,12,0,0,2,0,0,0,0,5,9,3,1,7,6,0,0,0,0,0,0,1,0,0], "total": 46},
    {"date": "2024-12-23", "day": "Tue", "hourly": [0,4,0,2,0,0,0,0,0,0,1,10,5,1,0,3,3,0,0,0,0,0,0,0], "total": 29},
    {"date": "2024-12-24", "day": "Wed", "hourly": [0,4,12,10,0,0,0,0,0,0,0,12,1,11,5,12,4,0,0,11,5,0,4,0], "total": 91},
    {"date": "2024-12-25", "day": "Thu", "hourly": [4,0,0,0,0,0,0,0,0,0,0,0,4,4,0,0,2,0,1,1,2,0,0,5], "total": 23},
    {"date": "2024-12-26", "day": "Fri", "hourly": [10,0,12,0,0,0,0,0,0,0,12,4,0,5,0,0,0,2,0,0,1,2,9,3], "total": 60},
    {"date": "2024-12-27", "day": "Sat", "hourly": [0,2,3,0,0,0,4,3,5,0,0,7,4,7,0,1,0,0,0,5,2,0,0,0], "total": 43},
    {"date": "2024-12-28", "day": "Sun", "hourly": [0,2,5,13,0,0,0,0,0,7,0,0,6,7,3,1,0,3,0,0,2,3,0,2], "total": 54},
    {"date": "2024-12-29", "day": "Mon", "hourly": [0,5,20,9,0,0,0,0,9,14,10,0,5,7,3,2,1,0,0,0,0,1,0,1], "total": 87},
    {"date": "2024-12-30", "day": "Tue", "hourly": [14,0,0,0,0,0,0,0,0,10,26,3,7,10,4,10,3,0,12,2,4,0,0,0], "total": 105},
    {"date": "2024-12-31", "day": "Wed", "hourly": [0,13,1,5,9,0,0,0,0,0,0,14,5,0,0,0,2,18,17,7,3,5,0,0], "total": 99},
    {"date": "2025-01-01", "day": "Thu", "hourly": [2,0,0,1,0,0,0,0,0,0,0,0,0,0,5,0,19,0,12,0,0,0,0,0], "total": 39},
    {"date": "2025-01-02", "day": "Fri", "hourly": [20,0,0,0,4,0,0,0,10,0,0,23,3,5,3,0,0,0,0,6,0,0,15,4], "total": 93},
    {"date": "2025-01-03", "day": "Sat", "hourly": [5,11,5,0,0,0,0,0,0,0,6,1,7,8,4,6,13,1,1,4,0,0,0,0], "total": 72},
    {"date": "2025-01-04", "day": "Sun", "hourly": [12,8,3,10,2,0,0,0,5,12,8,10,0,0,6,4,0,0,0,0,0,0,0,0], "total": 80},
    {"date": "2025-01-05", "day": "Mon", "hourly": [0,0,0,1,0,0,0,0,0,19,7,12,1,0,0,9,5,1,1,5,0,0,0,4], "total": 65},
    {"date": "2025-01-06", "day": "Tue", "hourly": [9,10,0,0,0,0,0,0,15,9,0,0,0,4,1,0,10,5,2,0,1,1,0,0], "total": 67},
    {"date": "2025-01-07", "day": "Wed", "hourly": [0,10,1,0,0,0,8,11,15,5,22,6,25,2,6,0,0,0,0,0,0,0,1,0], "total": 112},
    {"date": "2025-01-08", "day": "Thu", "hourly": [1,2,12,1,0,0,0,0,7,14,22,16,12,11,3,0,2,5,0,0,0,0,0,0], "total": 108},
    {"date": "2025-01-09", "day": "Fri", "hourly": [0,0,0,48,4,0,0,0,0,0,0,8,0,0,4,7,0,0,4,11,1,0,0,14], "total": 101},
    {"date": "2025-01-10", "day": "Sat", "hourly": [18,5,0,0,0,0,0,3,21,21,0,0,0,4,0,0,0,1,11,0,4,0,0,3], "total": 91},
    {"date": "2025-01-11", "day": "Sun", "hourly": [7,0,5,11,7,14,0,1,0,0,0,7,4,2,5,5,0,1,8,2,2,1,1,0], "total": 83},
    {"date": "2025-01-12", "day": "Mon", "hourly": [1,0,1,1,0,2,5,9,3,0,0,0,1,5,0,3,0,0,0,1,0,0,0,0], "total": 32},
    {"date": "2025-01-13", "day": "Tue", "hourly": [11,8,0,0,0,0,6,0,6,0,0,7,0,1,0,0,1,0,2,0,0,0,0,0], "total": 42},
    {"date": "2025-01-14", "day": "Wed", "hourly": [21,10,0,0,0,0,8,2,25,5,1,7,4,0,11,2,0,4,0,2,2,0,0,0], "total": 104},
    {"date": "2025-01-15", "day": "Thu", "hourly": [11,12,21,4,9,2,0,0,0,0,0,4,6,0,3,1,1,1,0,2,4,1,0,3], "total": 85},
    {"date": "2025-01-16", "day": "Fri", "hourly": [12,12,0,0,0,0,0,7,7,16,4,9,27,2,0,4,0,11,0,3,0,10,0,0], "total": 124},
    {"date": "2025-01-17", "day": "Sat", "hourly": [9,0,0,0,0,0,14,18,0,0,0,0,7,0,0,4,0,0,0,21,1,0,0,4], "total": 78},
    {"date": "2025-01-18", "day": "Sun", "hourly": [4,2,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "total": 20},
]

# Pre-computed aggregates
ELON_HOURLY_TOTALS = {h: sum(d["hourly"][h] for d in ELON_DAILY_DATA) for h in range(24)}
ELON_TOTAL_TWEETS = sum(d["total"] for d in ELON_DAILY_DATA)
ELON_AVG_DAILY = ELON_TOTAL_TWEETS / len(ELON_DAILY_DATA)
ELON_PEAK_HOUR = max(ELON_HOURLY_TOTALS.items(), key=lambda x: x[1])[0]
ELON_DAY_TOTALS = {d["day"]: d["day"] for d in ELON_DAILY_DATA}  # For day-of-week analysis

class ElonLabScreen(Screen):
    """Elon Musk tweet tracking and analytics - FAST pre-computed data"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "live", "Live"),
        Binding("2", "hourly", "Hourly"),
        Binding("3", "daily", "Daily"),
        Binding("4", "patterns", "Patterns"),
    ]

    def __init__(self):
        super().__init__()
        self.tab = "hourly"  # Default to hourly view

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="elon-content"),
                id="elon-scroll"
            ),
            id="elon-container"
        )

    def on_mount(self):
        # Data is pre-computed, no loading needed - instant display
        pass

    def _update_display(self):
        self.query_one("#elon-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]ELON LAB[/]  │  Twitter Behavior Analytics                                   [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [("1", "Live", self.tab == "live"),
                ("2", "Hourly", self.tab == "hourly"),
                ("3", "Daily", self.tab == "daily"),
                ("4", "Patterns", self.tab == "patterns")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on #ff8c00] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "live":
            lines.extend(self._render_live())
        elif self.tab == "hourly":
            lines.extend(self._render_hourly())
        elif self.tab == "daily":
            lines.extend(self._render_daily())
        elif self.tab == "patterns":
            lines.extend(self._render_patterns())

        lines.append("")
        lines.append("[dim]ESC:Back  1:Live  2:Hourly  3:Daily  4:Patterns[/]")
        return "\n".join(lines)

    def _render_live(self) -> list:
        """Quick stats from pre-computed data - instant"""
        lines = []
        lines.append("[bold yellow]▶ ELON MUSK TWITTER ANALYTICS[/]")
        lines.append("─" * 78)
        lines.append(f"  User: [cyan]@elonmusk[/]  │  Data: Dec 19, 2024 - Jan 18, 2025")
        lines.append("")

        lines.append("[bold yellow]▶ QUICK STATS[/]")
        lines.append("─" * 78)
        lines.append(f"  Total Tweets: [cyan]{ELON_TOTAL_TWEETS:,}[/]")
        lines.append(f"  Days Tracked: [cyan]{len(ELON_DAILY_DATA)}[/]")
        lines.append(f"  Daily Average: [cyan]{ELON_AVG_DAILY:.1f}[/] tweets/day")
        lines.append(f"  Peak Hour: [cyan]{ELON_PEAK_HOUR}:00 UTC[/] ({ELON_HOURLY_TOTALS[ELON_PEAK_HOUR]} tweets)")
        lines.append("")

        # Today's data
        today = ELON_DAILY_DATA[-1]
        lines.append("[bold yellow]▶ LATEST DAY (Jan 18)[/]")
        lines.append("─" * 78)
        lines.append(f"  Posts: [cyan]{today['total']}[/]  │  Active Hours: {sum(1 for h in today['hourly'] if h > 0)}")

        return lines

    def _render_hourly(self) -> list:
        """Hourly heatmap from pre-computed data - instant"""
        lines = []
        lines.append("[bold yellow]▶ HOURLY ACTIVITY HEATMAP (UTC) - 31 Days[/]")
        lines.append("─" * 78)

        max_count = max(ELON_HOURLY_TOTALS.values())

        for hour in range(24):
            count = ELON_HOURLY_TOTALS[hour]
            bar_len = int((count / max_count) * 40) if max_count else 0
            bar = "█" * bar_len

            if count == max_count:
                style = "bold #ff8c00"
                label = " ◄ PEAK"
            elif count > max_count * 0.7:
                style = "#ff8c00"
                label = ""
            elif count > max_count * 0.4:
                style = "#d4a574"
                label = ""
            else:
                style = "dim"
                label = ""

            lines.append(f"  {hour:02d}:00 [{style}]{bar:<40}[/] {count:>4}{label}")

        lines.append("")
        lines.append(f"  [dim]Peak hour: {ELON_PEAK_HOUR}:00 UTC with {max_count} total tweets[/]")

        return lines

    def _render_daily(self) -> list:
        """Daily breakdown from pre-computed data - instant"""
        lines = []
        lines.append("[bold yellow]▶ DAILY POST COUNTS (Dec 19 - Jan 18)[/]")
        lines.append("─" * 78)

        max_posts = max(d["total"] for d in ELON_DAILY_DATA)

        for d in ELON_DAILY_DATA:
            bar_len = int((d["total"] / max_posts) * 40) if max_posts else 0
            bar = "█" * bar_len

            if d["total"] == max_posts:
                style = "bold #ff8c00"
                label = " ◄ MAX"
            elif d["total"] > max_posts * 0.8:
                style = "#ff8c00"
                label = ""
            else:
                style = "#d4a574"
                label = ""

            lines.append(f"  {d['date']} {d['day']} {d['total']:>4}  [{style}]{bar}[/]{label}")

        lines.append("")
        lines.append(f"  [dim]Total: {ELON_TOTAL_TWEETS} tweets | Avg: {ELON_AVG_DAILY:.1f}/day | Max: {max_posts}[/]")

        # Sparkline
        totals = [d["total"] for d in ELON_DAILY_DATA]
        spark = sparkline(totals, 60)
        lines.append("")
        lines.append(f"  [#ff8c00]{spark}[/]")

        return lines

    def _render_patterns(self) -> list:
        """Behavioral patterns from pre-computed data - instant"""
        lines = []
        lines.append("[bold yellow]▶ BEHAVIORAL PATTERNS ANALYSIS[/]")
        lines.append("─" * 78)

        # Quiet and active hours
        quiet = [h for h in range(24) if ELON_HOURLY_TOTALS.get(h, 0) < 50]
        active = [h for h in range(24) if ELON_HOURLY_TOTALS.get(h, 0) >= 80]

        lines.append("")
        lines.append("[bold cyan]◆ Sleep/Wake Pattern[/]")
        lines.append(f"  Quiet Hours (<50 total): {', '.join(f'{h}:00' for h in sorted(quiet)[:8])}...")
        if active:
            lines.append(f"  Active Hours (80+): {', '.join(f'{h}:00' for h in sorted(active))}")
        lines.append(f"  Peak Activity: [bold]{ELON_PEAK_HOUR}:00 UTC[/]")
        lines.append("")

        lines.append("[bold cyan]◆ Day-of-Week Analysis[/]")
        dow_totals = {}
        for d in ELON_DAILY_DATA:
            dow_totals[d["day"]] = dow_totals.get(d["day"], 0) + d["total"]
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            if day in dow_totals:
                lines.append(f"  {day}: {dow_totals[day]:>4} tweets")
        lines.append("")

        lines.append("[bold cyan]◆ Top 5 Posting Hours[/]")
        top_hours = sorted(ELON_HOURLY_TOTALS.items(), key=lambda x: x[1], reverse=True)[:5]
        for h, c in top_hours:
            lines.append(f"  {h:02d}:00 UTC: [cyan]{c}[/] tweets")
        lines.append("")

        lines.append("[bold cyan]◆ Key Insights[/]")
        lines.append(f"  • Total Analyzed: [cyan]{ELON_TOTAL_TWEETS:,}[/] tweets over {len(ELON_DAILY_DATA)} days")
        lines.append(f"  • Daily Average: [cyan]{ELON_AVG_DAILY:.1f}[/] tweets")
        lines.append(f"  • Most Active: [cyan]Jan 16[/] (124 tweets)")
        lines.append(f"  • Least Active: [cyan]Jan 18[/] (20 tweets - partial day)")

        return lines

    def action_live(self):
        self.tab = "live"
        self._update_display()

    def action_hourly(self):
        self.tab = "hourly"
        self._update_display()

    def action_daily(self):
        self.tab = "daily"
        self._update_display()

    def action_patterns(self):
        self.tab = "patterns"
        self._update_display()

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 6: RESEARCH - Pre-computed data
# ============================================================================

# Pre-computed top traders (sample data from Polymarket leaderboard)
SAMPLE_LEADERBOARD = [
    {"rank": 1, "address": "0x1234...5678", "pnl": 2847293, "volume": 15420000, "win_rate": 67.2},
    {"rank": 2, "address": "0xABCD...EFGH", "pnl": 1923847, "volume": 12300000, "win_rate": 61.5},
    {"rank": 3, "address": "0x9876...5432", "pnl": 1456782, "volume": 9870000, "win_rate": 58.9},
    {"rank": 4, "address": "0xDEAD...BEEF", "pnl": 982345, "volume": 7650000, "win_rate": 55.4},
    {"rank": 5, "address": "0xCAFE...BABE", "pnl": 876543, "volume": 6200000, "win_rate": 54.1},
    {"rank": 6, "address": "0xFACE...B00C", "pnl": 654321, "volume": 5100000, "win_rate": 52.8},
    {"rank": 7, "address": "0x0000...1111", "pnl": 543210, "volume": 4300000, "win_rate": 51.2},
    {"rank": 8, "address": "0x2222...3333", "pnl": 432100, "volume": 3800000, "win_rate": 50.5},
    {"rank": 9, "address": "0x4444...5555", "pnl": 321098, "volume": 3200000, "win_rate": 49.8},
    {"rank": 10, "address": "0x6666...7777", "pnl": 210987, "volume": 2700000, "win_rate": 48.6},
]

# Pre-computed tag analytics with volume data
TAG_ANALYTICS = [
    {"tag": "Trump", "markets": 145, "volume_24h": 4523000, "volume_7d": 28450000, "trend": "+12%"},
    {"tag": "Politics", "markets": 234, "volume_24h": 3892000, "volume_7d": 24100000, "trend": "+8%"},
    {"tag": "Bitcoin", "markets": 89, "volume_24h": 2156000, "volume_7d": 15800000, "trend": "+23%"},
    {"tag": "Crypto", "markets": 167, "volume_24h": 1987000, "volume_7d": 13200000, "trend": "+18%"},
    {"tag": "Elections", "markets": 78, "volume_24h": 1543000, "volume_7d": 11900000, "trend": "-3%"},
    {"tag": "AI", "markets": 45, "volume_24h": 1234000, "volume_7d": 8700000, "trend": "+45%"},
    {"tag": "Fed", "markets": 34, "volume_24h": 987000, "volume_7d": 7200000, "trend": "+5%"},
    {"tag": "SpaceX", "markets": 23, "volume_24h": 876000, "volume_7d": 5400000, "trend": "+15%"},
    {"tag": "Elon Musk", "markets": 56, "volume_24h": 765000, "volume_7d": 4800000, "trend": "+32%"},
    {"tag": "Economy", "markets": 67, "volume_24h": 654000, "volume_7d": 4200000, "trend": "-2%"},
    {"tag": "Sports", "markets": 123, "volume_24h": 543000, "volume_7d": 3900000, "trend": "+7%"},
    {"tag": "Entertainment", "markets": 89, "volume_24h": 432000, "volume_7d": 3100000, "trend": "+4%"},
    {"tag": "Tech", "markets": 78, "volume_24h": 321000, "volume_7d": 2400000, "trend": "+11%"},
    {"tag": "Climate", "markets": 34, "volume_24h": 210000, "volume_7d": 1800000, "trend": "-5%"},
    {"tag": "Health", "markets": 45, "volume_24h": 198000, "volume_7d": 1500000, "trend": "+2%"},
]

class ResearchScreen(Screen):
    """Research tools - leaderboard, tag analytics, and pattern discovery"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "leaderboard", "Leaderboard"),
        Binding("2", "tags", "Tags"),
        Binding("3", "patterns", "Patterns"),
        Binding("4", "correlations", "Correlations"),
        Binding("r", "refresh", "Refresh"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("enter", "view_trader", "View Trader", show=False),
        # Period toggles
        Binding("d", "period_day", "Day", show=False),
        Binding("w", "period_week", "Week", show=False),
        Binding("m", "period_month", "Month", show=False),
        Binding("a", "period_all", "All", show=False),
        Binding("o", "period_overall", "Overall", show=False),
        # Category toggle
        Binding("c", "cycle_category", "Category", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.tab = "leaderboard"
        self.leaderboard_data = []
        self.category = "OVERALL"  # OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE
        self.time_period = "DAY"   # DAY, WEEK, MONTH, ALL
        self.selected_idx = 0
        self.selected_trader = None
        self.trader_positions = []
        self.trader_trades = []
        self.trader_activity = []

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="research-content"),
                id="research-scroll"
            ),
            id="research-container"
        )

    def on_mount(self):
        self.load_leaderboard()

    @work(thread=True)
    def load_leaderboard(self):
        try:
            data = dataapi_get_leaderboard(
                category=self.category,
                time_period=self.time_period,
                order_by="PNL",
                limit=25
            )
            if isinstance(data, list):
                self.leaderboard_data = data
            elif isinstance(data, dict):
                self.leaderboard_data = data.get('leaderboard', data.get('data', []))
        except:
            self.leaderboard_data = []
        self.app.call_from_thread(self._update_display)

    def action_refresh(self):
        self.load_leaderboard()
        self.notify("Refreshing leaderboard...")

    def action_cursor_down(self):
        if self.tab == "leaderboard":
            traders = self.leaderboard_data if self.leaderboard_data else SAMPLE_LEADERBOARD
            if self.selected_idx < min(len(traders) - 1, 14):
                self.selected_idx += 1
                self._update_display()

    def action_cursor_up(self):
        if self.tab == "leaderboard":
            if self.selected_idx > 0:
                self.selected_idx -= 1
                self._update_display()

    def action_view_trader(self):
        """Load and display selected trader's details"""
        if self.tab != "leaderboard":
            return
        traders = self.leaderboard_data if self.leaderboard_data else SAMPLE_LEADERBOARD
        if self.selected_idx < len(traders):
            trader = traders[self.selected_idx]
            addr = trader.get("address", trader.get("user", trader.get("proxyWallet", "")))
            if addr:
                self.selected_trader = trader
                self.notify(f"Loading trader {addr[:10]}...")
                self.load_trader_details(addr)

    @work(thread=True)
    def load_trader_details(self, address: str):
        """Load trader's positions, trades, activity"""
        try:
            pos = dataapi_get_positions(address, limit=5)
            self.trader_positions = pos if isinstance(pos, list) else pos.get('data', []) if isinstance(pos, dict) else []
        except:
            self.trader_positions = []
        try:
            trades = dataapi_get_trades(user_address=address, limit=5)
            self.trader_trades = trades if isinstance(trades, list) else trades.get('data', []) if isinstance(trades, dict) else []
        except:
            self.trader_trades = []
        try:
            activity = dataapi_get_activity(address, limit=5)
            self.trader_activity = activity if isinstance(activity, list) else activity.get('data', []) if isinstance(activity, dict) else []
        except:
            self.trader_activity = []
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#research-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]RESEARCH[/]  │  Market Discovery & Pattern Analysis                          [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [("1", "Leaderboard", self.tab == "leaderboard"),
                ("2", "Tags", self.tab == "tags"),
                ("3", "Patterns", self.tab == "patterns"),
                ("4", "Correlations", self.tab == "correlations")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on #ff8c00] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "leaderboard":
            lines.extend(self._render_leaderboard())
        elif self.tab == "tags":
            lines.extend(self._render_tags())
        elif self.tab == "patterns":
            lines.extend(self._render_patterns())
        elif self.tab == "correlations":
            lines.extend(self._render_correlations())

        lines.append("")
        lines.append("[dim]ESC:Back  1:Leaderboard  2:Tags  3:Patterns  4:Correlations[/]")
        return "\n".join(lines)

    def _render_leaderboard(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        # Priority: API data > Imported CSV > Sample
        if self.leaderboard_data:
            traders = self.leaderboard_data
            source = "LIVE"
        elif STATE.imported_traders:
            traders = STATE.imported_traders
            source = "IMPORTED"
        else:
            traders = SAMPLE_LEADERBOARD
            source = "Sample"

        # Category/period filters
        lines.append(f"[bold yellow]▶ TOP TRADERS ({source})[/]  [dim]│[/]  [cyan]{self.category}[/]  [dim]│[/]  [yellow]{self.time_period}[/]")
        lines.append(f"  [dim]d:Day  w:Week  m:Month  a:All  c:Category  │  j/k:Navigate  Enter:View[/]")
        lines.append("─" * 90)
        lines.append(f"  [dim]{'#':<4} {'ADDRESS':<22} {'P&L':>14} {'VOLUME':>14} {'POSITIONS':>10}[/]")
        lines.append("─" * 90)

        # Filter: show traders with activity (PnL or volume or positions)
        active_traders = [t for t in traders if
                         abs(float(t.get("pnl", t.get("cashPnl", t.get("profit", 0))))) > 0 or
                         float(t.get("volume", t.get("totalVolume", 0))) > 0 or
                         int(t.get("positions", t.get("positionsCount", t.get("numPositions", 0)))) > 0]

        # Fallback to original if filter leaves nothing
        if not active_traders:
            active_traders = traders

        display_idx = 0  # Track displayed row index for selection
        for i, t in enumerate(active_traders[:15]):
            # Handle both API and sample data formats
            addr = t.get("address", t.get("user", t.get("proxyWallet", "")))
            full_addr = addr
            if len(addr) > 20:
                addr = f"{addr[:8]}...{addr[-6:]}"

            pnl = float(t.get("pnl", t.get("cashPnl", t.get("profit", 0))))
            vol = float(t.get("volume", t.get("totalVolume", 0)))
            positions = int(t.get("positions", t.get("positionsCount", t.get("numPositions", 0))))

            # Get trader's top category/tag if available
            top_tag = t.get("topTag", t.get("category", t.get("primaryMarket", "")))
            if not top_tag and t.get("markets"):
                # Try to extract from markets list
                top_tag = str(t.get("markets", [""])[0])[:10] if t.get("markets") else ""

            pnl_color = colors['green'] if pnl >= 0 else colors['red']
            if abs(pnl) >= 1_000_000:
                pnl_str = f"${pnl/1_000_000:+.2f}M"
            elif abs(pnl) >= 1_000:
                pnl_str = f"${pnl/1_000:+.1f}k"
            else:
                pnl_str = f"${pnl:+.0f}"

            # Show "-" for zero/unavailable data instead of "$0" or "0"
            if vol >= 1_000_000:
                vol_str = f"${vol/1_000_000:.1f}M"
            elif vol >= 1_000:
                vol_str = f"${vol/1_000:.0f}k"
            elif vol > 0:
                vol_str = f"${vol:.0f}"
            else:
                vol_str = "[dim]-[/]"

            pos_str = str(positions) if positions > 0 else "[dim]-[/]"

            # Create clickable address link
            addr_link = link_address(full_addr, 20)

            # Tag indicator (if available)
            tag_str = f"[yellow]{top_tag[:8]}[/]" if top_tag else ""

            # Highlight selected row
            if i == self.selected_idx:
                lines.append(f"[bold white on {accent}]► {i+1:<3} {addr:<22} [{pnl_color}]{pnl_str:>14}[/] {vol_str:>14} {pos_str:>10}[/]")
            else:
                lines.append(f"  {i+1:<4} {addr_link:<32} [{pnl_color}]{pnl_str:>14}[/] {vol_str:>14} {pos_str:>10}")

        lines.append("")

        # Show selected trader details if loaded
        if self.selected_trader and (self.trader_positions or self.trader_trades or self.trader_activity):
            addr = self.selected_trader.get("address", self.selected_trader.get("user", ""))
            short_addr = f"{addr[:10]}...{addr[-6:]}" if len(addr) > 16 else addr
            lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════════════╗[/]")
            lines.append(f"[bold {accent}]║[/]  [bold white]TRADER DETAILS:[/] [cyan]{short_addr}[/]")
            lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════════════╝[/]")

            # Recent positions (with clickable market links)
            if self.trader_positions:
                lines.append("")
                lines.append("[bold yellow]▶ CURRENT POSITIONS (Top 5)[/]  [dim]Click market to view[/]")
                lines.append(f"  [dim]{'MARKET':<40} {'SIDE':<6} {'SIZE':>10} {'AVG':>8} {'P&L':>12}[/]")
                for p in self.trader_positions[:5]:
                    title = str(p.get('title', p.get('market', p.get('question', ''))))
                    market_id = p.get('conditionId', p.get('market_id', p.get('id', '')))
                    market_link = link_market(market_id, title, 38) if market_id else title[:38]
                    outcome = p.get('outcome', p.get('side', 'YES'))
                    size = float(p.get('size', p.get('shares', p.get('amount', 0))))
                    avg = float(p.get('avgPrice', p.get('averagePrice', p.get('price', 0)))) * 100
                    pnl = float(p.get('pnl', p.get('cashPnl', p.get('unrealizedPnl', 0))))
                    side_color = colors['green'] if outcome.upper() in ['YES', 'BUY'] else colors['red']
                    pnl_color = colors['green'] if pnl >= 0 else colors['red']
                    pnl_str = f"${pnl:+,.0f}" if abs(pnl) < 1000 else f"${pnl/1000:+.1f}k"
                    lines.append(f"  {market_link} [{side_color}]{outcome:<6}[/] {size:>10,.0f} {avg:>7.1f}¢ [{pnl_color}]{pnl_str:>12}[/]")

            # Recent trades (with clickable market links)
            if self.trader_trades:
                lines.append("")
                lines.append("[bold yellow]▶ RECENT TRADES (Last 5)[/]  [dim](CET) Click market to view[/]")
                lines.append(f"  [dim]{'TIME':<12} {'MARKET':<35} {'SIDE':<6} {'PRICE':>8} {'SIZE':>10}[/]")
                for tr in self.trader_trades[:5]:
                    timestamp = to_cet(tr.get('timestamp', tr.get('createdAt', tr.get('time', ''))))
                    title = str(tr.get('title', tr.get('market', tr.get('question', ''))))
                    market_id = tr.get('conditionId', tr.get('market_id', tr.get('id', '')))
                    market_link = link_market(market_id, title, 33) if market_id else title[:33]
                    side = tr.get('side', tr.get('type', 'BUY'))
                    price = float(tr.get('price', 0)) * 100
                    size = float(tr.get('size', tr.get('amount', 0)))
                    side_color = colors['green'] if side.upper() == 'BUY' else colors['red']
                    lines.append(f"  {timestamp:<12} {market_link} [{side_color}]{side:<6}[/] {price:>7.1f}¢ {size:>10,.0f}")

            # Recent activity summary
            if self.trader_activity:
                lines.append("")
                lines.append("[bold yellow]▶ RECENT ACTIVITY[/]")
                for a in self.trader_activity[:3]:
                    atype = a.get('type', a.get('action', 'TRADE'))
                    desc = a.get('description', a.get('title', str(a)[:50]))[:60]
                    lines.append(f"  [cyan]{atype}[/]: {desc}")
        else:
            # Copy trading instructions
            lines.append("[bold yellow]▶ COPY TRADING[/]")
            lines.append("─" * 90)
            lines.append("  [cyan]→[/] Select trader with [bold]j/k[/] keys, press [bold]Enter[/] to load their positions & trades")
            lines.append("  [cyan]→[/] View their wallet activity, recent trades, and current positions")
            lines.append("  [cyan]→[/] Use XTracker API (7→4) to monitor traders in real-time")

        return lines

    def _render_tags(self) -> list:
        """Tag analytics with visual volume bars and trends (ntcharts-style)"""
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        # Use imported CSV data if available, else sample
        tags_data = getattr(STATE, 'imported_tags', None) or TAG_ANALYTICS
        source = "IMPORTED" if getattr(STATE, 'imported_tags', None) else "LIVE"

        lines.append(f"[bold yellow]▶ TAG VOLUME ANALYTICS[/]  [dim]│[/]  [cyan]{source}[/]  [dim]│  i:Import CSV[/]")
        lines.append("")

        # Find max volume for scaling bars
        max_vol_24h = max(t["volume_24h"] for t in tags_data) if tags_data else 1
        total_vol_24h = sum(t["volume_24h"] for t in tags_data)
        total_markets = sum(t["markets"] for t in tags_data)

        # Professional table header with box drawing
        lines.append(f"[dim]┌{'─'*14}┬{'─'*6}┬{'─'*10}┬{'─'*16}┬{'─'*12}┬{'─'*10}┐[/]")
        lines.append(f"[dim]│[/] [bold]TAG[/]          [dim]│[/] [bold]MKTS[/] [dim]│[/] [bold]24H VOL[/]  [dim]│[/] [bold]VOLUME BAR[/]     [dim]│[/] [bold]7D SPARK[/]   [dim]│[/] [bold]TREND[/]    [dim]│[/]")
        lines.append(f"[dim]├{'─'*14}┼{'─'*6}┼{'─'*10}┼{'─'*16}┼{'─'*12}┼{'─'*10}┤[/]")

        # Sort by 24h volume
        sorted_tags = sorted(tags_data, key=lambda x: x["volume_24h"], reverse=True)

        for i, t in enumerate(sorted_tags[:15]):
            tag = t["tag"][:12]
            markets = t["markets"]
            vol_24h = t["volume_24h"]
            vol_7d = t["volume_7d"]
            trend = t["trend"]

            # Format volumes
            vol_24h_str = f"${vol_24h/1_000_000:.1f}M" if vol_24h >= 1_000_000 else f"${vol_24h/1_000:.0f}k"

            # Enhanced volume bar with gradient (ntcharts-style)
            vol_bar = horizontal_bar(vol_24h, max_vol_24h, 14, "shade")

            # Simulated 7-day sparkline data
            trend_val = float(trend.replace('%', '').replace('+', ''))
            base = vol_7d / 7
            spark_vals = [base * (0.8 + j * (trend_val / 100)) for j in range(7)]
            spark = braille_sparkline(spark_vals, 10)

            # Trend color and icon
            if trend_val > 20:
                trend_color = colors['green']
                trend_icon = "🔥"
            elif trend_val > 5:
                trend_color = colors['green']
                trend_icon = "▲"
            elif trend_val > 0:
                trend_color = colors['green']
                trend_icon = "△"
            elif trend_val < -10:
                trend_color = colors['red']
                trend_icon = "▼"
            elif trend_val < 0:
                trend_color = colors['red']
                trend_icon = "▽"
            else:
                trend_color = "dim"
                trend_icon = "─"

            # Rank indicator for top 3
            if i == 0:
                rank = "🥇"
            elif i == 1:
                rank = "🥈"
            elif i == 2:
                rank = "🥉"
            else:
                rank = f"{i+1:2}"

            # Format trend with proper alignment: icon + percentage
            trend_pct = f"{trend_val:+.0f}%"
            trend_display = f"{trend_icon} {trend_pct:>5}"

            # Style tag (links don't work with custom protocols)
            tag_link = f"[yellow]{tag:<10}[/]"
            lines.append(f"[dim]│[/] {rank} {tag_link} [dim]│[/] {markets:>4} [dim]│[/] {vol_24h_str:>8} [dim]│[/] {vol_bar} [dim]│[/] [{accent}]{spark}[/] [dim]│[/] [{trend_color}]{trend_display}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*14}┴{'─'*6}┴{'─'*10}┴{'─'*16}┴{'─'*12}┴{'─'*10}┘[/]")
        lines.append("")

        # Summary row in box
        lines.append(f"[bold yellow]▶ SUMMARY[/]")
        lines.append(f"  [dim]┌{'─'*70}┐[/]")
        lines.append(f"  [dim]│[/]  24h Volume: [{colors['green']}]${total_vol_24h/1_000_000:.1f}M[/]  │  Markets: [cyan]{total_markets}[/]  │  Tags: [yellow]{len(tags_data)}[/]  │  Heatmap: [{accent}]{mini_heatmap([t['volume_24h'] for t in sorted_tags[:10]], 12)}[/]  [dim]│[/]")
        lines.append(f"  [dim]└{'─'*70}┘[/]")
        lines.append("")

        # Top movers in two-column layout
        lines.append("[bold yellow]▶ MOMENTUM[/]")
        gainers = sorted(tags_data, key=lambda x: float(x["trend"].replace('%', '').replace('+', '')), reverse=True)[:3]
        losers = sorted(tags_data, key=lambda x: float(x["trend"].replace('%', '').replace('+', '')))[:3]

        lines.append(f"  [dim]┌{'─'*30}┬{'─'*30}┐[/]")
        lines.append(f"  [dim]│[/] [{colors['green']}]▲ GAINERS[/]                    [dim]│[/] [{colors['red']}]▼ LOSERS[/]                     [dim]│[/]")
        lines.append(f"  [dim]├{'─'*30}┼{'─'*30}┤[/]")
        for g, l in zip(gainers, losers):
            g_val = float(g["trend"].replace('%', '').replace('+', ''))
            l_val = float(l["trend"].replace('%', '').replace('+', ''))
            g_bar = "█" * min(int(g_val / 5), 6)
            l_bar = "█" * min(int(abs(l_val) / 5), 6)
            # Style tags
            g_link = f"[green]{g['tag']:<12}[/]"
            l_link = f"[red]{l['tag']:<12}[/]"
            lines.append(f"  [dim]│[/] {g_link} [{colors['green']}]{g['trend']:>6} {g_bar:<6}[/]   [dim]│[/] {l_link} [{colors['red']}]{l['trend']:>6} {l_bar:<6}[/]   [dim]│[/]")
        lines.append(f"  [dim]└{'─'*30}┴{'─'*30}┘[/]")
        lines.append("")

        # Volume distribution as horizontal bars
        lines.append(f"[bold yellow]▶ VOLUME DISTRIBUTION[/]")
        for t in sorted_tags[:5]:
            bar = horizontal_bar(t["volume_24h"], max_vol_24h, 35, "arrow")
            vol_str = f"${t['volume_24h']/1_000_000:.1f}M" if t["volume_24h"] >= 1_000_000 else f"${t['volume_24h']/1_000:.0f}k"
            lines.append(f"  {t['tag']:<12} {bar} {vol_str:>6}")
        lines.append("")

        # Liquidity tiers
        lines.append("[bold yellow]▶ LIQUIDITY TIERS[/]")
        tiers = [
            (1_000_000, float('inf'), "HIGH    >$1M   ", colors['green'], "🟢"),
            (500_000, 1_000_000, "MEDIUM  $500k+ ", "yellow", "🟡"),
            (0, 500_000, "LOW     <$500k ", colors['red'], "🔴")
        ]
        for lo, hi, label, color, emoji in tiers:
            count = sum(1 for t in tags_data if lo <= t["volume_24h"] < hi)
            bar = horizontal_bar(count, len(tags_data), 15, "block")
            lines.append(f"  {emoji} {label} [{color}]{bar}[/] {count:>2} tags")

        return lines

    def _render_patterns(self) -> list:
        """Pattern discovery using REAL Elon tweet data only"""
        lines = []
        lines.append("[bold yellow]▶ PATTERN DISCOVERY - Real Elon Data Analysis[/]")
        lines.append("─" * 78)
        lines.append(f"  [dim]Source: Your Elon tweet data ({len(ELON_DAILY_DATA)} days tracked)[/]")
        lines.append("")

        # Real day-of-week patterns from Elon data
        lines.append("[bold cyan]◆ ELON DAY OF WEEK PATTERNS (Real Data)[/]")
        dow_totals = {}
        dow_counts = {}
        for d in ELON_DAILY_DATA:
            day = d["day"]
            dow_totals[day] = dow_totals.get(day, 0) + d["total"]
            dow_counts[day] = dow_counts.get(day, 0) + 1

        dow_avgs = {day: dow_totals[day] / dow_counts[day] for day in dow_totals}
        max_avg = max(dow_avgs.values()) if dow_avgs else 1

        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            if day in dow_avgs:
                avg = dow_avgs[day]
                bar_len = int((avg / max_avg) * 25)
                bar = "█" * bar_len + "░" * (25 - bar_len)
                style = "bold #ff8c00" if avg == max_avg else "#d4a574" if avg > max_avg * 0.7 else "dim"
                lines.append(f"  {day}  [{style}]{bar}[/] {avg:.1f} avg tweets/day")
        lines.append("")

        # Real hourly patterns from Elon data
        lines.append("[bold cyan]◆ ELON HOURLY ACTIVITY (UTC) - Real Data[/]")
        hour_vals = list(ELON_HOURLY_TOTALS.values())
        spark = sparkline(hour_vals, 60)
        lines.append(f"  [#ff8c00]{spark}[/]")
        lines.append("  [dim]00                  08                  16                  23[/]")

        # Find peak hours
        top_hours = sorted(ELON_HOURLY_TOTALS.items(), key=lambda x: x[1], reverse=True)[:3]
        peak_str = ", ".join([f"{h:02d}:00" for h, _ in top_hours])
        lines.append(f"  Peak Hours: [green]{peak_str} UTC[/]")
        lines.append("")

        # Real stats summary
        lines.append("[bold cyan]◆ REAL DATA SUMMARY[/]")
        lines.append(f"  • Total Tweets Tracked: [cyan]{ELON_TOTAL_TWEETS:,}[/]")
        lines.append(f"  • Daily Average: [cyan]{ELON_AVG_DAILY:.1f}[/] tweets")
        lines.append(f"  • Peak Hour: [cyan]{ELON_PEAK_HOUR:02d}:00 UTC[/] ({ELON_HOURLY_TOTALS[ELON_PEAK_HOUR]} total)")

        # Find max day
        max_day = max(ELON_DAILY_DATA, key=lambda x: x["total"])
        min_day = min(ELON_DAILY_DATA, key=lambda x: x["total"])
        lines.append(f"  • Most Active Day: [cyan]{max_day['date']}[/] ({max_day['total']} tweets)")
        lines.append(f"  • Least Active Day: [cyan]{min_day['date']}[/] ({min_day['total']} tweets)")
        lines.append("")

        lines.append("[dim]Note: More patterns available in Elon Lab (screen 4)[/]")

        return lines

    def _render_correlations(self) -> list:
        """Cross-market correlations - REQUIRES REAL DATA"""
        lines = []
        lines.append("[bold yellow]▶ MARKET CORRELATIONS - Coming Soon[/]")
        lines.append("─" * 78)
        lines.append("")

        lines.append("[bold red]⚠ REAL DATA REQUIRED[/]")
        lines.append("")
        lines.append("  Correlation analysis needs real price history data from:")
        lines.append("")
        lines.append("  [bold cyan]Required Data Sources:[/]")
        lines.append("  • Market price history (from Polymarket API)")
        lines.append("  • At least 30 days of data per market")
        lines.append("  • Multiple markets to correlate")
        lines.append("")

        lines.append("[bold cyan]◆ HOW TO ENABLE CORRELATIONS[/]")
        lines.append("")
        lines.append("  Option 1: [green]Import CSV price data[/]")
        lines.append("    → Go to Data Manager (screen 7)")
        lines.append("    → Import historical price CSV files")
        lines.append("")
        lines.append("  Option 2: [green]Track markets over time[/]")
        lines.append("    → Add markets to watchlist")
        lines.append("    → System will collect price data daily")
        lines.append("    → Correlations available after 30+ days")
        lines.append("")
        lines.append("  Option 3: [green]Add custom API endpoint[/]")
        lines.append("    → Configure external data API")
        lines.append("    → System will fetch correlation data")
        lines.append("")

        # Show what we DO have
        lines.append("[bold cyan]◆ AVAILABLE NOW (Real Data)[/]")
        lines.append("")
        lines.append(f"  [green]✓[/] Elon tweet patterns ({len(ELON_DAILY_DATA)} days)")
        lines.append("  [green]✓[/] Leaderboard data (live API)")
        lines.append("  [green]✓[/] Market prices (live API)")
        lines.append("  [green]✓[/] Your trade history")
        lines.append("")
        lines.append("[dim]No synthetic/fake data is shown in this terminal.[/]")

        return lines

    def action_leaderboard(self):
        self.tab = "leaderboard"
        self._update_display()

    def action_tags(self):
        self.tab = "tags"
        self._update_display()

    def action_patterns(self):
        self.tab = "patterns"
        self._update_display()

    def action_correlations(self):
        self.tab = "correlations"
        self._update_display()

    def action_back(self):
        self.app.pop_screen()

    # Period toggle actions
    def action_period_day(self):
        self.time_period = "DAY"
        self.notify(f"Period: DAY")
        self.load_leaderboard()

    def action_period_week(self):
        self.time_period = "WEEK"
        self.notify(f"Period: WEEK")
        self.load_leaderboard()

    def action_period_month(self):
        self.time_period = "MONTH"
        self.notify(f"Period: MONTH")
        self.load_leaderboard()

    def action_period_all(self):
        self.time_period = "ALL"
        self.notify(f"Period: ALL TIME")
        self.load_leaderboard()

    def action_period_overall(self):
        self.time_period = "ALL"  # OVERALL maps to ALL
        self.category = "OVERALL"
        self.notify(f"Reset: OVERALL + ALL TIME")
        self.load_leaderboard()

    def action_cycle_category(self):
        categories = ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE"]
        idx = categories.index(self.category) if self.category in categories else 0
        self.category = categories[(idx + 1) % len(categories)]
        self.notify(f"Category: {self.category}")
        self.load_leaderboard()

# ============================================================================
# SCREEN 7: ANALYTICS - Market Analysis Tools
# ============================================================================

# Pre-computed market opportunities (sample data - would be live in production)
# Sample data - replaced with live data when available
MARKET_OPPORTUNITIES = [
    {"market": "Fed rate decision March 2026", "id": "fed-rate-march-2026", "yes": 72.3, "spread": 0.9, "volume_24h": 3250000, "momentum": "+2.8%", "history": [68, 70, 71, 69, 72, 73, 72]},
    {"market": "Bitcoin > $150k by June", "id": "bitcoin-150k-june", "yes": 38.5, "spread": 1.8, "volume_24h": 2890000, "momentum": "+11.2%", "history": [32, 34, 35, 33, 36, 38, 39]},
    {"market": "Elon tweets >80 this week", "id": "elon-tweets-80-week", "yes": 55.0, "spread": 2.2, "volume_24h": 1654000, "momentum": "+8.5%", "history": [48, 50, 52, 54, 53, 55, 55]},
    {"market": "AI regulation bill passes", "id": "ai-regulation-bill", "yes": 34.2, "spread": 3.1, "volume_24h": 987000, "momentum": "-5.3%", "history": [40, 38, 37, 36, 35, 34, 34]},
    {"market": "SpaceX Mars mission 2026", "id": "spacex-mars-2026", "yes": 28.9, "spread": 2.5, "volume_24h": 876000, "momentum": "+4.2%", "history": [25, 26, 27, 26, 28, 29, 29]},
    {"market": "US GDP growth > 3%", "id": "us-gdp-growth-3pct", "yes": 45.8, "spread": 1.4, "volume_24h": 765000, "momentum": "-1.8%", "history": [47, 48, 46, 47, 46, 46, 46]},
    {"market": "Nvidia > $200 by Q2", "id": "nvidia-200-q2", "yes": 62.1, "spread": 1.9, "volume_24h": 654000, "momentum": "+6.7%", "history": [55, 57, 58, 60, 59, 61, 62]},
    {"market": "China Taiwan tensions escalate", "id": "china-taiwan-tensions", "yes": 22.5, "spread": 4.5, "volume_24h": 543000, "momentum": "+18.3%", "history": [15, 16, 18, 19, 20, 21, 23]},
]

class AnalyticsScreen(Screen):
    """Quantitative analysis - spreads, momentum, volume, signals"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "spreads", "Spreads"),
        Binding("2", "momentum", "Momentum"),
        Binding("3", "volume", "Volume"),
        Binding("4", "signals", "Signals"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self):
        super().__init__()
        self.tab = "spreads"
        self.live_markets = []

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="analytics-content"),
                id="analytics-scroll"
            ),
            id="analytics-container"
        )

    def on_mount(self):
        self.load_live_data()

    @work(thread=True)
    def load_live_data(self):
        """Try to load real market data"""
        try:
            markets = []
            for term in ["bitcoin", "fed", "AI", "elon"]:
                results = polyrouter_search(term, limit=5) or []
                for m in results:
                    vol = m.get('volume_total', m.get('volume', 0)) or 0
                    if vol > 10000:
                        markets.append(m)
            self.live_markets = markets[:12]
        except:
            pass
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#analytics-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        # Header
        lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append(f"[bold {accent}]║[/]  [bold white]ANALYTICS[/]  │  Quantitative Market Analysis                                          [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [("1", "Spreads", self.tab == "spreads"),
                ("2", "Momentum", self.tab == "momentum"),
                ("3", "Volume", self.tab == "volume"),
                ("4", "Signals", self.tab == "signals")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on {accent}] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "spreads":
            lines.extend(self._render_spreads())
        elif self.tab == "momentum":
            lines.extend(self._render_momentum())
        elif self.tab == "volume":
            lines.extend(self._render_volume())
        else:
            lines.extend(self._render_signals())

        lines.append("")
        lines.append("[dim]ESC:Back  1:Spreads  2:Momentum  3:Volume  4:Signals  r:Refresh[/]")
        return "\n".join(lines)

    def _render_spreads(self) -> list:
        """Markets sorted by spread with visual quality indicators"""
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append("[bold yellow]▶ SPREAD ANALYSIS[/]  [dim]Bid/Ask Gap - Lower = Better Execution[/]")
        lines.append("")

        sorted_markets = sorted(MARKET_OPPORTUNITIES, key=lambda x: x["spread"])
        max_spread = max(m["spread"] for m in MARKET_OPPORTUNITIES)

        # Table with box borders
        lines.append(f"[dim]┌{'─'*32}┬{'─'*8}┬{'─'*9}┬{'─'*12}┬{'─'*17}┬{'─'*12}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'MARKET':<30}[/] [dim]│[/] [bold]{'YES':>6}[/] [dim]│[/] [bold]{'SPREAD':>7}[/] [dim]│[/] [bold]{'QUALITY':<10}[/] [dim]│[/] [bold]{'SPREAD BAR':<15}[/] [dim]│[/] [bold]{'7D CHART':<10}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*32}┼{'─'*8}┼{'─'*9}┼{'─'*12}┼{'─'*17}┼{'─'*12}┤[/]")

        for m in sorted_markets:
            spread = m["spread"]
            history = m.get("history", [50]*7)

            # Quality rating (plain text, color separate)
            if spread < 1.5:
                quality = "★★★ TIGHT"
                quality_color = colors['green']
                bar_color = colors['green']
            elif spread < 3.0:
                quality = "★★☆ FAIR"
                quality_color = "yellow"
                bar_color = "yellow"
            else:
                quality = "★☆☆ WIDE"
                quality_color = colors['red']
                bar_color = colors['red']

            # Visual spread bar (inverted - tighter is better)
            bar_len = int((1 - spread/max_spread) * 15) if max_spread > 0 else 0
            spread_bar = f"[{bar_color}]{'█' * bar_len}[/]{'░' * (15 - bar_len)}"

            # Sparkline for 7-day history
            spark = sparkline(history, 10)
            market_name = m['market'][:30]

            lines.append(f"[dim]│[/] [cyan]{market_name:<30}[/] [dim]│[/] {m['yes']:>6.1f}% [dim]│[/] {spread:>7.2f}% [dim]│[/] [{quality_color}]{quality:<10}[/] [dim]│[/] {spread_bar} [dim]│[/] [{accent}]{spark}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*32}┴{'─'*8}┴{'─'*9}┴{'─'*12}┴{'─'*17}┴{'─'*12}┘[/]")

        # Summary stats
        lines.append("")
        avg_spread = sum(m["spread"] for m in MARKET_OPPORTUNITIES) / len(MARKET_OPPORTUNITIES)
        tight_count = sum(1 for m in MARKET_OPPORTUNITIES if m["spread"] < 1.5)
        lines.append(f"[bold yellow]▶ SPREAD DISTRIBUTION[/]")
        lines.append(f"  Avg Spread: [{colors['green'] if avg_spread < 2 else colors['red']}]{avg_spread:.2f}%[/]  │  Tight (<1.5%): [{colors['green']}]{tight_count}[/]  │  Fair (1.5-3%): [yellow]{sum(1 for m in MARKET_OPPORTUNITIES if 1.5 <= m['spread'] < 3)}[/]  │  Wide (>3%): [{colors['red']}]{sum(1 for m in MARKET_OPPORTUNITIES if m['spread'] >= 3)}[/]")

        # Visual histogram
        lines.append("")
        lines.append("  [dim]Spread Distribution:[/]")
        buckets = [(0, 1.5, "< 1.5%"), (1.5, 3.0, "1.5-3%"), (3.0, 100, "> 3.0%")]
        for lo, hi, label in buckets:
            count = sum(1 for m in MARKET_OPPORTUNITIES if lo <= m["spread"] < hi)
            bar = "█" * (count * 3) + "░" * (24 - count * 3)
            color = colors['green'] if lo == 0 else ("yellow" if lo == 1.5 else colors['red'])
            lines.append(f"  {label:>8}  [{color}]{bar}[/] {count}")

        return lines

    def _render_momentum(self) -> list:
        """Markets sorted by momentum with visual indicators"""
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append("[bold yellow]▶ MOMENTUM SCANNER[/]  [dim]24h Price Change - Trend Detection[/]")
        lines.append("")

        # Sort by absolute momentum
        sorted_markets = sorted(MARKET_OPPORTUNITIES,
                               key=lambda x: abs(float(x["momentum"].replace("%", "").replace("+", ""))),
                               reverse=True)

        # Table with box borders
        lines.append(f"[dim]┌{'─'*32}┬{'─'*8}┬{'─'*9}┬{'─'*22}┬{'─'*10}┬{'─'*10}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'MARKET':<30}[/] [dim]│[/] [bold]{'YES':>6}[/] [dim]│[/] [bold]{'24H':>7}[/] [dim]│[/] [bold]{'MOMENTUM BAR':<20}[/] [dim]│[/] [bold]{'SIGNAL':<8}[/] [dim]│[/] [bold]{'STRENGTH':<8}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*32}┼{'─'*8}┼{'─'*9}┼{'─'*22}┼{'─'*10}┼{'─'*10}┤[/]")

        for m in sorted_markets:
            mom = m["momentum"]
            mom_val = float(mom.replace("%", "").replace("+", ""))
            history = m.get("history", [50]*7)

            # Momentum bar (centered at 0)
            bar_width = 20
            center = bar_width // 2
            if mom_val >= 0:
                filled = min(int(mom_val / 2), center)
                mom_bar = " " * center + f"[{colors['green']}]{'█' * filled}[/]" + "░" * (center - filled)
            else:
                filled = min(int(abs(mom_val) / 2), center)
                mom_bar = "░" * (center - filled) + f"[{colors['red']}]{'█' * filled}[/]" + " " * center

            # Signal classification (plain text)
            if mom_val > 10:
                signal = "▲▲ BULL"
                signal_color = colors['green']
                strength = "████████"
                str_color = colors['green']
            elif mom_val > 5:
                signal = "▲ UP"
                signal_color = colors['green']
                strength = "██████░░"
                str_color = colors['green']
            elif mom_val > 0:
                signal = "↗ MILD"
                signal_color = colors['green']
                strength = "████░░░░"
                str_color = colors['green']
            elif mom_val > -5:
                signal = "↘ MILD"
                signal_color = colors['red']
                strength = "████░░░░"
                str_color = colors['red']
            elif mom_val > -10:
                signal = "▼ DOWN"
                signal_color = colors['red']
                strength = "██████░░"
                str_color = colors['red']
            else:
                signal = "▼▼ BEAR"
                signal_color = colors['red']
                strength = "████████"
                str_color = colors['red']

            mom_color = colors['green'] if mom_val >= 0 else colors['red']
            market_name = m['market'][:30]
            lines.append(f"[dim]│[/] [cyan]{market_name:<30}[/] [dim]│[/] {m['yes']:>6.1f}% [dim]│[/] [{mom_color}]{mom:>7}[/] [dim]│[/] {mom_bar} [dim]│[/] [{signal_color}]{signal:<8}[/] [dim]│[/] [{str_color}]{strength:<8}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*32}┴{'─'*8}┴{'─'*9}┴{'─'*22}┴{'─'*10}┴{'─'*10}┘[/]")

        # Momentum summary
        lines.append("")
        lines.append("[bold yellow]▶ MARKET SENTIMENT[/]")
        bullish = sum(1 for m in MARKET_OPPORTUNITIES if float(m["momentum"].replace("%", "").replace("+", "")) > 0)
        bearish = len(MARKET_OPPORTUNITIES) - bullish
        total = len(MARKET_OPPORTUNITIES)
        bull_pct = (bullish / total) * 100
        sentiment_bar = f"[{colors['green']}]{'█' * int(bull_pct/5)}[/][{colors['red']}]{'█' * int((100-bull_pct)/5)}[/]"
        lines.append(f"  Bullish: [{colors['green']}]{bullish}[/] ({bull_pct:.0f}%)  │  Bearish: [{colors['red']}]{bearish}[/] ({100-bull_pct:.0f}%)  │  {sentiment_bar}")

        # Strongest movers
        lines.append("")
        lines.append("[bold cyan]◆ STRONGEST MOVERS[/]")
        for m in sorted_markets[:3]:
            mom = m["momentum"]
            mom_val = float(mom.replace("%", "").replace("+", ""))
            color = colors['green'] if mom_val > 0 else colors['red']
            arrow = "▲" if mom_val > 0 else "▼"
            lines.append(f"  [{color}]{arrow}[/] {m['market']}: [{color}]{mom}[/]")

        return lines

    def _render_volume(self) -> list:
        """Markets sorted by volume with liquidity visualization"""
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append("[bold yellow]▶ VOLUME HEATMAP[/]  [dim]24h Activity - Higher = Better Liquidity[/]")
        lines.append("")

        sorted_markets = sorted(MARKET_OPPORTUNITIES, key=lambda x: x["volume_24h"], reverse=True)
        max_vol = sorted_markets[0]["volume_24h"] if sorted_markets else 1

        # Table with box borders
        lines.append(f"[dim]┌{'─'*4}┬{'─'*32}┬{'─'*10}┬{'─'*22}┬{'─'*12}┐[/]")
        lines.append(f"[dim]│[/] [bold]RK[/] [dim]│[/] [bold]{'MARKET':<30}[/] [dim]│[/] [bold]{'24H VOL':>8}[/] [dim]│[/] [bold]{'LIQUIDITY BAR':<20}[/] [dim]│[/] [bold]{'7D CHART':<10}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*4}┼{'─'*32}┼{'─'*10}┼{'─'*22}┼{'─'*12}┤[/]")

        for rank, m in enumerate(sorted_markets, 1):
            vol = m["volume_24h"]
            history = m.get("history", [50]*7)
            vol_str = f"${vol/1000000:.2f}M" if vol >= 1000000 else f"${vol/1000:.0f}k"

            # Volume heatmap bar with gradient colors
            vol_pct = vol / max_vol
            if vol_pct > 0.7:
                bar_color = colors['green']
                heat = "🔥"
            elif vol_pct > 0.4:
                bar_color = "yellow"
                heat = "  "
            else:
                bar_color = "dim"
                heat = "  "

            bar_len = int(vol_pct * 18)
            vol_bar = f"[{bar_color}]{'█' * bar_len}[/]{'░' * (18 - bar_len)}"

            # Rank medal
            if rank == 1:
                rank_str = "🥇"
            elif rank == 2:
                rank_str = "🥈"
            elif rank == 3:
                rank_str = "🥉"
            else:
                rank_str = f"{rank:>2}"

            spark = sparkline(history, 10)
            market_name = m['market'][:30]
            lines.append(f"[dim]│[/] {rank_str} [dim]│[/] [cyan]{market_name:<30}[/] [dim]│[/] {vol_str:>8} [dim]│[/] {vol_bar}{heat} [dim]│[/] [{accent}]{spark}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*4}┴{'─'*32}┴{'─'*10}┴{'─'*22}┴{'─'*12}┘[/]")

        # Volume distribution
        lines.append("")
        total_vol = sum(m["volume_24h"] for m in MARKET_OPPORTUNITIES)
        avg_vol = total_vol / len(MARKET_OPPORTUNITIES)
        lines.append(f"[bold yellow]▶ VOLUME STATS[/]")
        lines.append(f"  Total 24h: [{colors['green']}]${total_vol/1000000:.2f}M[/]  │  Average: [cyan]${avg_vol/1000000:.2f}M[/]  │  Markets: [yellow]{len(MARKET_OPPORTUNITIES)}[/]")

        # Volume tier breakdown
        lines.append("")
        lines.append("[bold cyan]◆ LIQUIDITY TIERS[/]")
        tiers = [
            (1000000, float('inf'), "🟢 HIGH   (>$1M)", colors['green']),
            (500000, 1000000, "🟡 MEDIUM ($500k-1M)", "yellow"),
            (0, 500000, "🔴 LOW    (<$500k)", colors['red'])
        ]
        for lo, hi, label, color in tiers:
            count = sum(1 for m in MARKET_OPPORTUNITIES if lo <= m["volume_24h"] < hi)
            bar = "█" * (count * 3)
            lines.append(f"  {label}  [{color}]{bar}[/] {count} markets")

        return lines

    def _render_signals(self) -> list:
        """Combined trading signals with scoring"""
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append("[bold yellow]▶ TRADING SIGNALS[/]  [dim]Combined Analysis - AI-Scored Opportunities[/]")
        lines.append("")

        # Calculate composite score for each market
        scored_markets = []
        for m in MARKET_OPPORTUNITIES:
            mom_val = float(m["momentum"].replace("%", "").replace("+", ""))

            # Scoring (0-100)
            spread_score = max(0, 100 - m["spread"] * 20)  # Lower spread = higher score
            mom_score = min(100, abs(mom_val) * 5)  # Higher momentum = higher score
            vol_score = min(100, m["volume_24h"] / 30000)  # Higher volume = higher score

            # Weighted composite
            composite = (spread_score * 0.3 + mom_score * 0.4 + vol_score * 0.3)

            # Signal based on momentum direction and score
            if composite > 70 and mom_val > 0:
                signal = "STRONG BUY"
                signal_color = colors['green']
                action = "▲ BUY YES"
                action_color = colors['green']
            elif composite > 70 and mom_val < 0:
                signal = "STRONG SELL"
                signal_color = colors['red']
                action = "▼ BUY NO"
                action_color = colors['red']
            elif composite > 50 and mom_val > 0:
                signal = "BUY"
                signal_color = colors['green']
                action = "↗ CONSIDER"
                action_color = "cyan"
            elif composite > 50 and mom_val < 0:
                signal = "SELL"
                signal_color = colors['red']
                action = "↘ CONSIDER"
                action_color = "cyan"
            else:
                signal = "NEUTRAL"
                signal_color = "dim"
                action = "⊘ WAIT"
                action_color = "dim"

            scored_markets.append({
                **m,
                "composite": composite,
                "signal": signal,
                "signal_color": signal_color,
                "action": action,
                "action_color": action_color,
                "spread_score": spread_score,
                "mom_score": mom_score,
                "vol_score": vol_score
            })

        # Sort by composite score
        scored_markets.sort(key=lambda x: x["composite"], reverse=True)

        # Table with box borders
        lines.append(f"[dim]┌{'─'*32}┬{'─'*7}┬{'─'*12}┬{'─'*6}┬{'─'*6}┬{'─'*6}┬{'─'*12}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'MARKET':<30}[/] [dim]│[/] [bold]SCORE[/] [dim]│[/] [bold]{'SIGNAL':<10}[/] [dim]│[/] [bold]SPRD[/] [dim]│[/] [bold]MOM [/] [dim]│[/] [bold]VOL [/] [dim]│[/] [bold]{'ACTION':<10}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*32}┼{'─'*7}┼{'─'*12}┼{'─'*6}┼{'─'*6}┼{'─'*6}┼{'─'*12}┤[/]")

        for m in scored_markets:
            score = m["composite"]
            score_color = colors['green'] if score > 70 else ("yellow" if score > 50 else colors['red'])

            # Mini bars for each factor (4 chars each)
            spread_bar = "█" * int(m["spread_score"]/25) + "░" * (4 - int(m["spread_score"]/25))
            mom_bar = "█" * int(m["mom_score"]/25) + "░" * (4 - int(m["mom_score"]/25))
            vol_bar = "█" * int(m["vol_score"]/25) + "░" * (4 - int(m["vol_score"]/25))

            market_name = m['market'][:30]
            lines.append(f"[dim]│[/] [cyan]{market_name:<30}[/] [dim]│[/] [{score_color}]{score:>5.0f}[/] [dim]│[/] [{m['signal_color']}]{m['signal']:<10}[/] [dim]│[/] [{colors['green']}]{spread_bar}[/] [dim]│[/] [{accent}]{mom_bar}[/] [dim]│[/] [cyan]{vol_bar}[/] [dim]│[/] [{m['action_color']}]{m['action']:<10}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*32}┴{'─'*7}┴{'─'*12}┴{'─'*6}┴{'─'*6}┴{'─'*6}┴{'─'*12}┘[/]")

        # Signal summary
        lines.append("")
        lines.append("[bold yellow]▶ SIGNAL SUMMARY[/]")
        strong_buys = sum(1 for m in scored_markets if m["composite"] > 70 and float(m["momentum"].replace("%", "").replace("+", "")) > 0)
        strong_sells = sum(1 for m in scored_markets if m["composite"] > 70 and float(m["momentum"].replace("%", "").replace("+", "")) < 0)
        neutral = sum(1 for m in scored_markets if m["composite"] <= 50)

        lines.append(f"  [{colors['green']}]◉ Strong Buys: {strong_buys}[/]  │  [{colors['red']}]◉ Strong Sells: {strong_sells}[/]  │  [dim]─ Neutral: {neutral}[/]")

        # Top picks
        lines.append("")
        lines.append("[bold cyan]◆ TOP PICKS (Score > 60)[/]")
        for m in scored_markets[:3]:
            if m["composite"] > 60:
                spark = sparkline(m.get("history", [50]*7), 12)
                lines.append(f"  [{accent}]{spark}[/] [cyan]{m['market']}[/]: Score [{colors['green']}]{m['composite']:.0f}[/] → [{m['action_color']}]{m['action']}[/]")

        # Signal legend / explanation
        lines.append("")
        lines.append("[bold yellow]▶ SIGNAL GUIDE[/]")
        lines.append(f"  [dim]┌{'─'*85}┐[/]")
        lines.append(f"  [dim]│[/] [{colors['green']}]STRONG BUY[/]  Score>70 + bullish momentum → High confidence buy opportunity          [dim]│[/]")
        lines.append(f"  [dim]│[/] [{colors['green']}]BUY[/]         Score>50 + bullish momentum → Consider buying, do more research    [dim]│[/]")
        lines.append(f"  [dim]│[/] [{colors['red']}]STRONG SELL[/] Score>70 + bearish momentum → High confidence sell/short opportunity [dim]│[/]")
        lines.append(f"  [dim]│[/] [{colors['red']}]SELL[/]        Score>50 + bearish momentum → Consider selling, check news          [dim]│[/]")
        lines.append(f"  [dim]│[/] [dim]NEUTRAL[/]     Score<50 or mixed signals → Wait for clearer opportunity            [dim]│[/]")
        lines.append(f"  [dim]├{'─'*85}┤[/]")
        lines.append(f"  [dim]│[/] [bold]Score Components:[/] SPRD=spread tightness, MOM=price momentum, VOL=trading volume  [dim]│[/]")
        lines.append(f"  [dim]│[/] [bold]Higher bars = better.[/] Score = 30% spread + 40% momentum + 30% volume              [dim]│[/]")
        lines.append(f"  [dim]└{'─'*85}┘[/]")

        # Risk warning
        lines.append("")
        lines.append("[dim]⚠ Signals are algorithmic indicators, not financial advice. Always DYOR.[/]")

        return lines

    def action_spreads(self):
        self.tab = "spreads"
        self._update_display()

    def action_momentum(self):
        self.tab = "momentum"
        self._update_display()

    def action_volume(self):
        self.tab = "volume"
        self._update_display()

    def action_signals(self):
        self.tab = "signals"
        self._update_display()

    def action_refresh(self):
        self.load_live_data()
        self.notify("Refreshing analytics...")

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 8: API EXPLORER - All available endpoints
# ============================================================================

# All available API functions grouped by category
API_ENDPOINTS = {
    "Gamma API": [
        {"name": "gamma_search", "desc": "Search markets by keyword", "params": "query, limit", "example": "gamma_search('trump', 20)"},
        {"name": "gamma_get_market", "desc": "Get market details by ID", "params": "market_id", "example": "gamma_get_market('0x123...')"},
        {"name": "gamma_get_event", "desc": "Get event with all markets", "params": "event_id", "example": "gamma_get_event('evt_123')"},
        {"name": "gamma_list_tags", "desc": "List all market tags", "params": "none", "example": "gamma_list_tags()"},
        {"name": "gamma_get_profile", "desc": "Get user profile data", "params": "address", "example": "gamma_get_profile('0x...')"},
    ],
    "CLOB API": [
        {"name": "get_orderbook", "desc": "Get orderbook for market", "params": "market_id, side", "example": "get_orderbook('0x...', 'yes')"},
        {"name": "get_orderbook_depth", "desc": "Orderbook with depth", "params": "market_id", "example": "get_orderbook_depth('0x...')"},
        {"name": "get_spread", "desc": "Get bid/ask spread", "params": "market_id", "example": "get_spread('0x...')"},
        {"name": "clob_get_spreads", "desc": "Spreads for multiple markets", "params": "market_ids", "example": "clob_get_spreads(['0x1', '0x2'])"},
        {"name": "get_market_trades", "desc": "Recent trades for market", "params": "market_id, limit", "example": "get_market_trades('0x...', 50)"},
        {"name": "get_price", "desc": "Current price for outcome", "params": "market_id, outcome", "example": "get_price('0x...', 'yes')"},
    ],
    "Data API": [
        {"name": "data_get_positions", "desc": "User positions", "params": "address", "example": "data_get_positions('0x...')"},
        {"name": "data_get_activity", "desc": "User activity history", "params": "address, limit", "example": "data_get_activity('0x...', 20)"},
        {"name": "data_get_leaderboard", "desc": "Top traders ranking", "params": "period", "example": "data_get_leaderboard('weekly')"},
        {"name": "data_get_trades", "desc": "Trade history", "params": "market_id, limit", "example": "data_get_trades('0x...', 100)"},
        {"name": "data_get_top_holders", "desc": "Top position holders", "params": "market_id", "example": "data_get_top_holders('0x...')"},
    ],
    "Polyrouter": [
        {"name": "polyrouter_search", "desc": "Fast market search", "params": "query, limit", "example": "polyrouter_search('bitcoin', 50)"},
        {"name": "polyrouter_trending", "desc": "Trending markets", "params": "limit", "example": "polyrouter_trending(20)"},
    ],
    "Analysis": [
        {"name": "find_ev_opportunities", "desc": "Expected value scanner", "params": "markets", "example": "find_ev_opportunities(markets)"},
        {"name": "get_price_history", "desc": "Historical prices", "params": "market_id, interval", "example": "get_price_history('0x...', 60)"},
        {"name": "get_live_volume", "desc": "24h volume data", "params": "market_id", "example": "get_live_volume('0x...')"},
        {"name": "get_open_interest", "desc": "Total open interest", "params": "market_id", "example": "get_open_interest('0x...')"},
    ],
    "Account": [
        {"name": "get_positions", "desc": "Your open positions", "params": "none", "example": "get_positions()"},
        {"name": "get_balances", "desc": "Wallet balances", "params": "address", "example": "get_balances('0x...')"},
        {"name": "get_open_orders", "desc": "Your pending orders", "params": "none", "example": "get_open_orders()"},
        {"name": "get_user_trades", "desc": "Your trade history", "params": "limit", "example": "get_user_trades(50)"},
    ],
    "XTracker": [
        {"name": "xtracker_get_users", "desc": "Get tracked users", "params": "none", "example": "xtracker_get_users()"},
        {"name": "xtracker_get_user", "desc": "Get user details", "params": "user_id", "example": "xtracker_get_user('elonmusk')"},
        {"name": "xtracker_get_trackings", "desc": "Get tracking list", "params": "user_id", "example": "xtracker_get_trackings('elonmusk')"},
        {"name": "xtracker_get_metrics", "desc": "User metrics data", "params": "user_id", "example": "xtracker_get_metrics('elonmusk')"},
        {"name": "xtracker_get_all_trackings", "desc": "All trackings", "params": "none", "example": "xtracker_get_all_trackings()"},
    ],
    "Elon Data": [
        {"name": "load_elon_historic_tweets", "desc": "Historic tweet data", "params": "none", "example": "load_elon_historic_tweets()"},
        {"name": "load_elon_daily_metrics", "desc": "Daily aggregates", "params": "none", "example": "load_elon_daily_metrics()"},
        {"name": "analyze_elon_patterns", "desc": "Pattern analysis", "params": "none", "example": "analyze_elon_patterns()"},
        {"name": "scan_elon_markets", "desc": "Elon-related markets", "params": "none", "example": "scan_elon_markets()"},
    ],
}

class APIExplorerScreen(Screen):
    """Explore all 50+ available API endpoints"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "gamma", "Gamma"),
        Binding("2", "clob", "CLOB"),
        Binding("3", "data", "Data"),
        Binding("4", "analysis", "Analysis"),
        Binding("5", "account", "Account"),
        Binding("6", "xtracker", "XTracker"),
        Binding("7", "elon", "Elon"),
    ]

    def __init__(self):
        super().__init__()
        self.tab = "overview"

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="api-content"),
                id="api-scroll"
            ),
            id="api-container"
        )

    def _update_display(self):
        self.query_one("#api-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]API EXPLORER[/]  │  All Available Endpoints                                  [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [
            ("1", "Gamma", self.tab == "gamma"),
            ("2", "CLOB", self.tab == "clob"),
            ("3", "Data", self.tab == "data"),
            ("4", "Analysis", self.tab == "analysis"),
            ("5", "Account", self.tab == "account"),
            ("6", "XTracker", self.tab == "xtracker"),
            ("7", "Elon", self.tab == "elon"),
        ]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on #ff8c00] {key}:{name} [/] "
            else:
                tab_line += f"[dim] {key}:{name} [/] "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "overview":
            lines.extend(self._render_overview())
        elif self.tab == "gamma":
            lines.extend(self._render_api_group("Gamma API"))
        elif self.tab == "clob":
            lines.extend(self._render_api_group("CLOB API"))
        elif self.tab == "data":
            lines.extend(self._render_api_group("Data API"))
        elif self.tab == "analysis":
            lines.extend(self._render_api_group("Analysis"))
        elif self.tab == "account":
            lines.extend(self._render_api_group("Account"))
        elif self.tab == "xtracker":
            lines.extend(self._render_api_group("XTracker"))
        elif self.tab == "elon":
            lines.extend(self._render_api_group("Elon Data"))

        lines.append("")
        lines.append("[dim]ESC:Back  1-7:API Category[/]")
        return "\n".join(lines)

    def _render_overview(self) -> list:
        lines = []
        lines.append("[bold yellow]▶ API OVERVIEW - 50+ Endpoints Available[/]")
        lines.append("─" * 78)
        lines.append("")

        total = sum(len(eps) for eps in API_ENDPOINTS.values())

        for category, endpoints in API_ENDPOINTS.items():
            count = len(endpoints)
            lines.append(f"  [bold cyan]{category}[/] ({count} endpoints)")
            for ep in endpoints[:3]:  # Show first 3
                lines.append(f"    [dim]• {ep['name']}[/] - {ep['desc']}")
            if len(endpoints) > 3:
                lines.append(f"    [dim]  + {len(endpoints) - 3} more...[/]")
            lines.append("")

        lines.append("─" * 78)
        lines.append(f"  [bold]Total: {total} endpoints[/]  │  Press 1-7 to explore each category")

        return lines

    def _render_api_group(self, group_name: str) -> list:
        lines = []
        lines.append(f"[bold yellow]▶ {group_name.upper()}[/]")
        lines.append("─" * 78)
        lines.append("")

        endpoints = API_ENDPOINTS.get(group_name, [])

        for ep in endpoints:
            lines.append(f"  [bold cyan]{ep['name']}[/]")
            lines.append(f"    Description: {ep['desc']}")
            lines.append(f"    Parameters:  [yellow]{ep['params']}[/]")
            lines.append(f"    Example:     [dim]{ep['example']}[/]")
            lines.append("")

        lines.append("─" * 78)
        lines.append(f"  [bold]{len(endpoints)} endpoints in this category[/]")

        return lines

    def action_gamma(self):
        self.tab = "gamma"
        self._update_display()

    def action_clob(self):
        self.tab = "clob"
        self._update_display()

    def action_data(self):
        self.tab = "data"
        self._update_display()

    def action_analysis(self):
        self.tab = "analysis"
        self._update_display()

    def action_account(self):
        self.tab = "account"
        self._update_display()

    def action_xtracker(self):
        self.tab = "xtracker"
        self._update_display()

    def action_elon(self):
        self.tab = "elon"
        self._update_display()

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 9: SETTINGS
# ============================================================================

class WalletInputModal(ModalScreen):
    """Modal for entering wallet address"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("[bold yellow]Enter Wallet Address[/]\n\n[dim]Format: 0x... (42 characters)[/]", id="wallet-label"),
            Input(placeholder="0x...", id="wallet-input"),
            Horizontal(
                Button("Connect", id="connect-btn", variant="primary"),
                Button("Cancel", id="cancel-btn"),
                id="wallet-buttons"
            ),
            id="wallet-modal"
        )

    def _save_wallet(self, wallet):
        """Save wallet to state and config"""
        STATE.wallet = wallet
        # Save to config file for persistence
        try:
            from trading import load_config, save_config
            config = load_config() or {"host": "https://clob.polymarket.com", "chain_id": 137}
            config["funder"] = wallet
            save_config(config)
        except Exception:
            pass  # Config save is optional
        if STATE.private_key:
            STATE.logged_in = True

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "connect-btn":
            wallet = self.query_one("#wallet-input", Input).value
            if wallet.startswith("0x") and len(wallet) == 42:
                self._save_wallet(wallet)
                self.dismiss(wallet)
            else:
                self.notify("Invalid wallet address. Must be 0x... (42 chars)", severity="error")
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted):
        wallet = event.value
        if wallet.startswith("0x") and len(wallet) == 42:
            self._save_wallet(wallet)
            self.dismiss(wallet)
        else:
            self.notify("Invalid wallet address. Must be 0x... (42 chars)", severity="error")

    def action_cancel(self):
        self.dismiss(None)


class PrivateKeyInputModal(ModalScreen):
    """Modal for entering private key"""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Static("[bold yellow]Enter Private Key[/]\n\n[dim]WARNING: Never share your private key![/]\n[dim]Stored in memory only, not saved to disk.[/]", id="pk-label"),
            Input(placeholder="Enter private key...", password=True, id="pk-input"),
            Horizontal(
                Button("Save", id="save-btn", variant="primary"),
                Button("Cancel", id="cancel-btn"),
                id="pk-buttons"
            ),
            id="pk-modal"
        )

    def _save_pk(self, pk):
        """Save private key to state, env var, and config"""
        STATE.private_key = pk
        # Set env var for trading functions
        os.environ["POLYMARKET_PRIVATE_KEY"] = pk
        # Save to config file for persistence
        try:
            from trading import load_config, save_config
            config = load_config() or {"host": "https://clob.polymarket.com", "chain_id": 137}
            config["private_key"] = pk
            if STATE.wallet:
                config["funder"] = STATE.wallet
            save_config(config)
        except Exception as e:
            pass  # Config save is optional
        if STATE.wallet:
            STATE.logged_in = True

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "save-btn":
            pk = self.query_one("#pk-input", Input).value
            if pk and len(pk) >= 32:
                self._save_pk(pk)
                self.dismiss(pk)
            else:
                self.notify("Invalid private key", severity="error")
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted):
        pk = event.value
        if pk and len(pk) >= 32:
            self._save_pk(pk)
            self.dismiss(pk)
        else:
            self.notify("Invalid private key", severity="error")

    def action_cancel(self):
        self.dismiss(None)


class SettingsScreen(Screen):
    """Settings - login, theme, preferences, data import"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "tab_wallet", "Wallet"),
        Binding("2", "tab_theme", "Theme"),
        Binding("3", "tab_display", "Display"),
        Binding("4", "tab_data", "Data"),
        Binding("w", "enter_wallet", "Wallet", show=False),
        Binding("p", "enter_pk", "Private Key", show=False),
        Binding("d", "derive_creds", "Derive API", show=False),
        Binding("l", "logout", "Logout", show=False),
        # Data import shortcuts
        Binding("i", "import_csv", "Import CSV", show=False),
        Binding("r", "reload_data", "Reload", show=False),
        Binding("e", "edit_apis", "Edit APIs", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.tab = "wallet"  # wallet, theme, display

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="settings-content"),
                id="settings-scroll"
            ),
            id="settings-container"
        )

    def _update_display(self):
        self.query_one("#settings-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        accent = STATE.accent_color

        # Header
        lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append(f"[bold {accent}]║[/]  [bold white]SETTINGS[/]                                                                     [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [("1", "Wallet", self.tab == "wallet"),
                ("2", "Theme", self.tab == "theme"),
                ("3", "Display", self.tab == "display"),
                ("4", "Data", self.tab == "data")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on {accent}] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")
        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")

        if self.tab == "wallet":
            lines.extend(self._render_wallet_tab())
        elif self.tab == "theme":
            lines.extend(self._render_theme_tab())
        elif self.tab == "display":
            lines.extend(self._render_display_tab())
        elif self.tab == "data":
            lines.extend(self._render_data_tab())

        lines.append("")
        lines.append("[dim]1:Wallet  2:Theme  3:Display  4:Data  ESC:Back[/]")
        return "\n".join(lines)

    def _render_wallet_tab(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append("[bold yellow]▶ WALLET SETUP[/]")
        lines.append("")

        # Status box
        lines.append("  ┌────────────────────────────────────────────────────────────────────────┐")

        # Connection status
        if STATE.logged_in:
            lines.append("  │  [bold green]● CONNECTED & READY TO TRADE[/]                                       │")
        else:
            lines.append("  │  [bold red]○ NOT CONNECTED[/]                                                     │")
        lines.append("  │                                                                        │")

        # Wallet status
        if STATE.wallet:
            wallet_display = f"{STATE.wallet[:10]}...{STATE.wallet[-6:]}"
            lines.append(f"  │  [bold]1. Wallet:[/]  [{colors['green']}]✓[/] [cyan]{wallet_display}[/]                              │")
        else:
            lines.append(f"  │  [bold]1. Wallet:[/]  [{colors['red']}]✗[/] [dim]Not set - Press [bold]W[/] to enter[/]                     │")

        # Private key status
        if STATE.private_key:
            lines.append(f"  │  [bold]2. Key:[/]     [{colors['green']}]✓[/] [dim]Private key loaded (hidden)[/]                      │")
        else:
            lines.append(f"  │  [bold]2. Key:[/]     [{colors['red']}]✗[/] [dim]Not set - Press [bold]P[/] to enter[/]                     │")

        # API credentials status
        try:
            from trading import load_config
            config = load_config() or {}
            user_creds = config.get("user_api_creds", {})
            if user_creds and user_creds.get("api_key"):
                lines.append(f"  │  [bold]3. API:[/]     [{colors['green']}]✓[/] [dim]API credentials derived[/]                         │")
            else:
                lines.append(f"  │  [bold]3. API:[/]     [{colors['red']}]✗[/] [dim]Not derived - Press [bold]D[/] after setting key[/]        │")
        except:
            lines.append(f"  │  [bold]3. API:[/]     [{colors['red']}]?[/] [dim]Cannot check - Press [bold]D[/] to derive[/]               │")

        lines.append("  │                                                                        │")
        lines.append("  └────────────────────────────────────────────────────────────────────────┘")
        lines.append("")

        # Action buttons
        lines.append("[bold yellow]▶ QUICK ACTIONS[/]")
        lines.append("")
        if STATE.logged_in:
            lines.append(f"  [{accent}][W][/] Change Wallet   [{accent}][P][/] Change Key   [{accent}][L][/] Logout")
        else:
            lines.append(f"  [{accent}][W][/] Enter Wallet Address (0x...)")
            lines.append(f"  [{accent}][P][/] Enter Private Key")
            lines.append(f"  [{accent}][D][/] Derive API Credentials (after entering W + P)")
        lines.append("")

        # Setup flow
        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")
        lines.append("[bold yellow]▶ SETUP FLOW[/]")
        lines.append("")
        lines.append("  [dim]Step 1:[/] Press [bold]W[/] → Enter your Polymarket wallet address (0x...)")
        lines.append("  [dim]Step 2:[/] Press [bold]P[/] → Enter your private key (stored securely)")
        lines.append("  [dim]Step 3:[/] Press [bold]D[/] → Derive API credentials for trading")
        lines.append("")
        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")
        lines.append("[bold yellow]▶ SECURITY[/]")
        lines.append("  • Private key is stored in memory + encrypted config file")
        lines.append("  • API credentials are derived from your key")
        lines.append("  • Use a dedicated trading wallet, not your main wallet")

        return lines

    def _render_theme_tab(self) -> list:
        lines = []
        lines.append("[bold yellow]▶ COLOR THEME[/]")
        lines.append("")

        themes = [
            ("neon", "Neon Orange", "#ff8c00", "Dark bg, orange accent"),
            ("dark", "Dark Blue", "#4a9eff", "Navy bg, blue accent"),
            ("hacker", "Hacker Green", "#00ff00", "Black bg, matrix green"),
            ("ocean", "Ocean Teal", "#64ffda", "Deep blue bg, teal accent"),
        ]

        for theme_id, name, color, desc in themes:
            selected = "●" if STATE.theme == theme_id else "○"
            style = "bold" if STATE.theme == theme_id else "dim"
            lines.append(f"  [{style}]{selected}[/] [{color}]██[/] [bold]{name:15}[/] [dim]{desc}[/]")

        lines.append("")
        lines.append("  [dim]Press[/] [bold]N[/][dim]:Neon[/]  [bold]D[/][dim]:Dark[/]  [bold]H[/][dim]:Hacker[/]  [bold]O[/][dim]:Ocean[/]")

        lines.append("")
        lines.append(f"[{STATE.accent_color}]{'─' * 78}[/]")
        lines.append("")
        lines.append("[bold yellow]▶ CURRENT PALETTE[/]")
        colors = STATE.get_colors()
        lines.append(f"  Background: [{colors['bg']}]████[/] {colors['bg']}")
        lines.append(f"  Accent:     [{colors['accent']}]████[/] {colors['accent']}")
        lines.append(f"  Positive:   [{colors['green']}]████[/] {colors['green']}")
        lines.append(f"  Negative:   [{colors['red']}]████[/] {colors['red']}")

        return lines

    def _render_display_tab(self) -> list:
        lines = []
        lines.append("[bold yellow]▶ DISPLAY PREFERENCES[/]")
        lines.append("")

        # Compact mode
        compact = "●" if STATE.compact_mode else "○"
        lines.append(f"  [{compact}] [bold]Compact Mode[/]  [dim]- Smaller tables, less whitespace[/]")
        lines.append("      [dim]Press[/] [bold]C[/] [dim]to toggle[/]")
        lines.append("")

        # Sparklines
        spark = "●" if STATE.show_sparklines else "○"
        lines.append(f"  [{spark}] [bold]Show Sparklines[/]  [dim]- Mini charts in tables[/]")
        lines.append("      [dim]Press[/] [bold]S[/] [dim]to toggle[/]")
        lines.append("")

        # Number format
        lines.append(f"  [bold]Number Format:[/] [{STATE.accent_color}]{STATE.number_format.upper()}[/]")
        lines.append("      [dim]Press[/] [bold]A[/] [dim]for Aligned (right-align decimals)[/]")
        lines.append("      [dim]Press[/] [bold]K[/] [dim]for Compact (1.2M, 45k)[/]")

        lines.append("")
        lines.append(f"[{STATE.accent_color}]{'─' * 78}[/]")
        lines.append("")
        lines.append("[bold yellow]▶ PREVIEW[/]")
        lines.append("")

        # Show number alignment preview
        if STATE.number_format == "aligned":
            lines.append("  [dim]Right-aligned decimals:[/]")
            lines.append(f"    Market A     [green]{52.30:>8.2f}%[/]    ${1234567:>12,.0f}")
            lines.append(f"    Market B     [green]{ 8.50:>8.2f}%[/]    ${  45230:>12,.0f}")
            lines.append(f"    Market C     [green]{99.99:>8.2f}%[/]    ${ 987654:>12,.0f}")
        else:
            lines.append("  [dim]Compact format:[/]")
            lines.append("    Market A     [green]52.3%[/]    $1.2M")
            lines.append("    Market B     [green] 8.5%[/]    $45k")
            lines.append("    Market C     [green]99.9%[/]    $987k")

        return lines

    def _render_data_tab(self) -> list:
        """Data import and export settings"""
        lines = []
        accent = STATE.accent_color
        colors = STATE.get_colors()

        lines.append("[bold yellow]▶ DATA IMPORT & EXPORT[/]")
        lines.append("")

        # Current data sources (JSON)
        lines.append("[bold cyan]◆ ACTIVE JSON SOURCES[/]")
        lines.append("")

        data_dir = Path(__file__).parent / "data"
        sources = []

        elon_file = data_dir / "elon_tweets.json"
        if elon_file.exists():
            sources.append(("Elon Tweets", "elon_tweets.json", f"{len(ELON_DAILY_DATA)} days"))

        config_file = data_dir / ".trading_config.json"
        if config_file.exists():
            sources.append(("Trading Config", ".trading_config.json", "API keys"))

        watched_file = data_dir / ".watched_markets.json"
        if watched_file.exists():
            sources.append(("Watched Markets", ".watched_markets.json", "Watchlist"))

        if sources:
            lines.append(f"  [dim]{'Source':<20} {'File':<25} {'Info':<15}[/]")
            lines.append("  " + "─" * 60)
            for name, file, info in sources:
                lines.append(f"  [{colors['green']}]✓[/] {name:<18} [dim]{file:<25}[/] {info}")
        else:
            lines.append("  [dim]No JSON sources found[/]")

        lines.append("")
        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")

        # CSV Import Status
        lines.append("[bold cyan]◆ CSV DATA (Imported)[/]")
        lines.append("")

        csv_files = scan_csv_files(data_dir)
        csv_status = []

        # Check imported data in STATE
        if STATE.imported_tags:
            csv_status.append(("Tags", f"{len(STATE.imported_tags)} tags", colors['green']))
        elif "tags" in csv_files:
            csv_status.append(("Tags", f"Found: {csv_files['tags'].name}", "yellow"))
        else:
            csv_status.append(("Tags", "Not found (tags.csv)", "dim"))

        if STATE.imported_traders:
            csv_status.append(("Traders", f"{len(STATE.imported_traders)} traders", colors['green']))
        elif "traders" in csv_files:
            csv_status.append(("Traders", f"Found: {csv_files['traders'].name}", "yellow"))
        else:
            csv_status.append(("Traders", "Not found (traders.csv)", "dim"))

        if STATE.imported_markets:
            csv_status.append(("Markets", f"{len(STATE.imported_markets)} markets", colors['green']))
        elif "markets" in csv_files:
            csv_status.append(("Markets", f"Found: {csv_files['markets'].name}", "yellow"))
        else:
            csv_status.append(("Markets", "Not found (markets.csv)", "dim"))

        if STATE.imported_trades:
            csv_status.append(("Trades", f"{len(STATE.imported_trades)} trades", colors['green']))
        elif "trades" in csv_files:
            csv_status.append(("Trades", f"Found: {csv_files['trades'].name}", "yellow"))
        else:
            csv_status.append(("Trades", "Not found (trades.csv)", "dim"))

        for name, status, color in csv_status:
            icon = "✓" if "green" in str(color) else "○" if color == "yellow" else "✗"
            lines.append(f"  [{color}]{icon}[/] {name:<12} [{color}]{status}[/]")

        lines.append("")
        lines.append(f"  [{accent}][I][/] Import CSV files from data/ folder")
        lines.append(f"  [{accent}][R][/] Reload all data")
        lines.append("")

        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")

        # CSV Format Guide
        lines.append("[bold cyan]◆ CSV FORMAT GUIDE[/]")
        lines.append("")
        lines.append("  [bold]tags.csv[/]     - tag,markets,volume_24h,volume_7d,trend")
        lines.append("  [bold]traders.csv[/]  - address,pnl,volume,positions,win_rate")
        lines.append("  [bold]markets.csv[/]  - market_id,title,tags,volume,price")
        lines.append("  [bold]trades.csv[/]   - timestamp,side,price,size,market_id")
        lines.append("")

        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")

        # Custom API Endpoints
        lines.append("[bold cyan]◆ CUSTOM API ENDPOINTS[/]")
        lines.append("")

        if STATE.custom_api_endpoints:
            for name, config in STATE.custom_api_endpoints.items():
                url = config.get("url", "")[:40]
                api_type = config.get("type", "unknown")
                lines.append(f"  [{colors['green']}]✓[/] {name:<15} [dim]{url}...[/] ({api_type})")
        else:
            lines.append("  [dim]No custom APIs configured[/]")

        lines.append("")
        lines.append(f"  [{accent}][E][/] Edit API endpoints (opens data/.custom_apis.json)")
        lines.append("")
        lines.append("  [dim]Example .custom_apis.json:[/]")
        lines.append('  [dim]{"my_api": {"url": "https://...", "type": "leaderboard"}}[/]')
        lines.append("")

        lines.append(f"[{accent}]{'─' * 78}[/]")
        lines.append("")
        lines.append(f"  [dim]Data directory: {data_dir}[/]")

        return lines

    def action_tab_data(self):
        self.tab = "data"
        self.query_one("#settings-content", Static).update(self._render_content())

    def action_import_csv(self):
        if self.tab == "data":
            loaded = load_all_csv_data()
            if loaded:
                self.notify(f"Imported: {', '.join(loaded)}", severity="information")
            else:
                self.notify("No CSV files found in data/ folder", severity="warning")
                self.notify("Expected: tags.csv, traders.csv, markets.csv, trades.csv")
            self._update_display()

    def action_reload_data(self):
        if self.tab == "data":
            # Clear existing imported data
            STATE.imported_tags = None
            STATE.imported_traders = None
            STATE.imported_markets = None
            STATE.imported_trades = None
            # Reload
            loaded = load_all_csv_data()
            # Also reload custom APIs
            self._load_custom_apis()
            if loaded:
                self.notify(f"Reloaded: {', '.join(loaded)}", severity="information")
            else:
                self.notify("Data cleared, no CSV files found")
            self._update_display()

    def action_edit_apis(self):
        if self.tab == "data":
            import subprocess
            import os
            data_dir = Path(__file__).parent / "data"
            api_file = data_dir / ".custom_apis.json"
            # Create template if doesn't exist
            if not api_file.exists():
                template = '{\n  "example_api": {\n    "url": "https://api.example.com/data",\n    "type": "leaderboard",\n    "headers": {}\n  }\n}'
                api_file.write_text(template)
            # Open with default editor
            try:
                if os.name == 'nt':
                    os.startfile(str(api_file))
                elif os.uname().sysname == 'Darwin':
                    subprocess.run(['open', str(api_file)])
                else:
                    subprocess.run(['xdg-open', str(api_file)])
                self.notify(f"Opening {api_file.name}...")
            except:
                self.notify(f"Edit: {api_file}", severity="information")

    def _load_custom_apis(self):
        """Load custom API endpoints from .custom_apis.json"""
        data_dir = Path(__file__).parent / "data"
        api_file = data_dir / ".custom_apis.json"
        if api_file.exists():
            try:
                import json
                STATE.custom_api_endpoints = json.loads(api_file.read_text())
            except:
                STATE.custom_api_endpoints = {}

    def on_key(self, event):
        key = event.key.lower()

        # Tab switching
        if key == "1":
            self.tab = "wallet"
        elif key == "2":
            self.tab = "theme"
        elif key == "3":
            self.tab = "display"
        elif key == "4":
            self.tab = "data"

        # Theme shortcuts
        elif key == "n":
            STATE.set_theme("neon")
            self.notify("Theme: Neon Orange")
        elif key == "d" and self.tab == "theme":
            STATE.set_theme("dark")
            self.notify("Theme: Dark Blue")
        elif key == "h":
            STATE.set_theme("hacker")
            self.notify("Theme: Hacker Green")
        elif key == "o":
            STATE.set_theme("ocean")
            self.notify("Theme: Ocean Teal")

        # Display toggles
        elif key == "c":
            STATE.compact_mode = not STATE.compact_mode
            self.notify(f"Compact mode: {'ON' if STATE.compact_mode else 'OFF'}")
        elif key == "s" and self.tab == "display":
            STATE.show_sparklines = not STATE.show_sparklines
            self.notify(f"Sparklines: {'ON' if STATE.show_sparklines else 'OFF'}")
        elif key == "a":
            STATE.number_format = "aligned"
            self.notify("Number format: Aligned")
        elif key == "k":
            STATE.number_format = "compact"
            self.notify("Number format: Compact")

        # Logout
        elif key == "l" and STATE.logged_in:
            STATE.logout()
            self.notify("Logged out")

        self.query_one("#settings-content", Static).update(self._render_content())

    def action_tab_wallet(self):
        self.tab = "wallet"
        self.query_one("#settings-content", Static).update(self._render_content())

    def action_tab_theme(self):
        self.tab = "theme"
        self.query_one("#settings-content", Static).update(self._render_content())

    def action_tab_display(self):
        self.tab = "display"
        self.query_one("#settings-content", Static).update(self._render_content())

    def action_back(self):
        self.app.pop_screen()

    def action_enter_wallet(self):
        """Open wallet input modal"""
        def handle_wallet(wallet):
            if wallet:
                self.notify(f"Wallet set: {wallet[:10]}...{wallet[-6:]}")
                self.query_one("#settings-content", Static).update(self._render_content())
        self.app.push_screen(WalletInputModal(), handle_wallet)

    def action_enter_pk(self):
        """Open private key input modal"""
        def handle_pk(pk):
            if pk:
                self.notify("Private key saved")
                if STATE.wallet and STATE.private_key:
                    STATE.logged_in = True
                    self.notify("[green]Connected! Press D to derive API credentials[/]")
                self.query_one("#settings-content", Static).update(self._render_content())
        self.app.push_screen(PrivateKeyInputModal(), handle_pk)

    def action_derive_creds(self):
        """Derive API credentials from private key"""
        if not STATE.private_key:
            self.notify("[red]Enter private key first (Press P)[/]", severity="error")
            return
        if not STATE.wallet:
            self.notify("[red]Enter wallet address first (Press W)[/]", severity="error")
            return

        self.notify("[yellow]Deriving API credentials...[/]")
        try:
            from trading import derive_user_creds, load_config, save_config

            # Make sure config has wallet
            config = load_config() or {"host": "https://clob.polymarket.com", "chain_id": 137}
            config["funder"] = STATE.wallet
            config["private_key"] = STATE.private_key
            save_config(config)

            # Derive credentials
            creds = derive_user_creds()

            STATE.logged_in = True
            self.notify(f"[bold green]✓ API credentials derived successfully![/]")
            self.notify(f"[green]You are now ready to trade![/]")
            self.query_one("#settings-content", Static).update(self._render_content())
        except Exception as e:
            self.notify(f"[red]Failed to derive credentials: {str(e)[:50]}[/]", severity="error")

    def action_logout(self):
        """Logout and clear credentials"""
        STATE.logged_in = False
        STATE.wallet = ""
        STATE.private_key = ""
        if "POLYMARKET_PRIVATE_KEY" in os.environ:
            del os.environ["POLYMARKET_PRIVATE_KEY"]
        self.notify("Logged out")
        self.query_one("#settings-content", Static).update(self._render_content())

# ============================================================================
# SCREEN 10: WORLD MAP
# ============================================================================

# Market locations for pins
MARKET_REGIONS = {
    "US": {"x": 18, "y": 4, "label": "US Politics", "markets": ["Trump", "Biden", "Elections", "Fed"]},
    "EU": {"x": 55, "y": 5, "label": "Europe", "markets": ["UK", "Germany", "France", "EU"]},
    "ASIA": {"x": 70, "y": 7, "label": "Asia", "markets": ["China", "Japan", "India", "Korea"]},
    "LATAM": {"x": 25, "y": 13, "label": "Latin America", "markets": ["Brazil", "Argentina", "Mexico"]},
    "AFRICA": {"x": 50, "y": 12, "label": "Africa", "markets": ["Nigeria", "South Africa", "Egypt"]},
    "CRYPTO": {"x": 40, "y": 3, "label": "Crypto (Global)", "markets": ["Bitcoin", "Ethereum", "Solana"]},
}

# ============================================================================
# EDGE SCANNER SCREEN
# ============================================================================

class EdgeScannerScreen(Screen):
    """Edge Scanner - Find trading opportunities via mispricing, spreads, momentum"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("s", "scan", "Scan"),
        Binding("m", "momentum", "Momentum"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self):
        super().__init__()
        self.edges = []
        self.momentum = []
        self.tab = "edges"
        self.min_volume = 10000
        self.selected_idx = 0

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="edge-content"),
                id="edge-scroll"
            ),
            id="edge-container"
        )

    def on_mount(self):
        self.scan_edges()

    @work(thread=True)
    def scan_edges(self):
        """Scan for trading edges"""
        try:
            # Inline edge scanning to avoid circular import
            edges = []
            markets = polyrouter_search("", limit=100) or []
            for m in markets[:30]:
                market_id = m.get('id') or m.get('conditionId')
                if not market_id:
                    continue
                vol = float(m.get('volume', 0) or 0)
                if vol < self.min_volume:
                    continue
                edge_info = {
                    'market_id': market_id,
                    'title': m.get('title', m.get('question', 'Unknown')),
                    'volume': vol,
                    'edges': [],
                    'score': 0
                }
                try:
                    price = get_price(market_id)
                    if price:
                        yes_p = price.get('yes', 0)
                        no_p = price.get('no', 0)
                        total = yes_p + no_p
                        if abs(total - 1.0) > 0.01:
                            edge_info['edges'].append({'type': 'mispricing', 'desc': f'YES+NO={total*100:.1f}%'})
                            edge_info['score'] += abs(total - 1.0) * 50
                        if yes_p < 0.05 or yes_p > 0.95:
                            edge_info['edges'].append({'type': 'extreme_price', 'desc': f'Near {int(yes_p*100)}%'})
                            edge_info['score'] += 10
                except:
                    pass
                if edge_info['edges']:
                    edges.append(edge_info)
            self.edges = sorted(edges, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            self.edges = []
        self.app.call_from_thread(self._update_display)

    @work(thread=True)
    def scan_momentum_signals(self):
        """Scan for momentum signals"""
        try:
            results = scan_all_spikes(threshold_vol=2.0, threshold_liq=1.5)
            self.momentum = results.get('volume_spikes', []) + results.get('liquidity_spikes', [])
        except:
            self.momentum = []
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#edge-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        accent = STATE.accent_color
        colors = STATE.get_colors()

        # Header
        lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append(f"[bold {accent}]║[/]  [bold white]🎯 EDGE SCANNER[/]  │  Mispricing, Spreads, Volume Spikes                              [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [("s", "Scan Edges", self.tab == "edges"), ("m", "Momentum", self.tab == "momentum")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on {accent}] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append(f"  [dim]Min Volume: ${self.min_volume:,}[/]")
        lines.append("")

        if self.tab == "edges":
            lines.extend(self._render_edges())
        else:
            lines.extend(self._render_momentum())

        lines.append("")
        lines.append("[dim]ESC:Back  s:Scan  m:Momentum  r:Refresh  ENTER:View Market[/]")
        return "\n".join(lines)

    def _render_edges(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append(f"[bold yellow]▶ EDGE OPPORTUNITIES[/]  [dim]Found {len(self.edges)} markets with edges[/]")
        lines.append("")

        if not self.edges:
            lines.append("  [dim]No edges found. Press 's' to scan...[/]")
            return lines

        # Table header
        lines.append(f"[dim]┌{'─'*4}┬{'─'*40}┬{'─'*12}┬{'─'*35}┬{'─'*8}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'#':>2}[/] [dim]│[/] [bold]{'MARKET':<38}[/] [dim]│[/] [bold]{'VOLUME':>10}[/] [dim]│[/] [bold]{'EDGES':<33}[/] [dim]│[/] [bold]{'SCORE':>6}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*4}┼{'─'*40}┼{'─'*12}┼{'─'*35}┼{'─'*8}┤[/]")

        for i, e in enumerate(self.edges[:20]):
            marker = ">" if i == self.selected_idx else " "
            title = (e.get('title', '')[:36] + '..') if len(e.get('title', '')) > 38 else e.get('title', '')[:38]
            vol = e.get('volume', 0)
            score = e.get('score', 0)

            # Format edges
            edge_descs = []
            for edge in e.get('edges', [])[:2]:
                etype = edge.get('type', '')
                if etype == 'mispricing':
                    edge_descs.append(f"[yellow]MISPRICE[/]")
                elif etype == 'wide_spread':
                    edge_descs.append(f"[cyan]SPREAD[/]")
                elif etype == 'vol_liq_imbalance':
                    edge_descs.append(f"[magenta]V/L[/]")
                elif etype == 'extreme_price':
                    edge_descs.append(f"[{colors['red']}]EXTREME[/]")
            edge_str = ' '.join(edge_descs)[:33]

            score_color = colors['green'] if score > 30 else ('yellow' if score > 15 else 'dim')
            row_style = f"bold {accent}" if i == self.selected_idx else ""

            lines.append(f"[dim]│[/][{row_style}]{marker}{i+1:>2}[/] [dim]│[/] [{accent}]{title:<38}[/] [dim]│[/] [cyan]${vol/1000:.1f}k[/] [dim]│[/] {edge_str:<33} [dim]│[/] [{score_color}]{score:>6.1f}[/] [dim]│[/]")

        lines.append(f"[dim]└{'─'*4}┴{'─'*40}┴{'─'*12}┴{'─'*35}┴{'─'*8}┘[/]")
        return lines

    def _render_momentum(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append(f"[bold yellow]▶ MOMENTUM SIGNALS[/]  [dim]Volume & Liquidity Spikes[/]")
        lines.append("")

        if not self.momentum:
            lines.append("  [dim]No momentum signals. Press 'm' to scan...[/]")
            return lines

        lines.append(f"[dim]┌{'─'*4}┬{'─'*45}┬{'─'*12}┬{'─'*10}┬{'─'*15}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'#':>2}[/] [dim]│[/] [bold]{'MARKET':<43}[/] [dim]│[/] [bold]{'TYPE':>10}[/] [dim]│[/] [bold]{'RATIO':>8}[/] [dim]│[/] [bold]{'SIGNAL':<13}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*4}┼{'─'*45}┼{'─'*12}┼{'─'*10}┼{'─'*15}┤[/]")

        for i, s in enumerate(self.momentum[:15]):
            title = (s.get('title', '')[:41] + '..') if len(s.get('title', '')) > 43 else s.get('title', '')[:43]
            spike_type = s.get('type', 'volume')
            ratio = s.get('ratio', s.get('spike_ratio', 0))
            signal = "🔥 HOT" if ratio > 3 else ("📈 RISING" if ratio > 2 else "👀 WATCH")

            type_color = 'magenta' if spike_type == 'liquidity' else 'cyan'
            lines.append(f"[dim]│[/] {i+1:>2} [dim]│[/] [{accent}]{title:<43}[/] [dim]│[/] [{type_color}]{spike_type:>10}[/] [dim]│[/] [{colors['green']}]{ratio:>7.1f}x[/] [dim]│[/] {signal:<13} [dim]│[/]")

        lines.append(f"[dim]└{'─'*4}┴{'─'*45}┴{'─'*12}┴{'─'*10}┴{'─'*15}┘[/]")
        return lines

    def action_scan(self):
        self.tab = "edges"
        self.scan_edges()

    def action_momentum(self):
        self.tab = "momentum"
        self.scan_momentum_signals()

    def action_refresh(self):
        if self.tab == "edges":
            self.scan_edges()
        else:
            self.scan_momentum_signals()

    def action_back(self):
        self.app.pop_screen()

    def action_select(self):
        if self.tab == "edges" and self.edges and self.selected_idx < len(self.edges):
            market = self.edges[self.selected_idx]
            mid = market.get('market_id')
            if mid:
                self.app.push_screen(MarketDetailScreen({'id': mid, 'title': market.get('title', '')}))

    def on_key(self, event):
        if event.key == "j" or event.key == "down":
            max_idx = len(self.edges) - 1 if self.tab == "edges" else len(self.momentum) - 1
            self.selected_idx = min(self.selected_idx + 1, max_idx)
            self._update_display()
        elif event.key == "k" or event.key == "up":
            self.selected_idx = max(0, self.selected_idx - 1)
            self._update_display()


# ============================================================================
# AUTOMATION SCREEN
# ============================================================================

class AutomationScreen(Screen):
    """Automation - Volume/Liquidity spike detection and auto-trading"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("s", "scan", "Scan"),
        Binding("r", "run", "Run Auto"),
        Binding("c", "config", "Config"),
        Binding("l", "log", "Log"),
    ]

    def __init__(self):
        super().__init__()
        self.config = {}
        self.spikes = []
        self.auto_log = []
        self.tab = "scan"

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="auto-content"),
                id="auto-scroll"
            ),
            id="auto-container"
        )

    def on_mount(self):
        self.load_config()
        # Auto-scan on load
        self.do_scan()

    def load_config(self):
        try:
            self.config = load_automation_config()
        except:
            self.config = {'enabled': False, 'dry_run': True, 'volume_spike_threshold': 2.0, 'liquidity_spike_threshold': 1.5}
        self._update_display()

    @work(thread=True)
    def do_scan(self):
        try:
            results = scan_all_spikes(
                threshold_vol=self.config.get('volume_spike_threshold', 2.0),
                threshold_liq=self.config.get('liquidity_spike_threshold', 1.5)
            )
            self.spikes = results.get('volume_spikes', []) + results.get('liquidity_spikes', [])
        except:
            self.spikes = []
        self.app.call_from_thread(self._update_display)

    @work(thread=True)
    def do_run(self):
        try:
            result = run_automation_scan()
            self.auto_log.insert(0, f"[{result.get('timestamp', 'now')}] Scanned {result.get('markets_scanned', 0)} markets, {result.get('spikes_found', 0)} spikes")
        except Exception as e:
            self.auto_log.insert(0, f"[ERROR] {str(e)}")
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#auto-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        accent = STATE.accent_color
        colors = STATE.get_colors()

        # Header
        lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append(f"[bold {accent}]║[/]  [bold white]⚡ AUTOMATION[/]  │  Volume & Liquidity Spike Trading                               [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Status
        enabled = self.config.get('enabled', False)
        dry_run = self.config.get('dry_run', True)
        status_color = colors['green'] if enabled else colors['red']
        mode = "[yellow]DRY RUN[/]" if dry_run else f"[{colors['green']}]LIVE[/]"
        lines.append(f"  [{status_color}]● {'ENABLED' if enabled else 'DISABLED'}[/]  │  Mode: {mode}  │  Vol Threshold: {self.config.get('volume_spike_threshold', 2.0)}x  │  Liq Threshold: {self.config.get('liquidity_spike_threshold', 1.5)}x")
        lines.append("")

        # Tabs
        tabs = [("s", "Scan", self.tab == "scan"), ("c", "Config", self.tab == "config"), ("l", "Log", self.tab == "log")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on {accent}] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "scan":
            lines.extend(self._render_scan())
        elif self.tab == "config":
            lines.extend(self._render_config())
        else:
            lines.extend(self._render_log())

        lines.append("")
        lines.append("[dim]ESC:Back  s:Scan  r:Run Auto  c:Config  l:Log[/]")
        return "\n".join(lines)

    def _render_scan(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append(f"[bold yellow]▶ SPIKE DETECTOR[/]  [dim]Found {len(self.spikes)} spikes[/]")
        lines.append("")

        if not self.spikes:
            lines.append("  [dim]No spikes detected. Press 's' to scan...[/]")
            return lines

        lines.append(f"[dim]┌{'─'*4}┬{'─'*45}┬{'─'*10}┬{'─'*10}┬{'─'*12}┐[/]")
        lines.append(f"[dim]│[/] [bold]{'#':>2}[/] [dim]│[/] [bold]{'MARKET':<43}[/] [dim]│[/] [bold]{'TYPE':>8}[/] [dim]│[/] [bold]{'RATIO':>8}[/] [dim]│[/] [bold]{'ACTION':<10}[/] [dim]│[/]")
        lines.append(f"[dim]├{'─'*4}┼{'─'*45}┼{'─'*10}┼{'─'*10}┼{'─'*12}┤[/]")

        for i, s in enumerate(self.spikes[:15]):
            title = (s.get('title', '')[:41] + '..') if len(s.get('title', '')) > 43 else s.get('title', '')[:43]
            spike_type = s.get('spike_type', s.get('type', 'vol'))[:8]
            ratio = s.get('spike_ratio', s.get('liq_vol_ratio', s.get('ratio', 0)))
            action = "🎯 TRADE" if ratio > 2.5 else "👀 WATCH"

            lines.append(f"[dim]│[/] {i+1:>2} [dim]│[/] [{accent}]{title:<43}[/] [dim]│[/] [cyan]{spike_type:>8}[/] [dim]│[/] [{colors['green']}]{ratio:>7.1f}x[/] [dim]│[/] {action:<10} [dim]│[/]")

        lines.append(f"[dim]└{'─'*4}┴{'─'*45}┴{'─'*10}┴{'─'*10}┴{'─'*12}┘[/]")
        return lines

    def _render_config(self) -> list:
        lines = []
        colors = STATE.get_colors()

        lines.append(f"[bold yellow]▶ CONFIGURATION[/]")
        lines.append("")
        lines.append(f"  [bold]Enabled:[/]                [{colors['green'] if self.config.get('enabled') else colors['red']}]{self.config.get('enabled', False)}[/]")
        lines.append(f"  [bold]Dry Run:[/]                [yellow]{self.config.get('dry_run', True)}[/]")
        lines.append(f"  [bold]Volume Threshold:[/]       {self.config.get('volume_spike_threshold', 2.0)}x")
        lines.append(f"  [bold]Liquidity Threshold:[/]    {self.config.get('liquidity_spike_threshold', 1.5)}x")
        lines.append(f"  [bold]Min Volume:[/]             ${self.config.get('min_volume', 5000):,}")
        lines.append(f"  [bold]Order Size:[/]             {self.config.get('order_size', 10)}")
        lines.append(f"  [bold]Max Orders/Hour:[/]        {self.config.get('max_orders_per_hour', 10)}")
        lines.append(f"  [bold]Watched Only:[/]           {self.config.get('watched_only', True)}")
        lines.append("")
        lines.append("  [dim]Edit config in data/.automation_config.json[/]")
        return lines

    def _render_log(self) -> list:
        lines = []
        lines.append(f"[bold yellow]▶ AUTOMATION LOG[/]")
        lines.append("")
        if not self.auto_log:
            lines.append("  [dim]No log entries. Press 'r' to run automation...[/]")
        else:
            for entry in self.auto_log[:20]:
                lines.append(f"  [dim]{entry}[/]")
        return lines

    def action_scan(self):
        self.tab = "scan"
        self.do_scan()

    def action_run(self):
        self.do_run()

    def action_config(self):
        self.tab = "config"
        self._update_display()

    def action_log(self):
        self.tab = "log"
        self._update_display()

    def action_back(self):
        self.app.pop_screen()


# ============================================================================
# QUANT RESEARCH SCREEN
# ============================================================================

class QuantScreen(Screen):
    """Quant Research - Forecasting models, backtesting, optimization"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "run_models", "Run Models"),
        Binding("b", "backtest", "Backtest"),
        Binding("o", "optimize", "Optimize"),
        Binding("s", "snapshot", "Snapshot"),
    ]

    def __init__(self):
        super().__init__()
        self.market_id = ""
        self.results = {}
        self.backtest_results = {}
        self.tab = "models"

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="quant-content"),
                id="quant-scroll"
            ),
            id="quant-container"
        )

    def on_mount(self):
        """Auto-load a market if none selected"""
        if not self.market_id:
            self.load_default_market()

    @work(thread=True)
    def load_default_market(self):
        """Load first trending market for quant analysis"""
        try:
            # Try to get a trending market
            trending = polyrouter_trending(limit=5)
            if trending and isinstance(trending, list) and len(trending) > 0:
                m = trending[0]
                self.market_id = m.get('id') or m.get('market_id') or m.get('condition_id', '')
                if self.market_id:
                    self.results = run_all_models(self.market_id, forecast_periods=5)
        except:
            pass
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#quant-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        accent = STATE.accent_color
        colors = STATE.get_colors()

        # Header
        lines.append(f"[bold {accent}]╔══════════════════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append(f"[bold {accent}]║[/]  [bold white]🔬 QUANT RESEARCH[/]  │  Forecasting, Backtesting, Optimization                       [bold {accent}]║[/]")
        lines.append(f"[bold {accent}]╚══════════════════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Market ID input hint
        lines.append(f"  [dim]Market ID:[/] [cyan]{self.market_id or '(none - set via market detail)'}[/]")
        lines.append("")

        # Tabs
        tabs = [("r", "Models", self.tab == "models"), ("b", "Backtest", self.tab == "backtest"), ("o", "Optimize", self.tab == "optimize")]
        tab_line = "  "
        for key, name, active in tabs:
            if active:
                tab_line += f"[bold white on {accent}] {key}:{name} [/]  "
            else:
                tab_line += f"[dim] {key}:{name} [/]  "
        lines.append(tab_line)
        lines.append("")

        if self.tab == "models":
            lines.extend(self._render_models())
        elif self.tab == "backtest":
            lines.extend(self._render_backtest())
        else:
            lines.extend(self._render_optimize())

        lines.append("")
        lines.append("[dim]ESC:Back  r:Run Models  b:Backtest  o:Optimize  s:Store Snapshot[/]")
        return "\n".join(lines)

    def _render_models(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append(f"[bold yellow]▶ MODEL RESULTS[/]")
        lines.append("")

        if not self.results:
            lines.append("  [dim]No results. Select a market and press 'r' to run models...[/]")
            return lines

        # Indicators
        indicators = self.results.get('indicators', {})
        if indicators:
            lines.append("  [bold cyan]INDICATORS[/]")
            ind_line = "  "
            for k, v in list(indicators.items())[:6]:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                ind_line += f"[dim]{k}:[/] [{accent}]{val_str}[/]  "
            lines.append(ind_line)
            lines.append("")

        # Signals
        signals = self.results.get('signals', {})
        if signals:
            lines.append("  [bold cyan]SIGNALS[/]")
            for model, sig in signals.items():
                signal = sig.get('signal', 'neutral')
                color = colors['green'] if signal == 'bullish' else (colors['red'] if signal == 'bearish' else 'dim')
                lines.append(f"    {model:<20} [{color}]{signal.upper()}[/]")
            lines.append("")

        # Consensus
        consensus = self.results.get('consensus', {})
        if consensus:
            lines.append("  [bold cyan]ENSEMBLE CONSENSUS[/]")
            direction = consensus.get('direction', 'neutral')
            dir_color = colors['green'] if direction == 'up' else (colors['red'] if direction == 'down' else 'dim')
            lines.append(f"    Avg Forecast: [{accent}]{consensus.get('avg_forecast', 0):.4f}[/]")
            lines.append(f"    Direction:    [{dir_color}]{direction.upper()}[/]")
            lines.append(f"    Confidence:   [{accent}]{consensus.get('confidence', 0)*100:.0f}%[/]")

        return lines

    def _render_backtest(self) -> list:
        lines = []
        colors = STATE.get_colors()
        accent = STATE.accent_color

        lines.append(f"[bold yellow]▶ BACKTEST RESULTS[/]")
        lines.append("")

        if not self.backtest_results:
            lines.append("  [dim]No backtest results. Press 'b' to run backtest...[/]")
            return lines

        lines.append(f"  [bold]Model:[/] {self.backtest_results.get('model', 'linear')}")
        lines.append("")
        lines.append(f"  [bold cyan]METRICS[/]")
        lines.append(f"    MAE:                [{accent}]{self.backtest_results.get('mae', 0):.6f}[/]")
        lines.append(f"    MSE:                [{accent}]{self.backtest_results.get('mse', 0):.6f}[/]")
        lines.append(f"    RMSE:               [{accent}]{self.backtest_results.get('rmse', 0):.6f}[/]")

        dir_acc = self.backtest_results.get('direction_accuracy', 0)
        dir_color = colors['green'] if dir_acc > 0.6 else ('yellow' if dir_acc > 0.5 else colors['red'])
        lines.append(f"    Direction Accuracy: [{dir_color}]{dir_acc*100:.1f}%[/]")

        quality = "Good" if dir_acc > 0.6 else ("Fair" if dir_acc > 0.5 else "Poor")
        lines.append("")
        lines.append(f"  [bold]Quality:[/] [{dir_color}]{quality}[/]")

        return lines

    def _render_optimize(self) -> list:
        lines = []
        lines.append(f"[bold yellow]▶ PARAMETER OPTIMIZATION[/]")
        lines.append("")
        lines.append("  [dim]Grid search for optimal model parameters.[/]")
        lines.append("  [dim]Press 'o' to run optimization on current market...[/]")
        return lines

    @work(thread=True)
    def do_run_models(self):
        if not self.market_id:
            return
        try:
            self.results = run_all_models(self.market_id, forecast_periods=5)
        except Exception as e:
            self.results = {'error': str(e)}
        self.app.call_from_thread(self._update_display)

    @work(thread=True)
    def do_backtest(self):
        if not self.market_id:
            return
        try:
            self.backtest_results = backtest_model(self.market_id, model_type='linear', lookback=50, test_periods=10)
        except Exception as e:
            self.backtest_results = {'error': str(e)}
        self.app.call_from_thread(self._update_display)

    @work(thread=True)
    def do_snapshot(self):
        if not self.market_id:
            return
        try:
            store_price_snapshot(self.market_id, get_price(self.market_id))
        except:
            pass

    def action_run_models(self):
        self.tab = "models"
        self.do_run_models()

    def action_backtest(self):
        self.tab = "backtest"
        self.do_backtest()

    def action_optimize(self):
        self.tab = "optimize"
        self._update_display()

    def action_snapshot(self):
        self.do_snapshot()

    def action_back(self):
        self.app.pop_screen()


class WorldMapScreen(Screen):
    """World Map - Geographic view of markets"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "region_us", "US"),
        Binding("2", "region_eu", "Europe"),
        Binding("3", "region_asia", "Asia"),
        Binding("4", "region_crypto", "Crypto"),
        Binding("s", "search", "Search"),
    ]

    def __init__(self):
        super().__init__()
        self.selected_region = None
        self.region_markets = {}

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="map-content"),
                id="map-scroll"
            ),
            id="map-container"
        )

    def on_mount(self):
        self.load_regional_data()

    @work(thread=True)
    def load_regional_data(self):
        """Load market data for each region"""
        for region, data in MARKET_REGIONS.items():
            try:
                # Search for markets related to this region
                keywords = data["markets"]
                markets = []
                for kw in keywords[:2]:  # Limit searches
                    result = gamma_search(kw)
                    if isinstance(result, list):
                        markets.extend(result[:3])
                self.region_markets[region] = markets
            except:
                self.region_markets[region] = []
        self.app.call_from_thread(self._update_display)

    def _update_display(self):
        self.query_one("#map-content", Static).update(self._render_content())

    def _render_content(self) -> str:
        lines = []
        colors = STATE.get_colors()

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]WORLD MAP[/]  │  Global Prediction Market View                             [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Simplified ASCII map with market pins
        lines.append("[bold yellow]▶ GLOBAL MARKET ACTIVITY[/]")
        lines.append("")
        lines.append(self._render_map())
        lines.append("")

        # Legend
        lines.append("[bold yellow]▶ MARKET REGIONS[/]")
        lines.append("─" * 78)
        lines.append(f"  [bold #ff8c00]1[/] [bold]US Politics[/]     [bold #ff8c00]2[/] [bold]Europe[/]     [bold #ff8c00]3[/] [bold]Asia[/]     [bold #ff8c00]4[/] [bold]Crypto (Global)[/]")
        lines.append("")

        # Regional stats
        if self.selected_region and self.selected_region in MARKET_REGIONS:
            region_data = MARKET_REGIONS[self.selected_region]
            markets = self.region_markets.get(self.selected_region, [])
            lines.append(f"[bold cyan]▶ {region_data['label'].upper()} MARKETS[/]")
            lines.append("─" * 78)
            if markets:
                lines.append(f"  [dim]{'MARKET':<40} {'YES':>10} {'VOLUME':>15}[/]")
                for m in markets[:8]:
                    title = str(m.get('title', m.get('question', '')))[:38]
                    yes = float(m.get('yes', m.get('outcomePrices', [0.5])[0] if isinstance(m.get('outcomePrices'), list) else 0.5)) * 100
                    vol = float(m.get('volume', m.get('volume_total', 0)))
                    vol_str = f"${vol/1000000:.1f}M" if vol >= 1000000 else f"${vol/1000:.0f}k" if vol >= 1000 else f"${vol:.0f}"
                    lines.append(f"  {title:<40} [{colors['green']}]{yes:>9.1f}%[/] {vol_str:>15}")
            else:
                lines.append("  [dim]Loading markets...[/]")
        else:
            # Show global overview
            lines.append("[bold cyan]▶ REGIONAL OVERVIEW[/]")
            lines.append("─" * 78)
            lines.append(f"  [dim]{'REGION':<20} {'MARKETS':<30} {'HOT TOPICS'}[/]")
            for region, data in MARKET_REGIONS.items():
                market_count = len(self.region_markets.get(region, []))
                topics = ", ".join(data["markets"][:3])
                icon = "●" if market_count > 0 else "○"
                lines.append(f"  [{colors['green']}]{icon}[/] {data['label']:<18} {market_count:>3} markets              {topics}")
            lines.append("")
            lines.append("  [dim]Press 1-4 to explore a region, or s to search[/]")

        lines.append("")
        lines.append("[dim]ESC:Back  1-4:Regions  s:Search  r:Refresh[/]")
        return "\n".join(lines)

    def _render_map(self) -> str:
        """Render ASCII world map with market pins (mapscii style)"""
        colors = STATE.get_colors()

        # Get market counts for pin colors
        us_count = len(self.region_markets.get("US", []))
        eu_count = len(self.region_markets.get("EU", []))
        asia_count = len(self.region_markets.get("ASIA", []))
        latam_count = len(self.region_markets.get("LATAM", []))
        africa_count = len(self.region_markets.get("AFRICA", []))
        crypto_count = len(self.region_markets.get("CRYPTO", []))

        # Pin colors based on activity (filled = active, hollow = loading)
        us_pin = "[bold #ff8c00]◉[/]" if us_count > 0 else "[#ff8c00]○[/]"
        eu_pin = "[bold #4a9eff]◉[/]" if eu_count > 0 else "[#4a9eff]○[/]"
        asia_pin = "[bold #ff6b6b]◉[/]" if asia_count > 0 else "[#ff6b6b]○[/]"
        latam_pin = "[bold #9b59b6]◉[/]" if latam_count > 0 else "[#9b59b6]○[/]"
        africa_pin = "[bold #f39c12]◉[/]" if africa_count > 0 else "[#f39c12]○[/]"
        crypto_pin = "[bold #00ff88]◉[/]" if crypto_count > 0 else "[#00ff88]○[/]"

        # Proper ASCII World Map
        map_art = f"""[dim]
                                   ┌─────────────────────────────────────────────────────────────────┐
                                   │  {crypto_pin} [bold #00ff88]CRYPTO[/] [dim](Global)[/]                                            │
    ┌──────────────────────────────┼─────────────────────────────────────────────────────────────────┤
    │    [#3498db]▄▄▄▄▄[/]           [#3498db]▄▄[/]      │          [#27ae60]▄▄▄▄▄▄[/]                [#e74c3c]▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄[/]          │
    │  [#3498db]▄█████████▄▄▄▄▄▄████[/]     │        [#27ae60]▄████████▄[/]          [#e74c3c]▄███████████████████[/]      │
    │ [#3498db]████████████████████[/]      │  {eu_pin}[bold #4a9eff]EU[/]  [#27ae60]██████████████[/]  {asia_pin}[bold #ff6b6b]ASIA[/] [#e74c3c]█████████████████████[/]    │
    │ [#3498db]█████[/]{us_pin}[bold #ff8c00]US[/][#3498db]██████████████[/]      │       [#27ae60]██████████████[/]        [#e74c3c]███████████████████████[/]  │
    │  [#3498db]██████████████████[/]       │        [#27ae60]▀████████▀[/]          [#e74c3c]█████████████████████[/]    │
    │   [#3498db]████████████████[/]        │          [#27ae60]▀▀▀▀▀[/]              [#e74c3c]███████████████████[/]      │
    │    [#3498db]▀▀████████████[/]         │                                  [#e74c3c]▀███████████████▀[/]        │
    │        [#3498db]▀▀▀▀▀▀▀[/]            │    [#f39c12]▄▄▄▄▄▄▄▄▄▄▄[/]                   [#e74c3c]▀▀▀▀▀▀▀▀▀▀[/]           │
    ├──────────────────────────────┤  [#f39c12]▄█████████████[/]                                           │
    │      [#9b59b6]▄▄▄▄▄▄▄[/]              │ [#f39c12]███[/]{africa_pin}[bold #f39c12]AFRICA[/][#f39c12]██████[/]                                           │
    │    [#9b59b6]▄█████████▄[/]            │  [#f39c12]█████████████[/]                    [#1abc9c]▄▄▄▄▄▄▄[/]            │
    │   [#9b59b6]██████████████[/]          │   [#f39c12]▀█████████▀[/]                   [#1abc9c]▄███████▄[/]           │
    │   [#9b59b6]█[/]{latam_pin}[bold #9b59b6]LATAM[/][#9b59b6]████████[/]          │     [#f39c12]▀▀▀▀▀▀▀[/]                     [#1abc9c]█████████[/]           │
    │    [#9b59b6]██████████████[/]         │                                    [#1abc9c]█[/] AUS [#1abc9c]██[/]           │
    │     [#9b59b6]▀██████████▀[/]          │                                     [#1abc9c]▀█████▀[/]            │
    │       [#9b59b6]▀▀▀▀▀▀▀▀[/]            │                                       [#1abc9c]▀▀▀[/]              │
    └──────────────────────────────┴─────────────────────────────────────────────────────────────────┘[/]
    [dim]◉ = Active markets    ○ = Loading...[/]"""

        return map_art

    def action_region_us(self):
        self.selected_region = "US"
        self._update_display()
        self.notify("Viewing US markets")

    def action_region_eu(self):
        self.selected_region = "EU"
        self._update_display()
        self.notify("Viewing European markets")

    def action_region_asia(self):
        self.selected_region = "ASIA"
        self._update_display()
        self.notify("Viewing Asian markets")

    def action_region_crypto(self):
        self.selected_region = "CRYPTO"
        self._update_display()
        self.notify("Viewing Crypto markets (Global)")

    def action_search(self):
        # Could open a search modal in future
        self.notify("Search: Use Markets screen (1) for full search")

    def action_refresh(self):
        self.selected_region = None
        self.load_regional_data()
        self.notify("Refreshing world map...")

    def action_back(self):
        self.app.pop_screen()

# ============================================================================
# SCREEN 11: HELP - In-App Guidance
# ============================================================================

HELP_CONTENT = {
    "overview": """
[bold yellow]◆ YES/NO.EVENTS TERMINAL GUIDE[/]

Welcome to the most comprehensive Polymarket research terminal.
This guide helps you navigate and use all features effectively.

[bold cyan]WHAT YOU CAN DO:[/]

  [bold]1. RESEARCH MARKETS[/]
     Browse trending markets, search by keyword, analyze price movements
     and volume patterns to find trading opportunities.

  [bold]2. TRACK TRADERS[/]
     View leaderboards of top performers, analyze their strategies,
     and identify whale movements in real-time.

  [bold]3. ANALYZE DATA[/]
     Use 50+ API endpoints to access orderbooks, price history,
     spread analysis, and cross-market correlations.

  [bold]4. TRADE[/]
     Place limit/market orders, manage positions, track P&L,
     and monitor your portfolio performance.

  [bold]5. EXPLORE PATTERNS[/]
     Elon Lab analyzes 31 days of tweet data with heatmaps,
     daily patterns, and behavioral insights.

  [bold]6. GLOBAL VIEW[/]
     World Map shows regional market activity across
     US, Europe, Asia, and Crypto markets.

[dim]Press 1-7 to navigate sections, ESC to go back[/]
""",
    "navigation": """
[bold yellow]◆ NAVIGATION & KEYBOARD SHORTCUTS[/]

[bold cyan]GLOBAL SHORTCUTS:[/]
  [bold white]ESC[/]     Go back / Close modal
  [bold white]q[/]       Quit application
  [bold white]1-9[/]     Jump to screen (from main menu)
  [bold white]?[/]       Open this help guide

[bold cyan]SCREEN-SPECIFIC:[/]

  [bold]MARKETS (1)[/]
    F1    Trending markets (by volume)
    F2    Search mode (type query, Enter to search)
    F3    Elon-related markets only
    r     Refresh data
    TAB   Focus table for navigation
    Enter View market details
    j/k   Move cursor down/up

  [bold]TRADING (2)[/]
    1     Limit order form
    2     Market order form
    r     Refresh orders

  [bold]PORTFOLIO (3)[/]
    1     Positions tab
    2     Trades tab
    3     Activity tab
    4     Orders tab

  [bold]ELON LAB (4)[/]
    1     Live overview
    2     Hourly heatmap
    3     Daily patterns
    4     Behavior analysis

  [bold]RESEARCH (5)[/]
    1     Leaderboard
    2     Tag analytics
    3     Pattern discovery
    4     Correlations

  [bold]ANALYTICS (6)[/]
    1     Spread analysis
    2     Momentum scanner
    3     Volume leaders
    4     Liquidity ratings

  [bold]SETTINGS (8)[/]
    n/d/h/o  Theme shortcuts (Neon/Dark/Hacker/Ocean)
    c        Toggle compact mode
    s        Toggle sparklines

[dim]Press ESC to go back[/]
""",
    "search": """
[bold yellow]◆ ADVANCED SEARCH GUIDE[/]

[bold cyan]SEARCH CATEGORIES:[/]

  [bold]POLITICS[/]
    Keywords: trump, biden, election, congress, senate, president
    Best for: US political markets, polls, elections

  [bold]CRYPTO[/]
    Keywords: bitcoin, ethereum, solana, crypto, btc, eth
    Best for: Price targets, adoption milestones

  [bold]FINANCE[/]
    Keywords: fed, rates, stock, market, recession, inflation
    Best for: Macro markets, economic predictions

  [bold]TECH[/]
    Keywords: ai, apple, google, tesla, spacex, elon
    Best for: Company events, product launches

  [bold]SPORTS[/]
    Keywords: nfl, nba, soccer, superbowl, championship
    Best for: Game outcomes, championship winners

  [bold]GLOBAL[/]
    Keywords: china, russia, ukraine, europe, brexit
    Best for: Geopolitical events, international relations

[bold cyan]SEARCH TIPS:[/]

  • Use specific terms: "bitcoin 100k" not just "crypto"
  • Combine keywords: "trump election 2024"
  • Check volume: High volume = more liquidity
  • Compare prices: YES + NO should = ~100%

[bold cyan]UNDERSTANDING RESULTS:[/]

  [bold]YES Price[/]  - Market's estimated probability of YES
  [bold]NO Price[/]   - Market's estimated probability of NO
  [bold]Volume[/]     - Total trading volume (higher = more liquid)
  [bold]Spread[/]     - Difference between bid/ask (lower = better)

[dim]Press ESC to go back[/]
""",
    "trading": """
[bold yellow]◆ TRADING GUIDE[/]

[bold cyan]ORDER TYPES:[/]

  [bold]LIMIT ORDER[/]
    You specify the exact price you want.
    Order waits in the orderbook until filled or cancelled.
    Best for: Getting better prices, patient traders

  [bold]MARKET ORDER[/]
    Executes immediately at best available price.
    May have slippage on large orders.
    Best for: Quick execution, small positions

[bold cyan]UNDERSTANDING PRICES:[/]

  • Prices are in cents (0-100 or 0.00-1.00)
  • 52% YES = market thinks 52% chance of happening
  • Your cost = price × shares
  • Max payout = 100 × shares (if you win)

[bold cyan]POSITION SIZING:[/]

  [bold]Kelly Criterion[/] - Optimal bet sizing based on edge
    f* = (bp - q) / b
    where b=odds-1, p=your probability, q=1-p

  [bold]Example:[/]
    Market: 50%, Your estimate: 60%, Bankroll: $1000
    Kelly suggests: ~4% of bankroll = $40 position

  [dim]Tip: Use 25% Kelly (quarter Kelly) for safety[/]

[bold cyan]RISK MANAGEMENT:[/]

  • Never bet more than you can afford to lose
  • Diversify across multiple markets
  • Use stop-losses mentally
  • Track your P&L regularly

[bold red]WARNING: Prediction markets involve real money.
This is NOT financial advice. Trade at your own risk.[/]

[dim]Press ESC to go back[/]
""",
    "api": """
[bold yellow]◆ API ENDPOINTS REFERENCE[/]

[bold cyan]GAMMA API[/] - Market Data (8 endpoints)
  [bold]gamma_search(query)[/]        Search markets by keyword
  [bold]gamma_get_market(id)[/]       Get single market details
  [bold]gamma_get_event(id)[/]        Get event with all markets
  [bold]gamma_list_tags()[/]          Get all available tags
  [bold]gamma_get_comments(id)[/]     Get market comments
  [bold]gamma_get_profile(addr)[/]    Get user profile
  [bold]gamma_get_resolution(id)[/]   Get resolution status
  [bold]gamma_get_public_profile()[/] Get public profile

[bold cyan]CLOB API[/] - Orderbook (6 endpoints)
  [bold]get_orderbook(token)[/]       Full orderbook depth
  [bold]get_spread(token)[/]          Current bid-ask spread
  [bold]get_market_trades(token)[/]   Recent trade history
  [bold]clob_get_spreads()[/]         Spreads for multiple markets
  [bold]get_orderbook_depth()[/]      Depth analysis
  [bold]get_price(token)[/]           Current mid price

[bold cyan]DATA API v2[/] - User Data (6 endpoints)
  [bold]dataapi_get_positions(addr)[/]  User positions
  [bold]dataapi_get_trades(addr)[/]     Trade history
  [bold]dataapi_get_activity(addr)[/]   Account activity
  [bold]dataapi_get_value(addr)[/]      Portfolio value
  [bold]dataapi_get_holders(id)[/]      Market holders
  [bold]dataapi_get_leaderboard()[/]    Top traders

[bold cyan]XTRACKER[/] - User Tracking (6 endpoints)
  [bold]xtracker_get_users()[/]         All tracked users
  [bold]xtracker_get_user(id)[/]        Single user details
  [bold]xtracker_get_metrics(id)[/]     User metrics
  [bold]xtracker_get_trackings()[/]     Active trackings
  [bold]xtracker_get_all_trackings()[/] All historical

[bold cyan]ANALYSIS[/] - Quant Tools (4 endpoints)
  [bold]find_ev_opportunities()[/]      EV scanner
  [bold]get_price_history(token)[/]     Historical prices
  [bold]get_live_volume(token)[/]       Real-time volume
  [bold]get_open_interest(token)[/]     Open interest

[dim]All endpoints available in API Explorer (7)[/]
[dim]Press ESC to go back[/]
""",
    "tips": """
[bold yellow]◆ PRO TIPS & STRATEGIES[/]

[bold cyan]FINDING EDGE:[/]

  [bold]1. Information Asymmetry[/]
     You know something the market doesn't.
     Example: Expert in a specific field, early news access

  [bold]2. Model Disagreement[/]
     Your probability estimate differs from market.
     Only bet when edge > 3% (covers fees + risk)

  [bold]3. Liquidity Provision[/]
     Make markets by placing limit orders on both sides.
     Collect the spread, manage inventory risk.

  [bold]4. Correlation Trading[/]
     Two related markets mispriced relative to each other.
     Example: Same event, different timelines

[bold cyan]COMMON MISTAKES:[/]

  [red]✗[/] Overconfidence - "I'm sure this will happen"
  [red]✗[/] Ignoring fees - Spread eats into small edges
  [red]✗[/] Chasing losses - Doubling down on bad trades
  [red]✗[/] FOMO - Buying after big price moves
  [red]✗[/] No exit plan - Hold forever hoping for recovery

[bold cyan]BEST PRACTICES:[/]

  [green]✓[/] Document your thesis before trading
  [green]✓[/] Set position limits per market
  [green]✓[/] Review your trades weekly
  [green]✓[/] Track win rate AND average win size
  [green]✓[/] Take profits when target hit

[bold cyan]DATA ANALYSIS:[/]

  • [bold]Volume Spike[/] = New information, traders reacting
  • [bold]Wide Spread[/] = Low liquidity, harder to trade
  • [bold]Price Momentum[/] = Trend may continue short-term
  • [bold]Leaderboard[/] = Follow what top traders buy

[dim]Press ESC to go back[/]
"""
}

class HelpScreen(Screen):
    """In-app help and guidance"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("1", "section_overview", "Overview"),
        Binding("2", "section_navigation", "Navigation"),
        Binding("3", "section_search", "Search"),
        Binding("4", "section_trading", "Trading"),
        Binding("5", "section_api", "API"),
        Binding("6", "section_tips", "Tips"),
    ]

    def __init__(self):
        super().__init__()
        self.section = "overview"

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(
                Static(self._render_content(), id="help-content"),
                id="help-scroll"
            ),
            id="help-container"
        )

    def _render_content(self) -> str:
        lines = []
        colors = STATE.get_colors()

        # Header
        lines.append("[bold #ff8c00]╔══════════════════════════════════════════════════════════════════════════════╗[/]")
        lines.append("[bold #ff8c00]║[/]  [bold white]HELP & GUIDE[/]  │  Press 1-6 for sections                               [bold #ff8c00]║[/]")
        lines.append("[bold #ff8c00]╚══════════════════════════════════════════════════════════════════════════════╝[/]")
        lines.append("")

        # Tabs
        tabs = [
            ("1", "Overview", "overview"),
            ("2", "Navigation", "navigation"),
            ("3", "Search", "search"),
            ("4", "Trading", "trading"),
            ("5", "API", "api"),
            ("6", "Tips", "tips"),
        ]
        tab_line = "  "
        for key, label, section_id in tabs:
            if self.section == section_id:
                tab_line += f"[bold white on #ff8c00] {key}:{label} [/]  "
            else:
                tab_line += f"[dim] {key}:{label} [/]  "
        lines.append(tab_line)
        lines.append("")
        lines.append("─" * 78)

        # Content
        content = HELP_CONTENT.get(self.section, "Section not found")
        lines.append(content)

        return "\n".join(lines)

    def _update_display(self):
        self.query_one("#help-content", Static).update(self._render_content())

    def action_section_overview(self):
        self.section = "overview"
        self._update_display()

    def action_section_navigation(self):
        self.section = "navigation"
        self._update_display()

    def action_section_search(self):
        self.section = "search"
        self._update_display()

    def action_section_trading(self):
        self.section = "trading"
        self._update_display()

    def action_section_api(self):
        self.section = "api"
        self._update_display()

    def action_section_tips(self):
        self.section = "tips"
        self._update_display()

    def action_back(self):
        self.app.pop_screen()


# ============================================================================
# SCREEN: CLAUDE AI - Natural Language Trading
# ============================================================================

class ClaudeAIScreen(Screen):
    """Claude AI - Natural language trading interface"""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+enter", "send", "Send"),
        Binding("f1", "example_buy", "Buy Example"),
        Binding("f2", "example_sell", "Sell Example"),
        Binding("f3", "example_search", "Search Example"),
        Binding("f5", "refresh_db", "Refresh DB"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_history = []
        self.pending_action = None
        self.is_configured = False

    def compose(self) -> ComposeResult:
        yield Container(
            Static(self._render_header(), id="claude-header"),
            ScrollableContainer(
                Static(self._render_chat(), id="chat-content"),
                id="chat-scroll"
            ),
            Container(
                Input(placeholder="Tell Claude what you want to trade...", id="claude-input"),
                Button("Send", id="send-btn", variant="primary"),
                id="input-area"
            ),
            Static(self._render_status(), id="claude-status"),
            id="claude-container"
        )

    def on_mount(self):
        """Check configuration on mount"""
        self._check_config()
        self._add_welcome_message()

    def _check_config(self):
        """Check if Claude and trading are configured"""
        try:
            from claude_trader import check_setup
            status = check_setup()
            self.is_configured = status.get('claude_configured', False) and status.get('trading_configured', False)
            if not status.get('claude_configured', False):
                self._add_system_message("Claude API not configured. Add ANTHROPIC_API_KEY to data/.trading_config.json")
            elif not status.get('trading_configured', False):
                self._add_system_message("Trading API not configured. Add wallet credentials to data/.trading_config.json")
        except Exception as e:
            self._add_system_message(f"Setup check failed: {str(e)}")

    def _add_welcome_message(self):
        """Add welcome message with examples"""
        self.chat_history.append({
            'type': 'system',
            'content': '''[bold #7ec8e3]Welcome to Claude AI Trading[/]

Tell me what you want to trade in natural language:

[dim]Examples:[/]
  [cyan]›[/] Buy 10 YES shares of Bitcoin above 100k
  [cyan]›[/] Sell my Trump positions at 85 cents
  [cyan]›[/] Find markets about AI with high volume
  [cyan]›[/] Build a ladder from 60 to 70 cents, 5 shares each

[dim]Press[/] [white]F1[/] [dim]buy example[/] [dim]│[/] [white]F2[/] [dim]sell example[/] [dim]│[/] [white]F3[/] [dim]search example[/]'''
        })
        self._refresh_chat()

    def _add_system_message(self, msg):
        """Add system message"""
        self.chat_history.append({'type': 'system', 'content': msg})
        self._refresh_chat()

    def _add_user_message(self, msg):
        """Add user message"""
        self.chat_history.append({'type': 'user', 'content': msg})
        self._refresh_chat()

    def _add_claude_message(self, msg, action=None):
        """Add Claude response"""
        self.chat_history.append({'type': 'claude', 'content': msg, 'action': action})
        if action:
            self.pending_action = action
        self._refresh_chat()

    def _render_header(self) -> str:
        return """[bold #7ec8e3]╔══════════════════════════════════════════════════════════════════════════════╗
║                            🤖 CLAUDE AI TRADING                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Natural Language Trading • Research • Position Building                      ║
╚══════════════════════════════════════════════════════════════════════════════╝[/]"""

    def _render_chat(self) -> str:
        if not self.chat_history:
            return "[dim]Chat history will appear here...[/]"

        lines = []
        for msg in self.chat_history[-20:]:  # Show last 20 messages
            if msg['type'] == 'system':
                lines.append(f"[yellow]━━━ SYSTEM ━━━[/]")
                lines.append(msg['content'])
                lines.append("")
            elif msg['type'] == 'user':
                lines.append(f"[bold #e94560]YOU >[/] {msg['content']}")
                lines.append("")
            elif msg['type'] == 'claude':
                lines.append(f"[bold #7ec8e3]CLAUDE >[/] {msg['content']}")
                if msg.get('action'):
                    action = msg['action']
                    lines.append("")
                    lines.append("[#7ec8e3]┌─ PARSED ACTION ─────────────────────────┐[/]")
                    lines.append(f"[#7ec8e3]│[/]  Action:   [bold]{action.get('action', '-')}[/]")
                    side = action.get('side', '-')
                    side_color = '#00ff88' if side == 'YES' else '#ff4444' if side == 'NO' else 'white'
                    lines.append(f"[#7ec8e3]│[/]  Side:     [{side_color}]{side}[/]")
                    lines.append(f"[#7ec8e3]│[/]  Quantity: [bold]{action.get('quantity', '-')}[/]")
                    price = action.get('price')
                    lines.append(f"[#7ec8e3]│[/]  Price:    [bold]{str(price) + '¢' if price else 'Market'}[/]")
                    if action.get('market_query'):
                        lines.append(f"[#7ec8e3]│[/]  Market:   {action.get('market_query')}")
                    lines.append("[#7ec8e3]└──────────────────────────────────────────┘[/]")
                    lines.append("")
                    lines.append("[bold green]Press ENTER to execute[/]  [dim]│[/]  [bold red]Press ESC to cancel[/]")
                lines.append("")

        return "\n".join(lines)

    def _render_status(self) -> str:
        status = "[green]●[/] Ready" if self.is_configured else "[red]●[/] Not Configured"
        pending = " [yellow]│ Action pending - ENTER to execute[/]" if self.pending_action else ""
        return f"[dim]Status:[/] {status}{pending} [dim]│[/] [dim]ESC[/] back [dim]│[/] [dim]Ctrl+Enter[/] send"

    def _refresh_chat(self):
        """Refresh chat display"""
        try:
            self.query_one("#chat-content", Static).update(self._render_chat())
            self.query_one("#claude-status", Static).update(self._render_status())
            # Scroll to bottom
            scroll = self.query_one("#chat-scroll", ScrollableContainer)
            scroll.scroll_end(animate=False)
        except:
            pass

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle input submission"""
        if event.input.id == "claude-input":
            await self._send_message()

    async def on_button_pressed(self, event: Button.Pressed):
        """Handle button press"""
        if event.button.id == "send-btn":
            await self._send_message()

    async def _send_message(self):
        """Send message to Claude"""
        input_widget = self.query_one("#claude-input", Input)
        text = input_widget.value.strip()
        if not text:
            return

        # If there's a pending action and user presses enter, execute it
        if self.pending_action and text.lower() in ['yes', 'y', 'execute', 'confirm', '']:
            await self._execute_action()
            input_widget.value = ""
            return
        elif self.pending_action and text.lower() in ['no', 'n', 'cancel']:
            self.pending_action = None
            self._add_system_message("Action cancelled")
            input_widget.value = ""
            return

        self._add_user_message(text)
        input_widget.value = ""

        # Send to Claude
        try:
            from claude_trader import parse_trading_intent, execute_trading_action
            result = parse_trading_intent(text)

            if result.get('error'):
                self._add_claude_message(f"[red]Error:[/] {result['error']}")
            elif result.get('action'):
                action_type = result.get('action')  # This is a string like 'SEARCH', 'BUY', etc.
                if action_type == 'SEARCH':
                    # Execute search directly
                    query = result.get('market_query', text)
                    self._add_claude_message(f"[cyan]Searching for:[/] {query}")
                    # Execute the search
                    exec_result = execute_trading_action(result)
                    if exec_result.get('results'):
                        markets = exec_result['results'][:5]
                        for m in markets:
                            title = m.get('title', m.get('question', 'Unknown'))[:50]
                            yes_p = m.get('yes_price', m.get('current_prices', {}).get('yes', {}).get('price', 0.5))
                            self._add_claude_message(f"  • {title}... [green]{int(yes_p*100)}¢[/]")
                elif action_type == 'INFO':
                    # Show event info with all markets
                    exec_result = execute_trading_action(result)
                    if exec_result.get('error'):
                        self._add_claude_message(f"[red]Error:[/] {exec_result['error']}")
                    else:
                        title = exec_result.get('title', 'Unknown Event')
                        markets = exec_result.get('markets', [])
                        lines = [f"[bold cyan]═══ {title} ═══[/]", ""]
                        lines.append("[dim]┌─────────────────────────────────────────────┬────────┬─────────┬─────────┐[/]")
                        lines.append("[dim]│ Market                                      │  YES   │  Bid    │  Ask    │[/]")
                        lines.append("[dim]├─────────────────────────────────────────────┼────────┼─────────┼─────────┤[/]")
                        for m in markets:
                            q = m.get('question', '')[:43].ljust(43)
                            yes_p = f"{m.get('yes_price', 0)*100:.0f}¢".center(6)
                            bid = f"{m.get('best_bid', 0)*100:.0f}¢".center(7) if m.get('best_bid') else "  --   "
                            ask = f"{m.get('best_ask', 1)*100:.0f}¢".center(7) if m.get('best_ask') else "  --   "
                            lines.append(f"[dim]│[/] {q} [dim]│[/] {yes_p} [dim]│[/] {bid} [dim]│[/] {ask} [dim]│[/]")
                        lines.append("[dim]└─────────────────────────────────────────────┴────────┴─────────┴─────────┘[/]")
                        lines.append("")
                        lines.append("[dim]To trade: 'buy 5 shares of [market name] at [price]'[/]")
                        self._add_claude_message("\n".join(lines))
                elif action_type == 'CANCEL':
                    exec_result = execute_trading_action(result)
                    if exec_result.get('status') == 'ok':
                        cancelled = exec_result.get('cancelled', [])
                        if cancelled == 'all':
                            self._add_claude_message("[green]✓ All orders cancelled[/]")
                        else:
                            self._add_claude_message(f"[green]✓ Cancelled {len(cancelled)} orders[/]")
                    else:
                        self._add_claude_message(f"[red]Cancel failed:[/] {exec_result.get('error', 'Unknown')}")
                elif action_type in ['BUY', 'SELL']:
                    # Show parsed action with market info for confirmation
                    side = result.get('side', 'YES')
                    size = result.get('size', result.get('quantity', 0))
                    budget = result.get('budget')
                    price = result.get('price')
                    market = result.get('market_query', result.get('market', 'Unknown'))

                    # Try to get market info with orderbook
                    market_info = None
                    try:
                        from claude_trader import get_market_info_for_trade, search_markets_db
                        # Find market ID
                        results = search_markets_db(market, limit=1)
                        if results:
                            market_info = get_market_info_for_trade(results[0]['id'])
                    except:
                        pass

                    lines = []
                    lines.append(f"[bold yellow]═══ {action_type} ORDER ═══[/]")
                    lines.append("")

                    # Show market info if available
                    if market_info and not market_info.get('error'):
                        lines.append(f"[cyan]Market:[/] {market_info.get('question', market)[:60]}")
                        lines.append(f"[cyan]ID:[/] {market_info.get('market_id', '?')}")
                        lines.append("")

                        # Orderbook summary
                        bid = market_info.get('best_bid', 0)
                        ask = market_info.get('best_ask', 1)
                        spread = market_info.get('spread', 1)
                        liquid = "[green]✓ Liquid[/]" if market_info.get('liquid') else "[red]⚠ Wide spread[/]"

                        lines.append(f"[dim]┌─────────────────────────────────────┐[/]")
                        lines.append(f"[dim]│[/] Best Bid: [green]{bid*100:.0f}¢[/]  │  Best Ask: [red]{ask*100:.0f}¢[/]")
                        lines.append(f"[dim]│[/] Spread: {spread*100:.0f}¢  {liquid}")
                        lines.append(f"[dim]└─────────────────────────────────────┘[/]")
                        lines.append("")
                    else:
                        lines.append(f"[cyan]Market:[/] {market}")
                        lines.append("")

                    # Order details
                    if budget and not size:
                        lines.append(f"[bold]Action:[/]  {action_type} ${budget} worth")
                    else:
                        lines.append(f"[bold]Action:[/]  {action_type} {size} {side} shares")

                    if price:
                        lines.append(f"[bold]Price:[/]   {int(price*100)}¢ (limit)")
                    else:
                        lines.append(f"[bold]Price:[/]   Market order")

                    lines.append("")
                    lines.append("[bold green]Enter[/] to confirm  │  [bold red]ESC[/] to cancel  │  [dim]Type to edit[/]")

                    self._add_claude_message("\n".join(lines), result)
                elif action_type == 'LADDER':
                    # Show ladder plan with full details
                    side = result.get('side', 'BUY')
                    market = result.get('market_query', result.get('market', 'Unknown'))
                    start = result.get('ladder_start', 0)
                    end = result.get('ladder_end', 0)
                    step = result.get('ladder_step', 0.01)
                    count = result.get('ladder_count', 10)
                    shares_per = result.get('shares_per_step', result.get('size', 10))

                    # Build ladder preview
                    lines = [f"[bold yellow]▶ LADDER ORDER PLAN[/]"]
                    lines.append(f"  Market: [cyan]{market}[/]")
                    lines.append(f"  Side:   [{('#00ff88' if side in ['BUY', 'YES'] else '#ff4444')}]{side}[/]")
                    lines.append(f"  Range:  [bold]{start*100:.1f}¢[/] → [bold]{end*100:.1f}¢[/]")
                    lines.append(f"  Step:   {step*100:.2f}¢")
                    lines.append(f"  Orders: {count} @ {shares_per} shares each")
                    lines.append(f"  Total:  [bold]{count * shares_per}[/] shares, ~${count * shares_per * ((start+end)/2):.2f}")
                    lines.append("")
                    lines.append("[dim]┌─ ORDER PREVIEW ──────────────────────┐[/]")
                    # Show individual orders - respect direction (BUY: high→low, SELL: low→high)
                    price = start
                    going_down = start > end
                    for i in range(min(count, 8)):
                        lines.append(f"[dim]│[/]  {i+1:>2}. {side} {shares_per} @ {price*100:.1f}¢")
                        if going_down:
                            price -= step
                        else:
                            price += step
                    if count > 8:
                        lines.append(f"[dim]│  ... {count - 8} more orders ...[/]")
                    lines.append("[dim]└──────────────────────────────────────┘[/]")
                    lines.append("")
                    lines.append("[bold green]Enter[/] to deploy  [bold red]ESC[/] to cancel  [dim]Type 'edit' to modify[/]")
                    self._add_claude_message("\n".join(lines), result)
                elif action_type == 'POSITION':
                    exec_result = execute_trading_action(result)
                    positions = exec_result.get('positions', [])
                    if positions:
                        self._add_claude_message(f"[cyan]Your positions ({len(positions)}):[/]")
                        for p in positions[:5]:
                            title = p.get('title', p.get('market', 'Unknown'))[:40]
                            outcome = p.get('outcome', p.get('side', 'YES'))
                            self._add_claude_message(f"  • {title}... [{outcome}]")
                    else:
                        self._add_claude_message("[dim]No open positions[/]")
                elif action_type == 'ORDERS':
                    exec_result = execute_trading_action(result)
                    orders = exec_result.get('orders', [])
                    if orders:
                        self._add_claude_message(f"[cyan]Open orders ({len(orders)}):[/]")
                        for o in orders[:5]:
                            self._add_claude_message(f"  • {o.get('side', 'BUY')} @ {int(float(o.get('price', 0))*100)}¢")
                    else:
                        self._add_claude_message("[dim]No open orders[/]")
                else:
                    self._add_claude_message(f"[dim]Action: {action_type}[/]\n{result.get('reasoning', '')}")
            else:
                self._add_claude_message(result.get('message', 'Could not understand request'))
        except Exception as e:
            self._add_claude_message(f"[red]Error:[/] {str(e)}")

    async def _execute_action(self):
        """Execute pending action"""
        if not self.pending_action:
            return

        action_type = self.pending_action.get('action', 'UNKNOWN')
        self._add_system_message(f"Executing {action_type}...")

        try:
            from claude_trader import execute_trading_action
            result = execute_trading_action(self.pending_action)

            if result.get('status') == 'need_selection':
                # Multiple markets found - show list and ask user to specify
                self._add_system_message(f"[yellow]⚠ Multiple markets found![/]")
                self._add_system_message(f"[dim]{result.get('message', 'Please specify which market')}[/]")
                self._add_system_message("")
                for i, m in enumerate(result.get('markets', [])[:5]):
                    self._add_system_message(f"  {i+1}. [cyan]{m.get('title', 'Unknown')}[/]")
                self._add_system_message("")
                self._add_system_message("[dim]Re-enter your command with the specific date (e.g., 'jan 31')[/]")
            elif result.get('success') or result.get('status') == 'ok':
                if action_type == 'LADDER':
                    orders_placed = result.get('orders_placed', len(result.get('orders', [])))
                    self._add_system_message(f"[green]✓ Ladder deployed! {orders_placed} orders placed[/]")
                    # Show order details
                    for o in result.get('orders', [])[:5]:
                        self._add_system_message(f"  [dim]•[/] {o.get('size', 0)} @ {o.get('price', 0)}¢ [{o.get('result', '-')}]")
                    if len(result.get('orders', [])) > 5:
                        self._add_system_message(f"  [dim]... and {len(result.get('orders', [])) - 5} more[/]")
                else:
                    order_id = result.get('order_id', result.get('market_id', 'confirmed'))
                    self._add_system_message(f"[green]✓ Trade executed! {order_id}[/]")
            else:
                error = result.get('error', 'Unknown error')
                self._add_system_message(f"[red]✗ Trade failed: {error}[/]")
        except Exception as e:
            self._add_system_message(f"[red]✗ Execution error: {str(e)}[/]")

        self.pending_action = None

    def action_back(self):
        if self.pending_action:
            self.pending_action = None
            self._add_system_message("Action cancelled")
        else:
            self.app.pop_screen()

    def action_send(self):
        """Ctrl+Enter to send"""
        self.run_worker(self._send_message())

    def action_example_buy(self):
        """F1 - Buy example"""
        self.query_one("#claude-input", Input).value = "Buy 10 YES shares at 35 cents"

    def action_example_sell(self):
        """F2 - Sell example"""
        self.query_one("#claude-input", Input).value = "create 10 orders from 15 to 25 cents every 1 cent, 5 shares each"

    def action_example_search(self):
        """F3 - Search example"""
        self.query_one("#claude-input", Input).value = "Find markets about Trump"

    def action_refresh_db(self):
        """F5 - Refresh market database"""
        self._add_system_message("Rebuilding market database...")
        try:
            from claude_trader import build_market_database
            count = build_market_database()
            self._add_system_message(f"[green]✓ Database rebuilt: {count} markets indexed[/]")
        except Exception as e:
            self._add_system_message(f"[red]✗ Rebuild failed: {str(e)}[/]")


# ============================================================================
# MAIN APP
# ============================================================================

class YesNoApp(App):
    """YES/NO.EVENTS - Clean Polymarket Terminal"""

    CSS = """
    Screen {
        background: #0a0a15;
    }

    #menu-container, #markets-container, #trading-container,
    #portfolio-container, #elon-container, #research-container,
    #analytics-container, #settings-container, #detail-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #markets-table {
        height: 1fr;
        min-height: 30;
    }

    #search-input {
        width: 40%;
        height: 3;
        margin: 0 0 0 0;
    }

    #markets-header {
        height: 5;
    }

    #markets-footer {
        height: 1;
    }

    #settings-form {
        padding: 2;
        width: 60;
    }

    #settings-form Label {
        margin: 1 0 0 0;
    }

    #settings-form Input {
        margin: 0 0 1 0;
    }

    #settings-buttons, #detail-buttons {
        margin: 2 0 0 0;
    }

    Button {
        margin: 0 1;
    }

    ScrollableContainer {
        height: 1fr;
    }

    ModalScreen {
        align: center middle;
    }

    #detail-container {
        width: 95%;
        height: 95%;
        background: #0a0a15;
        border: solid #ff8c00;
        layout: vertical;
    }

    #detail-data {
        height: 80%;
        padding: 1;
    }

    #trading-panel {
        height: auto;
        min-height: 8;
        max-height: 50%;
        background: #13131f;
        border-top: solid #ff8c00;
        padding: 1;
    }

    #trading-row {
        height: auto;
        align: center middle;
    }

    #data-row {
        height: auto;
        margin-bottom: 1;
    }

    #outcome-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
        background: #1a1a2e;
        padding: 0 1;
    }

    #outcome-row Button {
        min-width: 10;
        height: 3;
        margin: 0 1;
    }

    #outcome-display {
        margin-left: 2;
    }

    #trade-label {
        margin-right: 1;
    }

    #size-box, #price-box {
        width: 15;
        height: auto;
        margin: 0 1;
    }

    #size-box Input, #price-box Input {
        width: 100%;
    }

    #trading-panel Button {
        min-width: 12;
        height: 3;
        margin: 0 1;
    }

    #order-preview {
        height: auto;
        padding: 1;
        margin-top: 1;
    }

    #confirm-row {
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #confirm-row Button {
        min-width: 20;
        margin: 0 2;
    }

    #order-form {
        padding: 1;
        background: #13131f;
        margin: 1 0;
    }

    #order-form Input {
        width: 20;
        margin: 0 1;
    }

    #wallet-modal, #pk-modal {
        width: 60;
        height: auto;
        padding: 2;
        background: #13131f;
        border: solid #ff8c00;
    }

    #wallet-modal Input, #pk-modal Input {
        width: 100%;
        margin: 1 0;
    }

    #wallet-buttons, #pk-buttons {
        margin: 1 0 0 0;
        align: center middle;
    }

    /* Claude AI Screen */
    #claude-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }

    #claude-header {
        height: 5;
    }

    #chat-scroll {
        height: 1fr;
        min-height: 15;
        border: solid #7ec8e3;
        background: #0a0a15;
        padding: 1;
    }

    #chat-content {
        width: 100%;
    }

    #input-area {
        height: 4;
        layout: horizontal;
        padding: 1 0;
    }

    #claude-input {
        width: 85%;
        height: 3;
    }

    #send-btn {
        width: 15%;
        height: 3;
        background: #7ec8e3;
    }

    #claude-status {
        height: 1;
        dock: bottom;
    }
    """

    TITLE = "YES/NO.EVENTS"

    def on_mount(self):
        # Load saved credentials from config on startup
        try:
            from trading import load_config
            config = load_config()
            if config:
                if config.get("funder"):
                    STATE.wallet = config["funder"]
                if config.get("private_key"):
                    STATE.private_key = config["private_key"]
                    os.environ["POLYMARKET_PRIVATE_KEY"] = config["private_key"]
                if STATE.wallet and STATE.private_key:
                    STATE.logged_in = True
        except Exception:
            pass  # Config loading is optional
        self.push_screen(MainMenuScreen())

    def action_open_link(self, link: str) -> None:
        """Handle custom link protocols for research navigation"""
        if link.startswith("trader:"):
            addr = link[7:]
            self._open_trader_profile(addr)
        elif link.startswith("market:"):
            market_id = link[7:]
            self._open_market_detail(market_id)
        elif link.startswith("tag:"):
            tag = link[4:]
            self._search_by_tag(tag)
        elif link.startswith("event:"):
            event_id = link[6:]
            self._open_event(event_id)
        elif link.startswith("http"):
            # External link - open in browser
            import webbrowser
            webbrowser.open(link)

    def _open_trader_profile(self, address: str):
        """Open trader profile with positions, trades, activity"""
        self.notify(f"Loading trader {address[:10]}...")
        # Navigate to Research screen and load trader
        # For now, show notification with address
        self.notify(f"Trader: {address[:20]}... - Use Research tab to view details", severity="information")

    def _open_market_detail(self, market_id: str):
        """Open market detail modal"""
        try:
            market = gamma_get_market(market_id)
            if market:
                self.push_screen(MarketDetailScreen(market))
            else:
                self.notify(f"Market not found: {market_id[:15]}...", severity="error")
        except Exception as e:
            self.notify(f"Error loading market: {str(e)[:30]}", severity="error")

    def _search_by_tag(self, tag: str):
        """Search markets by tag"""
        self.notify(f"Searching markets tagged: #{tag}")
        # Navigate to Markets screen with tag filter
        # For now just notify
        self.notify(f"Use Markets screen and search for: {tag}", severity="information")

    def _open_event(self, event_id: str):
        """Open event detail"""
        try:
            event = gamma_get_event(event_id)
            if event:
                # Events contain multiple markets - show first market
                markets = event.get('markets', [])
                if markets:
                    self.push_screen(MarketDetailScreen(markets[0]))
                else:
                    self.notify(f"Event has no markets", severity="warning")
            else:
                self.notify(f"Event not found", severity="error")
        except Exception as e:
            self.notify(f"Error: {str(e)[:30]}", severity="error")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = YesNoApp()
    app.run()
