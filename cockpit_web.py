#!/usr/bin/env python3
"""
Polymarket Web Trading Cockpit
==============================
Full-featured web trading interface for Claude Code.

Features:
- Market search and discovery
- Trade preview and confirmation
- Live WebSocket prices and trades
- Chart.js price visualization
- Alert system integration
- Light/dark theme with oklch colors

Usage:
    python cockpit_web.py              # Start cockpit on :8891
    python cockpit_web.py --port 8000  # Custom port
"""

import json
import time
import threading
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import sys

# Add parent for imports
PARENT = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT))

# Import trading functions
try:
    from polymarket_api import (
        get_event_by_slug, get_gamma_market, get_price, get_best_prices,
        get_orderbook, place_order, cancel_order, cancel_all_orders,
        get_balances, get_positions, get_open_orders, search_markets,
        get_market_info, get_recent_trades, get_clob_token_id
    )
    HAS_API = True
except ImportError:
    HAS_API = False

try:
    from market_db import search_db, get_trending, get_categories, db_stats
    HAS_DB = True
except ImportError:
    HAS_DB = False

try:
    from rtds_client import RealTimeDataClient, Message
    HAS_RTDS = True
except ImportError:
    HAS_RTDS = False

try:
    from alerts import AlertManager, AlertType
    HAS_ALERTS = True
except ImportError:
    HAS_ALERTS = False

# =============================================================================
# GLOBAL STATE
# =============================================================================

STATE = {
    "pending_trades": [],
    "active_market": None,
    "last_search": [],
    "ws_connected": False,
    "latest_trades": [],
    "crypto_prices": {},
}

SSE_CLIENTS: Set = set()
MAX_TRADES = 30

# =============================================================================
# HTML TEMPLATE WITH OKLCH STYLING
# =============================================================================

def get_html_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polymarket Trading Cockpit</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            /* Light theme (default) */
            --bg-primary: oklch(98% 0.01 240);
            --bg-secondary: oklch(95% 0.015 240);
            --bg-tertiary: oklch(92% 0.02 240);
            --bg-card: oklch(99% 0.005 240);

            --text-primary: oklch(15% 0.02 240);
            --text-secondary: oklch(40% 0.02 240);
            --text-muted: oklch(55% 0.01 240);

            --border-color: oklch(85% 0.02 240);
            --border-strong: oklch(75% 0.03 240);

            --accent: oklch(55% 0.25 250);
            --accent-hover: oklch(50% 0.28 250);
            --accent-light: oklch(92% 0.08 250);

            --success: oklch(55% 0.2 145);
            --success-light: oklch(92% 0.06 145);
            --danger: oklch(55% 0.22 25);
            --danger-light: oklch(92% 0.08 25);
            --warning: oklch(65% 0.18 85);
            --warning-light: oklch(92% 0.06 85);

            --shadow-sm: 0 1px 2px oklch(0% 0 0 / 0.05);
            --shadow-md: 0 4px 6px oklch(0% 0 0 / 0.07);
            --shadow-lg: 0 10px 15px oklch(0% 0 0 / 0.1);

            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;

            --font-mono: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        [data-theme="dark"] {
            --bg-primary: oklch(12% 0.02 260);
            --bg-secondary: oklch(16% 0.025 260);
            --bg-tertiary: oklch(20% 0.03 260);
            --bg-card: oklch(14% 0.02 260);

            --text-primary: oklch(92% 0.01 260);
            --text-secondary: oklch(72% 0.02 260);
            --text-muted: oklch(55% 0.02 260);

            --border-color: oklch(25% 0.03 260);
            --border-strong: oklch(35% 0.04 260);

            --accent: oklch(65% 0.22 250);
            --accent-hover: oklch(70% 0.24 250);
            --accent-light: oklch(25% 0.08 250);

            --success-light: oklch(22% 0.06 145);
            --danger-light: oklch(22% 0.08 25);
            --warning-light: oklch(22% 0.06 85);

            --shadow-sm: 0 1px 2px oklch(0% 0 0 / 0.2);
            --shadow-md: 0 4px 6px oklch(0% 0 0 / 0.25);
            --shadow-lg: 0 10px 15px oklch(0% 0 0 / 0.3);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: var(--font-sans);
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
        }

        /* Header */
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: var(--shadow-sm);
        }

        .logo {
            font-family: var(--font-mono);
            font-size: 1.2em;
            font-weight: 700;
            color: var(--accent);
        }

        .status-bar {
            display: flex;
            gap: 20px;
            align-items: center;
            font-size: 0.85em;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }

        .status-dot.connected { background: var(--success); }
        .status-dot.pending { background: var(--warning); }

        /* Theme toggle */
        .theme-toggle {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            font-size: 0.85em;
        }

        .theme-toggle:hover {
            background: var(--bg-secondary);
        }

        /* Main layout */
        .main {
            display: grid;
            grid-template-columns: 320px 1fr 300px;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }

        @media (max-width: 1200px) {
            .main {
                grid-template-columns: 1fr;
            }
        }

        /* Panels */
        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }

        .panel-header {
            background: var(--bg-secondary);
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 0.9em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .panel-body {
            padding: 16px;
        }

        /* Search */
        .search-box {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }

        .search-input {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 14px;
            border-radius: var(--radius-md);
            font-size: 0.9em;
            font-family: var(--font-mono);
        }

        .search-input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-light);
        }

        .btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: var(--radius-md);
            cursor: pointer;
            font-weight: 500;
            font-size: 0.9em;
            transition: background 0.15s;
        }

        .btn:hover {
            background: var(--accent-hover);
        }

        .btn-success {
            background: var(--success);
        }

        .btn-danger {
            background: var(--danger);
        }

        .btn-sm {
            padding: 6px 10px;
            font-size: 0.8em;
        }

        /* Market list */
        .market-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .market-item {
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
            transition: background 0.15s;
        }

        .market-item:hover {
            background: var(--bg-secondary);
        }

        .market-item.active {
            background: var(--accent-light);
            border-left: 3px solid var(--accent);
        }

        .market-title {
            font-size: 0.9em;
            margin-bottom: 4px;
            color: var(--text-primary);
        }

        .market-meta {
            font-size: 0.8em;
            color: var(--text-muted);
            display: flex;
            gap: 12px;
        }

        .market-price {
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--accent);
        }

        /* Trading panel */
        .market-card {
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
            padding: 16px;
            margin-bottom: 16px;
        }

        .market-card-title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 12px;
        }

        .price-display {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }

        .price-box {
            text-align: center;
            padding: 12px;
            background: var(--bg-card);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
        }

        .price-label {
            font-size: 0.75em;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .price-value {
            font-family: var(--font-mono);
            font-size: 1.4em;
            font-weight: 700;
        }

        .price-value.bid { color: var(--success); }
        .price-value.ask { color: var(--danger); }

        /* Order form */
        .order-form {
            display: grid;
            gap: 12px;
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .form-label {
            font-size: 0.8em;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .form-input {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 0.9em;
        }

        .form-input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .form-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px 12px;
            border-radius: var(--radius-sm);
            font-size: 0.9em;
        }

        /* Preview table */
        .preview-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
            margin: 16px 0;
        }

        .preview-table th,
        .preview-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .preview-table th {
            background: var(--bg-secondary);
            font-weight: 600;
            color: var(--text-secondary);
        }

        .preview-table td {
            font-family: var(--font-mono);
        }

        .status-pending {
            color: var(--warning);
        }

        .status-success {
            color: var(--success);
        }

        .status-error {
            color: var(--danger);
        }

        /* Live feed */
        .trade-feed {
            max-height: 400px;
            overflow-y: auto;
        }

        .trade-item {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85em;
        }

        .trade-item.buy {
            border-left: 3px solid var(--success);
        }

        .trade-item.sell {
            border-left: 3px solid var(--danger);
        }

        .trade-title {
            color: var(--text-secondary);
            font-size: 0.8em;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .trade-details {
            font-family: var(--font-mono);
            display: flex;
            justify-content: space-between;
        }

        .trade-side-buy { color: var(--success); }
        .trade-side-sell { color: var(--danger); }

        /* Chart */
        .chart-container {
            padding: 16px;
        }

        #priceChart {
            max-height: 250px;
        }

        /* Orderbook */
        .orderbook {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            font-size: 0.85em;
        }

        .orderbook-side h4 {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .orderbook-row {
            display: flex;
            justify-content: space-between;
            padding: 4px 8px;
            font-family: var(--font-mono);
        }

        .orderbook-bid {
            background: var(--success-light);
            border-radius: 2px;
            margin-bottom: 2px;
        }

        .orderbook-ask {
            background: var(--danger-light);
            border-radius: 2px;
            margin-bottom: 2px;
        }

        /* Alerts */
        .alert-item {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .alert-info {
            font-size: 0.85em;
        }

        .alert-type {
            font-size: 0.75em;
            color: var(--text-muted);
        }

        /* Portfolio */
        .position-item {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }

        .position-market {
            font-size: 0.85em;
            margin-bottom: 4px;
        }

        .position-details {
            font-family: var(--font-mono);
            font-size: 0.8em;
            color: var(--text-secondary);
        }

        /* Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        .tab {
            padding: 10px 16px;
            cursor: pointer;
            font-size: 0.85em;
            color: var(--text-secondary);
            border-bottom: 2px solid transparent;
            transition: all 0.15s;
        }

        .tab:hover {
            color: var(--text-primary);
        }

        .tab.active {
            color: var(--accent);
            border-bottom-color: var(--accent);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Utility */
        .text-success { color: var(--success); }
        .text-danger { color: var(--danger); }
        .text-warning { color: var(--warning); }
        .text-muted { color: var(--text-muted); }
        .mono { font-family: var(--font-mono); }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-muted);
        }

        /* Confirmation modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: oklch(0% 0 0 / 0.5);
            z-index: 200;
            align-items: center;
            justify-content: center;
        }

        .modal-overlay.active {
            display: flex;
        }

        .modal {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            padding: 24px;
            min-width: 400px;
            max-width: 500px;
            box-shadow: var(--shadow-lg);
        }

        .modal-title {
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 16px;
        }

        .modal-actions {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="logo">POLY COCKPIT</div>
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot" id="wsStatus"></span>
                <span id="wsLabel">Connecting...</span>
            </div>
            <div class="status-item">
                <span class="mono" id="balance">--</span>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
        </div>
    </header>

    <main class="main">
        <!-- Left Panel: Search & Markets -->
        <aside>
            <div class="panel">
                <div class="panel-header">Market Search</div>
                <div class="panel-body">
                    <div class="search-box">
                        <input type="text" class="search-input" id="searchInput"
                               placeholder="Search markets..." onkeypress="handleSearchKey(event)">
                        <button class="btn" onclick="searchMarkets()">Search</button>
                    </div>
                    <div class="market-list" id="marketList">
                        <div class="empty-state">Search for markets or load trending</div>
                    </div>
                </div>
            </div>

            <div class="panel" style="margin-top: 16px;">
                <div class="panel-header">Quick Actions</div>
                <div class="panel-body">
                    <button class="btn btn-sm" onclick="loadTrending()" style="margin-right: 8px;">Trending</button>
                    <button class="btn btn-sm" onclick="loadPortfolio()">Portfolio</button>
                    <button class="btn btn-sm" onclick="loadOrders()" style="margin-left: 8px;">Orders</button>
                </div>
            </div>
        </aside>

        <!-- Center Panel: Trading -->
        <section>
            <div class="panel">
                <div class="tabs">
                    <div class="tab active" onclick="showTab('trading')">Trading</div>
                    <div class="tab" onclick="showTab('chart')">Chart</div>
                    <div class="tab" onclick="showTab('orderbook')">Orderbook</div>
                    <div class="tab" onclick="showTab('portfolio')">Portfolio</div>
                </div>

                <!-- Trading Tab -->
                <div class="tab-content active" id="tab-trading">
                    <div class="panel-body">
                        <div class="market-card" id="selectedMarket">
                            <div class="empty-state">Select a market to trade</div>
                        </div>

                        <div id="orderSection" style="display: none;">
                            <div class="price-display" id="priceDisplay">
                                <div class="price-box">
                                    <div class="price-label">Bid</div>
                                    <div class="price-value bid" id="bidPrice">--</div>
                                </div>
                                <div class="price-box">
                                    <div class="price-label">YES Price</div>
                                    <div class="price-value" id="yesPrice">--</div>
                                </div>
                                <div class="price-box">
                                    <div class="price-label">Ask</div>
                                    <div class="price-value ask" id="askPrice">--</div>
                                </div>
                            </div>

                            <div class="order-form">
                                <div class="form-row">
                                    <div class="form-group">
                                        <label class="form-label">Side</label>
                                        <select class="form-select" id="orderSide">
                                            <option value="BUY">BUY</option>
                                            <option value="SELL">SELL</option>
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label class="form-label">Outcome</label>
                                        <select class="form-select" id="orderOutcome">
                                            <option value="yes">YES</option>
                                            <option value="no">NO</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="form-row">
                                    <div class="form-group">
                                        <label class="form-label">Size (shares)</label>
                                        <input type="number" class="form-input" id="orderSize" value="10" min="1">
                                    </div>
                                    <div class="form-group">
                                        <label class="form-label">Price (cents)</label>
                                        <input type="number" class="form-input" id="orderPrice" value="50" min="1" max="99" step="1">
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Estimated Cost</label>
                                    <div class="mono" id="estimatedCost">$0.00</div>
                                </div>
                                <button class="btn" onclick="previewOrder()">Preview Order</button>
                            </div>
                        </div>

                        <!-- Preview Section -->
                        <div id="previewSection" style="display: none;">
                            <h3 style="margin: 16px 0 8px;">Order Preview</h3>
                            <table class="preview-table">
                                <thead>
                                    <tr>
                                        <th>Action</th>
                                        <th>Details</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="previewBody"></tbody>
                            </table>
                            <div style="display: flex; gap: 12px;">
                                <button class="btn btn-success" onclick="executeOrder()">Confirm & Execute</button>
                                <button class="btn btn-danger" onclick="cancelPreview()">Cancel</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Chart Tab -->
                <div class="tab-content" id="tab-chart">
                    <div class="chart-container">
                        <canvas id="priceChart"></canvas>
                    </div>
                </div>

                <!-- Orderbook Tab -->
                <div class="tab-content" id="tab-orderbook">
                    <div class="panel-body">
                        <div class="orderbook" id="orderbookDisplay">
                            <div class="orderbook-side">
                                <h4>Bids (Buyers)</h4>
                                <div id="bidsList"></div>
                            </div>
                            <div class="orderbook-side">
                                <h4>Asks (Sellers)</h4>
                                <div id="asksList"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Tab -->
                <div class="tab-content" id="tab-portfolio">
                    <div class="panel-body">
                        <h4 style="margin-bottom: 12px;">Positions</h4>
                        <div id="positionsList">
                            <div class="empty-state">Loading positions...</div>
                        </div>
                        <h4 style="margin: 16px 0 12px;">Open Orders</h4>
                        <div id="ordersList">
                            <div class="empty-state">Loading orders...</div>
                        </div>
                        <div style="margin-top: 16px;">
                            <button class="btn btn-danger btn-sm" onclick="cancelAllOrders()">Cancel All Orders</button>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Right Panel: Live Feed & Alerts -->
        <aside>
            <div class="panel">
                <div class="panel-header">Live Trades</div>
                <div class="trade-feed" id="tradeFeed">
                    <div class="empty-state">Connecting to live feed...</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 16px;">
                <div class="panel-header">Crypto Prices</div>
                <div class="panel-body" id="cryptoPrices">
                    <div class="text-muted">Waiting for data...</div>
                </div>
            </div>

            <div class="panel" style="margin-top: 16px;">
                <div class="panel-header">Alerts</div>
                <div class="panel-body" id="alertsPanel">
                    <div class="empty-state">No alerts configured</div>
                </div>
            </div>
        </aside>
    </main>

    <!-- Confirmation Modal -->
    <div class="modal-overlay" id="confirmModal">
        <div class="modal">
            <div class="modal-title">Confirm Trade</div>
            <div id="confirmContent"></div>
            <div class="modal-actions">
                <button class="btn btn-danger" onclick="closeModal()">Cancel</button>
                <button class="btn btn-success" onclick="confirmTrade()">Execute</button>
            </div>
        </div>
    </div>

    <script>
        // State
        let activeMarket = null;
        let pendingOrder = null;
        let priceChart = null;
        let eventSource = null;

        // Theme
        function toggleTheme() {
            const body = document.body;
            if (body.getAttribute('data-theme') === 'dark') {
                body.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
            } else {
                body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
            }
        }

        // Load saved theme
        if (localStorage.getItem('theme') === 'dark') {
            document.body.setAttribute('data-theme', 'dark');
        }

        // Tabs
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');

            if (tabName === 'chart' && activeMarket) {
                loadChart(activeMarket.id);
            } else if (tabName === 'orderbook' && activeMarket) {
                loadOrderbook(activeMarket.id);
            } else if (tabName === 'portfolio') {
                loadPortfolio();
            }
        }

        // Search
        function handleSearchKey(e) {
            if (e.key === 'Enter') searchMarkets();
        }

        async function searchMarkets() {
            const query = document.getElementById('searchInput').value;
            if (!query) return;

            try {
                const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                const data = await resp.json();
                renderMarketList(data.markets || []);
            } catch (e) {
                console.error('Search error:', e);
            }
        }

        async function loadTrending() {
            try {
                const resp = await fetch('/api/trending');
                const data = await resp.json();
                renderMarketList(data.markets || []);
            } catch (e) {
                console.error('Trending error:', e);
            }
        }

        function renderMarketList(markets) {
            const list = document.getElementById('marketList');
            if (!markets.length) {
                list.innerHTML = '<div class="empty-state">No markets found</div>';
                return;
            }

            list.innerHTML = markets.map(m => `
                <div class="market-item" onclick="selectMarket('${m.id}')">
                    <div class="market-title">${m.title || m.question || ''}</div>
                    <div class="market-meta">
                        <span class="market-price">${formatPrice(m.yes_price || m.yes || 0.5)}</span>
                        <span>${formatVolume(m.volume || 0)}</span>
                    </div>
                </div>
            `).join('');
        }

        // Market selection
        async function selectMarket(marketId) {
            try {
                const resp = await fetch(`/api/market/${marketId}`);
                const data = await resp.json();

                activeMarket = data;

                // Update UI
                document.getElementById('selectedMarket').innerHTML = `
                    <div class="market-card-title">${data.question || data.title}</div>
                    <div class="text-muted">ID: ${data.id}</div>
                `;

                document.getElementById('orderSection').style.display = 'block';
                document.getElementById('bidPrice').textContent = formatPrice(data.bid);
                document.getElementById('askPrice').textContent = formatPrice(data.ask);
                document.getElementById('yesPrice').textContent = formatPrice(data.yes_price);

                // Set default price to best ask for buy
                document.getElementById('orderPrice').value = Math.round(data.ask * 100);
                updateEstimatedCost();

                // Highlight in list
                document.querySelectorAll('.market-item').forEach(el => el.classList.remove('active'));

            } catch (e) {
                console.error('Market load error:', e);
            }
        }

        // Order form
        function updateEstimatedCost() {
            const size = parseInt(document.getElementById('orderSize').value) || 0;
            const price = parseInt(document.getElementById('orderPrice').value) / 100;
            const cost = size * price;
            document.getElementById('estimatedCost').textContent = `$${cost.toFixed(2)}`;
        }

        document.getElementById('orderSize')?.addEventListener('input', updateEstimatedCost);
        document.getElementById('orderPrice')?.addEventListener('input', updateEstimatedCost);

        async function previewOrder() {
            if (!activeMarket) return;

            const order = {
                market_id: activeMarket.id,
                side: document.getElementById('orderSide').value,
                outcome: document.getElementById('orderOutcome').value,
                size: parseInt(document.getElementById('orderSize').value),
                price: parseInt(document.getElementById('orderPrice').value) / 100
            };

            pendingOrder = order;

            const cost = order.size * order.price;

            document.getElementById('previewBody').innerHTML = `
                <tr>
                    <td>${order.side} ${order.outcome.toUpperCase()}</td>
                    <td>${order.size} @ ${order.price * 100}c ($${cost.toFixed(2)})</td>
                    <td class="status-pending">PENDING</td>
                </tr>
                <tr>
                    <td>Market</td>
                    <td>${activeMarket.question?.substring(0, 40)}...</td>
                    <td></td>
                </tr>
            `;

            document.getElementById('previewSection').style.display = 'block';
        }

        async function executeOrder() {
            if (!pendingOrder) return;

            try {
                const resp = await fetch('/api/order', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(pendingOrder)
                });

                const result = await resp.json();

                const status = result.status || (result.error ? 'ERROR' : 'PLACED');
                const statusClass = result.error ? 'status-error' : 'status-success';

                document.getElementById('previewBody').innerHTML = `
                    <tr>
                        <td>${pendingOrder.side} ${pendingOrder.outcome.toUpperCase()}</td>
                        <td>${pendingOrder.size} @ ${pendingOrder.price * 100}c</td>
                        <td class="${statusClass}">${status}</td>
                    </tr>
                `;

                pendingOrder = null;

                // Refresh orders
                setTimeout(loadOrders, 1000);

            } catch (e) {
                console.error('Order error:', e);
            }
        }

        function cancelPreview() {
            pendingOrder = null;
            document.getElementById('previewSection').style.display = 'none';
        }

        // Chart
        async function loadChart(marketId) {
            try {
                const resp = await fetch(`/api/chart/${marketId}`);
                const data = await resp.json();

                if (!data.history || !data.history.length) return;

                const labels = data.history.map(h => {
                    const d = new Date(h.timestamp);
                    return d.getHours() + ':00';
                });
                const prices = data.history.map(h => (h.price * 100).toFixed(1));

                if (priceChart) priceChart.destroy();

                const ctx = document.getElementById('priceChart').getContext('2d');
                const isDark = document.body.getAttribute('data-theme') === 'dark';
                const gridColor = isDark ? '#333' : '#ddd';
                const textColor = isDark ? '#888' : '#666';

                priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Price (cents)',
                            data: prices,
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { labels: { color: textColor } }
                        },
                        scales: {
                            x: { ticks: { color: textColor }, grid: { color: gridColor } },
                            y: { ticks: { color: textColor }, grid: { color: gridColor } }
                        }
                    }
                });

            } catch (e) {
                console.error('Chart error:', e);
            }
        }

        // Orderbook
        async function loadOrderbook(marketId) {
            try {
                const resp = await fetch(`/api/orderbook/${marketId}`);
                const data = await resp.json();

                const bidsHtml = (data.bids || []).slice(0, 8).map(b => `
                    <div class="orderbook-row orderbook-bid">
                        <span>${b.size}</span>
                        <span>${(b.price * 100).toFixed(0)}c</span>
                    </div>
                `).join('');

                const asksHtml = (data.asks || []).slice(0, 8).map(a => `
                    <div class="orderbook-row orderbook-ask">
                        <span>${(a.price * 100).toFixed(0)}c</span>
                        <span>${a.size}</span>
                    </div>
                `).join('');

                document.getElementById('bidsList').innerHTML = bidsHtml || '<div class="text-muted">No bids</div>';
                document.getElementById('asksList').innerHTML = asksHtml || '<div class="text-muted">No asks</div>';

            } catch (e) {
                console.error('Orderbook error:', e);
            }
        }

        // Portfolio
        async function loadPortfolio() {
            try {
                const resp = await fetch('/api/portfolio');
                const data = await resp.json();

                // Positions
                const posHtml = (data.positions || []).length ?
                    data.positions.map(p => `
                        <div class="position-item">
                            <div class="position-market">${p.asset?.substring(0, 20)}...</div>
                            <div class="position-details">${p.side} ${p.size} @ ${formatPrice(p.avgPrice)}</div>
                        </div>
                    `).join('') :
                    '<div class="empty-state">No positions</div>';

                document.getElementById('positionsList').innerHTML = posHtml;

                // Orders
                const ordHtml = (data.orders || []).length ?
                    data.orders.map(o => `
                        <div class="position-item">
                            <div class="position-market">${o.side} @ ${formatPrice(o.price)}</div>
                            <div class="position-details">Size: ${o.original_size || o.size}</div>
                        </div>
                    `).join('') :
                    '<div class="empty-state">No open orders</div>';

                document.getElementById('ordersList').innerHTML = ordHtml;

                // Balance
                if (data.balance) {
                    document.getElementById('balance').textContent = `Balance: ${data.balance}`;
                }

            } catch (e) {
                console.error('Portfolio error:', e);
            }
        }

        async function loadOrders() {
            await loadPortfolio();
        }

        async function cancelAllOrders() {
            if (!confirm('Cancel all open orders?')) return;

            try {
                await fetch('/api/cancel-all', { method: 'POST' });
                setTimeout(loadPortfolio, 1000);
            } catch (e) {
                console.error('Cancel error:', e);
            }
        }

        // Live feed via SSE
        function connectLiveFeed() {
            eventSource = new EventSource('/api/stream');

            eventSource.onopen = () => {
                document.getElementById('wsStatus').classList.add('connected');
                document.getElementById('wsLabel').textContent = 'Live';
            };

            eventSource.onerror = () => {
                document.getElementById('wsStatus').classList.remove('connected');
                document.getElementById('wsLabel').textContent = 'Reconnecting...';
            };

            eventSource.addEventListener('trade', (e) => {
                const trade = JSON.parse(e.data);
                addTradeToFeed(trade);
            });

            eventSource.addEventListener('crypto', (e) => {
                const crypto = JSON.parse(e.data);
                updateCryptoPrice(crypto);
            });
        }

        function addTradeToFeed(trade) {
            const feed = document.getElementById('tradeFeed');
            const isEmpty = feed.querySelector('.empty-state');
            if (isEmpty) feed.innerHTML = '';

            const sideClass = trade.side?.toLowerCase() === 'buy' ? 'buy' : 'sell';
            const html = `
                <div class="trade-item ${sideClass}">
                    <div class="trade-title">${trade.title || 'Market Trade'}</div>
                    <div class="trade-details">
                        <span class="trade-side-${sideClass}">${trade.side}</span>
                        <span>${trade.size} @ ${formatPrice(trade.price)}</span>
                        <span>$${(trade.usd || trade.size * trade.price).toFixed(0)}</span>
                    </div>
                </div>
            `;

            feed.insertAdjacentHTML('afterbegin', html);

            // Keep only last 30 trades
            while (feed.children.length > 30) {
                feed.lastChild.remove();
            }
        }

        function updateCryptoPrice(crypto) {
            const panel = document.getElementById('cryptoPrices');
            const existing = panel.querySelector(`[data-symbol="${crypto.symbol}"]`);

            const html = `
                <div data-symbol="${crypto.symbol}" style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border-color);">
                    <span class="mono">${crypto.symbol}</span>
                    <span class="mono">$${parseFloat(crypto.price).toLocaleString()}</span>
                </div>
            `;

            if (existing) {
                existing.outerHTML = html;
            } else {
                if (panel.querySelector('.text-muted')) {
                    panel.innerHTML = '';
                }
                panel.insertAdjacentHTML('beforeend', html);
            }
        }

        // Helpers
        function formatPrice(p) {
            if (p === null || p === undefined) return '--';
            const cents = (parseFloat(p) * 100).toFixed(0);
            return `${cents}c`;
        }

        function formatVolume(v) {
            if (!v) return '$0';
            v = parseFloat(v);
            if (v >= 1000000) return `$${(v/1000000).toFixed(1)}M`;
            if (v >= 1000) return `$${(v/1000).toFixed(0)}K`;
            return `$${v.toFixed(0)}`;
        }

        function closeModal() {
            document.getElementById('confirmModal').classList.remove('active');
        }

        function confirmTrade() {
            executeOrder();
            closeModal();
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            connectLiveFeed();
            loadTrending();
            loadPortfolio();
        });
    </script>
</body>
</html>'''


# =============================================================================
# API HANDLERS
# =============================================================================

class CockpitHandler(BaseHTTPRequestHandler):
    """HTTP handler for cockpit endpoints"""

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/" or path == "/cockpit":
            self.serve_html()
        elif path == "/api/search":
            self.handle_search(query)
        elif path == "/api/trending":
            self.handle_trending()
        elif path.startswith("/api/market/"):
            market_id = path.split("/")[-1]
            self.handle_market(market_id)
        elif path.startswith("/api/chart/"):
            market_id = path.split("/")[-1]
            self.handle_chart(market_id)
        elif path.startswith("/api/orderbook/"):
            market_id = path.split("/")[-1]
            self.handle_orderbook(market_id)
        elif path == "/api/portfolio":
            self.handle_portfolio()
        elif path == "/api/stream":
            self.handle_sse()
        elif path == "/api/status":
            self.handle_status()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode() if content_length else "{}"

        try:
            data = json.loads(body)
        except:
            data = {}

        if path == "/api/order":
            self.handle_order(data)
        elif path == "/api/cancel-all":
            self.handle_cancel_all()
        else:
            self.send_error(404)

    def serve_html(self):
        html = get_html_template()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def handle_search(self, query):
        q = query.get("q", [""])[0]
        if not q:
            self.send_json({"markets": []})
            return

        results = []

        # Search DB first
        if HAS_DB:
            try:
                db_results = search_db(q, limit=10)
                for m in db_results:
                    results.append({
                        "id": m["id"],
                        "title": m.get("title", m.get("question", "")),
                        "yes_price": m.get("yes_price", 0.5),
                        "volume": m.get("volume", 0)
                    })
            except:
                pass

        # Also search API
        if HAS_API:
            try:
                api_results = search_markets(q, limit=10)
                seen = {r["id"] for r in results}
                for m in api_results:
                    mid = str(m.get("id", ""))
                    if mid and mid not in seen:
                        prices = m.get("outcomePrices", "[0.5]")
                        if isinstance(prices, str):
                            try:
                                prices = json.loads(prices)
                            except:
                                prices = [0.5]
                        results.append({
                            "id": mid,
                            "title": m.get("question", m.get("title", "")),
                            "yes_price": float(prices[0]) if prices else 0.5,
                            "volume": m.get("volume", 0)
                        })
            except:
                pass

        self.send_json({"markets": results[:20]})

    def handle_trending(self):
        results = []

        if HAS_DB:
            try:
                trending = get_trending(limit=15)
                for m in trending:
                    results.append({
                        "id": m["id"],
                        "title": m.get("title", ""),
                        "yes_price": m.get("yes_price", 0.5),
                        "volume": m.get("volume", 0)
                    })
            except:
                pass

        self.send_json({"markets": results})

    def handle_market(self, market_id):
        if not HAS_API:
            self.send_json({"error": "API not available"}, 500)
            return

        try:
            info = get_market_info(market_id)
            self.send_json(info)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_chart(self, market_id):
        # Generate price history (mock for now, can be enhanced)
        import random

        history = []
        now = datetime.now()
        price = 0.5

        if HAS_API:
            try:
                current = get_price(market_id)
                price = current.get("yes", 0.5)
            except:
                pass

        for i in range(24, 0, -1):
            change = random.uniform(-0.03, 0.03)
            price = max(0.01, min(0.99, price + change))
            timestamp = now.replace(hour=(now.hour - i) % 24)
            history.append({
                "timestamp": timestamp.isoformat(),
                "price": round(price, 4),
                "volume": random.randint(1000, 50000)
            })

        # Add current price
        if HAS_API:
            try:
                current = get_price(market_id)
                history.append({
                    "timestamp": now.isoformat(),
                    "price": round(current.get("yes", 0.5), 4),
                    "volume": random.randint(1000, 50000)
                })
            except:
                pass

        self.send_json({"market_id": market_id, "history": history})

    def handle_orderbook(self, market_id):
        if not HAS_API:
            self.send_json({"bids": [], "asks": []})
            return

        try:
            ob = get_orderbook(market_id, "yes")
            bids = sorted([{"price": float(b.price), "size": float(b.size)}
                          for b in ob.bids], key=lambda x: x["price"], reverse=True)[:10]
            asks = sorted([{"price": float(a.price), "size": float(a.size)}
                          for a in ob.asks], key=lambda x: x["price"])[:10]
            self.send_json({"bids": bids, "asks": asks})
        except Exception as e:
            self.send_json({"bids": [], "asks": [], "error": str(e)})

    def handle_portfolio(self):
        result = {"positions": [], "orders": [], "balance": None}

        if HAS_API:
            try:
                result["positions"] = get_positions() or []
            except:
                pass

            try:
                result["orders"] = get_open_orders() or []
            except:
                pass

            try:
                bal = get_balances()
                result["balance"] = str(bal) if bal else None
            except:
                pass

        self.send_json(result)

    def handle_order(self, data):
        if not HAS_API:
            self.send_json({"error": "API not available"}, 500)
            return

        market_id = data.get("market_id")
        side = data.get("side", "BUY")
        price = float(data.get("price", 0.5))
        size = int(data.get("size", 1))
        outcome = data.get("outcome", "yes")

        try:
            result = place_order(market_id, side, price, size, outcome)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_cancel_all(self):
        if not HAS_API:
            self.send_json({"error": "API not available"}, 500)
            return

        try:
            cancel_all_orders()
            self.send_json({"status": "cancelled"})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def handle_sse(self):
        """Server-Sent Events stream"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        SSE_CLIENTS.add(self.wfile)

        try:
            while True:
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                time.sleep(15)
        except:
            SSE_CLIENTS.discard(self.wfile)

    def handle_status(self):
        self.send_json({
            "ws_connected": STATE["ws_connected"],
            "trade_count": len(STATE["latest_trades"]),
            "has_api": HAS_API,
            "has_db": HAS_DB,
            "has_rtds": HAS_RTDS
        })


# =============================================================================
# WEBSOCKET INTEGRATION
# =============================================================================

def broadcast_sse(event: str, data: dict):
    """Broadcast to all SSE clients"""
    message = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead_clients = []

    for client in SSE_CLIENTS:
        try:
            client.write(message.encode())
            client.flush()
        except:
            dead_clients.append(client)

    for client in dead_clients:
        SSE_CLIENTS.discard(client)


def on_ws_message(msg):
    """Handle WebSocket messages"""
    timestamp = datetime.now().isoformat()

    if msg.topic == "activity" and msg.type == "trades":
        trade = {
            "side": msg.payload.get("side", ""),
            "size": msg.payload.get("size", 0),
            "price": msg.payload.get("price", 0),
            "title": msg.payload.get("title", "")[:60],
            "timestamp": timestamp,
            "usd": msg.payload.get("size", 0) * msg.payload.get("price", 0)
        }

        STATE["latest_trades"].append(trade)
        if len(STATE["latest_trades"]) > MAX_TRADES:
            STATE["latest_trades"].pop(0)

        broadcast_sse("trade", trade)

    elif msg.topic == "crypto_prices":
        crypto = {
            "symbol": msg.payload.get("symbol", ""),
            "price": msg.payload.get("value", 0),
            "timestamp": timestamp
        }
        STATE["crypto_prices"][crypto["symbol"]] = crypto
        broadcast_sse("crypto", crypto)


def on_ws_connect(client):
    """Handle WebSocket connection"""
    STATE["ws_connected"] = True
    print("[WS] Connected to Polymarket")
    client.subscribe_trades()
    client.subscribe_crypto_price("BTCUSDT")
    client.subscribe_crypto_price("ETHUSDT")


def on_ws_close():
    """Handle WebSocket disconnection"""
    STATE["ws_connected"] = False
    print("[WS] Disconnected")


def start_websocket():
    """Start WebSocket client in background"""
    if not HAS_RTDS:
        print("[WS] rtds_client not available")
        return

    client = RealTimeDataClient(
        on_message=on_ws_message,
        on_connect=on_ws_connect,
        on_close=on_ws_close,
        auto_reconnect=True
    )

    try:
        client.connect(blocking=True)
    except Exception as e:
        print(f"[WS] Error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Cockpit")
    parser.add_argument("--port", type=int, default=8891, help="HTTP port")
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket")
    args = parser.parse_args()

    print("=" * 60)
    print("POLYMARKET WEB TRADING COCKPIT")
    print("=" * 60)
    print(f"Server: http://localhost:{args.port}")
    print(f"API: {'Available' if HAS_API else 'Not available'}")
    print(f"Database: {'Available' if HAS_DB else 'Not available'}")
    print(f"WebSocket: {'Available' if HAS_RTDS else 'Not available'}")
    print("")
    print("Press Ctrl+C to stop")
    print("")

    # Start WebSocket in background
    if HAS_RTDS and not args.no_ws:
        ws_thread = threading.Thread(target=start_websocket, daemon=True)
        ws_thread.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), CockpitHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
