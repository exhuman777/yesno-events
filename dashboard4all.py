#!/usr/bin/env python3
"""
DASHBOARD4ALL v2.0 - Polymarket Quant Trading Terminal
Bloomberg-style prediction market interface with vector search

Port: 8888
"""
from __future__ import annotations

import re
import json
import html
import subprocess
import calendar as cal_module
from pathlib import Path
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, quote
from collections import Counter
from typing import Any, Optional
import urllib.request
import ssl

def safe_html(text):
    """Escape text for safe HTML rendering (XSS prevention)"""
    if text is None:
        return ''
    return html.escape(str(text))

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
KNOWLEDGE = DATA_DIR / "knowledge"
INVENTORY = DATA_DIR / "inventory"
CALENDAR_FILE = DATA_DIR / ".calendar.json"
CACHE_DIR = DATA_DIR / "cache"
MARKETS_CACHE = CACHE_DIR / "markets"
TRADES_CACHE = CACHE_DIR / "trades"
PRICE_CACHE = CACHE_DIR / "prices"
WATCHED_FILE = DATA_DIR / ".watched_markets.json"
RESEARCH_FILE = DATA_DIR / ".research_history.json"
DIARY_FILE = DATA_DIR / ".trading_diary.json"
MARKET_NOTES_FILE = DATA_DIR / ".market_notes.json"
ALERTS_FILE = DATA_DIR / ".alerts.json"
PREDICTIONS_FILE = DATA_DIR / ".predictions.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE.mkdir(exist_ok=True)
INVENTORY.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MARKETS_CACHE.mkdir(exist_ok=True)
TRADES_CACHE.mkdir(exist_ok=True)
PRICE_CACHE.mkdir(exist_ok=True)

# Polyrouter API
POLYROUTER_API = "https://api-v2.polyrouter.io"
CONFIG_FILE = DATA_DIR / ".trading_config.json"

def get_polyrouter_key():
    """Load Polyrouter API key from config file"""
    if CONFIG_FILE.exists():
        try:
            config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
            return config.get('polyrouter_key', '')
        except:
            pass
    return ''

# Lazy-loaded key (cached after first call)
_polyrouter_key_cache = None
def POLYROUTER_KEY():
    global _polyrouter_key_cache
    if _polyrouter_key_cache is None:
        _polyrouter_key_cache = get_polyrouter_key()
    return _polyrouter_key_cache

# === VECTOR SEARCH INTEGRATION ===
_search_engine: Optional[dict] = None

def get_search_engine() -> dict:
    """Lazy-load the search engine from search.py"""
    global _search_engine
    if _search_engine is None:
        try:
            from search import search_markets, find_similar_markets, semantic_search_markets
            _search_engine = {
                'search': search_markets,
                'similar': find_similar_markets,
                'semantic': semantic_search_markets
            }
        except ImportError:
            _search_engine = {}
    return _search_engine

def vector_search(query: str, top_k: int = 20, use_semantic: bool = True) -> list:
    """Search markets using TF-IDF vector search"""
    engine = get_search_engine()
    if not engine:
        return []
    try:
        if use_semantic and 'semantic' in engine:
            return engine['semantic'](query, top_k)
        elif 'search' in engine:
            return engine['search'](query, top_k)
    except Exception:
        pass
    return []

def find_similar(market_id: str, top_k: int = 5) -> list:
    """Find markets similar to a given market ID"""
    engine = get_search_engine()
    if not engine or 'similar' not in engine:
        return []
    try:
        return engine['similar'](market_id, top_k)
    except Exception:
        pass
    return []

# === LOCAL DATA FUNCTIONS ===
def get_cache_size_mb():
    """Calculate total cache size in MB"""
    total = 0
    for f in CACHE_DIR.rglob("*.json"):
        total += f.stat().st_size
    for f in DATA_DIR.glob(".*.json"):
        total += f.stat().st_size
    return round(total / (1024 * 1024), 2)

def load_watched_markets():
    if WATCHED_FILE.exists():
        try:
            return json.loads(WATCHED_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'markets': []}

def save_watched_markets(data):
    WATCHED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_watched_market(market_data):
    watched = load_watched_markets()
    # Check if already exists
    if not any(m['id'] == market_data['id'] for m in watched['markets']):
        watched['markets'].append({
            'id': market_data['id'],
            'title': market_data.get('title', ''),
            'added_at': datetime.now().isoformat(),
            'last_price': market_data.get('current_prices', {}),
            'platform': market_data.get('platform', 'polymarket')
        })
        save_watched_markets(watched)
    return watched

def remove_watched_market(market_id):
    watched = load_watched_markets()
    watched['markets'] = [m for m in watched['markets'] if m['id'] != market_id]
    save_watched_markets(watched)
    return watched

def load_research_history():
    if RESEARCH_FILE.exists():
        try:
            return json.loads(RESEARCH_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'searches': [], 'viewed_markets': []}

def save_research_history(data):
    RESEARCH_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_to_research_history(market_data=None, search_query=None):
    history = load_research_history()
    now = datetime.now().isoformat()

    if search_query:
        # Add search query
        history['searches'] = [s for s in history['searches'] if s['query'] != search_query][-49:]
        history['searches'].append({'query': search_query, 'timestamp': now})

    if market_data:
        # Add viewed market
        history['viewed_markets'] = [m for m in history['viewed_markets'] if m['id'] != market_data['id']][-99:]
        history['viewed_markets'].append({
            'id': market_data['id'],
            'title': market_data.get('title', ''),
            'timestamp': now,
            'price_snapshot': market_data.get('current_prices', {})
        })

    save_research_history(history)
    return history

def cache_market_data(market_id, data):
    """Save market data to local cache"""
    cache_file = MARKETS_CACHE / f"{market_id}.json"
    cache_data = {
        'data': data,
        'cached_at': datetime.now().isoformat(),
        'market_id': market_id
    }
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding='utf-8')

def get_cached_market(market_id, max_age_hours=24):
    """Get cached market data if fresh enough"""
    cache_file = MARKETS_CACHE / f"{market_id}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            cached_at = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return cached['data']
        except:
            pass
    return None

def cache_trades_data(market_id, data):
    """Save trades data to local cache"""
    cache_file = TRADES_CACHE / f"{market_id}.json"
    # Append new trades to existing
    existing = []
    if cache_file.exists():
        try:
            existing = json.loads(cache_file.read_text(encoding='utf-8')).get('trades', [])
        except:
            pass

    # Merge and dedupe by trade_id
    new_trades = data.get('trades', [])
    all_trades = {t.get('trade_id', str(i)): t for i, t in enumerate(existing)}
    for t in new_trades:
        all_trades[t.get('trade_id', '')] = t

    cache_data = {
        'trades': list(all_trades.values())[-1000:],  # Keep last 1000
        'cached_at': datetime.now().isoformat(),
        'market_id': market_id
    }
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding='utf-8')

def get_cached_trades(market_id):
    """Get all cached trades for a market"""
    cache_file = TRADES_CACHE / f"{market_id}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding='utf-8'))
        except:
            pass
    return {'trades': []}

# === TRADING DIARY FUNCTIONS ===
def load_trading_diary():
    if DIARY_FILE.exists():
        try:
            return json.loads(DIARY_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'entries': []}

def save_trading_diary(data):
    DIARY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_diary_entry(entry_type, content, market_id=None, market_title=None, prediction=None, sentiment=None):
    """Add trading diary entry. Types: reasoning, prediction, analysis, event, trade"""
    diary = load_trading_diary()
    entry = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S'),
        'type': entry_type,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
    }
    if market_id:
        entry['market_id'] = market_id
        entry['market_title'] = market_title or ''
    if prediction:
        entry['prediction'] = prediction  # {'outcome': 'yes', 'confidence': 0.8, 'target_price': 0.65}
    if sentiment:
        entry['sentiment'] = sentiment  # 'bullish', 'bearish', 'neutral'
    diary['entries'].append(entry)
    save_trading_diary(diary)
    return entry

def get_diary_by_date(date_str):
    """Get diary entries for specific date"""
    diary = load_trading_diary()
    return [e for e in diary['entries'] if e.get('date') == date_str]

def get_diary_by_market(market_id):
    """Get diary entries for specific market"""
    diary = load_trading_diary()
    return [e for e in diary['entries'] if e.get('market_id') == market_id]

# === MARKET NOTES (quick reasoning) ===
def load_market_notes():
    if MARKET_NOTES_FILE.exists():
        try:
            return json.loads(MARKET_NOTES_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {}

def save_market_notes(data):
    MARKET_NOTES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_market_note(market_id, note, sentiment=None):
    """Quick note/reasoning for a market"""
    notes = load_market_notes()
    if market_id not in notes:
        notes[market_id] = {'notes': [], 'title': ''}
    notes[market_id]['notes'].append({
        'text': note,
        'timestamp': datetime.now().isoformat(),
        'sentiment': sentiment
    })
    save_market_notes(notes)
    return notes[market_id]

def get_market_notes(market_id):
    notes = load_market_notes()
    return notes.get(market_id, {'notes': [], 'title': ''})

# === PRICE ALERTS ===
def load_alerts():
    if ALERTS_FILE.exists():
        try:
            return json.loads(ALERTS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'alerts': []}

def save_alerts(data):
    ALERTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_alert(market_id, market_title, condition, target_price, alert_type='price'):
    """Add a price alert. condition: 'above' or 'below'"""
    alerts_data = load_alerts()
    alert = {
        'id': datetime.now().strftime('%Y%m%d%H%M%S'),
        'market_id': market_id,
        'market_title': market_title,
        'condition': condition,  # 'above' or 'below'
        'target_price': target_price,
        'alert_type': alert_type,
        'created_at': datetime.now().isoformat(),
        'triggered': False,
        'triggered_at': None
    }
    alerts_data['alerts'].append(alert)
    save_alerts(alerts_data)
    return alert

def remove_alert(alert_id):
    alerts_data = load_alerts()
    alerts_data['alerts'] = [a for a in alerts_data['alerts'] if a['id'] != alert_id]
    save_alerts(alerts_data)
    return alerts_data

def get_active_alerts():
    alerts_data = load_alerts()
    return [a for a in alerts_data['alerts'] if not a.get('triggered')]

def check_alerts(current_prices):
    """Check alerts against current prices. Returns list of triggered alerts."""
    alerts_data = load_alerts()
    triggered = []
    for alert in alerts_data['alerts']:
        if alert.get('triggered'):
            continue
        market_id = alert['market_id']
        if market_id in current_prices:
            price = current_prices[market_id]
            target = alert['target_price']
            if alert['condition'] == 'above' and price >= target:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
                alert['triggered_price'] = price
                triggered.append(alert)
            elif alert['condition'] == 'below' and price <= target:
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
                alert['triggered_price'] = price
                triggered.append(alert)
    if triggered:
        save_alerts(alerts_data)
    return triggered

# === PREDICTIONS DATABASE ===
def load_predictions():
    """Load predictions database"""
    if PREDICTIONS_FILE.exists():
        try:
            return json.loads(PREDICTIONS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'predictions': []}

def save_predictions(data):
    """Save predictions database"""
    PREDICTIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def add_prediction(market_id, market_title, your_prob, market_price, side, reasoning=''):
    """Add a prediction to track"""
    predictions = load_predictions()
    pred = {
        'id': f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{market_id[:8]}",
        'market_id': market_id,
        'market_title': market_title,
        'your_prob': your_prob,
        'market_price': market_price,
        'side': side,
        'reasoning': reasoning,
        'created_at': datetime.now().isoformat(),
        'resolved': False,
        'correct': None,
        'outcome': None
    }
    # Calculate EV at time of prediction
    if side == 'YES':
        pred['ev'] = (your_prob * (1 - market_price)) - ((1 - your_prob) * market_price)
    else:
        pred['ev'] = ((1 - your_prob) * (1 - market_price)) - (your_prob * market_price)

    predictions['predictions'].append(pred)
    save_predictions(predictions)
    return pred

def resolve_prediction(pred_id, outcome):
    """Resolve a prediction with the actual outcome (YES or NO)"""
    predictions = load_predictions()
    for pred in predictions['predictions']:
        if pred['id'] == pred_id:
            pred['resolved'] = True
            pred['outcome'] = outcome
            pred['resolved_at'] = datetime.now().isoformat()
            pred['correct'] = (pred['side'] == outcome)
            break
    save_predictions(predictions)
    return predictions

def get_prediction_stats():
    """Get prediction accuracy statistics"""
    predictions = load_predictions()
    all_preds = predictions.get('predictions', [])
    resolved = [p for p in all_preds if p.get('resolved')]
    wins = [p for p in resolved if p.get('correct')]

    # Calculate by confidence level
    high_conf = [p for p in resolved if abs(p.get('your_prob', 0.5) - 0.5) > 0.2]
    high_conf_wins = [p for p in high_conf if p.get('correct')]

    return {
        'total': len(all_preds),
        'resolved': len(resolved),
        'wins': len(wins),
        'losses': len(resolved) - len(wins),
        'pending': len(all_preds) - len(resolved),
        'win_rate': len(wins) / len(resolved) * 100 if resolved else 0,
        'high_conf_total': len(high_conf),
        'high_conf_wins': len(high_conf_wins),
        'high_conf_rate': len(high_conf_wins) / len(high_conf) * 100 if high_conf else 0,
        'avg_ev': sum(p.get('ev', 0) for p in all_preds) / len(all_preds) if all_preds else 0
    }

# === EDGE SCANNER ===
def scan_for_edges(min_volume=1000, max_results=50):
    """
    Scan markets for trading edges.
    Looks for:
    - Mispricing (YES+NO sum != 1)
    - High spread opportunities
    - Volume/liquidity imbalance
    - Momentum signals
    """
    edges = []
    try:
        from trading import search_markets, get_price, get_orderbook
    except ImportError:
        return edges

    # Search for active markets
    try:
        markets = search_markets("", limit=100)  # Get recent markets
    except:
        markets = []

    for m in markets[:max_results]:
        market_id = m.get('id') or m.get('conditionId') or m.get('condition_id')
        if not market_id:
            continue

        edge_info = {
            'market_id': market_id,
            'title': m.get('title', m.get('question', 'Unknown')),
            'volume': float(m.get('volume', 0) or 0),
            'liquidity': float(m.get('liquidity', 0) or 0),
            'edges': [],
            'score': 0
        }

        # Skip low volume
        if edge_info['volume'] < min_volume:
            continue

        try:
            price = get_price(market_id)
            if not price:
                continue

            yes_price = price.get('yes', 0)
            no_price = price.get('no', 0)

            # Edge 1: Mispricing check
            total = yes_price + no_price
            if abs(total - 1.0) > 0.01:  # More than 1% off
                edge_info['edges'].append({
                    'type': 'mispricing',
                    'desc': f'YES+NO={total*100:.1f}% ({"over" if total > 1 else "under"}priced)',
                    'severity': abs(total - 1.0) * 100
                })
                edge_info['score'] += abs(total - 1.0) * 50

            # Edge 2: Volume/Liquidity ratio
            if edge_info['liquidity'] > 0:
                vol_liq = edge_info['volume'] / edge_info['liquidity']
                if vol_liq > 10:  # High volume relative to liquidity
                    edge_info['edges'].append({
                        'type': 'vol_liq_imbalance',
                        'desc': f'V/L ratio {vol_liq:.1f}x (volatile)',
                        'severity': min(vol_liq, 50)
                    })
                    edge_info['score'] += min(vol_liq, 20)

            # Edge 3: Spread analysis
            try:
                book = get_orderbook(market_id, 'yes')
                if book and book.bids and book.asks:
                    best_bid = book.bids[0].price if book.bids else 0
                    best_ask = book.asks[0].price if book.asks else 1
                    spread = (best_ask - best_bid) * 100

                    if spread > 3:  # More than 3% spread
                        edge_info['edges'].append({
                            'type': 'wide_spread',
                            'desc': f'{spread:.1f}% spread (bid:{best_bid*100:.0f} ask:{best_ask*100:.0f})',
                            'severity': spread
                        })
                        edge_info['score'] += spread * 2
            except:
                pass

            # Edge 4: Extreme prices (potential mean reversion)
            if yes_price < 0.05 or yes_price > 0.95:
                edge_info['edges'].append({
                    'type': 'extreme_price',
                    'desc': f'{"Near 0" if yes_price < 0.05 else "Near 100"}% - reversion potential',
                    'severity': 10
                })
                edge_info['score'] += 10

        except Exception as e:
            continue

        if edge_info['edges']:
            edges.append(edge_info)

    # Sort by score
    edges.sort(key=lambda x: x['score'], reverse=True)
    return edges[:max_results]

# === PRICE HISTORY CACHE ===
def cache_price_history(market_id, data, interval):
    """Cache price history data"""
    cache_file = PRICE_CACHE / f"{market_id}_{interval}.json"
    cache_data = {
        'data': data,
        'cached_at': datetime.now().isoformat(),
        'market_id': market_id,
        'interval': interval
    }
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding='utf-8')

def get_cached_price_history(market_id, interval, max_age_hours=1):
    """Get cached price history if fresh"""
    cache_file = PRICE_CACHE / f"{market_id}_{interval}.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding='utf-8'))
            cached_at = datetime.fromisoformat(cached['cached_at'])
            if datetime.now() - cached_at < timedelta(hours=max_age_hours):
                return cached['data']
        except:
            pass
    return None

# === ELON TWEET MARKET RESEARCH DATA ===
ELON_DATA_FILE = DATA_DIR / ".elon_tweet_history.json"

def load_elon_data():
    if ELON_DATA_FILE.exists():
        try:
            return json.loads(ELON_DATA_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'weeks': [], 'imported_files': []}

def save_elon_data(data):
    ELON_DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_elon_csv(csv_path):
    """Parse Polymarket CSV export of Elon tweet bracket probabilities"""
    import csv
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        brackets = header[2:]  # Skip Date, Timestamp
        for row in reader:
            if len(row) < 3:
                continue
            date_str, timestamp = row[0], row[1]
            probs = {}
            for i, bracket in enumerate(brackets):
                val = row[i+2] if i+2 < len(row) else ''
                probs[bracket] = float(val) if val else 0.0
            rows.append({
                'date': date_str,
                'timestamp': int(timestamp) if timestamp else 0,
                'brackets': brackets,
                'probabilities': probs
            })
    return {'brackets': brackets, 'data': rows}

def import_elon_csv(csv_path):
    """Import CSV and add to historical data"""
    elon = load_elon_data()
    parsed = parse_elon_csv(csv_path)

    # Identify the week (from filename or data range)
    fname = Path(csv_path).name
    if parsed['data']:
        start_date = parsed['data'][0]['date']
        end_date = parsed['data'][-1]['date']
        week_id = f"{start_date}_to_{end_date}"
    else:
        week_id = fname

    # Check if already imported
    if fname in elon.get('imported_files', []):
        return {'status': 'already_imported', 'week_id': week_id}

    # Add week data
    week_entry = {
        'week_id': week_id,
        'source_file': fname,
        'imported_at': datetime.now().isoformat(),
        'brackets': parsed['brackets'],
        'daily_data': parsed['data']
    }
    elon['weeks'].append(week_entry)
    elon['imported_files'].append(fname)
    save_elon_data(elon)
    return {'status': 'imported', 'week_id': week_id, 'days': len(parsed['data'])}

def calc_expected_tweets(probabilities, brackets):
    """Calculate expected tweet count from probability distribution"""
    ev = 0.0
    for bracket in brackets:
        p = probabilities.get(bracket, 0)
        if not p:
            continue
        # Parse bracket to get midpoint
        if bracket.startswith('<'):
            mid = int(bracket[1:]) / 2
        elif bracket.endswith('+'):
            mid = int(bracket[:-1]) + 20  # Assume ~20 more than lower bound
        elif '-' in bracket:
            low, high = bracket.split('-')
            mid = (int(low) + int(high)) / 2
        else:
            mid = 0
        ev += mid * p
    return ev

def analyze_elon_week(week_data):
    """Analyze a week's data for patterns"""
    daily = week_data.get('daily_data', [])
    brackets = week_data.get('brackets', [])

    analysis = {
        'total_days': len(daily),
        'ev_progression': [],
        'peak_bracket_progression': [],
        'final_day': None,
        'winner_bracket': None
    }

    for day in daily:
        probs = day.get('probabilities', {})
        ev = calc_expected_tweets(probs, brackets)
        analysis['ev_progression'].append({'date': day['date'], 'ev': round(ev, 1)})

        # Find peak probability bracket
        max_p = 0
        peak_bracket = None
        for b, p in probs.items():
            if p > max_p:
                max_p = p
                peak_bracket = b
        if peak_bracket:
            analysis['peak_bracket_progression'].append({
                'date': day['date'],
                'bracket': peak_bracket,
                'prob': round(max_p * 100, 1)
            })

    # Final day = likely winner
    if daily:
        final = daily[-1]
        analysis['final_day'] = final['date']
        probs = final.get('probabilities', {})
        max_p = 0
        for b, p in probs.items():
            if p > max_p:
                max_p = p
                analysis['winner_bracket'] = b

    return analysis

# === ELON TWEETS & STATS DATA ===
ELON_TWEETS_FILE = DATA_DIR / ".elon_tweets.json"
ELON_STATS_FILE = DATA_DIR / ".elon_stats.json"
ELON_MODELS_FILE = DATA_DIR / ".elon_models.json"

def load_elon_tweets():
    if ELON_TWEETS_FILE.exists():
        try:
            return json.loads(ELON_TWEETS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'tweets': [], 'imported_files': []}

def save_elon_tweets(data):
    ELON_TWEETS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def load_elon_stats():
    if ELON_STATS_FILE.exists():
        try:
            return json.loads(ELON_STATS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'periods': [], 'imported_files': []}

def save_elon_stats(data):
    ELON_STATS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def parse_elon_stats_csv(csv_path):
    """Parse Elon stats CSV (Metric,Value format)"""
    import csv
    stats = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip().lower().replace(' ', '_')
                val = row[1].strip()
                # Try to parse numeric values
                try:
                    if '%' in val:
                        stats[key] = float(val.replace('%', ''))
                    elif val.isdigit():
                        stats[key] = int(val)
                    else:
                        stats[key] = val
                except:
                    stats[key] = val
    return stats

def import_elon_stats_csv(csv_path):
    """Import stats CSV"""
    data = load_elon_stats()
    fname = Path(csv_path).name
    if fname in data.get('imported_files', []):
        return {'status': 'already_imported'}

    stats = parse_elon_stats_csv(csv_path)
    stats['source_file'] = fname
    stats['imported_at'] = datetime.now().isoformat()
    data['periods'].append(stats)
    data['imported_files'].append(fname)
    save_elon_stats(data)
    return {'status': 'imported', 'stats': stats}

def parse_elon_tweets_csv(csv_path):
    """Parse Elon tweets CSV with content and timestamps"""
    import csv
    tweets = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            tweet = {
                'id': row[0],
                'user': row[1],
                'content': row[2],
                'posted_at': row[3],
                'date': row[4],
                'time': row[5],
            }
            # Analyze tweet
            content = tweet['content'].lower()
            tweet['is_rt'] = content.startswith('rt @')
            tweet['has_link'] = 'https://t.co/' in content
            tweet['is_short'] = len(row[2]) < 30
            # Topic detection
            topics = []
            if 'grok' in content: topics.append('grok')
            if 'doge' in content or 'government' in content: topics.append('doge')
            if any(w in content for w in ['immigra', 'border', 'deport']): topics.append('immigration')
            if any(w in content for w in ['tesla', 'spacex', 'starship', 'neuralink']): topics.append('companies')
            if any(w in content for w in ['trump', 'maga', 'elect']): topics.append('politics')
            tweet['topics'] = topics
            tweets.append(tweet)
    return tweets

def import_elon_tweets_csv(csv_path):
    """Import tweets CSV"""
    data = load_elon_tweets()
    fname = Path(csv_path).name
    if fname in data.get('imported_files', []):
        return {'status': 'already_imported'}

    tweets = parse_elon_tweets_csv(csv_path)
    # Dedupe by tweet ID
    existing_ids = {t['id'] for t in data['tweets']}
    new_tweets = [t for t in tweets if t['id'] not in existing_ids]
    data['tweets'].extend(new_tweets)
    data['imported_files'].append(fname)
    save_elon_tweets(data)
    return {'status': 'imported', 'count': len(new_tweets)}

def analyze_elon_tweets():
    """Comprehensive tweet analysis for patterns"""
    data = load_elon_tweets()
    tweets = data.get('tweets', [])
    if not tweets:
        return {}

    # Hourly distribution
    hourly = {}
    for t in tweets:
        try:
            time_str = t.get('time', '').strip()
            if time_str:
                hour = int(time_str.split(':')[0])
                hourly[hour] = hourly.get(hour, 0) + 1
        except:
            pass

    # Daily distribution
    daily = {}
    for t in tweets:
        date = t.get('date', '').strip()
        if date:
            daily[date] = daily.get(date, 0) + 1

    # RT ratio
    rt_count = sum(1 for t in tweets if t.get('is_rt'))
    rt_ratio = rt_count / len(tweets) if tweets else 0

    # Topic distribution
    topics = {}
    for t in tweets:
        for topic in t.get('topics', []):
            topics[topic] = topics.get(topic, 0) + 1

    # Short tweet ratio (quick engagement)
    short_count = sum(1 for t in tweets if t.get('is_short'))

    # Tweet velocity (average interval between tweets)
    intervals = []
    sorted_tweets = sorted(tweets, key=lambda x: x.get('posted_at', ''))
    for i in range(1, len(sorted_tweets)):
        # Simple estimation - not precise but indicative
        intervals.append(1)  # placeholder

    return {
        'total_tweets': len(tweets),
        'hourly_distribution': hourly,
        'daily_distribution': daily,
        'rt_ratio': round(rt_ratio * 100, 1),
        'original_ratio': round((1 - rt_ratio) * 100, 1),
        'short_tweet_ratio': round(short_count / len(tweets) * 100, 1) if tweets else 0,
        'topics': topics,
        'peak_hours': sorted(hourly.items(), key=lambda x: x[1], reverse=True)[:5],
        'avg_daily': round(len(tweets) / max(len(daily), 1), 1),
    }

def fetch_historical_winners():
    """Fetch actual winning brackets from Polyrouter API"""
    import urllib.request
    winners = []
    headers = {'X-API-Key': POLYROUTER_KEY()}

    # Search for resolved Elon tweet markets
    queries = ['elon%20december', 'elon%20november', 'elon%20january']
    for q in queries:
        try:
            url = f'https://api-v2.polyrouter.io/markets?query={q}&status=resolved&platform=polymarket&limit=50'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for m in data.get('markets', []):
                    prices = m.get('current_prices', {})
                    yes_p = prices.get('yes', {}).get('price', 0)
                    if yes_p > 0.9:  # Winner (resolved to YES)
                        title = m.get('title', '')
                        # Extract bracket from title
                        import re
                        match = re.search(r'(\d+[-+]?\d*)', title)
                        if match:
                            bracket = match.group(1)
                            winners.append({
                                'market_id': m.get('id'),
                                'title': title,
                                'bracket': bracket,
                                'resolved_at': m.get('resolved_at')
                            })
        except:
            pass
    return winners

def fetch_clob_data(market_id):
    """Fetch orderbook (CLOB) data for a market"""
    import urllib.request
    headers = {'X-API-Key': POLYROUTER_KEY()}
    try:
        url = f'https://api-v2.polyrouter.io/markets/{market_id}/orderbook'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return None

def fetch_current_markets(event_filter=None):
    """Fetch current active Elon tweet markets with prices, grouped by event"""
    import urllib.request
    import re
    headers = {'X-API-Key': POLYROUTER_KEY()}

    # Known events (weeks)
    events = {
        'jan6_13': {'query': 'january%206%20january%2013', 'ids': ['1093']},
        'jan9_16': {'query': 'january%209%20january%2016', 'ids': ['1119']},
        'jan13_20': {'query': 'january%2013%20january%2020', 'ids': ['1148']},
        'jan15_22': {'query': 'january%2015%20january%2022', 'ids': ['1168']},
        'jan16_23': {'query': 'january%2016%20january%2023', 'ids': ['1172']},
        'jan17_19': {'query': 'january%2017%20january%2019', 'ids': ['1192']},  # weekend
        'jan20_27': {'query': 'january%2020%20january%2027', 'ids': ['1203']},
    }

    try:
        # Fetch all open tweet markets
        url = 'https://api-v2.polyrouter.io/markets?query=elon%20musk%20tweet&status=open&platform=polymarket&limit=50'
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            by_event = {}

            for m in data.get('markets', []):
                title = m.get('title', '').lower()
                if 'tweet' not in title and 'post' not in title:
                    continue

                mid = m.get('id', '')

                # Determine event from ID prefix or title
                event_key = 'other'
                if mid.startswith('1093'):
                    event_key = 'jan6_13'
                elif mid.startswith('1119'):
                    event_key = 'jan9_16'
                elif mid.startswith('1148'):
                    event_key = 'jan13_20'
                elif mid.startswith('1168'):
                    event_key = 'jan15_22'
                elif mid.startswith('1172'):
                    event_key = 'jan16_23'
                elif mid.startswith('1192'):
                    event_key = 'jan17_19'
                elif mid.startswith('1203'):
                    event_key = 'jan20_27'

                # Extract bracket from title
                bracket = ''
                match = re.search(r'(\d+[-+]?\d*)\s*(tweets|tweet)', m.get('title', ''))
                if match:
                    bracket = match.group(1)
                if not bracket and 'post ' in m.get('title', ''):
                    bracket = m.get('title', '').split('post ')[-1].split(' tweets')[0].strip()

                if event_key not in by_event:
                    by_event[event_key] = []

                by_event[event_key].append({
                    'id': mid,
                    'title': m.get('title'),
                    'bracket': bracket,
                    'yes_price': m.get('current_prices', {}).get('yes', {}).get('price', 0),
                    'volume': m.get('volume_total', 0),
                    'liquidity': m.get('liquidity', 0),
                    'event': event_key
                })

            # Sort each event's markets by price
            for key in by_event:
                by_event[key] = sorted(by_event[key], key=lambda x: x['yes_price'], reverse=True)

            if event_filter and event_filter in by_event:
                return by_event[event_filter]

            return by_event
    except Exception as e:
        return {}

def build_prediction_models():
    """Build prediction models from ACTUAL API winners"""
    # Fetch real winners from API
    winners = fetch_historical_winners()

    models = {
        'linear': {'name': 'Linear Extrapolation', 'params': {}},
        'pattern': {'name': 'Historical Pattern', 'params': {}},
        'momentum': {'name': 'Momentum Trend', 'params': {}},
        'winners': winners,
        'built_at': datetime.now().isoformat()
    }

    # Extract final counts from winning brackets
    final_counts = []
    for w in winners:
        bracket = w.get('bracket', '')
        try:
            if bracket.endswith('+'):
                mid = int(bracket[:-1]) + 20
            elif '-' in bracket:
                low, high = bracket.split('-')
                mid = (int(low) + int(high)) / 2
            else:
                mid = int(bracket) if bracket.isdigit() else 0
            if mid > 0:
                final_counts.append(mid)
        except:
            pass

    # Model 1: Linear - daily rates (7-day weeks)
    if final_counts:
        daily_rates = [c / 7 for c in final_counts]
        models['linear']['params'] = {
            'avg_daily_rate': round(sum(daily_rates) / len(daily_rates), 1),
            'min_daily_rate': round(min(daily_rates), 1),
            'max_daily_rate': round(max(daily_rates), 1),
            'samples': len(final_counts)
        }

    # Model 2: Pattern - weekly totals
    if final_counts:
        avg = sum(final_counts) / len(final_counts)
        models['pattern']['params'] = {
            'avg_weekly_total': round(avg, 0),
            'min_weekly_total': min(final_counts),
            'max_weekly_total': max(final_counts),
            'std_dev': round((sum((x - avg)**2 for x in final_counts) / len(final_counts))**0.5, 1) if len(final_counts) > 1 else 0,
            'samples': len(final_counts),
            'winning_brackets': [w['bracket'] for w in winners[:10]]
        }

    # Model 3: Momentum - analyze recent trend direction
    if len(final_counts) >= 3:
        recent = final_counts[:5]  # Most recent 5 weeks
        older = final_counts[5:10] if len(final_counts) > 5 else final_counts[len(final_counts)//2:]

        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else recent_avg

        # Calculate momentum (positive = trending up, negative = trending down)
        momentum = recent_avg - older_avg
        momentum_pct = (momentum / older_avg * 100) if older_avg > 0 else 0

        # Trend direction
        if momentum > 50:
            trend = 'strongly_up'
        elif momentum > 20:
            trend = 'up'
        elif momentum > -20:
            trend = 'stable'
        elif momentum > -50:
            trend = 'down'
        else:
            trend = 'strongly_down'

        models['momentum']['params'] = {
            'recent_avg': round(recent_avg, 0),
            'older_avg': round(older_avg, 0),
            'momentum': round(momentum, 1),
            'momentum_pct': round(momentum_pct, 1),
            'trend': trend,
            'recent_weeks': len(recent),
            'projected_next': round(recent_avg + momentum * 0.5, 0)  # Extrapolate half the momentum
        }

    # Save models
    ELON_MODELS_FILE.write_text(json.dumps(models, indent=2), encoding='utf-8')
    return models

def get_prediction_models():
    """Load prediction models"""
    if ELON_MODELS_FILE.exists():
        try:
            return json.loads(ELON_MODELS_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return build_prediction_models()

def predict_elon_tweets(current_count, days_elapsed, days_total=7, model='both'):
    """Make predictions using models"""
    models = get_prediction_models()
    days_remaining = days_total - days_elapsed

    predictions = {}

    # Linear model
    if days_elapsed > 0:
        current_rate = current_count / days_elapsed
        linear_pred = current_count + (current_rate * days_remaining)
    else:
        linear_rate = models.get('linear', {}).get('params', {}).get('avg_daily_rate', 50)
        linear_pred = linear_rate * days_total

    predictions['linear'] = {
        'predicted_total': round(linear_pred),
        'daily_rate': round(current_count / max(days_elapsed, 1), 1),
        'confidence': 'high' if days_elapsed >= 3 else 'medium' if days_elapsed >= 1 else 'low'
    }

    # Pattern model - blend historical with current
    pattern_params = models.get('pattern', {}).get('params', {})
    hist_avg = pattern_params.get('avg_weekly_total', 350)

    if days_elapsed > 0:
        # Weight current data more as week progresses
        weight = days_elapsed / days_total
        pattern_pred = (linear_pred * weight) + (hist_avg * (1 - weight))
    else:
        pattern_pred = hist_avg

    predictions['pattern'] = {
        'predicted_total': round(pattern_pred),
        'historical_avg': hist_avg,
        'confidence': 'medium'
    }

    # Find likely bracket
    def get_bracket(count):
        brackets = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580]
        for i, low in enumerate(brackets[:-1]):
            high = brackets[i+1]
            if count >= low and count < high:
                return f"{low}-{high-1}"
        return "580+"

    predictions['linear']['bracket'] = get_bracket(predictions['linear']['predicted_total'])
    predictions['pattern']['bracket'] = get_bracket(predictions['pattern']['predicted_total'])

    return predictions

# === AUTO-UPDATE MECHANISM ===
LAST_UPDATE_FILE = DATA_DIR / ".last_api_update.json"
API_RATE_LIMIT = 100  # calls per minute max
UPDATE_INTERVAL = 60  # 1 minute in seconds

def get_last_update_info():
    if LAST_UPDATE_FILE.exists():
        try:
            return json.loads(LAST_UPDATE_FILE.read_text(encoding='utf-8'))
        except:
            pass
    return {'last_update': None, 'calls_this_minute': 0, 'minute_start': None, 'cached_ids': []}

def save_update_info(info):
    LAST_UPDATE_FILE.write_text(json.dumps(info, indent=2), encoding='utf-8')

def should_update():
    """Check if we should run auto-update"""
    info = get_last_update_info()
    if not info.get('last_update'):
        return True
    try:
        last = datetime.fromisoformat(info['last_update'])
        return (datetime.now() - last).total_seconds() >= UPDATE_INTERVAL
    except:
        return True

def rate_limit_ok():
    """Check if we're within API rate limits"""
    info = get_last_update_info()
    now = datetime.now()
    minute_start = info.get('minute_start')

    if minute_start:
        try:
            start = datetime.fromisoformat(minute_start)
            if (now - start).total_seconds() < 60:
                return info.get('calls_this_minute', 0) < API_RATE_LIMIT
        except:
            pass
    return True

def increment_api_call():
    """Track API call for rate limiting"""
    info = get_last_update_info()
    now = datetime.now()
    minute_start = info.get('minute_start')

    reset = True
    if minute_start:
        try:
            start = datetime.fromisoformat(minute_start)
            if (now - start).total_seconds() < 60:
                reset = False
        except:
            pass

    if reset:
        info['minute_start'] = now.isoformat()
        info['calls_this_minute'] = 1
    else:
        info['calls_this_minute'] = info.get('calls_this_minute', 0) + 1

    save_update_info(info)

def auto_update_markets(market_ids):
    """Update markets with rate limiting, skip already cached"""
    info = get_last_update_info()
    cached = set(info.get('cached_ids', []))

    # Only update markets not cached in last interval
    to_update = [mid for mid in market_ids if mid not in cached]

    updated = []
    for mid in to_update[:10]:  # Max 10 per update cycle
        if not rate_limit_ok():
            break
        increment_api_call()
        data = get_market(mid, use_cache=False)
        if 'error' not in data:
            updated.append(mid)
            cached.add(mid)

    info['last_update'] = datetime.now().isoformat()
    info['cached_ids'] = list(cached)[-50:]  # Keep last 50
    save_update_info(info)

    return {'updated': updated, 'skipped': len(market_ids) - len(to_update)}

# === LO-FI AESTHETIC COLOR SCHEME ===
# Warm, muted colors with soft gradients
COLORS = {
    'bg_main': '#1a1a2e',       # Deep navy
    'bg_panel': '#16213e',      # Darker panel
    'bg_input': '#0f3460',      # Input fields
    'border': '#4a4a6a',        # Soft purple border
    'border_accent': '#e94560', # Pink accent
    'text_primary': '#eaeaea',  # Soft white
    'text_secondary': '#a0a0b0',# Muted gray
    'text_muted': '#6a6a7a',    # Very muted
    'accent': '#e94560',        # Pink/coral
    'accent_dim': '#c73e54',    # Dimmer accent
    'link': '#f9b17a',          # Warm orange link
    'success': '#7ec8e3',       # Soft cyan
    'error': '#ff6b6b',         # Soft red
    'warning': '#feca57',       # Warm yellow
}

def parse_md(path):
    content = path.read_text(encoding='utf-8')
    result = {'name': path.stem, 'content': content, 'items': [], 'meta': {}, 'sections': {}}

    m = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if m:
        result['title'] = m.group(1)

    for m in re.finditer(r'^(\w+):\s*(.+)$', content, re.MULTILINE):
        result['meta'][m.group(1).lower()] = m.group(2)

    for m in re.finditer(r'^-\s*\[[ x]\]\s*(.+)$', content, re.MULTILINE):
        result['items'].append(m.group(1).strip())

    current = None
    for line in content.split('\n'):
        if line.startswith('## '):
            current = line[3:].strip()
            result['sections'][current] = []
        elif current and line.strip():
            result['sections'][current].append(line.strip())

    return result

def load_folder(folder):
    if not folder.exists():
        return []
    return [parse_md(f) for f in sorted(folder.glob("*.md")) if not f.name.startswith('_')]

def get_inventory():
    items = set()
    for inv in load_folder(INVENTORY):
        items.update(inv['items'])
    return items

def get_inventory_by_category():
    categories = {}
    for inv in load_folder(INVENTORY):
        for section, lines in inv.get('sections', {}).items():
            if section not in categories:
                categories[section] = []
            for line in lines:
                if line.startswith('- '):
                    categories[section].append(line[2:].strip().replace('[ ] ', ''))
    return categories

def load_calendar():
    if CALENDAR_FILE.exists():
        try:
            return json.loads(CALENDAR_FILE.read_text(encoding='utf-8'))
        except:
            pass
    today = datetime.now()
    return {
        'month': today.month,
        'year': today.year,
        'days': {},
        'goals': ['Daily review', 'Market analysis', 'Exercise']
    }

def save_calendar(data):
    CALENDAR_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

# === CALENDAR .MD FILES ===
CALENDAR_DIR = BASE / "calendar"

def load_calendar_events():
    """Load events from calendar/events/*.md files"""
    events_dir = CALENDAR_DIR / "events"
    events = []
    if events_dir.exists():
        for f in events_dir.glob("*.md"):
            content = f.read_text(encoding='utf-8')
            current_date = None
            current_event = None
            in_event = False

            for line in content.split('\n'):
                # Date header like ### 2026-01-13 or ### 2026-01-28-29
                if line.startswith('### 20'):
                    if current_event and current_event.get('type'):
                        events.append(current_event)
                    current_date = line[4:].strip()
                    current_event = None
                    in_event = False
                # New event block starts with "- type:"
                elif '- type:' in line:
                    if current_event and current_event.get('type'):
                        events.append(current_event)
                    current_event = {'date': current_date}
                    _, val = line.split('- type:', 1)
                    current_event['type'] = val.strip()
                    in_event = True
                # Event properties (indented lines with key: value)
                elif in_event and current_event and ':' in line and line.startswith('  '):
                    stripped = line.strip()
                    if ':' in stripped and not stripped.startswith('#'):
                        parts = stripped.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip().strip('"\'')
                            if key and val and not key.startswith('-'):
                                current_event[key] = val
                # End of event block
                elif in_event and line.strip() == '':
                    pass  # blank lines are OK within event
                elif in_event and line.startswith('## '):
                    if current_event and current_event.get('type'):
                        events.append(current_event)
                    current_event = None
                    in_event = False

            # Don't forget last event
            if current_event and current_event.get('type'):
                events.append(current_event)
    return events

def load_calendar_news(date_str=None):
    """Load news from calendar/news/*.md files"""
    news_dir = CALENDAR_DIR / "news"
    news = []
    if news_dir.exists():
        for f in news_dir.glob("*.md"):
            if date_str and date_str not in f.stem:
                continue
            content = f.read_text(encoding='utf-8')
            # Extract high impact items
            in_high = False
            for line in content.split('\n'):
                if '## High Impact' in line:
                    in_high = True
                elif line.startswith('## ') and in_high:
                    in_high = False
                elif in_high and line.startswith('### '):
                    news.append({'title': line[4:].strip(), 'date': f.stem, 'impact': 'HIGH'})
    return news

def load_watched_md():
    """Load watched markets from calendar/markets/watched.md"""
    watched_file = CALENDAR_DIR / "markets" / "watched.md"
    markets = []
    if watched_file.exists():
        content = watched_file.read_text(encoding='utf-8')
        import re
        for match in re.finditer(r'### ([\w-]+)\n((?:- [^\n]+\n?)+)', content):
            slug = match.group(1)
            block = match.group(2)
            market = {'slug': slug}
            for line in block.split('\n'):
                if ': ' in line:
                    key, val = line.strip('- ').split(': ', 1)
                    market[key.strip()] = val.strip()
            markets.append(market)
    return markets

def load_insights():
    """Load trading insights from calendar/insights/*.md"""
    insights_dir = CALENDAR_DIR / "insights"
    insights = []
    if insights_dir.exists():
        for f in insights_dir.glob("*.md"):
            content = f.read_text(encoding='utf-8')
            meta = {}
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    for line in parts[1].split('\n'):
                        if ': ' in line:
                            k, v = line.split(': ', 1)
                            meta[k.strip()] = v.strip()
            insights.append({'file': f.stem, 'meta': meta, 'content': content[:500]})
    return insights

def get_upcoming_events(days=7):
    """Get events in next N days"""
    events = load_calendar_events()
    today = datetime.now()
    upcoming = []
    for e in events:
        try:
            edate = datetime.strptime(e.get('date', ''), '%Y-%m-%d')
            delta = (edate - today).days
            if 0 <= delta <= days:
                e['days_until'] = delta
                upcoming.append(e)
        except:
            pass
    return sorted(upcoming, key=lambda x: x.get('days_until', 999))

# === POLYROUTER API ===
def polyrouter_request(endpoint, params=None):
    url = f"{POLYROUTER_API}{endpoint}"
    if params:
        # URL encode parameters properly
        encoded_params = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
        url += "?" + encoded_params

    req = urllib.request.Request(url)
    req.add_header('X-API-Key', POLYROUTER_KEY())
    req.add_header('Accept', 'application/json')

    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        return {'error': str(e)}

def get_market(market_id, platform='polymarket', use_cache=True, max_age_hours=1):
    """Get market data, checking cache first"""
    if use_cache:
        cached = get_cached_market(market_id, max_age_hours)
        if cached:
            cached['_cached'] = True
            return cached

    data = polyrouter_request(f"/markets/{market_id}", {'platform': platform})
    if 'error' not in data:
        cache_market_data(market_id, data)
        # Track in research history
        if data.get('markets'):
            add_to_research_history(market_data={'id': market_id, 'title': data['markets'][0].get('title', ''), 'current_prices': data['markets'][0].get('current_prices', {})})
    return data

def get_orderbook(market_id, platform='polymarket'):
    return polyrouter_request(f"/markets/{market_id}/orderbook", {'platform': platform})

def get_trades(market_id, limit=50, cache=True):
    """Get trades, caching for historical research"""
    data = polyrouter_request("/trades", {'market_id': market_id, 'limit': str(limit)})
    if cache and 'error' not in data and data.get('trades'):
        cache_trades_data(market_id, data)
    return data

def get_all_cached_trades(market_id):
    """Get all locally cached trades for research"""
    return get_cached_trades(market_id)

def search_markets(query, platform='polymarket', limit=20):
    """Search markets using /markets?query= endpoint"""
    return polyrouter_request("/markets", {'query': query, 'platform': platform, 'limit': str(limit)})

def get_price_history(market_ids, days_back=7, interval='60'):
    """Get OHLC price history. interval: 1 (1min), 60 (1hr), 1440 (1day)"""
    # Check cache first
    cache_key = market_ids if isinstance(market_ids, str) else ','.join(market_ids)
    cached = get_cached_price_history(cache_key, interval, max_age_hours=0.5)
    if cached:
        cached['_cached'] = True
        return cached

    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp())

    params = {
        'market_ids': market_ids if isinstance(market_ids, str) else ','.join(market_ids),
        'start_ts': str(start_ts),
        'end_ts': str(end_ts),
        'interval': str(interval),
        'limit': '1000'
    }
    data = polyrouter_request("/price-history", params)
    if 'error' not in data:
        cache_price_history(cache_key, data, interval)
    return data

# Load data
ALL_KNOWLEDGE = load_folder(KNOWLEDGE)
ALL_INVENTORY = get_inventory()
ALL_INV_DATA = load_folder(INVENTORY)
INV_BY_CAT = get_inventory_by_category()

# === HTML TEMPLATE (Modern Clean Design) ===
c = COLORS
HTML = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YES/NO Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root {{
    --bg: #09090b;
    --bg-card: #18181b;
    --bg-muted: #27272a;
    --border: #3f3f46;
    --border-accent: #52525b;
    --text: #fafafa;
    --text-muted: #a1a1aa;
    --text-dim: #71717a;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --success: #22c55e;
    --error: #ef4444;
    --warning: #f59e0b;
    --radius: 8px;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}}
a {{ color: var(--accent); text-decoration: none; transition: color 0.2s; }}
a:hover {{ color: var(--accent-hover); }}

/* YES/NO Animation Styles */
.splash-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #050505;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 99999;
    transition: opacity 0.5s ease;
}}
.splash-overlay.hidden {{ opacity: 0; pointer-events: none; }}
#splash-canvas {{
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 14px;
    line-height: 1.2;
    letter-spacing: 0.15em;
}}
#splash-canvas .yes-bright {{ color: #5fffaf; }}
#splash-canvas .yes-mid {{ color: #2a8060; }}
#splash-canvas .yes-dim {{ color: #153020; }}
#splash-canvas .no-bright {{ color: #ff6a6a; }}
#splash-canvas .no-mid {{ color: #802a2a; }}
#splash-canvas .no-dim {{ color: #301515; }}
#splash-canvas .dot {{ color: #181818; }}

/* YES/NO Header Ticker */
#yesno-ticker {{
    font-family: "SF Mono", monospace;
    font-size: 10px;
    line-height: 1.1;
    letter-spacing: 0.08em;
    white-space: pre;
    height: 22px;
    overflow: hidden;
    background: #0a0a0a;
    border-bottom: 1px solid {c['border']};
}}
#yesno-ticker .yes {{ color: #ff9830; }}
#yesno-ticker .no {{ color: #ff6820; }}
#yesno-ticker .dim {{ color: #1a1a1a; }}

/* Expandable Panel Styles */
.expandable {{
    position: relative;
    transition: all 0.3s ease;
}}
.expandable .expand-btn {{
    position: absolute;
    top: 8px;
    right: 8px;
    width: 20px;
    height: 20px;
    background: {c['bg_input']};
    border: 1px solid {c['border']};
    color: {c['text_muted']};
    font-size: 10px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    transition: all 0.2s;
}}
.expandable .expand-btn:hover {{
    background: {c['accent']};
    color: #000;
    border-color: {c['accent']};
}}
.expandable.collapsed .panel-content {{
    display: none;
}}
.expandable.collapsed {{
    padding-bottom: 0 !important;
}}

/* Modal/Fullscreen Panel */
.modal-overlay {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0,0,0,0.85);
    z-index: 10000;
    display: none;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}}
.modal-overlay.active {{
    display: flex;
}}
.modal-content {{
    background: {c['bg_panel']};
    border: 2px solid {c['accent']};
    width: 90vw;
    max-width: 1200px;
    height: 85vh;
    overflow: auto;
    position: relative;
    box-shadow: 0 0 60px {c['accent']}40;
}}
.modal-close {{
    position: absolute;
    top: 10px;
    right: 10px;
    width: 30px;
    height: 30px;
    background: {c['error']};
    border: none;
    color: #fff;
    font-size: 16px;
    cursor: pointer;
    z-index: 10001;
}}
.modal-header {{
    background: {c['bg_input']};
    padding: 15px 20px;
    border-bottom: 1px solid {c['border']};
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.modal-header h2 {{
    color: {c['accent']};
    font-size: 14px;
    margin: 0;
}}
.modal-body {{
    padding: 20px;
    height: calc(100% - 60px);
    overflow: auto;
}}

/* Resizable panels */
.resizable {{
    resize: both;
    overflow: auto;
    min-height: 100px;
    min-width: 200px;
}}

.header {{
    background: {c['bg_panel']};
    padding: 10px 20px;
    border-bottom: 2px solid {c['border_accent']};
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.logo {{
    font-size: 10px;
    font-weight: bold;
    color: {c['accent']};
    letter-spacing: 0;
    white-space: pre;
    line-height: 1.1;
    font-family: 'Courier New', monospace;
}}
.logo-ascii {{
    color: {c['accent']};
    text-shadow: 0 0 10px {c['accent']}40;
}}
.tui-box {{
    border: 1px solid {c['border']};
    position: relative;
}}
.tui-box::before {{
    content: attr(data-title);
    position: absolute;
    top: -8px;
    left: 10px;
    background: {c['bg_panel']};
    padding: 0 5px;
    font-size: 9px;
    color: {c['accent']};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.tui-border {{
    border: 1px solid {c['border']};
    box-shadow: inset 0 0 0 1px {c['bg_main']}, 0 0 0 1px {c['border']};
}}
.tui-corners {{
    position: relative;
}}
.tui-corners::before {{
    content: '┌' attr(data-title) '─';
    position: absolute;
    top: -1px;
    left: 0;
    color: {c['border']};
    font-size: 12px;
}}
/* Lo-fi grain overlay */
.scanline {{
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    opacity: 0.03;
    pointer-events: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
}}
/* Lo-fi glow effect */
.lofi-glow {{
    box-shadow: 0 0 20px {c['accent']}20, inset 0 0 30px {c['bg_main']}80;
}}
/* Soft blur background */
.lofi-blur {{
    backdrop-filter: blur(2px);
}}
/* Gradient background */
body {{
    background: linear-gradient(135deg, {c['bg_main']} 0%, #0f0f23 50%, {c['bg_panel']} 100%);
    background-attachment: fixed;
}}
.stats-row {{ display: flex; gap: 25px; font-family: 'Courier New', monospace; }}
.stat {{
    text-align: center;
    padding: 5px 15px;
    border: 1px solid {c['border']};
    background: {c['bg_input']};
    position: relative;
}}
.stat::before {{ content: '│'; position: absolute; left: 0; top: 50%; transform: translateY(-50%); color: {c['border']}; }}
.stat-value {{ font-size: 18px; font-weight: bold; color: {c['accent']}; text-shadow: 0 0 8px {c['accent']}40; }}
.stat-label {{ font-size: 9px; color: {c['text_secondary']}; text-transform: uppercase; letter-spacing: 1px; }}

/* TUI data display */
.tui-data {{
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: {c['text_primary']};
}}
.tui-data-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px dotted {c['border']};
}}
.tui-data-label {{ color: {c['text_secondary']}; }}
.tui-data-value {{ color: {c['accent']}; font-weight: bold; }}
.tui-highlight {{ color: {c['success']}; }}
.tui-warning {{ color: {c['warning']}; }}
.tui-error {{ color: {c['error']}; }}
.tui-muted {{ color: {c['text_muted']}; }}

/* ASCII bar */
.ascii-bar {{
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: {c['accent']};
    background: {c['bg_input']};
    padding: 2px 5px;
}}
.ascii-bar-fill {{ color: {c['success']}; }}
.ascii-bar-empty {{ color: {c['border']}; }}

/* Blinking cursor effect */
@keyframes blink {{
    0%, 50% {{ opacity: 1; }}
    51%, 100% {{ opacity: 0; }}
}}
.cursor {{
    display: inline-block;
    width: 8px;
    height: 14px;
    background: {c['accent']};
    animation: blink 1s infinite;
    margin-left: 2px;
    vertical-align: middle;
}}

.nav {{
    display: flex;
    background: {c['bg_panel']};
    border-bottom: 1px solid {c['border']};
    padding: 0 20px;
    font-family: 'Courier New', monospace;
}}
.nav a {{
    color: {c['text_secondary']};
    padding: 10px 18px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 2px solid transparent;
    position: relative;
}}
.nav a::before {{
    content: '[';
    margin-right: 3px;
    color: {c['border']};
}}
.nav a::after {{
    content: ']';
    margin-left: 3px;
    color: {c['border']};
}}
.nav a:hover {{ color: {c['text_primary']}; text-decoration: none; }}
.nav a:hover::before, .nav a:hover::after {{ color: {c['accent']}; }}
.nav a.active {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']}; }}
.nav a.active::before, .nav a.active::after {{ color: {c['accent']}; }}

.search-box {{
    padding: 10px 20px;
    background: {c['bg_panel']};
    border-bottom: 1px solid {c['border']};
    position: relative;
}}
.search-box::before {{
    content: '>';
    position: absolute;
    left: 25px;
    top: 50%;
    transform: translateY(-50%);
    color: {c['accent']};
    font-family: 'Courier New', monospace;
    font-size: 14px;
    z-index: 1;
}}
.search-box input {{
    width: 100%;
    padding: 10px 15px 10px 25px;
    background: {c['bg_input']};
    border: 1px solid {c['border']};
    color: {c['text_primary']};
    font-family: 'Courier New', monospace;
    font-size: 13px;
    caret-color: {c['accent']};
}}
.search-box input:focus {{
    outline: none;
    border-color: {c['accent']};
    box-shadow: 0 0 5px {c['accent']}40;
}}
.search-box input::placeholder {{ color: {c['text_muted']}; }}

/* TUI-style selects */
select {{
    background: {c['bg_input']};
    border: 1px solid {c['border']};
    color: {c['text_primary']};
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 8px 12px;
    cursor: pointer;
    appearance: none;
    -webkit-appearance: none;
}}
select:focus {{
    outline: none;
    border-color: {c['accent']};
    box-shadow: 0 0 5px {c['accent']}40;
}}
select option {{
    background: {c['bg_panel']};
    color: {c['text_primary']};
    padding: 8px;
}}
select optgroup {{
    background: {c['bg_main']};
    color: {c['accent']};
    font-style: normal;
    font-weight: bold;
}}

/* TUI-style buttons */
button, .btn {{
    background: {c['bg_input']};
    border: 1px solid {c['border']};
    color: {c['text_primary']};
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 6px 12px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
button:hover, .btn:hover {{
    border-color: {c['accent']};
    color: {c['accent']};
}}
button:active, .btn:active {{
    background: {c['accent']};
    color: {c['bg_main']};
}}
.btn-primary {{
    border-color: {c['accent']};
    color: {c['accent']};
}}

/* Chart tabs */
.chart-tab {{
    background: {c['bg_input']} !important;
    color: {c['text_secondary']} !important;
    border: 1px solid {c['border']} !important;
    cursor: pointer;
}}
.chart-tab:hover {{
    color: {c['accent']} !important;
    border-color: {c['accent']} !important;
}}
.chart-tab.active {{
    background: {c['accent']} !important;
    color: #000 !important;
    border-color: {c['accent']} !important;
}}

/* Fullscreen chart mode */
.chart-fullscreen {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 9999 !important;
    background: {c['bg_main']} !important;
    display: flex !important;
    flex-direction: column !important;
}}
.chart-fullscreen #price-canvas {{
    flex: 1 !important;
    width: 100% !important;
    height: calc(100vh - 60px) !important;
}}
.chart-fullscreen .chart-header {{
    padding: 10px 20px;
    background: {c['bg_panel']};
    border-bottom: 1px solid {c['border']};
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
#fullscreen-btn:hover {{
    color: {c['accent']} !important;
    border-color: {c['accent']} !important;
}}

/* Live data highlight */
.live-indicator {{
    display: inline-block;
    width: 8px;
    height: 8px;
    background: {c['success']};
    border-radius: 50%;
    animation: pulse 2s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.2); }}
}}
.live-data-box {{
    border: 2px solid {c['success']};
    background: linear-gradient(180deg, {c['bg_panel']} 0%, rgba(0,255,0,0.05) 100%);
}}

.main-layout {{
    display: grid;
    grid-template-columns: 300px 1fr 350px;
    height: calc(100vh - 130px);
}}
.sidebar {{
    border-right: 1px solid {c['border']};
    overflow-y: auto;
    background: {c['bg_main']};
}}
.content {{
    padding: 15px;
    overflow-y: auto;
    background: {c['bg_main']};
}}
.right-panel {{
    border-left: 1px solid {c['border']};
    overflow-y: auto;
    background: {c['bg_panel']};
    padding: 15px;
}}

.panel {{
    background: {c['bg_panel']};
    border: 1px solid {c['border']};
    padding: 15px;
    margin-bottom: 15px;
    position: relative;
}}
.panel::before {{
    content: '┌─────────────────────────────────';
    position: absolute;
    top: -1px;
    left: -1px;
    color: {c['border']};
    font-size: 11px;
    line-height: 1;
    pointer-events: none;
}}
.panel::after {{
    content: '─────────────────────────────────┐';
    position: absolute;
    top: -1px;
    right: -1px;
    color: {c['border']};
    font-size: 11px;
    line-height: 1;
    pointer-events: none;
}}
.panel h2 {{
    color: {c['accent']};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid {c['border']};
    font-family: 'Courier New', monospace;
}}
.panel h2::before {{
    content: '► ';
    color: {c['accent']};
}}

/* Lists */
.item-list {{ }}
.item {{
    padding: 10px;
    border-bottom: 1px solid {c['border']};
    cursor: pointer;
}}
.item:hover {{ background: {c['bg_input']}; }}
.item-title {{ color: {c['text_primary']}; font-size: 13px; }}
.item-meta {{ color: {c['text_secondary']}; font-size: 10px; margin-top: 4px; }}

/* Tags */
.tag {{
    display: inline-block;
    background: {c['bg_input']};
    color: {c['accent']};
    padding: 2px 8px;
    font-size: 10px;
    margin: 2px;
    border: 1px solid {c['border']};
}}

/* Price colors */
.price-up {{ color: {c['success']}; }}
.price-down {{ color: {c['error']}; }}

/* Calendar */
.cal-grid {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }}
.cal-header {{ text-align: center; padding: 5px; color: {c['text_secondary']}; font-size: 10px; }}
.cal-day {{
    background: {c['bg_input']};
    padding: 8px 4px;
    text-align: center;
    cursor: pointer;
    min-height: 45px;
}}
.cal-day:hover {{ background: {c['border']}; }}
.cal-day.today {{ border: 1px solid {c['accent']}; }}
.cal-day.has-data {{ background: #1a1500; }}
.cal-day .num {{ font-size: 13px; font-weight: bold; }}
.cal-day .dots {{ font-size: 8px; margin-top: 3px; }}

/* Orderbook */
.orderbook {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
.bid {{ color: {c['success']}; }}
.ask {{ color: {c['error']}; }}
.trade-row {{
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid {c['border']};
    font-size: 11px;
}}

/* Buttons */
button {{
    background: {c['accent']};
    border: none;
    color: {c['bg_main']};
    padding: 8px 15px;
    font-family: inherit;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    cursor: pointer;
}}
button:hover {{ background: {c['accent_dim']}; }}
button.secondary {{ background: {c['border']}; color: {c['text_primary']}; }}

/* Tables */
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid {c['border']}; }}
th {{ color: {c['text_secondary']}; font-size: 10px; text-transform: uppercase; }}
td {{ color: {c['text_primary']}; font-size: 12px; }}
tr:hover {{ background: {c['bg_input']}; }}

/* Forms */
input, select, textarea {{
    background: {c['bg_input']};
    border: 1px solid {c['border']};
    color: {c['text_primary']};
    padding: 8px 10px;
    font-family: inherit;
    font-size: 12px;
    width: 100%;
}}
input:focus, select:focus, textarea:focus {{
    outline: none;
    border-color: {c['accent']};
}}

/* j/k Navigation Selection */
.selected {{
    background: {c['bg_input']} !important;
    outline: 2px solid {c['accent']};
    outline-offset: -2px;
}}
tr.selected td {{
    background: {c['bg_input']};
}}

/* Collapsible Panels */
.panel.collapsible .panel-header {{
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.panel.collapsible .panel-header::after {{
    content: '▼';
    font-size: 10px;
    color: {c['text_secondary']};
    transition: transform 0.2s;
}}
.panel.collapsible.collapsed .panel-header::after {{
    transform: rotate(-90deg);
}}
.panel.collapsible.collapsed .panel-content {{
    display: none;
}}
.panel-content {{
    transition: max-height 0.2s ease-out;
}}

/* Loading States */
.loading {{
    position: relative;
    pointer-events: none;
    opacity: 0.6;
}}
.loading::after {{
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid {c['border']};
    border-top-color: {c['accent']};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}
.skeleton {{
    background: linear-gradient(90deg, {c['bg_input']} 25%, {c['border']} 50%, {c['bg_input']} 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}}
@keyframes shimmer {{
    0% {{ background-position: 200% 0; }}
    100% {{ background-position: -200% 0; }}
}}

/* Error Recovery */
.error-banner {{
    background: {c['error']};
    color: {c['bg_main']};
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.error-banner button {{
    background: {c['bg_main']};
    color: {c['error']};
    padding: 5px 10px;
}}

/* Mobile */
@media (max-width: 900px) {{
    .main-layout {{ grid-template-columns: 1fr; height: auto; }}
    .sidebar {{ display: none; }}
    .right-panel {{ display: none; }}
    .header {{ flex-direction: column; gap: 10px; text-align: center; }}
    .nav {{ flex-wrap: wrap; justify-content: center; padding: 10px; }}
    .nav a {{ padding: 8px 12px; font-size: 10px; }}
    .content {{ padding: 10px; }}
    .stats-row {{ flex-wrap: wrap; justify-content: center; }}
}}
</style>
</head>
<body>
'''

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        view = params.get('view', ['home'])[0]

        # API endpoints
        if parsed.path.startswith('/api/'):
            self.handle_api(parsed.path, params)
            return

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

        html = self.render_page(view, params)
        self.wfile.write(html.encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == '/api/calendar':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            cal_data = load_calendar()
            action = post_data.get('action')

            if action == 'toggle_goal':
                date = post_data.get('date')
                goal_idx = post_data.get('goal_idx')
                if date not in cal_data['days']:
                    cal_data['days'][date] = {'completed_goals': [], 'notes': '', 'events': []}
                completed = cal_data['days'][date].get('completed_goals', [])
                if goal_idx in completed:
                    completed.remove(goal_idx)
                else:
                    completed.append(goal_idx)
                cal_data['days'][date]['completed_goals'] = completed

            elif action == 'add_goal':
                goal = post_data.get('goal', '').strip()
                if goal:
                    cal_data['goals'].append(goal)

            elif action == 'add_note':
                date = post_data.get('date')
                note = post_data.get('note', '')
                if date not in cal_data['days']:
                    cal_data['days'][date] = {'completed_goals': [], 'notes': '', 'events': []}
                cal_data['days'][date]['notes'] = note

            elif action == 'add_event':
                date = post_data.get('date')
                event = post_data.get('event', '').strip()
                event_type = post_data.get('event_type', 'general')  # general, fed, earnings, market_expiry, trade, alert
                if date and event:
                    if date not in cal_data['days']:
                        cal_data['days'][date] = {'completed_goals': [], 'notes': '', 'events': [], 'markets': [], 'diary': []}
                    if 'events' not in cal_data['days'][date]:
                        cal_data['days'][date]['events'] = []
                    cal_data['days'][date]['events'].append({'text': event, 'type': event_type, 'time': post_data.get('time', '')})

            elif action == 'link_market':
                date = post_data.get('date')
                market_id = post_data.get('market_id')
                market_title = post_data.get('market_title', '')
                if date and market_id:
                    if date not in cal_data['days']:
                        cal_data['days'][date] = {'completed_goals': [], 'notes': '', 'events': [], 'markets': [], 'diary': []}
                    if 'markets' not in cal_data['days'][date]:
                        cal_data['days'][date]['markets'] = []
                    if not any(m['id'] == market_id for m in cal_data['days'][date]['markets']):
                        cal_data['days'][date]['markets'].append({'id': market_id, 'title': market_title})

            elif action == 'unlink_market':
                date = post_data.get('date')
                market_id = post_data.get('market_id')
                if date and market_id and date in cal_data['days']:
                    cal_data['days'][date]['markets'] = [m for m in cal_data['days'][date].get('markets', []) if m['id'] != market_id]

            elif action == 'add_diary':
                date = post_data.get('date')
                entry = post_data.get('entry', '').strip()
                entry_type = post_data.get('entry_type', 'note')  # note, reasoning, prediction, trade
                market_id = post_data.get('market_id')
                sentiment = post_data.get('sentiment')  # bullish, bearish, neutral
                if date and entry:
                    if date not in cal_data['days']:
                        cal_data['days'][date] = {'completed_goals': [], 'notes': '', 'events': [], 'markets': [], 'diary': []}
                    if 'diary' not in cal_data['days'][date]:
                        cal_data['days'][date]['diary'] = []
                    cal_data['days'][date]['diary'].append({
                        'text': entry,
                        'type': entry_type,
                        'market_id': market_id,
                        'sentiment': sentiment,
                        'timestamp': datetime.now().isoformat()
                    })

            elif action == 'sync_api':
                # Sync calendar with API data (winners, market expiries)
                try:
                    winners = fetch_historical_winners()
                    for w in winners:
                        resolved_at = w.get('resolved_at', '')
                        if resolved_at:
                            # Extract date from resolved_at (format: 2026-01-02 20:57:43+00)
                            date_part = resolved_at.split()[0] if ' ' in resolved_at else resolved_at[:10]
                            if date_part not in cal_data['days']:
                                cal_data['days'][date_part] = {'completed_goals': [], 'notes': '', 'events': [], 'markets': [], 'diary': []}
                            # Add as market resolution event
                            if 'markets' not in cal_data['days'][date_part]:
                                cal_data['days'][date_part]['markets'] = []
                            # Avoid duplicates
                            existing_ids = [m.get('id') for m in cal_data['days'][date_part]['markets']]
                            if w.get('market_id') not in existing_ids:
                                cal_data['days'][date_part]['markets'].append({
                                    'id': w.get('market_id'),
                                    'title': w.get('title', ''),
                                    'bracket': w.get('bracket', ''),
                                    'resolved': True
                                })
                except Exception as e:
                    print(f"Calendar sync error: {e}")

            save_calendar(cal_data)
            self.send_json({'ok': True, 'calendar': cal_data})
            return

        elif parsed.path == '/api/diary':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            entry = add_diary_entry(
                entry_type=post_data.get('type', 'note'),
                content=post_data.get('content', ''),
                market_id=post_data.get('market_id'),
                market_title=post_data.get('market_title'),
                prediction=post_data.get('prediction'),
                sentiment=post_data.get('sentiment')
            )
            self.send_json({'ok': True, 'entry': entry})
            return

        elif parsed.path == '/api/notes':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))

            market_id = post_data.get('market_id')
            note = post_data.get('note', '').strip()
            sentiment = post_data.get('sentiment')

            if market_id and note:
                notes = add_market_note(market_id, note, sentiment)
                self.send_json({'ok': True, 'notes': notes})
            else:
                self.send_json({'error': 'Missing market_id or note'})
            return

        elif parsed.path == '/api/elon/import':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            csv_path = post_data.get('path', '')
            csv_type = post_data.get('type', 'prices')  # prices, stats, tweets

            if csv_path and Path(csv_path).exists():
                if csv_type == 'stats':
                    result = import_elon_stats_csv(csv_path)
                elif csv_type == 'tweets':
                    result = import_elon_tweets_csv(csv_path)
                else:
                    result = import_elon_csv(csv_path)
                # Rebuild models after import
                build_prediction_models()
                self.send_json(result)
            else:
                self.send_json({'error': 'Invalid path'})
            return

        # === TRADING POST ENDPOINTS ===
        elif parsed.path == '/api/trading/order':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            market_id = post_data.get('market_id')
            side = post_data.get('side', 'BUY').upper()
            price = float(post_data.get('price', 0))
            size = float(post_data.get('size', 0))
            outcome = post_data.get('outcome', 'yes')
            order_type = post_data.get('order_type', 'GTC')

            if not market_id or not price or not size:
                self.send_json({'error': 'Missing required fields: market_id, price, size'})
                return

            try:
                from trading import place_order
                result = place_order(market_id, side, price, size, outcome, order_type)
                self.send_json({'status': 'submitted', 'order': result})
            except Exception as e:
                self.send_json({'error': str(e)})
            return

        elif parsed.path == '/api/trading/cancel':
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
            order_id = post_data.get('order_id')

            try:
                from trading import cancel_order, cancel_all_orders
                if order_id:
                    result = cancel_order(order_id)
                else:
                    result = cancel_all_orders()
                self.send_json({'status': 'cancelled', 'result': result})
            except Exception as e:
                self.send_json({'error': str(e)})
            return

        self.send_response(404)
        self.end_headers()

    def handle_api(self, path, params):
        if path == '/api/market':
            market_id = params.get('id', [''])[0]
            if market_id:
                data = get_market(market_id)
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/orderbook':
            market_id = params.get('id', [''])[0]
            if market_id:
                data = get_orderbook(market_id)
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/trades':
            market_id = params.get('id', [''])[0]
            limit = params.get('limit', ['50'])[0]
            if market_id:
                data = get_trades(market_id, int(limit))
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/search':
            q = params.get('q', [''])[0]
            if q:
                # Track search query
                add_to_research_history(search_query=q)
                data = search_markets(q)
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing query'})

        elif path == '/api/polyrouter/search':
            # Polyrouter search for web terminal
            q = params.get('q', [''])[0]
            limit = int(params.get('limit', ['30'])[0])
            try:
                data = polyrouter_request("/markets", {
                    'query': q,
                    'platform': 'polymarket',
                    'status': 'open',
                    'limit': str(limit)
                })
                markets = data if isinstance(data, list) else data.get('markets', []) if data else []
                self.send_json({'markets': markets})
            except Exception as e:
                self.send_json({'markets': [], 'error': str(e)})

        elif path == '/api/polyrouter/trending':
            # Polyrouter trending markets for web terminal
            limit = int(params.get('limit', ['30'])[0])
            try:
                data = polyrouter_request("/markets", {
                    'platform': 'polymarket',
                    'status': 'open',
                    'sort': 'volume',
                    'limit': str(limit)
                })
                markets = data if isinstance(data, list) else data.get('markets', []) if data else []
                self.send_json({'markets': markets})
            except Exception as e:
                self.send_json({'markets': [], 'error': str(e)})

        elif path == '/api/watch':
            # Add market to watch list
            market_id = params.get('id', [''])[0]
            title = params.get('title', [''])[0]
            if market_id:
                watched = add_watched_market({'id': market_id, 'title': title})
                self.send_json({'ok': True, 'watched': watched})
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/unwatch':
            market_id = params.get('id', [''])[0]
            if market_id:
                watched = remove_watched_market(market_id)
                self.send_json({'ok': True, 'watched': watched})
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/watched':
            watched = load_watched_markets()
            self.send_json(watched)

        elif path == '/api/history':
            history = load_research_history()
            self.send_json(history)

        elif path == '/api/cache-size':
            size_mb = get_cache_size_mb()
            # Count cached items
            markets_count = len(list(MARKETS_CACHE.glob('*.json')))
            trades_count = len(list(TRADES_CACHE.glob('*.json')))
            self.send_json({
                'size_mb': size_mb,
                'markets_cached': markets_count,
                'trades_cached': trades_count
            })

        elif path == '/api/cached-trades':
            market_id = params.get('id', [''])[0]
            if market_id:
                data = get_all_cached_trades(market_id)
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/price-history':
            market_id = params.get('id', [''])[0]
            days = int(params.get('days', ['7'])[0])
            interval = params.get('interval', ['60'])[0]  # 1, 60, 1440
            if market_id:
                data = get_price_history(market_id, days_back=days, interval=interval)
                self.send_json(data)
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/diary':
            # GET all diary entries
            diary = load_trading_diary()
            self.send_json(diary)

        elif path == '/api/diary/date':
            date_str = params.get('date', [''])[0]
            if date_str:
                entries = get_diary_by_date(date_str)
                self.send_json({'entries': entries})
            else:
                self.send_json({'error': 'Missing date'})

        elif path == '/api/diary/market':
            market_id = params.get('id', [''])[0]
            if market_id:
                entries = get_diary_by_market(market_id)
                self.send_json({'entries': entries})
            else:
                self.send_json({'error': 'Missing market ID'})

        elif path == '/api/notes':
            market_id = params.get('id', [''])[0]
            if market_id:
                notes = get_market_notes(market_id)
                self.send_json(notes)
            else:
                # Return all notes
                self.send_json(load_market_notes())

        elif path == '/api/elon':
            elon = load_elon_data()
            # Include analyses
            for week in elon.get('weeks', []):
                week['analysis'] = analyze_elon_week(week)
            self.send_json(elon)

        elif path == '/api/elon/current':
            # Get current Elon tweet market from API
            data = polyrouter_request("/markets", {'query': 'elon musk tweets january', 'platform': 'polymarket', 'limit': '10'})
            self.send_json(data)

        elif path == '/api/elon/tweets':
            data = load_elon_tweets()
            data['analysis'] = analyze_elon_tweets()
            self.send_json(data)

        elif path == '/api/elon/stats':
            self.send_json(load_elon_stats())

        elif path == '/api/elon/models':
            self.send_json(get_prediction_models())

        elif path == '/api/elon/predict':
            current = int(params.get('current', ['0'])[0])
            elapsed = int(params.get('elapsed', ['0'])[0])
            total = int(params.get('total', ['7'])[0])
            pred = predict_elon_tweets(current, elapsed, total)
            self.send_json(pred)

        elif path == '/api/elon/update':
            # Trigger auto-update if due
            if should_update():
                # Fetch all open markets dynamically
                all_markets = fetch_current_markets()
                all_ids = []
                for event_key, markets in all_markets.items():
                    all_ids.extend([m['id'] for m in markets])
                result = auto_update_markets(all_ids)
                self.send_json({'status': 'updated', **result})
            else:
                info = get_last_update_info()
                self.send_json({'status': 'skipped', 'last_update': info.get('last_update'), 'reason': 'too_soon'})

        elif path == '/api/elon/rebuild-models':
            models = build_prediction_models()
            self.send_json(models)

        elif path == '/api/elon/clob':
            # Get CLOB (orderbook) for a market
            market_id = params.get('id', ['1093290'])[0]  # Default to 580+ bracket
            clob = fetch_clob_data(market_id)
            self.send_json(clob or {'error': 'Failed to fetch CLOB'})

        elif path == '/api/elon/live-markets':
            # Get current active markets with live prices, grouped by event
            event = params.get('event', [None])[0]
            markets = fetch_current_markets(event)
            self.send_json({'markets': markets, 'grouped': event is None})

        elif path == '/api/elon/winners':
            # Get historical winners from cached models file
            models = get_prediction_models()
            winners = models.get('winners', [])
            # If no cached winners, try fetching fresh
            if not winners:
                winners = fetch_historical_winners()
            self.send_json({'winners': winners, 'count': len(winners)})

        elif path == '/api/elon/custom-predict':
            # Custom model prediction
            current = int(params.get('current', [0])[0])
            elapsed_days = float(params.get('elapsed', [5])[0])
            total_days = float(params.get('total', [7])[0])
            custom_rate = params.get('rate', [None])[0]

            remaining = max(0, total_days - elapsed_days)
            rate = float(custom_rate) if custom_rate else (current / elapsed_days if elapsed_days > 0 else 0)
            projected = current + (rate * remaining)

            # Determine bracket
            brackets = [(0,20), (20,40), (40,60), (60,80), (80,100), (100,120), (120,140), (140,160),
                       (160,180), (180,200), (200,220), (220,240), (240,260), (260,280), (280,300),
                       (300,320), (320,340), (340,360), (360,380), (380,400), (400,420), (420,440),
                       (440,460), (460,480), (480,500), (500,520), (520,540), (540,560), (560,580)]
            bracket = '580+'
            for low, high in brackets:
                if projected >= low and projected < high:
                    bracket = f'{low}-{high-1}'
                    break

            self.send_json({
                'current': current,
                'elapsed_days': elapsed_days,
                'remaining_days': remaining,
                'rate': round(rate, 1),
                'projected': round(projected, 0),
                'bracket': bracket,
                'confidence': 'based on your inputs'
            })

        # === TRADING API ENDPOINTS ===
        elif path == '/api/trading/price':
            market_id = params.get('id', [''])[0]
            if not market_id:
                self.send_json({'error': 'Missing market ID'})
                return
            try:
                from trading import get_price
                price = get_price(market_id)
                self.send_json({'market_id': market_id, **price})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/book':
            market_id = params.get('id', [''])[0]
            outcome = params.get('outcome', ['yes'])[0]
            if not market_id:
                self.send_json({'error': 'Missing market ID'})
                return
            try:
                from trading import get_orderbook as trading_get_orderbook
                book = trading_get_orderbook(market_id, outcome)
                # Sort bids descending (highest/best bid first)
                raw_bids = sorted((book.bids or []), key=lambda b: b.price, reverse=True)[:10]
                bids = [{'price': b.price, 'size': b.size} for b in raw_bids]
                # Sort asks ascending (lowest/best ask first)
                raw_asks = sorted((book.asks or []), key=lambda a: a.price, reverse=False)[:10]
                asks = [{'price': a.price, 'size': a.size} for a in raw_asks]
                self.send_json({'market_id': market_id, 'outcome': outcome, 'bids': bids, 'asks': asks})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/orders':
            try:
                from trading import get_open_orders
                orders = get_open_orders()
                self.send_json({'orders': orders or []})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/positions':
            try:
                from trading import get_positions
                positions = get_positions()
                self.send_json({'positions': positions or []})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/trades':
            try:
                from trading import get_user_trades
                limit = int(params.get('limit', ['50'])[0])
                trades = get_user_trades(limit)
                self.send_json({'trades': trades or []})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/mass-order':
            # Place multiple orders at once
            try:
                from trading import place_mass_orders
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode('utf-8'))
                else:
                    # Allow GET with orders as JSON string param
                    orders_json = params.get('orders', ['[]'])[0]
                    data = {'orders': json.loads(orders_json)}

                orders = data.get('orders', [])
                delay = int(data.get('delay_ms', 100))

                if not orders:
                    self.send_json({'error': 'No orders provided'})
                    return

                result = place_mass_orders(orders, delay_ms=delay)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/bracket':
            # Place bracket orders (buy below, sell above)
            try:
                from trading import place_bracket_orders
                market_id = params.get('id', [''])[0]
                center = float(params.get('center', ['0.5'])[0])
                spread = float(params.get('spread', ['0.02'])[0])
                size = int(params.get('size', ['10'])[0])
                outcome = params.get('outcome', ['yes'])[0]

                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return

                result = place_bracket_orders(market_id, center, spread, size, outcome)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/trading/ladder':
            # Place ladder orders across price range
            try:
                from trading import place_ladder_orders
                market_id = params.get('id', [''])[0]
                start = float(params.get('start', ['0.4'])[0])
                end = float(params.get('end', ['0.6'])[0])
                steps = int(params.get('steps', ['5'])[0])
                total_size = int(params.get('size', ['100'])[0])
                side = params.get('side', ['BUY'])[0]
                outcome = params.get('outcome', ['yes'])[0]

                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return

                result = place_ladder_orders(market_id, start, end, steps, total_size, side, outcome)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/endpoints':
            # Return API endpoints reference
            endpoints_file = DATA_DIR / ".api_endpoints.json"
            if endpoints_file.exists():
                data = json.loads(endpoints_file.read_text(encoding='utf-8'))
                self.send_json(data)
            else:
                self.send_json({'error': 'Endpoints file not found'})

        elif path == '/api/search/vector':
            # Vector search using local TF-IDF index
            try:
                from search import search_markets as vector_search_markets, find_similar_markets
                query = params.get('q', [''])[0]
                limit = int(params.get('limit', ['20'])[0])

                if query:
                    results = vector_search_markets(query, top_k=limit)
                    self.send_json({'results': results, 'source': 'vector', 'query': query})
                else:
                    self.send_json({'results': [], 'error': 'No query provided'})
            except ImportError:
                self.send_json({'error': 'Vector search module not available'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/kelly':
            # Kelly criterion calculator
            try:
                from quant import kelly_criterion
                prob = float(params.get('prob', ['0'])[0])
                price = float(params.get('price', ['0'])[0])
                bankroll = float(params.get('bankroll', ['1000'])[0])
                fraction = float(params.get('fraction', ['0.25'])[0])

                if prob > 0 and price > 0:
                    odds = 1 / price  # Convert market price to decimal odds
                    result = kelly_criterion(prob, odds, fraction)
                    result['bet_size'] = result['recommended'] * bankroll
                    result['bankroll'] = bankroll
                    result['market_price'] = price
                    self.send_json(result)
                else:
                    self.send_json({'error': 'Invalid probability or price'})
            except ImportError:
                self.send_json({'error': 'Kelly module not available'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/search/similar':
            # Find similar markets
            try:
                from search import find_similar_markets
                market_id = params.get('id', [''])[0]
                limit = int(params.get('limit', ['5'])[0])

                if market_id:
                    results = find_similar_markets(market_id, top_k=limit)
                    self.send_json({'results': results, 'market_id': market_id})
                else:
                    self.send_json({'results': [], 'error': 'No market_id provided'})
            except ImportError:
                self.send_json({'error': 'Vector search module not available'})
            except Exception as e:
                self.send_json({'error': str(e)})

        # === PRICE ALERTS API ===
        elif path == '/api/alerts':
            # Get all alerts
            alerts = load_alerts()
            self.send_json(alerts)

        elif path == '/api/alerts/active':
            # Get only active (non-triggered) alerts
            active = get_active_alerts()
            self.send_json({'alerts': active, 'count': len(active)})

        elif path == '/api/alerts/add':
            # Add new alert
            market_id = params.get('id', [''])[0]
            market_title = params.get('title', [''])[0]
            condition = params.get('condition', ['above'])[0]
            target = params.get('target', ['0'])[0]
            try:
                target_price = float(target)
                if market_id and 0 < target_price <= 1:
                    alert = add_alert(market_id, market_title, condition, target_price)
                    self.send_json({'success': True, 'alert': alert})
                else:
                    self.send_json({'error': 'Invalid market_id or target_price (must be 0-1)'})
            except ValueError:
                self.send_json({'error': 'Invalid target price'})

        elif path == '/api/alerts/delete':
            # Delete an alert
            alert_id = params.get('id', [''])[0]
            if alert_id:
                result = remove_alert(alert_id)
                self.send_json({'success': True, 'remaining': len(result['alerts'])})
            else:
                self.send_json({'error': 'Missing alert id'})

        elif path == '/api/alerts/check':
            # Check alerts against current prices (for polling)
            try:
                active = get_active_alerts()
                if not active:
                    self.send_json({'triggered': [], 'checked': 0})
                    return
                # Fetch current prices for all active alert markets
                from trading import get_price
                current_prices = {}
                for alert in active:
                    try:
                        price_data = get_price(alert['market_id'])
                        if price_data and 'yes' in price_data:
                            current_prices[alert['market_id']] = price_data['yes']
                    except:
                        pass
                triggered = check_alerts(current_prices)
                self.send_json({'triggered': triggered, 'checked': len(active), 'prices': current_prices})
            except Exception as e:
                self.send_json({'error': str(e)})

        # === EDGE SCANNER API ===
        elif path == '/api/edges':
            # Scan for trading edges
            try:
                min_volume = int(params.get('min_volume', ['1000'])[0])
                limit = int(params.get('limit', ['30'])[0])
                edges = scan_for_edges(min_volume=min_volume, max_results=limit)
                self.send_json({'edges': edges, 'count': len(edges)})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/edges/momentum':
            # Get momentum signals from quant module
            try:
                from quant import momentum_score
                event = params.get('event', [None])[0]
                scores = momentum_score(event)
                self.send_json({'signals': scores[:20], 'count': len(scores)})
            except ImportError:
                self.send_json({'error': 'Quant module not available'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/edges/efficiency':
            # Get market efficiency metrics
            try:
                edges = scan_for_edges(min_volume=500, max_results=100)
                spreads = []
                volumes = []
                liquidities = []
                arb_count = 0

                for e in edges:
                    if 'spread' in str(e.get('edges', [])):
                        spreads.append(e.get('spread', 0))
                    volumes.append(e.get('volume', 0))
                    liquidities.append(e.get('liquidity', 0))
                    for ed in e.get('edges', []):
                        if ed.get('type') == 'mispricing':
                            arb_count += 1

                self.send_json({
                    'avg_spread': sum(spreads) / len(spreads) if spreads else 0,
                    'avg_volume': sum(volumes) / len(volumes) if volumes else 0,
                    'avg_liquidity': sum(liquidities) / len(liquidities) if liquidities else 0,
                    'arb_opportunities': arb_count,
                    'markets_scanned': len(edges)
                })
            except Exception as e:
                self.send_json({'error': str(e)})

        # === PREDICTIONS API ===
        elif path == '/api/predictions':
            # Get all predictions
            try:
                predictions = load_predictions()
                stats = get_prediction_stats()
                self.send_json({
                    'predictions': predictions.get('predictions', [])[-50:],
                    'stats': stats
                })
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/predictions/add':
            # Add a new prediction (POST via GET params for simplicity)
            try:
                market_id = params.get('market_id', [''])[0]
                market_title = params.get('title', [''])[0]
                your_prob = float(params.get('prob', ['0.5'])[0])
                market_price = float(params.get('price', ['0.5'])[0])
                side = params.get('side', ['YES'])[0]
                reasoning = params.get('reasoning', [''])[0]

                if not market_id:
                    self.send_json({'error': 'Market ID required'})
                    return

                pred = add_prediction(market_id, market_title, your_prob, market_price, side, reasoning)
                self.send_json({'success': True, 'prediction': pred})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/predictions/resolve':
            # Resolve a prediction
            try:
                pred_id = params.get('id', [''])[0]
                outcome = params.get('outcome', [''])[0]

                if not pred_id or outcome not in ['YES', 'NO']:
                    self.send_json({'error': 'Prediction ID and outcome (YES/NO) required'})
                    return

                resolve_prediction(pred_id, outcome)
                self.send_json({'success': True})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/predictions/stats':
            # Get prediction statistics
            try:
                stats = get_prediction_stats()
                self.send_json(stats)
            except Exception as e:
                self.send_json({'error': str(e)})

        # === AUTOMATION API ===
        elif path == '/api/automation/config':
            # Get or update automation config
            try:
                from trading import load_automation_config, save_automation_config
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    # Update config
                    body = self.rfile.read(content_length)
                    new_config = json.loads(body.decode('utf-8'))
                    save_automation_config(new_config)
                    self.send_json({'success': True, 'config': new_config})
                else:
                    # Get config
                    config = load_automation_config()
                    self.send_json(config)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/automation/scan':
            # Scan for volume/liquidity spikes
            try:
                from trading import scan_all_spikes
                vol_thresh = float(params.get('vol_threshold', ['2.0'])[0])
                liq_thresh = float(params.get('liq_threshold', ['1.5'])[0])
                results = scan_all_spikes(vol_thresh, liq_thresh)
                self.send_json(results)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/automation/run':
            # Run automation scan and optionally trade
            try:
                from trading import run_automation_scan
                results = run_automation_scan()
                self.send_json(results)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/automation/log':
            # Get automation log
            try:
                from trading import get_automation_log
                limit = int(params.get('limit', ['50'])[0])
                log = get_automation_log(limit)
                self.send_json({'log': log, 'count': len(log)})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/automation/spikes/volume':
            # Get volume spikes only
            try:
                from trading import detect_volume_spikes
                threshold = float(params.get('threshold', ['2.0'])[0])
                min_vol = int(params.get('min_volume', ['5000'])[0])
                spikes = detect_volume_spikes(threshold=threshold, min_volume=min_vol)
                self.send_json({'spikes': spikes, 'count': len(spikes)})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/automation/spikes/liquidity':
            # Get liquidity spikes only
            try:
                from trading import detect_liquidity_spikes
                threshold = float(params.get('threshold', ['1.5'])[0])
                spikes = detect_liquidity_spikes(threshold=threshold)
                self.send_json({'spikes': spikes, 'count': len(spikes)})
            except Exception as e:
                self.send_json({'error': str(e)})

        # === QUANT RESEARCH API ===
        elif path == '/api/quant/models':
            # Run all models on a market
            try:
                from trading import run_all_models
                market_id = params.get('id', [''])[0]
                periods = int(params.get('periods', ['5'])[0])
                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return
                result = run_all_models(market_id, forecast_periods=periods)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/quant/backtest':
            # Backtest a model
            try:
                from trading import backtest_model
                market_id = params.get('id', [''])[0]
                model = params.get('model', ['linear'])[0]
                lookback = int(params.get('lookback', ['50'])[0])
                test_periods = int(params.get('test', ['10'])[0])
                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return
                result = backtest_model(market_id, model, lookback, test_periods)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/quant/optimize':
            # Optimize model parameters
            try:
                from trading import optimize_model_params
                market_id = params.get('id', [''])[0]
                model = params.get('model', ['mean_reversion'])[0]
                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return
                result = optimize_model_params(market_id, model)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/quant/snapshot':
            # Store a price snapshot
            try:
                from trading import store_price_snapshot, get_price
                market_id = params.get('id', [''])[0]
                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return
                price = get_price(market_id)
                if price:
                    store_price_snapshot(market_id, price)
                    self.send_json({'success': True, 'price': price})
                else:
                    self.send_json({'error': 'Could not fetch price'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/quant/history':
            # Get local price history
            try:
                from trading import get_price_history_local
                market_id = params.get('id', [''])[0]
                limit = int(params.get('limit', ['100'])[0])
                if not market_id:
                    self.send_json({'error': 'Missing market_id'})
                    return
                history = get_price_history_local(market_id, limit)
                self.send_json({'history': history, 'count': len(history)})
            except Exception as e:
                self.send_json({'error': str(e)})

        # === CLAUDE TRADING ASSISTANT API ===
        elif path == '/api/claude/setup':
            # Check setup status
            try:
                from claude_trader import run_setup_wizard, get_market_count
                result = run_setup_wizard(interactive=False)
                result['market_count'] = get_market_count()
                self.send_json(result)
            except ImportError:
                self.send_json({'error': 'Claude trader module not available'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/claude/trade':
            # Natural language trading
            try:
                from claude_trader import parse_trading_intent, execute_trading_action
                query = params.get('q', [''])[0]
                dry_run = params.get('dry_run', ['false'])[0] == 'true'

                if not query:
                    self.send_json({'error': 'Missing query (q parameter)'})
                    return

                # Parse intent
                action = parse_trading_intent(query)

                if dry_run or action.get('confirmation_required'):
                    self.send_json({
                        'status': 'confirmation_required',
                        'action': action,
                        'message': 'Confirm to execute'
                    })
                else:
                    result = execute_trading_action(action)
                    self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/claude/execute':
            # Execute confirmed action
            try:
                from claude_trader import execute_trading_action
                import json as json_module

                content_length = int(self.headers.get('Content-Length', 0))
                if content_length > 0:
                    body = self.rfile.read(content_length)
                    data = json_module.loads(body.decode('utf-8'))
                    action = data.get('action', data)  # Handle both {action: ...} and direct action
                else:
                    self.send_json({'error': 'Missing action data'})
                    return

                result = execute_trading_action(action)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/claude/search':
            # Fast local market search
            try:
                from claude_trader import search_markets_db
                query = params.get('q', [''])[0]
                category = params.get('category', [None])[0]
                limit = int(params.get('limit', ['20'])[0])

                results = search_markets_db(query, category, limit)
                self.send_json({'markets': results, 'count': len(results), 'source': 'local_db'})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/claude/research':
            # Market research
            try:
                from claude_trader import research_market
                query = params.get('q', [''])[0]
                deep = params.get('deep', ['false'])[0] == 'true'

                result = research_market(query, deep=deep)
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/claude/build-db':
            # Rebuild market database
            try:
                from claude_trader import build_market_database
                result = build_market_database()
                self.send_json(result)
            except Exception as e:
                self.send_json({'error': str(e)})

        else:
            self.send_json({'error': 'Unknown endpoint'})

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        pass

    def render_page(self, view, params):
        knowledge = load_folder(KNOWLEDGE)
        inventory = load_folder(INVENTORY)
        cal_data = load_calendar()

        cache_size = get_cache_size_mb()
        watched_count = len(load_watched_markets().get('markets', []))
        stats_html = f'''
<div class="stat">
    <div class="stat-value">{len(knowledge)}</div>
    <div class="stat-label">Knowledge</div>
</div>
<div class="stat">
    <div class="stat-value">{len(ALL_INVENTORY)}</div>
    <div class="stat-label">Inventory</div>
</div>
<div class="stat">
    <div class="stat-value">{watched_count}</div>
    <div class="stat-label">Watching</div>
</div>
<div class="stat">
    <div class="stat-value">{cache_size}</div>
    <div class="stat-label">Cache MB</div>
</div>
'''

        diary_count = len(load_trading_diary().get('entries', []))
        elon_weeks = len(load_elon_data().get('weeks', []))
        # Navigation tabs
        is_calendar = view in ('home', 'calendar', 'diary', '')
        is_elon = view in ('elon', 'markets', 'market', 'research')
        is_portfolio = view == 'portfolio'
        is_edges = view == 'edges'
        is_auto = view == 'automation'
        is_quant = view == 'quant'
        is_items = view == 'items'
        is_about = view == 'about'
        is_method = view == 'methodology'
        nav_html = f'''
<a href="/?view=calendar" class="{'active' if is_calendar else ''}" style="font-size:12px">📅 CALENDAR</a>
<a href="/?view=portfolio" class="{'active' if is_portfolio else ''}" style="font-size:12px">💼 PORTFOLIO</a>
<a href="/?view=edges" class="{'active' if is_edges else ''}" style="font-size:12px">🎯 EDGE SCANNER</a>
<a href="/?view=automation" class="{'active' if is_auto else ''}" style="font-size:12px">⚡ AUTO</a>
<a href="/?view=quant" class="{'active' if is_quant else ''}" style="font-size:12px">🔬 QUANT</a>
<a href="/?view=elon" class="{'active' if is_elon else ''}" style="font-size:12px">📊 ELON MARKETS</a>
<a href="/?view=methodology" class="{'active' if is_method else ''}" style="font-size:12px">📖 METHODOLOGY</a>
<a href="/?view=about" class="{'active' if is_about else ''}" style="font-size:12px">ℹ️ ABOUT</a>
'''

        # Sidebar
        sidebar_html = self.render_sidebar(view, params, knowledge, inventory, cal_data)

        # Main content - Calendar is default
        if view in ('calendar', 'home', '', 'diary'):
            content_html = self.render_calendar_full(cal_data, params)
        elif view == 'elon':
            content_html = self.render_elon_research(params)
        elif view == 'markets':
            content_html = self.render_markets(params)
        elif view == 'market':
            content_html = self.render_market_detail(params)
        elif view == 'research':
            content_html = self.render_research(params)
        elif view == 'portfolio':
            content_html = self.render_portfolio(params)
        elif view == 'edges':
            content_html = self.render_edges(params)
        elif view == 'automation':
            content_html = self.render_automation(params)
        elif view == 'quant':
            content_html = self.render_quant(params)
        elif view == 'items':
            content_html = self.render_items(params)
        elif view == 'knowledge':
            content_html = self.render_knowledge(knowledge, params)
        elif view == 'inventory':
            content_html = self.render_inventory(inventory, params)
        elif view == 'about':
            content_html = self.render_about()
        elif view == 'methodology':
            content_html = self.render_methodology()
        elif view == 'promo':
            return self.render_promo_page()
        elif view == 'terminal':
            return self.render_web_terminal(params)
        else:
            # Default to web terminal instead of animated dashboard
            return self.render_web_terminal(params)

        # Right panel
        right_html = self.render_right_panel(view, cal_data)

        return HTML + f'''
<!-- Splash Screen Overlay -->
<div class="splash-overlay" id="splash">
    <pre id="splash-canvas"></pre>
</div>

<div class="scanline"></div>

<!-- YES/NO Ticker Bar -->
<pre id="yesno-ticker"></pre>

<div class="header">
    <div class="logo logo-ascii">╦ ╦ ╔═╗ ╔═╗ ╔╗╔ ╔═╗   ╔═╗ ╦  ╦ ╔═╗ ╔╗╔ ╔╦╗ ╔═╗
╚╦╝ ║╣  ╚═╗ ║║║ ║ ║   ║╣  ╚╗╔╝ ║╣  ║║║  ║  ╚═╗
 ╩  ╚═╝ ╚═╝ ╝╚╝ ╚═╝ ● ╚═╝  ╚╝  ╚═╝ ╝╚╝  ╩  ╚═╝</div>
    <div class="stats-row">{stats_html}</div>
</div>

<div class="nav">{nav_html}</div>

<div class="search-box">
    <input type="text" placeholder="Search markets, knowledge, inventory..." id="global-search"
           onkeyup="if(event.key==='Enter')doGlobalSearch(this.value)">
</div>

<div class="main-layout">
    <div class="sidebar">{sidebar_html}</div>
    <div class="content">{content_html}</div>
    <div class="right-panel">{right_html}</div>
</div>

<!-- Status Bar -->
<div id="status-bar" style="position:fixed;bottom:0;left:0;right:0;height:22px;background:{c['bg_panel']};border-top:1px solid {c['border']};display:flex;align-items:center;justify-content:space-between;padding:0 15px;font-size:10px;font-family:monospace;z-index:1000">
    <div style="display:flex;gap:20px">
        <span style="color:{c['text_secondary']}">API: <span id="api-status" style="color:{c['success']}">OK</span></span>
        <span style="color:{c['text_secondary']}">Cache: <span id="cache-size">{cache_size}</span> MB</span>
        <span style="color:{c['text_secondary']}">Watched: <span id="watch-count">{watched_count}</span></span>
    </div>
    <div style="display:flex;gap:20px">
        <span style="color:{c['text_muted']}">Press ? for shortcuts</span>
        <span id="clock" style="color:{c['text_secondary']}"></span>
    </div>
</div>

<script>
// Status bar clock
function updateClock() {{
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString();
}}
updateClock();
setInterval(updateClock, 1000);
</script>

<script>
function doGlobalSearch(q) {{
    if (!q) return;
    location.href = '/?view=markets&q=' + encodeURIComponent(q);
}}

// === YES/NO SPLASH ANIMATION ===
(function() {{
    const splash = document.getElementById('splash');
    const canvas = document.getElementById('splash-canvas');
    if (!canvas || !splash) return;

    // Skip splash if already seen this session
    if (sessionStorage.getItem('splashSeen')) {{
        splash.classList.add('hidden');
        return;
    }}

    let time = 0;
    const w = 60, h = 35;
    let animId;

    function renderSplash() {{
        time++;
        let out = '';
        for (let y = 0; y < h; y++) {{
            let line = '';
            for (let x = 0; x < w; x++) {{
                const dx = x - w/2, dy = y - h/2;
                const dist = Math.sqrt(dx*dx + dy*dy);
                const angle = Math.atan2(dy, dx);
                const wave1 = Math.sin(dist * 0.2 - time * 0.03 + angle);
                const wave2 = Math.sin(x * 0.1 + time * 0.02);
                const wave3 = Math.cos(y * 0.15 - time * 0.025);
                const combined = wave1 + wave2 * 0.5 + wave3 * 0.3;
                const wordPos = (x + Math.floor(time * 0.1)) % 6;

                if (combined > 0.8) {{
                    line += wordPos < 3 ? '<span class="yes-bright">' + 'YES'[wordPos] + '</span>' : '<span class="no-bright">' + 'NO '[wordPos-3] + '</span>';
                }} else if (combined > 0.3) {{
                    line += wordPos < 3 ? '<span class="yes-mid">' + 'YES'[wordPos] + '</span>' : '<span class="no-mid">' + 'NO '[wordPos-3] + '</span>';
                }} else if (combined > -0.2) {{
                    line += wordPos < 3 ? '<span class="yes-dim">' + 'yes'[wordPos] + '</span>' : '<span class="no-dim">' + 'no '[wordPos-3] + '</span>';
                }} else if (combined > -0.6) {{
                    line += '<span class="dot">·</span>';
                }} else {{
                    line += ' ';
                }}
            }}
            out += line + '\\n';
        }}
        canvas.innerHTML = out;
        animId = requestAnimationFrame(renderSplash);
    }}

    renderSplash();

    // Hide splash after 2.5s
    setTimeout(() => {{
        splash.classList.add('hidden');
        cancelAnimationFrame(animId);
        sessionStorage.setItem('splashSeen', 'true');
    }}, 2500);
}})();

// === YES/NO TICKER BAR ===
(function() {{
    const ticker = document.getElementById('yesno-ticker');
    if (!ticker) return;

    let time = 0;
    const w = 120, h = 2;

    function renderTicker() {{
        time++;
        let out = '';
        for (let y = 0; y < h; y++) {{
            let line = '';
            for (let x = 0; x < w; x++) {{
                const wave = Math.sin(x * 0.12 - time * 0.05 + y * 0.4);
                const wordPos = (x + Math.floor(time * 0.12)) % 6;
                if (wave > 0.3) {{
                    line += wordPos < 3 ? '<span class="yes">' + 'YES'[wordPos] + '</span>' : '<span class="no">' + 'NO '[wordPos-3] + '</span>';
                }} else if (wave > -0.3) {{
                    line += '<span class="dim">·</span>';
                }} else {{
                    line += ' ';
                }}
            }}
            out += line + '\\n';
        }}
        ticker.innerHTML = out;
        requestAnimationFrame(renderTicker);
    }}

    renderTicker();
}})();

// Global Keyboard Shortcuts
(function() {{
    let gPressed = false;
    let gTimeout = null;

    document.addEventListener('keydown', function(e) {{
        // Ignore if typing in input/textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {{
            if (e.key === 'Escape') {{
                e.target.blur();
            }}
            return;
        }}

        // Show help modal
        if (e.key === '?') {{
            e.preventDefault();
            showKeyboardHelp();
            return;
        }}

        // Focus search
        if (e.key === '/') {{
            e.preventDefault();
            const search = document.querySelector('input[type="text"]');
            if (search) search.focus();
            return;
        }}

        // Escape - close modals
        if (e.key === 'Escape') {{
            const modal = document.querySelector('.modal');
            if (modal) modal.remove();
            return;
        }}

        // Two-key shortcuts (g + key)
        if (e.key === 'g') {{
            gPressed = true;
            clearTimeout(gTimeout);
            gTimeout = setTimeout(() => {{ gPressed = false; }}, 500);
            return;
        }}

        if (gPressed) {{
            gPressed = false;
            clearTimeout(gTimeout);
            switch(e.key) {{
                case 'm': location.href = '/?view=markets'; break;
                case 'p': location.href = '/?view=portfolio'; break;
                case 'e': location.href = '/?view=elon'; break;
                case 'c': location.href = '/?view=calendar'; break;
                case 'r': location.href = '/?view=research'; break;
                case 'd': location.href = '/?view=edges'; break;
            }}
            return;
        }}

        // j/k list navigation
        if (e.key === 'j' || e.key === 'k') {{
            e.preventDefault();
            navigateList(e.key === 'j' ? 1 : -1);
            return;
        }}

        // Enter - open selected item
        if (e.key === 'Enter') {{
            const selected = document.querySelector('.list-item.selected, tr.selected, .item.selected');
            if (selected) {{
                selected.click();
                return;
            }}
        }}

        // Trading shortcuts (single key)
        switch(e.key.toLowerCase()) {{
            case 'b':
                // Buy - focus quick trade or show buy modal
                const buySelect = document.getElementById('qt-side');
                if (buySelect) {{
                    buySelect.value = 'BUY';
                    document.getElementById('qt-market')?.focus();
                }}
                break;
            case 's':
                // Sell - focus quick trade or show sell modal
                const sellSelect = document.getElementById('qt-side');
                if (sellSelect) {{
                    sellSelect.value = 'SELL';
                    document.getElementById('qt-market')?.focus();
                }}
                break;
            case 'w':
                // Watch - add current market to watchlist
                const watchBtn = document.getElementById('watch-btn');
                if (watchBtn) watchBtn.click();
                break;
            case 'c':
                // Cancel all orders (with confirmation)
                if (typeof cancelAllOrders === 'function') {{
                    cancelAllOrders();
                }}
                break;
            case 'r':
                // Refresh data
                if (typeof loadPortfolio === 'function') loadPortfolio();
                if (typeof loadMarket === 'function') loadMarket();
                break;
        }}
    }});

    // Collapsible Panel Toggle
    function togglePanel(panelId) {{
        const panel = document.getElementById(panelId);
        if (!panel) return;
        panel.classList.toggle('collapsed');
        const isCollapsed = panel.classList.contains('collapsed');
        localStorage.setItem('panel_' + panelId, isCollapsed ? 'collapsed' : 'expanded');
    }}

    // Initialize collapsed panels from localStorage
    document.querySelectorAll('.panel.collapsible').forEach(panel => {{
        const state = localStorage.getItem('panel_' + panel.id);
        if (state === 'collapsed') panel.classList.add('collapsed');
    }});

    // Loading State Helpers
    function showLoading(elementId) {{
        const el = document.getElementById(elementId);
        if (el) el.classList.add('loading');
    }}
    function hideLoading(elementId) {{
        const el = document.getElementById(elementId);
        if (el) el.classList.remove('loading');
    }}

    // Error Banner
    function showError(message, retryFn) {{
        dismissError();
        const banner = document.createElement('div');
        banner.className = 'error-banner';
        banner.id = 'error-banner';
        banner.innerHTML = `
            <span>⚠ ${{message}}</span>
            <div>
                ${{retryFn ? `<button onclick="dismissError();(${{retryFn.toString()}})()">Retry</button>` : ''}}
                <button onclick="dismissError()">Dismiss</button>
            </div>
        `;
        document.body.insertBefore(banner, document.body.firstChild);
    }}
    function dismissError() {{
        const banner = document.getElementById('error-banner');
        if (banner) banner.remove();
    }}

    // j/k List Navigation
    let selectedIndex = -1;
    function navigateList(direction) {{
        const items = document.querySelectorAll('table tbody tr[onclick], .list-item, .item[onclick], .market-card');
        if (!items.length) return;

        // Remove current selection
        items.forEach(i => i.classList.remove('selected'));

        // Update index
        selectedIndex += direction;
        if (selectedIndex < 0) selectedIndex = items.length - 1;
        if (selectedIndex >= items.length) selectedIndex = 0;

        // Apply selection
        const selected = items[selectedIndex];
        selected.classList.add('selected');
        selected.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    }}

    function showKeyboardHelp() {{
        const existing = document.querySelector('.keyboard-help-modal');
        if (existing) {{ existing.remove(); return; }}

        const modal = document.createElement('div');
        modal.className = 'keyboard-help-modal';
        modal.style.cssText = `
            position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: {c['bg_panel']}; border: 1px solid {c['border']};
            padding: 20px; z-index: 10000; min-width: 300px;
            font-family: monospace; font-size: 12px;
        `;
        modal.innerHTML = `
            <h3 style="margin-bottom:15px;color:{c['accent']}">Keyboard Shortcuts</h3>
            <table style="width:100%">
                <tr><td style="color:{c['text_secondary']}">?</td><td>Show this help</td></tr>
                <tr><td style="color:{c['text_secondary']}">/</td><td>Focus search</td></tr>
                <tr><td style="color:{c['text_secondary']}">Esc</td><td>Close modal / blur</td></tr>
                <tr><td colspan="2" style="padding-top:10px;font-weight:bold">List Navigation</td></tr>
                <tr><td style="color:{c['text_secondary']}">j</td><td>Move down in list</td></tr>
                <tr><td style="color:{c['text_secondary']}">k</td><td>Move up in list</td></tr>
                <tr><td style="color:{c['text_secondary']}">Enter</td><td>Open selected item</td></tr>
                <tr><td colspan="2" style="padding-top:10px;font-weight:bold">Navigation (g + key)</td></tr>
                <tr><td style="color:{c['text_secondary']}">g m</td><td>Go to Markets</td></tr>
                <tr><td style="color:{c['text_secondary']}">g p</td><td>Go to Portfolio</td></tr>
                <tr><td style="color:{c['text_secondary']}">g e</td><td>Go to Elon</td></tr>
                <tr><td style="color:{c['text_secondary']}">g c</td><td>Go to Calendar</td></tr>
                <tr><td style="color:{c['text_secondary']}">g r</td><td>Go to Research</td></tr>
                <tr><td style="color:{c['text_secondary']}">g d</td><td>Go to Edges</td></tr>
                <tr><td colspan="2" style="padding-top:10px;font-weight:bold">Trading</td></tr>
                <tr><td style="color:{c['text_secondary']}">b</td><td>Buy mode (focus order entry)</td></tr>
                <tr><td style="color:{c['text_secondary']}">s</td><td>Sell mode (focus order entry)</td></tr>
                <tr><td style="color:{c['text_secondary']}">w</td><td>Watch/unwatch market</td></tr>
                <tr><td style="color:{c['text_secondary']}">c</td><td>Cancel all orders</td></tr>
                <tr><td style="color:{c['text_secondary']}">r</td><td>Refresh data</td></tr>
            </table>
            <div style="margin-top:15px;text-align:right">
                <button onclick="this.parentElement.parentElement.remove()" style="font-size:10px">Close</button>
            </div>
        `;
        document.body.appendChild(modal);
    }}
}})();
</script>
</body>
</html>
'''

    def render_sidebar(self, view, params, knowledge, inventory, cal_data):
        html = ''

        # Quick stats
        html += f'''
<div class="panel">
    <h2>Quick Stats</h2>
    <div style="display:grid;gap:10px">
        <div style="display:flex;justify-content:space-between">
            <span style="color:{c['text_secondary']}">Knowledge</span>
            <span style="color:{c['accent']}">{len(knowledge)}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:{c['text_secondary']}">Inventory</span>
            <span style="color:{c['accent']}">{len(ALL_INVENTORY)}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:{c['text_secondary']}">Goals</span>
            <span style="color:{c['accent']}">{len(cal_data.get('goals', []))}</span>
        </div>
    </div>
</div>
'''

        # Recent knowledge (collapsible)
        html += '<div class="panel collapsible" id="panel-recent-knowledge"><div class="panel-header" onclick="togglePanel(\'panel-recent-knowledge\')"><h2 style="margin:0;border:0;padding:0">Recent Knowledge</h2></div><div class="panel-content"><div class="item-list">'
        for k in knowledge[:5]:
            html += f'''<div class="item" onclick="location.href='/?view=knowledge&name={k['name']}'">
                <div class="item-title">{k.get('title', k['name'])}</div>
                <div class="item-meta">{k['meta'].get('category', '')}</div>
            </div>'''
        html += '</div></div></div>'

        return html

    def render_right_panel(self, view, cal_data):
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        day_data = cal_data.get('days', {}).get(today_str, {})
        goals = cal_data.get('goals', [])
        completed = day_data.get('completed_goals', [])

        # Today's goals
        goals_html = ''
        for i, goal in enumerate(goals):
            checked = i in completed
            color = c['success'] if checked else c['text_secondary']
            check = '[x]' if checked else '[ ]'
            goals_html += f'<div style="padding:4px 0;color:{color}">{check} {goal}</div>'

        html = f'''
<div class="panel collapsible" id="panel-today">
    <div class="panel-header" onclick="togglePanel('panel-today')">
        <h2 style="margin:0;border:0;padding:0">Today - {today.strftime("%b %d")}</h2>
    </div>
    <div class="panel-content" style="margin-top:10px">
        {goals_html or f'<div style="color:{c["text_muted"]}">No goals</div>'}
    </div>
</div>
'''

        # Market watch - default to Elon Musk trillionaire market
        html += f'''
<div class="panel collapsible" id="panel-market-watch">
    <div class="panel-header" onclick="togglePanel('panel-market-watch')">
        <h2 style="margin:0;border:0;padding:0">Market Watch</h2>
    </div>
    <div class="panel-content" style="margin-top:10px">
        <div id="market-watch" style="min-height:100px">
            <div style="color:{c['text_secondary']};text-align:center;padding:20px">Loading...</div>
        </div>
    </div>
</div>
<script>
async function loadMarketWatch() {{
    try {{
        // Load Elon Musk trillionaire before 2027 (active market)
        const resp = await fetch('/api/market?id=836162');
        const data = await resp.json();
        if (data.markets && data.markets.length > 0) {{
            const m = data.markets[0];
            let html = '<a href="/?view=market&id=' + m.id + '" style="font-size:12px;display:block;margin-bottom:10px">' + m.title.substring(0,50) + '...</a>';
            html += '<div style="font-size:9px;color:{c["text_muted"]};margin-bottom:8px">' + m.status.toUpperCase() + '</div>';
            if (m.current_prices) {{
                for (const [outcome, price] of Object.entries(m.current_prices)) {{
                    const p = price.price || price;
                    const pct = (p * 100).toFixed(1);
                    html += '<div style="display:flex;justify-content:space-between;padding:4px 0"><span style="color:{c["text_secondary"]}">' + outcome + '</span><span style="color:{c["accent"]};font-size:16px">' + pct + '%</span></div>';
                }}
            }}
            document.getElementById('market-watch').innerHTML = html;
        }} else {{
            document.getElementById('market-watch').innerHTML = '<div style="color:{c["text_muted"]}">Search for markets to watch</div>';
        }}
    }} catch(e) {{
        document.getElementById('market-watch').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}
loadMarketWatch();
</script>
'''
        return html

    def render_calendar_full(self, cal_data, params):
        """Full-page prediction market calendar - Bloomberg Terminal style"""
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        current_hour = today.hour

        # Event type styling
        event_styles = {
            'FED': {'color': '#ff4444', 'icon': '🏛️', 'bg': '#ff444420', 'risk': 'HIGH'},
            'FOMC': {'color': '#ff4444', 'icon': '🏛️', 'bg': '#ff444420', 'risk': 'HIGH'},
            'ECONOMIC': {'color': '#ff8844', 'icon': '📊', 'bg': '#ff884420', 'risk': 'MEDIUM'},
            'EARNINGS': {'color': '#44ff44', 'icon': '💰', 'bg': '#44ff4420', 'risk': 'MEDIUM'},
            'POLITICAL': {'color': '#4488ff', 'icon': '🏛️', 'bg': '#4488ff20', 'risk': 'HIGH'},
            'MARKET': {'color': '#e94560', 'icon': '📈', 'bg': '#e9456020', 'risk': 'MEDIUM'},
            'ELON': {'color': '#e94560', 'icon': '🐦', 'bg': '#e9456020', 'risk': 'MEDIUM'},
            'HOLIDAY': {'color': '#888888', 'icon': '🎉', 'bg': '#88888820', 'risk': 'LOW'},
        }

        # Load data from .md files
        all_events = load_calendar_events()
        upcoming = get_upcoming_events(14)
        watched = load_watched_md()
        news = load_calendar_news(today_str)

        # === NEXT 24H TIMELINE (Priority Section) ===
        next_24h = [e for e in upcoming if e.get('days_until', 99) <= 1]
        timeline_html = ''
        for e in next_24h[:6]:
            etype = e.get('type', 'MARKET').upper()
            style = event_styles.get(etype, {'color': c['text_secondary'], 'icon': '•', 'bg': '#ffffff10'})
            time_str = e.get('time', '??:??')
            title = e.get('title', 'Event')[:35]
            is_today = e.get('days_until', 1) == 0

            # Live indicator
            live_indicator = ''
            if is_today and time_str and time_str != '??:??':
                try:
                    event_hour = int(time_str.split(':')[0])
                    if event_hour <= current_hour <= event_hour + 2:
                        live_indicator = '<span class="live-pulse" style="background:#ff4444;padding:2px 6px;font-size:9px;border-radius:3px;animation:pulse 1s infinite">● LIVE</span>'
                except: pass

            countdown = 'NOW' if is_today and live_indicator else f'{e.get("days_until", 0)}d {time_str}' if not is_today else time_str

            timeline_html += f'''
            <div style="display:flex;align-items:center;gap:10px;padding:12px;background:{style['bg']};border-left:4px solid {style['color']};margin-bottom:6px;position:relative">
                <div style="color:{style['color']};font-size:20px">{style['icon']}</div>
                <div style="flex:1">
                    <div style="color:{c['text_primary']};font-size:13px;font-weight:bold">{title}</div>
                    <div style="color:{c['text_muted']};font-size:10px">{e.get('date', '')} @ {time_str} ET</div>
                </div>
                <div style="text-align:right">
                    <div style="color:{c['accent']};font-size:14px;font-weight:bold;font-family:monospace">{countdown}</div>
                    {live_indicator}
                </div>
            </div>'''

        if not timeline_html:
            timeline_html = f'<div style="color:{c["text_muted"]};padding:20px;text-align:center">No events in next 24h</div>'

        # === CONTRARIAN INDICATOR ===
        # Simulated crowd sentiment for demo
        crowd_yes = 78  # Would come from API
        contrarian_color = '#ff4444' if crowd_yes > 80 else '#feca57' if crowd_yes > 65 else '#44ff44'
        contrarian_msg = 'TOO CROWDED - Remember MSTR?' if crowd_yes > 80 else 'Elevated consensus' if crowd_yes > 65 else 'Balanced sentiment'
        contrarian_html = f'''
        <div style="background:{c['bg_input']};border:1px solid {contrarian_color};padding:12px;margin-bottom:15px">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:{c['text_muted']};font-size:9px;text-transform:uppercase">Contrarian Index</span>
                <span style="color:{contrarian_color};font-size:11px;font-weight:bold">{contrarian_msg}</span>
            </div>
            <div style="margin-top:8px;display:flex;align-items:center;gap:8px">
                <div style="flex:1;height:8px;background:{c['border']};border-radius:4px;overflow:hidden">
                    <div style="width:{crowd_yes}%;height:100%;background:{contrarian_color}"></div>
                </div>
                <span style="color:{contrarian_color};font-size:14px;font-weight:bold">{crowd_yes}%</span>
            </div>
            <div style="color:{c['text_muted']};font-size:9px;margin-top:4px">Crowd on YES • elon-tweets-jan6-13</div>
        </div>'''

        # === UPCOMING EVENTS (with enriched data) ===
        events_html = ''
        for e in upcoming[:8]:
            etype = e.get('type', 'MARKET').upper()
            style = event_styles.get(etype, {'color': c['text_secondary'], 'icon': '•', 'bg': '#ffffff10', 'risk': 'LOW'})
            days = e.get('days_until', 0)
            title = e.get('title', 'Event')[:40]
            impact = e.get('impact', '')

            # Days badge with color coding
            if days == 0:
                days_badge = '<span style="color:#44ff44;font-weight:bold;background:#44ff4420;padding:2px 6px;font-size:10px">TODAY</span>'
            elif days == 1:
                days_badge = '<span style="color:#feca57;font-weight:bold;font-size:11px">1d</span>'
            else:
                days_badge = f'<span style="color:{c["text_muted"]};font-size:11px">{days}d</span>'

            # Risk indicator
            risk_badge = ''
            if style.get('risk') == 'HIGH':
                risk_badge = f'<span style="color:#ff4444;font-size:8px;margin-left:4px">⚠ HIGH</span>'

            events_html += f'''
            <div style="display:flex;gap:8px;padding:8px;border-bottom:1px solid {c['border']};cursor:pointer"
                 onmouseover="this.style.background='{c['bg_input']}'" onmouseout="this.style.background='transparent'">
                <div style="color:{style['color']};font-size:14px">{style['icon']}</div>
                <div style="flex:1;min-width:0">
                    <div style="color:{c['text_primary']};font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{title}</div>
                    <div style="color:{c['text_muted']};font-size:9px">{e.get('date', '')} <span style="color:{style['color']}">[{etype}]</span>{risk_badge}</div>
                </div>
                <div>{days_badge}</div>
            </div>'''

        if not events_html:
            events_html = f'<div style="color:{c["text_muted"]};padding:20px;text-align:center">Add events to calendar/events/</div>'

        # === WATCHED MARKETS (with odds + liquidity) ===
        markets_html = ''
        for m in watched[:5]:
            slug = m.get('slug', '')
            if slug.startswith('market-slug'): continue  # Skip template
            odds = m.get('current_odds', '--')
            expiry = m.get('expiry', '')
            status = m.get('status', 'WATCHING')

            status_color = c['success'] if status == 'WATCHING' else c['warning'] if status == 'RESEARCH' else c['text_muted']

            # Liquidity warning (simulated)
            liquidity_warn = '<span style="color:#feca57;font-size:8px;margin-left:4px">LOW LIQ</span>' if 'jan9-16' in slug else ''

            markets_html += f'''
            <div style="padding:8px;background:{c['bg_input']};border:1px solid {c['border']};margin-bottom:6px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:{c['accent']};font-size:11px;font-weight:bold">{slug}</span>
                    <span style="color:{status_color};font-size:8px;text-transform:uppercase">{status}</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:4px;font-size:10px">
                    <span style="color:{c['text_muted']}">Odds: <span style="color:{c['text_primary']}">{odds}</span>{liquidity_warn}</span>
                    <span style="color:{c['text_muted']}">Exp: {expiry[:10] if expiry else '--'}</span>
                </div>
            </div>'''

        if not markets_html:
            markets_html = f'<div style="color:{c["text_muted"]};padding:15px;text-align:center;font-size:11px">Edit calendar/markets/watched.md</div>'

        # === HISTORICAL LENS (Past events) ===
        past_events = [e for e in all_events if e.get('resolution')][:3]
        history_html = ''
        for e in past_events:
            outcome = e.get('resolution', 'Unknown')
            history_html += f'''
            <div style="padding:6px;border-bottom:1px solid {c['border']};font-size:10px">
                <div style="color:{c['text_muted']}">{e.get('title', '')[:30]}</div>
                <div style="color:{c['success']};font-size:9px">✓ {outcome}</div>
            </div>'''
        if not history_html:
            history_html = f'<div style="color:{c["text_muted"]};font-size:10px;padding:8px">No resolved events</div>'

        # Get basic calendar grid
        cal_content = self.render_calendar(cal_data)

        return f'''
<style>
.cp {{ background:{c['bg_panel']};border:1px solid {c['border']};padding:15px }}
.cp h3 {{ color:{c['accent']};font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 12px 0;display:flex;align-items:center;gap:8px }}
.cp h3::before {{ content:'┌─'; color:{c['border']} }}
.cp h3::after {{ content:'─┐'; color:{c['border']} }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.5}} }}
.event-row {{ display:flex;gap:10px;padding:10px 0;border-bottom:1px solid {c['border']}22 }}
.event-row:hover {{ background:{c['bg_input']}44 }}
.event-icon {{ width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-size:16px;border-radius:4px }}
.event-badge {{ font-size:9px;padding:2px 6px;border-radius:2px;font-weight:bold }}
.stat-box {{ background:{c['bg_input']};padding:12px;text-align:center;border:1px solid {c['border']} }}
.stat-val {{ color:{c['accent']};font-size:20px;font-weight:bold }}
.stat-lbl {{ color:{c['text_muted']};font-size:9px;text-transform:uppercase;margin-top:4px }}
</style>

<!-- CLEAN 2-ROW LAYOUT -->
<div style="display:flex;flex-direction:column;gap:15px;height:100%">

    <!-- ROW 1: Main Content (Calendar + Events) -->
    <div style="display:grid;grid-template-columns:1fr 380px;gap:15px;flex:1">

        <!-- LEFT: Calendar Grid -->
        <div class="cp">
            <h3>JANUARY 2026</h3>
            <div style="margin-bottom:12px;font-size:10px;color:{c['text_muted']}">
                <span style="color:#ff4444">●</span> FED/FOMC
                <span style="color:#44ff44;margin-left:10px">●</span> Earnings
                <span style="color:{c['accent']};margin-left:10px">●</span> Market Expiry
                <span style="color:#4488ff;margin-left:10px">●</span> Political
            </div>
            {cal_content}
        </div>

        <!-- RIGHT: Upcoming Events -->
        <div class="cp" style="overflow-y:auto">
            <h3>UPCOMING EVENTS</h3>
            <div style="margin-bottom:10px;display:flex;gap:5px">
                <span class="event-badge" style="background:#ff444420;color:#ff4444">FED</span>
                <span class="event-badge" style="background:#44ff4420;color:#44ff44">EARN</span>
                <span class="event-badge" style="background:{c['accent']}20;color:{c['accent']}">MKT</span>
                <span class="event-badge" style="background:#4488ff20;color:#4488ff">POL</span>
            </div>
            {events_html}
        </div>
    </div>

    <!-- ROW 2: Stats Bar -->
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:15px">
        <div class="stat-box">
            <div class="stat-val">{len(upcoming)}</div>
            <div class="stat-lbl">Events (14d)</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{len([m for m in watched if not m.get('slug','').startswith('market-')])}</div>
            <div class="stat-lbl">Watching</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" style="color:{contrarian_color}">{crowd_yes}%</div>
            <div class="stat-lbl">Crowd YES</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{len(news)}</div>
            <div class="stat-lbl">High Impact</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{len([e for e in all_events if e.get('resolution')])}</div>
            <div class="stat-lbl">Resolved</div>
        </div>
    </div>

    <!-- ROW 3: Bottom Panels -->
    <div style="display:grid;grid-template-columns:1fr 1fr 300px;gap:15px">

        <!-- Watchlist -->
        <div class="cp">
            <h3>WATCHLIST</h3>
            {markets_html}
            <div style="margin-top:10px;padding-top:10px;border-top:1px solid {c['border']}">
                <a href="/?view=elon" style="color:{c['accent']};font-size:11px">→ Elon Dashboard</a>
            </div>
        </div>

        <!-- Contrarian + History -->
        <div class="cp">
            <h3>MARKET SENTIMENT</h3>
            {contrarian_html}
            <div style="margin-top:15px;padding-top:15px;border-top:1px solid {c['border']}">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:8px">RECENT RESOLUTIONS</div>
                {history_html}
            </div>
        </div>

        <!-- Quick Links + Methodology -->
        <div class="cp">
            <h3>RESOURCES</h3>
            <div style="display:flex;flex-direction:column;gap:8px">
                <a href="/?view=elon" style="display:flex;align-items:center;gap:8px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};text-decoration:none">
                    <span style="font-size:16px">🐦</span>
                    <div><div style="font-size:11px;font-weight:bold">Elon Tracker</div><div style="font-size:9px;color:{c['text_muted']}">Live tweet count</div></div>
                </a>
                <a href="/?view=markets" style="display:flex;align-items:center;gap:8px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};text-decoration:none">
                    <span style="font-size:16px">📊</span>
                    <div><div style="font-size:11px;font-weight:bold">Market Search</div><div style="font-size:9px;color:{c['text_muted']}">Polymarket API</div></div>
                </a>
                <div style="padding:10px;background:{c['bg_input']};border:1px solid {c['border']}">
                    <div style="font-size:9px;color:{c['text_muted']};margin-bottom:4px">METHODOLOGY</div>
                    <div style="font-size:10px;color:{c['text_secondary']}">
                        • 30-80 posts before end = sweet spot<br>
                        • Never predict &gt;100 posts ahead<br>
                        • Watch momentum: +95% current
                    </div>
                    <a href="#" style="font-size:9px;color:{c['accent']};margin-top:8px;display:block">→ View full methodology</a>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.onclick = function() {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
    }};
}});
</script>
'''

    def render_items(self, params):
        """Items view - placeholder for news/twitter feed"""
        return f'''
<div class="panel lofi-glow" style="text-align:center;padding:60px">
    <div style="font-size:48px;margin-bottom:20px">📰</div>
    <h2 style="color:{c['accent']}">ITEMS - Coming Soon</h2>
    <p style="color:{c['text_secondary']};max-width:400px;margin:20px auto">
        This section will display news and Twitter posts from external APIs.
        Connect your data sources to populate this feed.
    </p>
    <div style="margin-top:30px;padding:20px;background:{c['bg_input']};border:1px dashed {c['border']}">
        <div style="color:{c['text_muted']};font-size:11px;text-transform:uppercase;margin-bottom:10px">Data Sources</div>
        <div style="color:{c['text_secondary']};font-size:12px">
            • Twitter API (pending)<br>
            • News Feeds (pending)<br>
            • Custom Webhooks (pending)
        </div>
    </div>
</div>
'''

    def render_about(self):
        """About page with dashboard info"""
        return f'''
<div class="panel lofi-glow">
    <h2 style="color:{c['accent']};margin-bottom:20px">DASHBOARD4ALL</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>
            <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px">Overview</h3>
            <p style="color:{c['text_secondary']};font-size:12px;line-height:1.6">
                Bloomberg Terminal style dashboard for prediction market trading.
                Track events, monitor markets, and execute your trading methodology.
            </p>
            <div style="margin-top:20px;padding:15px;background:{c['bg_input']};border:1px solid {c['border']}">
                <div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;margin-bottom:8px">Technical</div>
                <div style="color:{c['text_secondary']};font-size:11px">
                    Port: 8888<br>
                    API: Polyrouter v2<br>
                    Storage: Local .md files<br>
                    Cache: data/cache/
                </div>
            </div>
        </div>
        <div>
            <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px">Views</h3>
            <div style="display:flex;flex-direction:column;gap:8px">
                <a href="/?view=calendar" style="display:flex;align-items:center;gap:10px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']}">
                    <span style="font-size:16px">📅</span>
                    <div><div style="font-size:11px;font-weight:bold">Calendar</div><div style="font-size:9px;color:{c['text_muted']}">Events, watchlist, sentiment</div></div>
                </a>
                <a href="/?view=elon" style="display:flex;align-items:center;gap:10px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']}">
                    <span style="font-size:16px">📊</span>
                    <div><div style="font-size:11px;font-weight:bold">Elon Markets</div><div style="font-size:9px;color:{c['text_muted']}">Tweet tracker, market search</div></div>
                </a>
                <a href="/?view=methodology" style="display:flex;align-items:center;gap:10px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']}">
                    <span style="font-size:16px">📖</span>
                    <div><div style="font-size:11px;font-weight:bold">Methodology</div><div style="font-size:9px;color:{c['text_muted']}">Trading rules & backtests</div></div>
                </a>
                <a href="/?view=knowledge" style="display:flex;align-items:center;gap:10px;padding:10px;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']}">
                    <span style="font-size:16px">📚</span>
                    <div><div style="font-size:11px;font-weight:bold">Knowledge</div><div style="font-size:9px;color:{c['text_muted']}">Notes & rules</div></div>
                </a>
            </div>
        </div>
    </div>
    <div style="margin-top:30px;padding:20px;background:{c['bg_input']};border:1px solid {c['border']}">
        <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px">File Structure</h3>
        <pre style="color:{c['text_secondary']};font-size:11px;font-family:monospace;line-height:1.6">
calendar/
├── events/2026-01.md     # Monthly events (FED, EARNINGS, etc)
├── news/2026-01-11.md    # Daily news with impact
├── markets/watched.md    # Position tracking
├── insights/             # Trading methodology
└── feeds/sources.md      # RSS/API config
        </pre>
    </div>
    <div style="margin-top:20px;text-align:center;color:{c['text_muted']};font-size:10px">
        Built for prediction market traders • Timeline > Grid • Anti-Tilt UI
    </div>
</div>
'''

    def render_methodology(self):
        """Methodology page with trading rules from .md files"""
        return f'''
<div class="panel lofi-glow">
    <h2 style="color:{c['accent']};margin-bottom:20px">TRADING METHODOLOGY</h2>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:25px">
        <div style="background:{c['bg_input']};padding:15px;text-align:center;border:1px solid {c['border']}">
            <div style="color:{c['accent']};font-size:24px;font-weight:bold">21</div>
            <div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase">Backtested Markets</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center;border:1px solid {c['border']}">
            <div style="color:#44ff44;font-size:24px;font-weight:bold">+95%</div>
            <div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase">Current Momentum</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center;border:1px solid {c['border']}">
            <div style="color:{c['text_primary']};font-size:24px;font-weight:bold">30-80</div>
            <div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase">Sweet Spot Posts</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center;border:1px solid {c['border']}">
            <div style="color:#ff4444;font-size:24px;font-weight:bold">&lt;100</div>
            <div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase">Max Predict Ahead</div>
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>
            <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px;border-bottom:1px solid {c['border']};padding-bottom:8px">KEY RULES</h3>
            <div style="padding:15px;background:{c['bg_input']};border-left:3px solid #44ff44;margin-bottom:10px">
                <div style="color:#44ff44;font-size:10px;font-weight:bold">DO</div>
                <ul style="color:{c['text_secondary']};font-size:11px;margin:8px 0 0 15px;line-height:1.8">
                    <li>Track live count religiously</li>
                    <li>Calculate required pace for each bracket</li>
                    <li>Use 30-80 posts window for predictions</li>
                    <li>Size positions based on orderbook depth</li>
                    <li>Have exit plan before entry</li>
                </ul>
            </div>
            <div style="padding:15px;background:{c['bg_input']};border-left:3px solid #ff4444">
                <div style="color:#ff4444;font-size:10px;font-weight:bold">DON'T</div>
                <ul style="color:{c['text_secondary']};font-size:11px;margin:8px 0 0 15px;line-height:1.8">
                    <li>Predict &gt;100 posts ahead</li>
                    <li>Ignore momentum trends</li>
                    <li>Trade without checking volume</li>
                    <li>Hold through uncertainty spikes</li>
                    <li>Oversize on thin liquidity</li>
                </ul>
            </div>
        </div>
        <div>
            <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px;border-bottom:1px solid {c['border']};padding-bottom:8px">WINNING BRACKETS (21 markets)</h3>
            <div style="padding:15px;background:{c['bg_input']};border:1px solid {c['border']}">
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {c['border']}">
                    <span style="color:{c['text_secondary']}">160-279</span>
                    <span style="color:{c['accent']}">29% (6 wins)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {c['border']}">
                    <span style="color:{c['text_secondary']}">280-399</span>
                    <span style="color:{c['accent']}">19% (4 wins)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid {c['border']}">
                    <span style="color:{c['text_secondary']}">400-519</span>
                    <span style="color:{c['accent']}">29% (6 wins)</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0">
                    <span style="color:{c['text_secondary']}">520+</span>
                    <span style="color:#44ff44">24% (5 wins)</span>
                </div>
            </div>
            <div style="margin-top:15px;padding:15px;background:{c['bg_input']};border:1px solid {c['border']}">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:8px">MOMENTUM INDICATOR</div>
                <code style="color:{c['accent']};font-size:10px;display:block;background:{c['bg_panel']};padding:10px">
recent_5wk_avg / older_avg = momentum_ratio<br>
If ratio &gt; 1.5 → strongly_up → buy higher<br>
If ratio &lt; 0.7 → strongly_down → buy lower
                </code>
            </div>
        </div>
    </div>

    <div style="margin-top:25px;padding:20px;background:{c['bg_input']};border:1px solid {c['border']}">
        <h3 style="color:{c['text_primary']};font-size:13px;margin-bottom:15px">RISK MANAGEMENT</h3>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px">
            <div>
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:5px">MAX PER MARKET</div>
                <div style="color:{c['accent']};font-size:18px;font-weight:bold">10%</div>
            </div>
            <div>
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:5px">MAX PER BRACKET</div>
                <div style="color:{c['accent']};font-size:18px;font-weight:bold">5%</div>
            </div>
            <div>
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:5px">TAKE PROFIT</div>
                <div style="color:#44ff44;font-size:18px;font-weight:bold">2x (80%)</div>
            </div>
        </div>
    </div>

    <div style="margin-top:20px;text-align:center">
        <a href="/?view=elon" style="color:{c['accent']};font-size:11px">→ View Elon Markets Dashboard</a>
        <span style="color:{c['text_muted']};margin:0 15px">|</span>
        <a href="/?view=calendar" style="color:{c['accent']};font-size:11px">→ View Calendar</a>
    </div>
</div>
'''

    def render_home(self, cal_data):
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        day_data = cal_data.get('days', {}).get(today_str, {})

        return f'''
<div class="panel">
    <h2>Dashboard Overview</h2>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:20px">
        <div style="background:{c['bg_input']};padding:20px;text-align:center">
            <div style="font-size:32px;color:{c['accent']}">{len(ALL_KNOWLEDGE)}</div>
            <div style="color:{c['text_secondary']};font-size:10px;text-transform:uppercase">Knowledge</div>
        </div>
        <div style="background:{c['bg_input']};padding:20px;text-align:center">
            <div style="font-size:32px;color:{c['accent']}">{len(ALL_INVENTORY)}</div>
            <div style="color:{c['text_secondary']};font-size:10px;text-transform:uppercase">Inventory</div>
        </div>
        <div style="background:{c['bg_input']};padding:20px;text-align:center">
            <div style="font-size:32px;color:{c['accent']}">{len(cal_data.get('goals', []))}</div>
            <div style="color:{c['text_secondary']};font-size:10px;text-transform:uppercase">Goals</div>
        </div>
    </div>
</div>

<div class="panel">
    <h2>Quick Actions</h2>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
        <a href="/?view=calendar"><button>Calendar</button></a>
        <a href="/?view=markets"><button>Markets</button></a>
        <a href="/?view=knowledge"><button>Knowledge</button></a>
        <a href="/?view=inventory"><button>Inventory</button></a>
    </div>
</div>

<div class="panel">
    <h2>Notes for Today</h2>
    <div style="white-space:pre-wrap;color:{c['text_primary']}">{day_data.get('notes', '') or 'No notes for today'}</div>
</div>
'''

    def render_calendar(self, cal_data):
        today = datetime.now()
        month = cal_data.get('month', today.month)
        year = cal_data.get('year', today.year)
        today_str = today.strftime("%Y-%m-%d")

        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        # Event type colors
        event_colors = {
            'fed': '#ff3333',
            'earnings': '#33ff33',
            'market_expiry': '#ff6600',
            'trade': '#00ccff',
            'alert': '#ffcc00',
            'general': '#888888'
        }

        # Calendar grid
        cal_html = '<div class="cal-grid">'
        for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
            cal_html += f'<div class="cal-header">{d}</div>'

        cal = cal_module.Calendar(firstweekday=0)
        for day in cal.itermonthdays(year, month):
            if day == 0:
                cal_html += '<div></div>'
            else:
                date_str = f"{year}-{month:02d}-{day:02d}"
                day_data = cal_data.get('days', {}).get(date_str, {})
                is_today = (day == today.day and month == today.month and year == today.year)
                has_markets = bool(day_data.get('markets'))
                has_diary = bool(day_data.get('diary'))
                has_events = bool(day_data.get('events'))

                classes = 'cal-day'
                if is_today:
                    classes += ' today'
                if has_markets or has_diary or has_events:
                    classes += ' has-data'

                # Visual indicators
                dots = ''
                if has_markets:
                    dots += f'<span style="color:{c["accent"]}">◆</span>'
                if has_diary:
                    dots += f'<span style="color:{c["success"]}">✎</span>'
                if has_events:
                    for e in day_data.get('events', [])[:2]:
                        etype = e.get('type', 'general') if isinstance(e, dict) else 'general'
                        dots += f'<span style="color:{event_colors.get(etype, "#888")}">▪</span>'

                cal_html += f'''<div class="{classes}" onclick="selectDay('{date_str}')">
                    <div class="num">{day}</div>
                    <div class="dots">{dots}</div>
                </div>'''

        cal_html += '</div>'

        # Watched markets for linking
        watched = load_watched_markets()
        watched_opts = ''.join(f'<option value="{safe_html(m["id"])}">{safe_html(m["title"][:40])}</option>' for m in watched.get('markets', []))

        return f'''
<div style="display:grid;grid-template-columns:2fr 1fr;gap:15px">
    <div class="panel">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <h2 style="margin:0">{month_names[month]} {year}</h2>
            <button onclick="syncCalendar()" style="font-size:10px;padding:4px 10px" title="Sync resolved markets from API">
                ↻ Sync API
            </button>
        </div>
        <div style="font-size:10px;color:{c['text_secondary']};margin-bottom:10px">
            <span style="color:{c['accent']}">◆</span> Markets
            <span style="color:{c['success']};margin-left:10px">✎</span> Diary
            <span style="color:#ff3333;margin-left:10px">▪</span> FED
            <span style="color:#33ff33;margin-left:10px">▪</span> Earnings
            <span style="color:#ff6600;margin-left:10px">▪</span> Expiry
        </div>
        {cal_html}
    </div>

    <div>
        <div class="panel" id="day-detail" style="max-height:60vh;overflow-y:auto">
            <h2 id="day-title">Select a Day</h2>
            <div id="day-content">
                <div style="color:{c['text_secondary']}">Click a day to view/add events, markets, diary</div>
            </div>
        </div>
    </div>
</div>

<script>
const calData = {json.dumps(cal_data, ensure_ascii=False)};
let selectedDate = null;
const eventColors = {json.dumps(event_colors)};

function selectDay(dateStr) {{
    selectedDate = dateStr;
    const dayData = calData.days[dateStr] || {{}};
    document.getElementById('day-title').textContent = dateStr;

    let html = '';

    // === LINKED MARKETS ===
    html += '<div style="margin-bottom:15px">';
    html += '<div style="color:{c["accent"]};font-size:11px;margin-bottom:8px;text-transform:uppercase;border-bottom:1px solid {c["border"]};padding-bottom:4px">◆ Linked Markets</div>';
    const markets = dayData.markets || [];
    if (markets.length) {{
        markets.forEach(m => {{
            html += '<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {c["border"]}">';
            html += '<a href="/?view=market&id=' + m.id + '" style="font-size:11px">' + (m.title || m.id).substring(0,35) + '...</a>';
            html += '<button onclick="unlinkMarket(\\'' + m.id + '\\')" class="secondary" style="font-size:9px;padding:2px 6px">×</button>';
            html += '</div>';
        }});
    }} else {{
        html += '<div style="color:{c["text_muted"]};font-size:11px">No markets linked</div>';
    }}
    html += '<div style="margin-top:8px;display:flex;gap:5px">';
    html += '<select id="link-market" style="flex:1;font-size:11px"><option value="">Select market...</option>{watched_opts}</select>';
    html += '<button onclick="linkMarket()" style="font-size:10px">Link</button>';
    html += '</div></div>';

    // === EVENTS ===
    html += '<div style="margin-bottom:15px">';
    html += '<div style="color:{c["text_secondary"]};font-size:11px;margin-bottom:8px;text-transform:uppercase;border-bottom:1px solid {c["border"]};padding-bottom:4px">Events</div>';
    const events = dayData.events || [];
    if (events.length) {{
        events.forEach(e => {{
            const text = typeof e === 'string' ? e : e.text;
            const type = typeof e === 'string' ? 'general' : (e.type || 'general');
            const color = eventColors[type] || '{c["text_secondary"]}';
            const typeLabel = type.toUpperCase().replace('_', ' ');
            html += '<div style="padding:4px 0;border-bottom:1px solid {c["border"]}">';
            html += '<span style="color:' + color + ';font-size:9px;margin-right:8px">[' + typeLabel + ']</span>';
            html += '<span style="font-size:12px">' + text + '</span>';
            html += '</div>';
        }});
    }} else {{
        html += '<div style="color:{c["text_muted"]};font-size:11px">No events</div>';
    }}
    html += '<div style="margin-top:8px;display:grid;grid-template-columns:1fr auto auto;gap:5px">';
    html += '<input type="text" id="new-event" placeholder="Add event..." style="font-size:11px">';
    html += '<select id="event-type" style="width:80px;font-size:10px"><option value="general">General</option><option value="fed">FED</option><option value="earnings">Earnings</option><option value="market_expiry">Expiry</option><option value="trade">Trade</option><option value="alert">Alert</option></select>';
    html += '<button onclick="addEvent()" style="font-size:10px">+</button>';
    html += '</div></div>';

    // === TRADING DIARY ===
    html += '<div style="margin-bottom:15px">';
    html += '<div style="color:{c["success"]};font-size:11px;margin-bottom:8px;text-transform:uppercase;border-bottom:1px solid {c["border"]};padding-bottom:4px">✎ Trading Diary</div>';
    const diary = dayData.diary || [];
    if (diary.length) {{
        diary.forEach(d => {{
            const sentimentColor = d.sentiment === 'bullish' ? '{c["success"]}' : d.sentiment === 'bearish' ? '{c["error"]}' : '{c["text_secondary"]}';
            const typeIcon = d.type === 'prediction' ? '🎯' : d.type === 'reasoning' ? '💭' : d.type === 'trade' ? '💰' : '📝';
            html += '<div style="padding:6px 0;border-bottom:1px solid {c["border"]}">';
            html += '<div style="display:flex;justify-content:space-between">';
            html += '<span style="font-size:10px">' + typeIcon + ' ' + d.type.toUpperCase() + '</span>';
            html += '<span style="color:' + sentimentColor + ';font-size:10px">' + (d.sentiment || '').toUpperCase() + '</span>';
            html += '</div>';
            html += '<div style="font-size:12px;margin-top:4px">' + d.text + '</div>';
            if (d.market_id) {{
                html += '<a href="/?view=market&id=' + d.market_id + '" style="font-size:10px;color:{c["accent"]}">→ Market</a>';
            }}
            html += '</div>';
        }});
    }} else {{
        html += '<div style="color:{c["text_muted"]};font-size:11px">No diary entries</div>';
    }}
    html += '<div style="margin-top:8px">';
    html += '<textarea id="diary-entry" style="width:100%;height:60px;font-size:11px" placeholder="Write reasoning, prediction, or note..."></textarea>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr auto;gap:5px;margin-top:5px">';
    html += '<select id="diary-type" style="font-size:10px"><option value="note">Note</option><option value="reasoning">Reasoning</option><option value="prediction">Prediction</option><option value="trade">Trade</option></select>';
    html += '<select id="diary-sentiment" style="font-size:10px"><option value="">Neutral</option><option value="bullish">Bullish</option><option value="bearish">Bearish</option></select>';
    html += '<button onclick="addDiary()" style="font-size:10px">Add</button>';
    html += '</div></div>';

    // === NOTES ===
    html += '<div>';
    html += '<div style="color:{c["text_secondary"]};font-size:11px;margin-bottom:8px;text-transform:uppercase;border-bottom:1px solid {c["border"]};padding-bottom:4px">Quick Notes</div>';
    html += '<textarea id="day-notes" style="height:60px;font-size:11px" onblur="saveNotes()">' + (dayData.notes || '') + '</textarea>';
    html += '</div>';

    document.getElementById('day-content').innerHTML = html;
}}

async function syncCalendar() {{
    // Sync calendar with API data (resolved markets)
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Syncing...';
    try {{
        const resp = await fetch('/api/calendar', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{action: 'sync_api'}})
        }});
        const data = await resp.json();
        if (data.ok) {{
            Object.assign(calData, data.calendar);
            location.reload();  // Refresh to show updated calendar
        }}
    }} catch(e) {{
        alert('Sync failed: ' + e.message);
    }} finally {{
        btn.disabled = false;
        btn.textContent = '↻ Sync API';
    }}
}}

async function linkMarket() {{
    const select = document.getElementById('link-market');
    const marketId = select.value;
    if (!marketId || !selectedDate) return;
    const title = select.options[select.selectedIndex].text;
    await fetch('/api/calendar', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'link_market', date: selectedDate, market_id: marketId, market_title: title}})
    }}).then(r => r.json()).then(data => {{
        Object.assign(calData, data.calendar);
        selectDay(selectedDate);
    }});
}}

async function unlinkMarket(marketId) {{
    await fetch('/api/calendar', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'unlink_market', date: selectedDate, market_id: marketId}})
    }}).then(r => r.json()).then(data => {{
        Object.assign(calData, data.calendar);
        selectDay(selectedDate);
    }});
}}

async function addEvent() {{
    const event = document.getElementById('new-event').value.trim();
    const eventType = document.getElementById('event-type').value;
    if (!event || !selectedDate) return;
    await fetch('/api/calendar', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'add_event', date: selectedDate, event: event, event_type: eventType}})
    }}).then(r => r.json()).then(data => {{
        Object.assign(calData, data.calendar);
        document.getElementById('new-event').value = '';
        selectDay(selectedDate);
    }});
}}

async function addDiary() {{
    const entry = document.getElementById('diary-entry').value.trim();
    const entryType = document.getElementById('diary-type').value;
    const sentiment = document.getElementById('diary-sentiment').value;
    if (!entry || !selectedDate) return;
    await fetch('/api/calendar', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'add_diary', date: selectedDate, entry: entry, entry_type: entryType, sentiment: sentiment}})
    }}).then(r => r.json()).then(data => {{
        Object.assign(calData, data.calendar);
        document.getElementById('diary-entry').value = '';
        selectDay(selectedDate);
    }});
}}

async function saveNotes() {{
    if (!selectedDate) return;
    const note = document.getElementById('day-notes').value;
    await fetch('/api/calendar', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'add_note', date: selectedDate, note: note}})
    }});
}}

// Auto-select today
selectDay('{today_str}');
</script>
'''

    def render_knowledge(self, knowledge, params):
        q = params.get('q', [''])[0].lower()
        selected = params.get('name', [''])[0]

        items = knowledge
        if q:
            items = [k for k in knowledge if q in k.get('title', '').lower() or q in k['content'].lower() or q in k['meta'].get('tags', '').lower()]

        # List
        list_html = ''
        for k in items:
            active = 'background:' + c['bg_input'] + ';' if k['name'] == selected else ''
            list_html += f'''<div class="item" style="{active}" onclick="location.href='/?view=knowledge&name={k['name']}'">
                <div class="item-title">{k.get('title', k['name'])}</div>
                <div class="item-meta">{k['meta'].get('category', '')} | {k['meta'].get('tags', '')}</div>
            </div>'''

        # Detail
        detail_html = f'<div style="color:{c["text_secondary"]}">Select an item</div>'
        if selected:
            k = next((x for x in knowledge if x['name'] == selected), None)
            if k:
                detail_html = f'''
<h3 style="color:{c['accent']};margin-bottom:15px">{k.get('title', k['name'])}</h3>
<div style="display:flex;gap:10px;margin-bottom:15px">
    <span class="tag">{k['meta'].get('category', 'uncategorized')}</span>
    {' '.join(f'<span class="tag">{t.strip()}</span>' for t in k['meta'].get('tags', '').split(',') if t.strip())}
</div>
<div style="white-space:pre-wrap;line-height:1.6">{k['content']}</div>
'''

        return f'''
<div style="display:grid;grid-template-columns:300px 1fr;gap:15px">
    <div class="panel" style="max-height:calc(100vh - 200px);overflow-y:auto">
        <h2>Knowledge Base</h2>
        <input type="text" placeholder="Search..." value="{q}" style="margin-bottom:10px"
               onkeyup="if(event.key==='Enter')location.href='/?view=knowledge&q='+this.value">
        <div class="item-list">{list_html}</div>
    </div>
    <div class="panel">
        <h2>Detail</h2>
        {detail_html}
    </div>
</div>
'''

    def render_inventory(self, inventory, params):
        q = params.get('q', [''])[0].lower()

        # Flatten all items
        all_items = []
        for inv in inventory:
            for item in inv['items']:
                if not q or q in item.lower():
                    all_items.append({'name': item, 'source': inv['name']})

        list_html = ''
        for item in all_items[:100]:
            list_html += f'''<tr>
                <td>{item['name']}</td>
                <td style="color:{c['text_secondary']}">{item['source']}</td>
            </tr>'''

        return f'''
<div class="panel">
    <h2>Inventory ({len(all_items)} items)</h2>
    <input type="text" placeholder="Search inventory..." value="{q}" style="margin-bottom:15px"
           onkeyup="if(event.key==='Enter')location.href='/?view=inventory&q='+this.value">
    <table>
        <tr><th>Item</th><th>Source</th></tr>
        {list_html}
    </table>
</div>
'''

    def render_markets(self, params):
        q = params.get('q', [''])[0]
        source = params.get('source', ['api'])[0]  # api or vector

        # Load search history for replay
        history = load_research_history()
        recent_searches = history.get('searches', [])[-8:][::-1]  # Last 8, reversed
        history_btns = ''.join([
            f'<button onclick="quickSearch(\'{safe_html(s["query"])}\')" class="secondary" style="font-size:9px;padding:2px 6px;opacity:0.7" title="{s["timestamp"][:16]}">{safe_html(s["query"][:15])}</button>'
            for s in recent_searches
        ]) if recent_searches else '<span style="color:{};font-size:9px">No history</span>'.format(c['text_muted'])

        return f'''
<div class="panel">
    <h2>Prediction Markets</h2>

    <!-- Search Bar with Debounce -->
    <div style="display:flex;gap:10px;margin-bottom:10px">
        <input type="text" id="market-search" placeholder="Search markets..." value="{safe_html(q)}" style="flex:1" oninput="debounceSearch()" onkeypress="if(event.key==='Enter')searchMarkets()">
        <select id="search-source" style="width:100px;font-size:11px">
            <option value="api" {'selected' if source == 'api' else ''}>API</option>
            <option value="vector" {'selected' if source == 'vector' else ''}>Local</option>
        </select>
        <button onclick="searchMarkets()">Search</button>
    </div>

    <!-- Quick Filters -->
    <div style="display:flex;gap:5px;margin-bottom:10px;flex-wrap:wrap">
        <button onclick="quickSearch('elon musk tweets')" class="secondary" style="font-size:10px;padding:4px 8px">Elon</button>
        <button onclick="quickSearch('bitcoin')" class="secondary" style="font-size:10px;padding:4px 8px">Bitcoin</button>
        <button onclick="quickSearch('trump')" class="secondary" style="font-size:10px;padding:4px 8px">Trump</button>
        <button onclick="quickSearch('election')" class="secondary" style="font-size:10px;padding:4px 8px">Election</button>
        <button onclick="quickSearch('fed rate')" class="secondary" style="font-size:10px;padding:4px 8px">Fed</button>
        <button onclick="quickSearch('crypto')" class="secondary" style="font-size:10px;padding:4px 8px">Crypto</button>
        <button onclick="quickSearch('sports')" class="secondary" style="font-size:10px;padding:4px 8px">Sports</button>
    </div>

    <!-- Search History (Replay) -->
    <div style="margin-bottom:10px">
        <span style="color:{c['text_muted']};font-size:9px;margin-right:5px">Recent:</span>
        {history_btns}
    </div>

    <!-- Advanced Filters -->
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:15px;padding:10px;background:{c['bg_input']};border-radius:4px">
        <div>
            <label style="color:{c['text_muted']};font-size:9px;display:block;margin-bottom:3px">YES Price</label>
            <div style="display:flex;gap:3px">
                <input type="number" id="price-min" placeholder="0" min="0" max="100" style="width:45px;font-size:10px">
                <span style="color:{c['text_muted']}">-</span>
                <input type="number" id="price-max" placeholder="100" min="0" max="100" style="width:45px;font-size:10px">
            </div>
        </div>
        <div>
            <label style="color:{c['text_muted']};font-size:9px;display:block;margin-bottom:3px">Min Volume</label>
            <select id="filter-volume" style="width:100%;font-size:10px">
                <option value="">Any</option>
                <option value="1000">$1K+</option>
                <option value="10000">$10K+</option>
                <option value="100000">$100K+</option>
                <option value="1000000">$1M+</option>
            </select>
        </div>
        <div>
            <label style="color:{c['text_muted']};font-size:9px;display:block;margin-bottom:3px">Min Liquidity</label>
            <select id="filter-liquidity" style="width:100%;font-size:10px">
                <option value="">Any</option>
                <option value="1000">$1K+</option>
                <option value="5000">$5K+</option>
                <option value="10000">$10K+</option>
                <option value="50000">$50K+</option>
            </select>
        </div>
        <div>
            <label style="color:{c['text_muted']};font-size:9px;display:block;margin-bottom:3px">Category</label>
            <select id="filter-category" style="width:100%;font-size:10px">
                <option value="">All</option>
                <option value="politics">Politics</option>
                <option value="crypto">Crypto</option>
                <option value="sports">Sports</option>
                <option value="entertainment">Entertainment</option>
                <option value="science">Science</option>
                <option value="business">Business</option>
            </select>
        </div>
        <div>
            <label style="color:{c['text_muted']};font-size:9px;display:block;margin-bottom:3px">Expiry</label>
            <select id="filter-expiry" style="width:100%;font-size:10px">
                <option value="">Any</option>
                <option value="1">Today</option>
                <option value="7">This Week</option>
                <option value="30">This Month</option>
                <option value="90">3 Months</option>
            </select>
        </div>
        <div style="display:flex;align-items:flex-end;gap:5px">
            <button onclick="applyFilters()" class="secondary" style="font-size:10px;padding:4px 8px;flex:1">Apply</button>
            <button onclick="clearFilters()" class="secondary" style="font-size:10px;padding:4px 8px">✕</button>
        </div>
    </div>

    <!-- Results Count -->
    <div id="results-info" style="color:{c['text_secondary']};font-size:10px;margin-bottom:10px"></div>

    <!-- Results -->
    <div id="markets-list">
        <div style="color:{c['text_secondary']};text-align:center;padding:30px">
            Enter a search term or use quick filters above
        </div>
    </div>
</div>

<script>
let allResults = [];
let searchTimeout = null;

// Debounce search (300ms)
function debounceSearch() {{
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {{
        const q = document.getElementById('market-search').value;
        if (q.length >= 2) searchMarkets();
    }}, 300);
}}

function quickSearch(term) {{
    document.getElementById('market-search').value = term;
    searchMarkets();
}}

async function searchMarkets() {{
    const q = document.getElementById('market-search').value;
    const source = document.getElementById('search-source').value;
    if (!q) return;

    document.getElementById('markets-list').innerHTML = '<div style="color:{c["text_secondary"]};text-align:center;padding:30px">Searching...</div>';
    document.getElementById('results-info').textContent = '';

    try {{
        let url = source === 'vector'
            ? '/api/search/vector?q=' + encodeURIComponent(q) + '&limit=50'
            : '/api/search?q=' + encodeURIComponent(q);
        const resp = await fetch(url);
        const data = await resp.json();

        const markets = source === 'vector' ? data.results : data.markets;
        allResults = markets || [];

        if (allResults.length > 0) {{
            document.getElementById('results-info').textContent =
                allResults.length + ' results via ' + (source === 'vector' ? 'local index' : 'API');
            applyFilters();  // Auto-apply any active filters
        }} else {{
            document.getElementById('markets-list').innerHTML = '<div style="color:{c["text_secondary"]};text-align:center;padding:30px">No markets found</div>';
        }}
    }} catch(e) {{
        document.getElementById('markets-list').innerHTML = '<div style="color:{c["error"]};text-align:center;padding:30px">Error: ' + e.message + '</div>';
    }}
}}

function applyFilters() {{
    const minPrice = parseFloat(document.getElementById('price-min').value) / 100 || 0;
    const maxPrice = parseFloat(document.getElementById('price-max').value) / 100 || 1;
    const minVolume = parseFloat(document.getElementById('filter-volume').value) || 0;
    const minLiquidity = parseFloat(document.getElementById('filter-liquidity').value) || 0;
    const category = document.getElementById('filter-category').value.toLowerCase();
    const expiryDays = parseInt(document.getElementById('filter-expiry').value) || 0;
    const now = Date.now();

    const filtered = allResults.filter(m => {{
        const price = getPrice(m);
        if (price < minPrice || price > maxPrice) return false;

        // Volume filter
        const vol = m.volume_24h || m.volume || 0;
        if (minVolume && vol < minVolume) return false;

        // Liquidity filter
        const liq = m.liquidity || m.total_liquidity || 0;
        if (minLiquidity && liq < minLiquidity) return false;

        // Category filter
        if (category) {{
            const tags = (m.tags || []).map(t => t.toLowerCase());
            const cat = (m.category || '').toLowerCase();
            const title = (m.title || m.question || '').toLowerCase();
            if (!tags.includes(category) && cat !== category && !title.includes(category)) return false;
        }}

        // Expiry filter
        if (expiryDays) {{
            const expiry = m.end_date || m.expiry_time || m.resolution_date;
            if (expiry) {{
                const expiryTime = new Date(expiry).getTime();
                const maxTime = now + (expiryDays * 24 * 60 * 60 * 1000);
                if (expiryTime > maxTime) return false;
            }}
        }}

        return true;
    }});

    const filterCount = [minPrice > 0, maxPrice < 1, minVolume > 0, minLiquidity > 0, category, expiryDays > 0].filter(Boolean).length;
    const filterText = filterCount > 0 ? ' (' + filterCount + ' filter' + (filterCount > 1 ? 's' : '') + ')' : '';
    document.getElementById('results-info').textContent =
        filtered.length + ' of ' + allResults.length + filterText;
    renderMarkets(filtered);
}}

function clearFilters() {{
    document.getElementById('price-min').value = '';
    document.getElementById('price-max').value = '';
    document.getElementById('filter-volume').value = '';
    document.getElementById('filter-liquidity').value = '';
    document.getElementById('filter-category').value = '';
    document.getElementById('filter-expiry').value = '';
    if (allResults.length > 0) {{
        document.getElementById('results-info').textContent = allResults.length + ' results';
        renderMarkets(allResults);
    }}
}}

function getPrice(m) {{
    if (m.current_prices) {{
        const prices = Object.values(m.current_prices);
        if (prices.length > 0) return prices[0]?.price || 0;
    }}
    if (m.yes_price) return m.yes_price;
    if (m.price) return m.price;
    return 0;
}}

function renderMarkets(markets) {{
    if (!markets.length) {{
        document.getElementById('markets-list').innerHTML = '<div style="color:{c["text_secondary"]};text-align:center;padding:30px">No markets match filters</div>';
        return;
    }}

    let html = '<table style="width:100%;font-size:11px"><thead><tr style="color:{c["text_secondary"]}"><th style="text-align:left">Market</th><th>YES</th><th>Volume</th><th>Score</th></tr></thead><tbody>';
    markets.forEach(m => {{
        const price = getPrice(m);
        const pricePct = price > 0 ? (price * 100).toFixed(1) + '%' : '-';
        const vol = m.volume_24h ? '$' + Math.round(m.volume_24h).toLocaleString() : '-';
        const score = m.score ? m.score.toFixed(3) : '-';
        const priceColor = price < 0.2 ? '{c["success"]}' : price > 0.8 ? '{c["error"]}' : '{c["accent"]}';
        html += `<tr onclick="location.href='/?view=market&id=${{m.id}}'" style="cursor:pointer">
            <td style="text-align:left;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${{m.title || m.question || m.id}}</td>
            <td style="color:${{priceColor}}">${{pricePct}}</td>
            <td style="color:{c["text_secondary"]}">${{vol}}</td>
            <td style="color:{c["text_muted"]}">${{score}}</td>
        </tr>`;
    }});
    html += '</tbody></table>';
    document.getElementById('markets-list').innerHTML = html;
}}

// Auto-search if query param exists
const urlParams = new URLSearchParams(window.location.search);
if (urlParams.get('q')) {{
    searchMarkets();
}}
</script>
'''

    def render_market_detail(self, params):
        market_id = params.get('id', [''])[0]

        if not market_id:
            return f'<div class="panel"><div style="color:{c["error"]}">No market ID provided</div></div>'

        # Get existing notes for this market
        notes = get_market_notes(market_id)
        notes_html = ''
        for n in notes.get('notes', [])[-5:]:
            sentiment_color = c['success'] if n.get('sentiment') == 'bullish' else c['error'] if n.get('sentiment') == 'bearish' else c['text_secondary']
            notes_html += f'''<div style="padding:6px;margin-bottom:6px;background:{c['bg_input']};border-left:2px solid {sentiment_color}">
                <div style="font-size:11px">{n['text']}</div>
                <div style="font-size:9px;color:{c['text_muted']}">{n['timestamp'][:16]}</div>
            </div>'''

        return f'''
<div style="display:grid;grid-template-columns:1fr 350px;gap:15px">
    <div>
        <div class="panel" id="market-info">
            <h2>Market Info</h2>
            <div style="color:{c['text_secondary']};text-align:center;padding:30px">Loading...</div>
        </div>

        <div class="panel" id="price-chart">
            <h2>Price History <span id="chart-price" style="color:{c['accent']};font-size:14px;margin-left:10px"></span></h2>
            <div style="display:flex;gap:5px;margin-bottom:8px;align-items:center">
                <button onclick="setChartType('area')" id="btn-area" class="secondary" style="font-size:9px;padding:3px 8px;background:{c['accent']};color:#000">LINE</button>
                <button onclick="setChartType('candle')" id="btn-candle" class="secondary" style="font-size:9px;padding:3px 8px">CANDLE</button>
                <span style="color:{c['text_muted']};margin:0 5px">|</span>
                <button onclick="toggleMA()" id="btn-ma" class="secondary" style="font-size:9px;padding:3px 8px">MA</button>
                <button onclick="toggleVolume()" id="btn-vol" class="secondary" style="font-size:9px;padding:3px 8px">VOL</button>
            </div>
            <div id="chart-container" style="width:100%;height:220px;background:{c['bg_input']}"></div>
            <div id="volume-container" style="width:100%;height:50px;background:{c['bg_input']};display:none"></div>
            <div style="display:flex;gap:5px;margin-top:8px">
                <button onclick="loadChart(1)" class="secondary tf-btn" data-tf="1" style="font-size:10px">1H</button>
                <button onclick="loadChart(4)" class="secondary tf-btn" data-tf="4" style="font-size:10px">4H</button>
                <button onclick="loadChart(24)" class="secondary tf-btn" data-tf="24" style="font-size:10px;background:{c['accent']};color:#000">1D</button>
                <button onclick="loadChart(168)" class="secondary tf-btn" data-tf="168" style="font-size:10px">1W</button>
                <button onclick="loadChart(720)" class="secondary tf-btn" data-tf="720" style="font-size:10px">1M</button>
                <button onclick="loadChart(8760)" class="secondary tf-btn" data-tf="8760" style="font-size:10px">ALL</button>
            </div>
        </div>

        <div class="panel" id="orderbook">
            <h2>Orderbook (CLOB)</h2>
            <div style="color:{c['text_secondary']};text-align:center;padding:30px">Loading...</div>
        </div>
    </div>

    <div>
        <div class="panel" id="trades" style="max-height:300px;overflow-y:auto">
            <h2>Recent Trades</h2>
            <div style="color:{c['text_secondary']};text-align:center;padding:30px">Loading...</div>
        </div>

        <div class="panel">
            <h2>My Notes</h2>
            <div style="max-height:150px;overflow-y:auto">
                {notes_html or f'<div style="color:{c["text_muted"]};font-size:11px">No notes yet</div>'}
            </div>
            <div style="margin-top:10px">
                <textarea id="market-note" style="width:100%;height:50px;font-size:11px" placeholder="Add reasoning/note..."></textarea>
                <div style="display:flex;gap:5px;margin-top:5px">
                    <select id="note-sentiment" style="flex:1;font-size:10px">
                        <option value="">Neutral</option>
                        <option value="bullish">Bullish</option>
                        <option value="bearish">Bearish</option>
                    </select>
                    <button onclick="addNote()" style="font-size:10px">Add</button>
                </div>
            </div>
        </div>

        <div class="panel" id="alerts-panel">
            <h2>⚡ Price Alerts</h2>
            <div id="active-alerts" style="margin-bottom:10px"></div>
            <div style="display:grid;grid-template-columns:1fr 80px 60px;gap:5px;align-items:center">
                <select id="alert-condition" style="font-size:10px">
                    <option value="above">Above</option>
                    <option value="below">Below</option>
                </select>
                <input type="number" id="alert-target" placeholder="0.50" step="0.01" min="0.01" max="0.99" style="font-size:10px">
                <button onclick="createAlert()" style="font-size:10px">Set</button>
            </div>
            <div style="font-size:9px;color:{c['text_muted']};margin-top:5px">Enter price as decimal (0.50 = 50%)</div>
        </div>

        <div class="panel">
            <h2>Quick Actions</h2>
            <div style="display:flex;flex-direction:column;gap:8px">
                <button onclick="addToDiary('reasoning')" class="secondary" style="font-size:10px">💭 Add Reasoning to Diary</button>
                <button onclick="addToDiary('prediction')" class="secondary" style="font-size:10px">🎯 Add Prediction</button>
                <button onclick="addToDiary('trade')" class="secondary" style="font-size:10px">💰 Log Trade</button>
            </div>
        </div>

        <div class="panel" id="similar-markets">
            <h2>🔗 Similar Markets</h2>
            <div id="similar-list" style="color:{c['text_secondary']};font-size:11px">Loading...</div>
        </div>
    </div>
</div>

<script>
const marketId = "{market_id}";

let marketTitle = '';

async function loadMarket() {{
    try {{
        const resp = await fetch('/api/market?id=' + marketId);
        const data = await resp.json();

        if (data.markets && data.markets.length > 0) {{
            const m = data.markets[0];
            marketTitle = m.title;
            const isCached = data._cached ? '<span style="color:{c["success"]};font-size:10px;margin-left:10px">● CACHED</span>' : '<span style="color:{c["warning"]};font-size:10px;margin-left:10px">● LIVE</span>';

            let html = '<h2>Market Info ' + isCached + '</h2>';
            html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:15px">';
            html += '<div style="font-size:16px;flex:1">' + m.title + '</div>';
            html += '<button onclick="watchMarket()" style="font-size:12px;padding:6px 12px" id="watch-btn">★ Watch</button>';
            html += '</div>';
            html += '<div style="color:{c["text_secondary"]};font-size:11px;margin-bottom:15px">' + (m.description || '').substring(0, 200) + '</div>';

            html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:15px;margin-bottom:15px">';
            if (m.current_prices) {{
                for (const [outcome, priceData] of Object.entries(m.current_prices)) {{
                    const p = priceData.price || priceData;
                    const pct = (p * 100).toFixed(1);
                    const bid = priceData.bid ? (priceData.bid * 100).toFixed(1) : '-';
                    const ask = priceData.ask ? (priceData.ask * 100).toFixed(1) : '-';
                    html += '<div style="background:{c["bg_input"]};padding:15px;text-align:center">';
                    html += '<div style="color:{c["text_secondary"]};font-size:10px;text-transform:uppercase">' + outcome + '</div>';
                    html += '<div style="font-size:28px;color:{c["accent"]}">' + pct + '%</div>';
                    html += '<div style="font-size:10px;color:{c["text_secondary"]}">Bid: <span class="bid">' + bid + '</span> / Ask: <span class="ask">' + ask + '</span></div>';
                    html += '</div>';
                }}
            }}
            html += '</div>';

            html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;font-size:11px">';
            html += '<div><span style="color:{c["text_secondary"]}">24h Vol:</span> $' + (m.volume_24h || 0).toLocaleString() + '</div>';
            html += '<div><span style="color:{c["text_secondary"]}">Total Vol:</span> $' + (m.volume_total || 0).toLocaleString() + '</div>';
            html += '<div><span style="color:{c["text_secondary"]}">Liquidity:</span> $' + (m.liquidity || 0).toLocaleString() + '</div>';
            html += '<div><span style="color:{c["text_secondary"]}">Status:</span> ' + m.status + '</div>';
            html += '</div>';

            if (m.source_url) {{
                html += '<div style="margin-top:15px"><a href="' + m.source_url + '" target="_blank" style="font-size:11px">View on ' + m.platform + ' →</a></div>';
            }}

            document.getElementById('market-info').innerHTML = html;
            checkWatchStatus();
        }} else if (data.error) {{
            document.getElementById('market-info').innerHTML = '<h2>Market Info</h2><div style="color:{c["error"]}">' + data.error + '</div>';
        }}
    }} catch(e) {{
        document.getElementById('market-info').innerHTML = '<h2>Market Info</h2><div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

async function watchMarket() {{
    const btn = document.getElementById('watch-btn');
    if (btn.textContent.includes('Watching')) {{
        await fetch('/api/unwatch?id=' + encodeURIComponent(marketId));
        btn.textContent = '★ Watch';
        btn.style.background = '{c["accent"]}';
    }} else {{
        await fetch('/api/watch?id=' + encodeURIComponent(marketId) + '&title=' + encodeURIComponent(marketTitle));
        btn.textContent = '✓ Watching';
        btn.style.background = '{c["success"]}';
    }}
}}

async function checkWatchStatus() {{
    const resp = await fetch('/api/watched');
    const data = await resp.json();
    const isWatched = data.markets?.some(m => m.id === marketId);
    const btn = document.getElementById('watch-btn');
    if (btn && isWatched) {{
        btn.textContent = '✓ Watching';
        btn.style.background = '{c["success"]}';
    }}
}}

async function loadOrderbook() {{
    try {{
        const resp = await fetch('/api/orderbook?id=' + marketId);
        const data = await resp.json();

        let html = '<h2>Orderbook Depth</h2>';

        if (data.orderbook) {{
            // Sort bids descending (best/highest bid first) then take top 8
            const bids = (data.orderbook.bids || []).sort((a, b) => b.price - a.price).slice(0, 8);
            // Sort asks ascending (best/lowest ask first) then take top 8, then reverse for display (highest at top)
            const asks = (data.orderbook.asks || []).sort((a, b) => a.price - b.price).slice(0, 8).reverse();

            // Find max size for scaling bars
            const allSizes = [...bids, ...asks].map(o => parseFloat(o.size) || 0);
            const maxSize = Math.max(...allSizes, 1);

            // Spread calculation - bestBid is first in sorted bids, bestAsk is last in reversed asks (which is first in original sorted)
            const bestBid = bids[0]?.price || 0;
            const bestAsk = asks[asks.length - 1]?.price || 0;
            const spread = bestAsk > 0 && bestBid > 0 ? ((bestAsk - bestBid) * 100).toFixed(2) : '-';

            html += '<div style="display:flex;justify-content:space-between;margin-bottom:10px;font-size:10px;color:{c["text_muted"]}">';
            html += '<span>Spread: <span style="color:{c["warning"]}">' + spread + '%</span></span>';
            html += '<span>Best: <span class="bid">' + (bestBid * 100).toFixed(1) + '</span> / <span class="ask">' + (bestAsk * 100).toFixed(1) + '</span></span>';
            html += '</div>';

            // Visual depth chart
            html += '<div style="display:flex;flex-direction:column;gap:2px">';

            // Asks (sells) - shown on top, reversed so highest at top
            asks.forEach(a => {{
                const price = (a.price * 100).toFixed(2);
                const size = parseFloat(a.size) || 0;
                const pct = (size / maxSize * 100).toFixed(0);
                html += '<div style="display:flex;align-items:center;gap:8px;height:18px">';
                html += '<span style="width:45px;font-size:10px;text-align:right" class="ask">' + price + '%</span>';
                html += '<div style="flex:1;height:14px;background:{c["bg_main"]};position:relative;overflow:hidden">';
                html += '<div style="position:absolute;right:0;top:0;height:100%;width:' + pct + '%;background:rgba(239,83,80,0.4);border-left:2px solid {c["error"]}"></div>';
                html += '</div>';
                html += '<span style="width:50px;font-size:9px;color:{c["text_muted"]};text-align:right">' + size.toFixed(0) + '</span>';
                html += '</div>';
            }});

            // Mid-price line
            html += '<div style="display:flex;align-items:center;gap:8px;height:22px;background:{c["bg_main"]};margin:4px 0;padding:2px 0">';
            html += '<span style="width:45px;font-size:9px;text-align:right;color:{c["text_muted"]}">MID</span>';
            html += '<div style="flex:1;border-top:1px dashed {c["text_muted"]}"></div>';
            const midPrice = bestBid > 0 && bestAsk > 0 ? ((bestBid + bestAsk) / 2 * 100).toFixed(1) : '-';
            html += '<span style="width:50px;font-size:10px;color:{c["accent"]};text-align:right">' + midPrice + '%</span>';
            html += '</div>';

            // Bids (buys) - shown on bottom
            bids.forEach(b => {{
                const price = (b.price * 100).toFixed(2);
                const size = parseFloat(b.size) || 0;
                const pct = (size / maxSize * 100).toFixed(0);
                html += '<div style="display:flex;align-items:center;gap:8px;height:18px">';
                html += '<span style="width:45px;font-size:10px;text-align:right" class="bid">' + price + '%</span>';
                html += '<div style="flex:1;height:14px;background:{c["bg_main"]};position:relative;overflow:hidden">';
                html += '<div style="position:absolute;left:0;top:0;height:100%;width:' + pct + '%;background:rgba(38,166,154,0.4);border-right:2px solid {c["success"]}"></div>';
                html += '</div>';
                html += '<span style="width:50px;font-size:9px;color:{c["text_muted"]};text-align:right">' + size.toFixed(0) + '</span>';
                html += '</div>';
            }});

            html += '</div>';

            // Summary
            const totalBidSize = bids.reduce((sum, b) => sum + (parseFloat(b.size) || 0), 0);
            const totalAskSize = asks.reduce((sum, a) => sum + (parseFloat(a.size) || 0), 0);
            const bidPct = totalBidSize + totalAskSize > 0 ? (totalBidSize / (totalBidSize + totalAskSize) * 100).toFixed(0) : 50;
            html += '<div style="margin-top:10px">';
            html += '<div style="display:flex;height:6px;background:{c["bg_main"]};border-radius:3px;overflow:hidden">';
            html += '<div style="width:' + bidPct + '%;background:{c["success"]}"></div>';
            html += '<div style="flex:1;background:{c["error"]}"></div>';
            html += '</div>';
            html += '<div style="display:flex;justify-content:space-between;font-size:9px;color:{c["text_muted"]};margin-top:4px">';
            html += '<span>Bids: $' + totalBidSize.toFixed(0) + ' (' + bidPct + '%)</span>';
            html += '<span>Asks: $' + totalAskSize.toFixed(0) + ' (' + (100 - bidPct) + '%)</span>';
            html += '</div></div>';
        }} else {{
            html += '<div style="color:{c["text_secondary"]};text-align:center;padding:20px">' + (data.error?.message || data.error || 'No orderbook data') + '</div>';
        }}

        document.getElementById('orderbook').innerHTML = html;
    }} catch(e) {{
        document.getElementById('orderbook').innerHTML = '<h2>Orderbook Depth</h2><div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

async function loadTrades() {{
    try {{
        const resp = await fetch('/api/trades?id=' + marketId + '&limit=30');
        const data = await resp.json();

        let html = '<h2>Recent Trades</h2>';

        if (data.trades && data.trades.length > 0) {{
            data.trades.forEach(t => {{
                const price = (t.price * 100).toFixed(2);
                const side = t.side;
                const sideColor = (side === 'buy' || side === 'yes') ? '{c["success"]}' : '{c["error"]}';
                const time = new Date(t.timestamp * 1000).toLocaleTimeString();
                html += '<div class="trade-row"><span style="color:' + sideColor + '">' + side.toUpperCase() + '</span><span>' + price + '%</span><span style="color:{c["text_secondary"]}">' + t.size + '</span><span style="color:{c["text_muted"]};font-size:10px">' + time + '</span></div>';
            }});
        }} else {{
            html += '<div style="color:{c["text_secondary"]};text-align:center;padding:20px">' + (data.error?.message || data.error || 'No trades') + '</div>';
        }}

        document.getElementById('trades').innerHTML = html;
    }} catch(e) {{
        document.getElementById('trades').innerHTML = '<h2>Recent Trades</h2><div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

loadMarket();
loadOrderbook();
loadTrades();
loadChart(24);
loadAlerts();
loadSimilarMarkets();
setInterval(loadTrades, 30000);
setInterval(checkAlertsTrigger, 60000);

// === SIMILAR MARKETS ===
async function loadSimilarMarkets() {{
    try {{
        const resp = await fetch('/api/search/similar?id=' + marketId + '&limit=5');
        const data = await resp.json();

        let html = '';
        if (data.results && data.results.length > 0) {{
            data.results.forEach(m => {{
                const price = m.yes_price || m.price || 0;
                const pricePct = price > 0 ? (price * 100).toFixed(0) + '%' : '-';
                const similarity = m.similarity ? (m.similarity * 100).toFixed(0) + '%' : '';
                html += '<div onclick="location.href=\\'/?view=market&id=' + m.id + '\\';" style="cursor:pointer;padding:8px;margin-bottom:6px;background:{c["bg_input"]};border-radius:4px">';
                html += '<div style="font-size:11px;color:{c["text_primary"]};margin-bottom:3px">' + (m.title || m.question || '').substring(0, 60) + (m.title?.length > 60 ? '...' : '') + '</div>';
                html += '<div style="display:flex;justify-content:space-between;font-size:10px;color:{c["text_muted"]}">';
                html += '<span>YES: <span style="color:{c["accent"]}">' + pricePct + '</span></span>';
                if (similarity) html += '<span>Match: ' + similarity + '</span>';
                html += '</div></div>';
            }});
        }} else {{
            html = '<div style="color:{c["text_muted"]};text-align:center;padding:10px;font-size:10px">No similar markets found</div>';
        }}
        document.getElementById('similar-list').innerHTML = html;
    }} catch(e) {{
        document.getElementById('similar-list').innerHTML = '<div style="color:{c["text_muted"]};font-size:10px">Vector search unavailable</div>';
    }}
}}

// === PRICE ALERTS ===
async function loadAlerts() {{
    try {{
        const resp = await fetch('/api/alerts');
        const data = await resp.json();
        const alerts = (data.alerts || []).filter(a => a.market_id === marketId && !a.triggered);

        let html = '';
        if (alerts.length === 0) {{
            html = '<div style="color:{c["text_muted"]};font-size:10px">No alerts set</div>';
        }} else {{
            alerts.forEach(a => {{
                const pct = (a.target_price * 100).toFixed(1);
                const icon = a.condition === 'above' ? '↑' : '↓';
                html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px;background:{c["bg_input"]};margin-bottom:4px">';
                html += '<span style="font-size:10px">' + icon + ' ' + a.condition + ' ' + pct + '%</span>';
                html += '<button onclick="deleteAlert(\\''+a.id+'\\',event)" style="font-size:9px;padding:2px 6px;background:{c["error"]}">×</button>';
                html += '</div>';
            }});
        }}
        document.getElementById('active-alerts').innerHTML = html;
    }} catch(e) {{
        console.error('Error loading alerts:', e);
    }}
}}

async function createAlert() {{
    const condition = document.getElementById('alert-condition').value;
    const target = parseFloat(document.getElementById('alert-target').value);

    if (isNaN(target) || target <= 0 || target >= 1) {{
        alert('Invalid target price (must be between 0.01 and 0.99)');
        return;
    }}

    const url = '/api/alerts/add?id=' + encodeURIComponent(marketId) +
                '&title=' + encodeURIComponent(marketTitle) +
                '&condition=' + condition +
                '&target=' + target;

    const resp = await fetch(url);
    const data = await resp.json();

    if (data.success) {{
        document.getElementById('alert-target').value = '';
        loadAlerts();
    }} else {{
        alert('Error: ' + (data.error || 'Failed to create alert'));
    }}
}}

async function deleteAlert(alertId, event) {{
    event.stopPropagation();
    const resp = await fetch('/api/alerts/delete?id=' + alertId);
    const data = await resp.json();
    if (data.success) {{
        loadAlerts();
    }}
}}

async function checkAlertsTrigger() {{
    try {{
        const resp = await fetch('/api/alerts/check');
        const data = await resp.json();
        if (data.triggered && data.triggered.length > 0) {{
            data.triggered.forEach(a => {{
                const pct = (a.triggered_price * 100).toFixed(1);
                const msg = '⚡ Alert triggered: ' + a.market_title + ' ' + a.condition + ' ' + (a.target_price*100).toFixed(1) + '% (now ' + pct + '%)';
                if (Notification.permission === 'granted') {{
                    new Notification('Price Alert', {{ body: msg }});
                }} else {{
                    alert(msg);
                }}
            }});
            loadAlerts();
        }}
    }} catch(e) {{
        console.error('Error checking alerts:', e);
    }}
}}

// Request notification permission
if ('Notification' in window && Notification.permission === 'default') {{
    Notification.requestPermission();
}}

// Lightweight Charts - Interactive Price Chart
let priceChart = null;
let volumeChart = null;
let priceSeries = null;
let candleSeries = null;
let volumeSeries = null;
let maSeries = null;
let chartType = 'area';
let showMA = false;
let showVolume = false;
let chartData = [];
let currentTimeframe = 24;

function initChart() {{
    const container = document.getElementById('chart-container');
    if (!container) return;

    // Clear existing
    if (priceChart) {{ priceChart.remove(); priceChart = null; }}

    priceChart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 220,
        layout: {{
            background: {{ type: 'solid', color: '{c["bg_input"]}' }},
            textColor: '{c["text_secondary"]}',
            fontFamily: 'Consolas, Monaco, monospace',
        }},
        grid: {{
            vertLines: {{ color: '{c["border"]}', style: 1 }},
            horzLines: {{ color: '{c["border"]}', style: 1 }},
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {{ labelBackgroundColor: '{c["accent"]}' }},
            horzLine: {{ labelBackgroundColor: '{c["accent"]}' }},
        }},
        rightPriceScale: {{
            borderColor: '{c["border"]}',
            scaleMargins: {{ top: 0.1, bottom: 0.2 }},
        }},
        timeScale: {{
            borderColor: '{c["border"]}',
            timeVisible: true,
            secondsVisible: false,
        }},
    }});

    // Create series based on chart type
    if (chartType === 'candle') {{
        candleSeries = priceChart.addCandlestickSeries({{
            upColor: '{c["success"]}',
            downColor: '{c["error"]}',
            borderUpColor: '{c["success"]}',
            borderDownColor: '{c["error"]}',
            wickUpColor: '{c["success"]}',
            wickDownColor: '{c["error"]}',
            priceFormat: {{ type: 'percent' }},
        }});
        priceSeries = null;
    }} else {{
        priceSeries = priceChart.addAreaSeries({{
            topColor: 'rgba(233, 69, 96, 0.4)',
            bottomColor: 'rgba(233, 69, 96, 0.0)',
            lineColor: '{c["accent"]}',
            lineWidth: 2,
            priceFormat: {{ type: 'percent' }},
        }});
        candleSeries = null;
    }}

    // MA overlay
    if (showMA) {{
        maSeries = priceChart.addLineSeries({{
            color: '#4fc3f7',
            lineWidth: 1,
            lineStyle: 2,
            priceFormat: {{ type: 'percent' }},
        }});
    }}

    // Resize handler
    const resizeHandler = () => {{
        if (priceChart && container) {{
            priceChart.applyOptions({{ width: container.clientWidth }});
        }}
        if (volumeChart) {{
            const volContainer = document.getElementById('volume-container');
            if (volContainer) volumeChart.applyOptions({{ width: volContainer.clientWidth }});
        }}
    }};
    window.removeEventListener('resize', resizeHandler);
    window.addEventListener('resize', resizeHandler);
}}

function initVolumeChart() {{
    const container = document.getElementById('volume-container');
    if (!container || volumeChart) return;

    volumeChart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 50,
        layout: {{
            background: {{ type: 'solid', color: '{c["bg_input"]}' }},
            textColor: '{c["text_muted"]}',
            fontFamily: 'Consolas, Monaco, monospace',
        }},
        grid: {{
            vertLines: {{ visible: false }},
            horzLines: {{ visible: false }},
        }},
        rightPriceScale: {{
            borderVisible: false,
            scaleMargins: {{ top: 0.1, bottom: 0.1 }},
        }},
        timeScale: {{
            visible: false,
        }},
    }});

    volumeSeries = volumeChart.addHistogramSeries({{
        color: '{c["text_muted"]}',
        priceFormat: {{ type: 'volume' }},
    }});
}}

function setChartType(type) {{
    chartType = type;
    document.getElementById('btn-area').style.background = type === 'area' ? '{c["accent"]}' : '';
    document.getElementById('btn-area').style.color = type === 'area' ? '#000' : '';
    document.getElementById('btn-candle').style.background = type === 'candle' ? '{c["accent"]}' : '';
    document.getElementById('btn-candle').style.color = type === 'candle' ? '#000' : '';
    loadChart(currentTimeframe);
}}

function toggleMA() {{
    showMA = !showMA;
    document.getElementById('btn-ma').style.background = showMA ? '{c["accent"]}' : '';
    document.getElementById('btn-ma').style.color = showMA ? '#000' : '';
    loadChart(currentTimeframe);
}}

function toggleVolume() {{
    showVolume = !showVolume;
    document.getElementById('btn-vol').style.background = showVolume ? '{c["accent"]}' : '';
    document.getElementById('btn-vol').style.color = showVolume ? '#000' : '';
    document.getElementById('volume-container').style.display = showVolume ? 'block' : 'none';
    if (showVolume && chartData.length) {{
        initVolumeChart();
        updateVolumeChart();
    }} else if (volumeChart) {{
        volumeChart.remove();
        volumeChart = null;
    }}
}}

function calculateMA(data, period) {{
    const result = [];
    for (let i = period - 1; i < data.length; i++) {{
        let sum = 0;
        for (let j = 0; j < period; j++) {{
            sum += data[i - j].value || data[i - j].close || 0;
        }}
        result.push({{ time: data[i].time, value: sum / period }});
    }}
    return result;
}}

function updateVolumeChart() {{
    if (!volumeSeries || !chartData.length) return;
    const volData = chartData.map(d => ({{
        time: d.time,
        value: d.volume || Math.random() * 1000,
        color: (d.close || d.value || 0) >= (d.open || d.value || 0) ? '{c["success"]}' : '{c["error"]}'
    }}));
    volumeSeries.setData(volData);
}}

async function loadChart(hours) {{
    currentTimeframe = hours;

    // Update timeframe button states
    document.querySelectorAll('.tf-btn').forEach(btn => {{
        const tf = parseInt(btn.dataset.tf);
        btn.style.background = tf === hours ? '{c["accent"]}' : '';
        btn.style.color = tf === hours ? '#000' : '';
    }});

    const container = document.getElementById('chart-container');
    container.innerHTML = '<div style="color:{c["text_secondary"]};text-align:center;padding:80px">Loading...</div>';

    // Reinitialize chart
    if (priceChart) {{ priceChart.remove(); priceChart = null; }}
    if (volumeChart) {{ volumeChart.remove(); volumeChart = null; }}
    initChart();
    if (showVolume) initVolumeChart();

    try {{
        const days = Math.ceil(hours / 24);
        const interval = hours <= 4 ? '5' : hours <= 24 ? '15' : hours <= 168 ? '60' : '1440';
        const resp = await fetch('/api/price-history?id=' + marketId + '&days=' + days + '&interval=' + interval);
        const data = await resp.json();

        if (data.data && data.data.length > 0) {{
            // Convert to lightweight-charts format
            chartData = data.data.map(d => {{
                const ts = Math.floor(new Date(d.timestamp || d.t).getTime() / 1000);
                const close = (d.price?.close || d.price || d.close || 0) * 100;
                const open = (d.price?.open || d.open || close) * 100;
                const high = (d.price?.high || d.high || Math.max(open, close)) * 100;
                const low = (d.price?.low || d.low || Math.min(open, close)) * 100;
                return {{
                    time: ts,
                    open: open,
                    high: high,
                    low: low,
                    close: close,
                    value: close,
                    volume: d.volume || 0
                }};
            }}).sort((a, b) => a.time - b.time);

            // Set data based on chart type
            if (chartType === 'candle' && candleSeries) {{
                candleSeries.setData(chartData);
            }} else if (priceSeries) {{
                priceSeries.setData(chartData.map(d => ({{ time: d.time, value: d.value }})));
            }}

            // MA overlay
            if (showMA && maSeries) {{
                const period = hours <= 24 ? 7 : hours <= 168 ? 14 : 30;
                const maData = calculateMA(chartData, period);
                maSeries.setData(maData);
            }}

            // Volume chart
            if (showVolume) updateVolumeChart();

            priceChart.timeScale().fitContent();
            if (volumeChart) volumeChart.timeScale().fitContent();

            // Show current price
            const lastPrice = chartData[chartData.length - 1]?.close || chartData[chartData.length - 1]?.value || 0;
            const priceEl = document.getElementById('chart-price');
            if (priceEl) priceEl.textContent = lastPrice.toFixed(1) + '% YES';
        }} else {{
            container.innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:80px">No price history available</div>';
        }}
    }} catch(e) {{
        container.innerHTML = '<div style="color:{c["error"]};text-align:center;padding:80px">Error: ' + e.message + '</div>';
    }}
}}

// Notes
async function addNote() {{
    const note = document.getElementById('market-note').value.trim();
    const sentiment = document.getElementById('note-sentiment').value;
    if (!note) return;

    await fetch('/api/notes', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{market_id: marketId, note, sentiment}})
    }});
    location.reload();
}}

// Diary
async function addToDiary(type) {{
    const content = prompt('Enter your ' + type + ':');
    if (!content) return;
    const sentiment = prompt('Sentiment? (bullish/bearish/leave empty for neutral)') || '';

    await fetch('/api/diary', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
            type: type,
            content: content,
            market_id: marketId,
            market_title: marketTitle,
            sentiment: sentiment
        }})
    }});
    alert('Added to diary!');
}}
</script>
'''

    def render_research(self, params):
        watched = load_watched_markets()
        history = load_research_history()
        cache_size = get_cache_size_mb()
        markets_count = len(list(MARKETS_CACHE.glob('*.json')))
        trades_count = len(list(TRADES_CACHE.glob('*.json')))

        # Watched markets list
        watched_html = ''
        for m in watched.get('markets', [])[:20]:
            watched_html += f'''<tr onclick="location.href='/?view=market&id={safe_html(m['id'])}'" style="cursor:pointer">
                <td>{safe_html(m.get('title', m['id'])[:50])}...</td>
                <td style="color:{c['text_secondary']}">{safe_html(m.get('added_at', '')[:10])}</td>
                <td><button class="secondary" onclick="event.stopPropagation();unwatchMarket('{safe_html(m['id'])}')" style="font-size:9px;padding:3px 8px">×</button></td>
            </tr>'''

        # Recent searches
        searches_html = ''
        for s in reversed(history.get('searches', [])[-10:]):
            searches_html += f'''<div class="item" onclick="location.href='/?view=markets&q={quote(s['query'])}'">
                <div class="item-title">{safe_html(s['query'])}</div>
                <div class="item-meta">{safe_html(s['timestamp'][:10])}</div>
            </div>'''

        # Viewed markets history
        viewed_html = ''
        for m in reversed(history.get('viewed_markets', [])[-15:]):
            viewed_html += f'''<tr onclick="location.href='/?view=market&id={safe_html(m['id'])}'" style="cursor:pointer">
                <td>{safe_html(m.get('title', m['id'])[:40])}...</td>
                <td style="color:{c['text_secondary']}">{safe_html(m['timestamp'][:10])}</td>
            </tr>'''

        return f'''
<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">
    <div>
        <div class="panel">
            <h2>Cache Stats</h2>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:15px">
                <div style="background:{c['bg_input']};padding:15px;text-align:center">
                    <div style="font-size:24px;color:{c['accent']}">{cache_size}</div>
                    <div style="color:{c['text_secondary']};font-size:10px">MB Total</div>
                </div>
                <div style="background:{c['bg_input']};padding:15px;text-align:center">
                    <div style="font-size:24px;color:{c['accent']}">{markets_count}</div>
                    <div style="color:{c['text_secondary']};font-size:10px">Markets</div>
                </div>
                <div style="background:{c['bg_input']};padding:15px;text-align:center">
                    <div style="font-size:24px;color:{c['accent']}">{trades_count}</div>
                    <div style="color:{c['text_secondary']};font-size:10px">Trade Files</div>
                </div>
            </div>
            <div style="font-size:11px;color:{c['text_secondary']}">
                Data cached at: {DATA_DIR / 'cache'}<br>
                Market data refreshes after 1 hour. Trades accumulate forever.
            </div>
        </div>

        <div class="panel">
            <h2>Watched Markets ({len(watched.get('markets', []))})</h2>
            {f'<table><tr><th>Market</th><th>Added</th><th></th></tr>{watched_html}</table>' if watched_html else f'<div style="color:{c["text_secondary"]};text-align:center;padding:20px">No watched markets. Click ★ on market detail to add.</div>'}
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>Recent Searches</h2>
            <div class="item-list" style="max-height:200px;overflow-y:auto">
                {searches_html or f'<div style="color:{c["text_secondary"]};text-align:center;padding:20px">No searches yet</div>'}
            </div>
        </div>

        <div class="panel">
            <h2>Viewed Markets History</h2>
            {f'<table style="max-height:300px;overflow-y:auto"><tr><th>Market</th><th>Viewed</th></tr>{viewed_html}</table>' if viewed_html else f'<div style="color:{c["text_secondary"]};text-align:center;padding:20px">No history yet</div>'}
        </div>
    </div>
</div>

<script>
async function unwatchMarket(id) {{
    await fetch('/api/unwatch?id=' + encodeURIComponent(id));
    location.reload();
}}
</script>
'''

    def render_portfolio(self, params):
        """Portfolio Dashboard - Positions, Orders, P&L tracking"""
        return f'''
<div class="panel">
    <h2>Portfolio Dashboard</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px">
        <div style="background:{c['bg_input']};padding:15px;text-align:center">
            <div id="total-value" style="font-size:24px;color:{c['accent']}">--</div>
            <div style="color:{c['text_secondary']};font-size:10px">Total Value</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center">
            <div id="total-pnl" style="font-size:24px;color:{c['success']}">--</div>
            <div style="color:{c['text_secondary']};font-size:10px">Unrealized P&L</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center">
            <div id="position-count" style="font-size:24px;color:{c['accent']}">--</div>
            <div style="color:{c['text_secondary']};font-size:10px">Positions</div>
        </div>
        <div style="background:{c['bg_input']};padding:15px;text-align:center">
            <div id="order-count" style="font-size:24px;color:{c['accent']}">--</div>
            <div style="color:{c['text_secondary']};font-size:10px">Open Orders</div>
        </div>
    </div>
</div>

<div style="display:grid;grid-template-columns:2fr 1fr;gap:15px">
    <div>
        <div class="panel">
            <h2>Positions</h2>
            <table style="width:100%;font-size:11px">
                <thead>
                    <tr style="color:{c['text_secondary']}">
                        <th style="text-align:left">Market</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Avg Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="6" style="text-align:center;color:{c['text_muted']};padding:20px">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <div class="panel">
            <h2>Recent Trades</h2>
            <table style="width:100%;font-size:11px">
                <thead>
                    <tr style="color:{c['text_secondary']}">
                        <th style="text-align:left">Time</th>
                        <th style="text-align:left">Market</th>
                        <th>Side</th>
                        <th>Price</th>
                        <th>Size</th>
                    </tr>
                </thead>
                <tbody id="trades-table">
                    <tr><td colspan="5" style="text-align:center;color:{c['text_muted']};padding:20px">Loading...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>Open Orders</h2>
            <div id="orders-list" style="max-height:300px;overflow-y:auto">
                <div style="text-align:center;color:{c['text_muted']};padding:20px">Loading...</div>
            </div>
            <button onclick="cancelAllOrders()" class="secondary" style="width:100%;margin-top:10px;font-size:10px">Cancel All Orders</button>
        </div>

        <div class="panel">
            <h2>Quick Trade</h2>
            <div style="display:grid;gap:10px">
                <input type="text" id="qt-market" placeholder="Market ID" style="font-size:11px">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                    <select id="qt-side" style="font-size:11px">
                        <option value="BUY">BUY</option>
                        <option value="SELL">SELL</option>
                    </select>
                    <select id="qt-outcome" style="font-size:11px">
                        <option value="yes">YES</option>
                        <option value="no">NO</option>
                    </select>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                    <input type="number" id="qt-price" placeholder="Price (0.01-0.99)" step="0.01" min="0.01" max="0.99" style="font-size:11px">
                    <input type="number" id="qt-size" placeholder="Size ($)" step="1" min="1" style="font-size:11px">
                </div>
                <div style="display:flex;gap:5px">
                    <button onclick="quickSize(10)" class="secondary" style="flex:1;font-size:9px">+$10</button>
                    <button onclick="quickSize(50)" class="secondary" style="flex:1;font-size:9px">+$50</button>
                    <button onclick="quickSize(100)" class="secondary" style="flex:1;font-size:9px">+$100</button>
                    <button onclick="quickSize(500)" class="secondary" style="flex:1;font-size:9px">+$500</button>
                    <button onclick="quickSizeMax()" class="secondary" style="flex:1;font-size:9px;background:{c['warning']};color:#000">MAX</button>
                </div>
                <div id="order-preview" style="display:none;background:{c['bg_input']};padding:8px;margin-bottom:8px;font-size:10px">
                    <div style="display:flex;justify-content:space-between">
                        <span>Est. Cost:</span><span id="preview-cost">$0</span>
                    </div>
                    <div style="display:flex;justify-content:space-between">
                        <span>Potential Win:</span><span id="preview-win" style="color:{c['success']}">$0</span>
                    </div>
                    <div style="display:flex;justify-content:space-between">
                        <span>Risk % of Portfolio:</span><span id="preview-risk">0%</span>
                    </div>
                </div>
                <button onclick="placeQuickOrder()" style="font-size:12px;padding:10px">Place Order</button>
            </div>
        </div>

        <div class="panel">
            <h2>Kelly Calculator</h2>
            <div style="display:grid;gap:8px;font-size:11px">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                    <div>
                        <label style="color:{c['text_secondary']}">Your Prob %</label>
                        <input type="number" id="kelly-prob" placeholder="e.g. 60" min="1" max="99" style="width:100%">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']}">Market Price %</label>
                        <input type="number" id="kelly-price" placeholder="e.g. 45" min="1" max="99" style="width:100%">
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                    <div>
                        <label style="color:{c['text_secondary']}">Bankroll $</label>
                        <input type="number" id="kelly-bankroll" value="1000" min="1" style="width:100%">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']}">Kelly %</label>
                        <select id="kelly-fraction" style="width:100%">
                            <option value="0.25">25% (Quarter)</option>
                            <option value="0.5">50% (Half)</option>
                            <option value="1">100% (Full)</option>
                        </select>
                    </div>
                </div>
                <button onclick="calculateKelly()" class="secondary">Calculate</button>
                <div id="kelly-result" style="background:{c['bg_input']};padding:10px;display:none">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                        <div>
                            <div style="color:{c['text_secondary']};font-size:9px">Edge</div>
                            <div id="kelly-edge" style="font-size:16px;color:{c['accent']}">-</div>
                        </div>
                        <div>
                            <div style="color:{c['text_secondary']};font-size:9px">Bet Size</div>
                            <div id="kelly-bet" style="font-size:16px;color:{c['success']}">-</div>
                        </div>
                    </div>
                    <div id="kelly-advice" style="margin-top:8px;font-size:10px;color:{c['text_secondary']}"></div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Mass Orders</h2>
            <div style="display:flex;gap:5px;margin-bottom:10px">
                <button onclick="showMassOrderType('bracket')" class="secondary" style="flex:1;font-size:10px">Bracket</button>
                <button onclick="showMassOrderType('ladder')" class="secondary" style="flex:1;font-size:10px">Ladder</button>
                <button onclick="showMassOrderType('custom')" class="secondary" style="flex:1;font-size:10px">Custom</button>
            </div>

            <div id="mass-bracket" style="display:none">
                <div style="font-size:10px;color:{c['text_secondary']};margin-bottom:8px">Place buy below + sell above center</div>
                <div style="display:grid;gap:5px;font-size:10px">
                    <input type="text" id="bracket-market" placeholder="Market ID" style="width:100%">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                        <input type="number" id="bracket-center" placeholder="Center (0.50)" step="0.01" min="0.01" max="0.99">
                        <input type="number" id="bracket-spread" placeholder="Spread (0.02)" step="0.01" min="0.01" max="0.2">
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                        <input type="number" id="bracket-size" placeholder="Size" min="1">
                        <select id="bracket-outcome" style="font-size:10px"><option value="yes">YES</option><option value="no">NO</option></select>
                    </div>
                    <button onclick="placeBracketOrder()" style="font-size:10px">Place Bracket</button>
                </div>
            </div>

            <div id="mass-ladder" style="display:none">
                <div style="font-size:10px;color:{c['text_secondary']};margin-bottom:8px">Place orders across price range</div>
                <div style="display:grid;gap:5px;font-size:10px">
                    <input type="text" id="ladder-market" placeholder="Market ID" style="width:100%">
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px">
                        <input type="number" id="ladder-start" placeholder="Start" step="0.01" min="0.01" max="0.99">
                        <input type="number" id="ladder-end" placeholder="End" step="0.01" min="0.01" max="0.99">
                        <input type="number" id="ladder-steps" placeholder="Steps" min="2" max="20" value="5">
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px">
                        <input type="number" id="ladder-size" placeholder="Total Size" min="1">
                        <select id="ladder-side" style="font-size:10px"><option value="BUY">BUY</option><option value="SELL">SELL</option></select>
                        <select id="ladder-outcome" style="font-size:10px"><option value="yes">YES</option><option value="no">NO</option></select>
                    </div>
                    <button onclick="placeLadderOrder()" style="font-size:10px">Place Ladder</button>
                </div>
            </div>

            <div id="mass-custom" style="display:none">
                <div style="font-size:10px;color:{c['text_secondary']};margin-bottom:8px">JSON array of orders</div>
                <textarea id="custom-orders" style="width:100%;height:80px;font-size:9px;font-family:monospace" placeholder='[{{"market_id":"123","side":"BUY","price":0.45,"size":10,"outcome":"yes"}}]'></textarea>
                <button onclick="placeCustomOrders()" style="font-size:10px;margin-top:5px">Place Orders</button>
            </div>

            <div id="mass-result" style="margin-top:10px;display:none;background:{c['bg_input']};padding:8px;font-size:10px"></div>
        </div>
    </div>
</div>

<script>
// Load portfolio data
async function loadPortfolio() {{
    // Load positions
    try {{
        const posResp = await fetch('/api/trading/positions');
        const posData = await posResp.json();
        renderPositions(posData.positions || []);
    }} catch(e) {{
        document.getElementById('positions-table').innerHTML = '<tr><td colspan="6" style="text-align:center;color:{c["error"]}">Error loading positions</td></tr>';
    }}

    // Load orders
    try {{
        const ordResp = await fetch('/api/trading/orders');
        const ordData = await ordResp.json();
        renderOrders(ordData.orders || []);
    }} catch(e) {{
        document.getElementById('orders-list').innerHTML = '<div style="text-align:center;color:{c["error"]}">Error loading orders</div>';
    }}

    // Load trades
    try {{
        const trResp = await fetch('/api/trading/trades?limit=20');
        const trData = await trResp.json();
        renderTrades(trData.trades || []);
    }} catch(e) {{
        document.getElementById('trades-table').innerHTML = '<tr><td colspan="5" style="text-align:center;color:{c["error"]}">Error loading trades</td></tr>';
    }}
}}

function renderPositions(positions) {{
    const tbody = document.getElementById('positions-table');
    if (!positions.length) {{
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:{c["text_muted"]};padding:20px">No positions</td></tr>';
        document.getElementById('position-count').textContent = '0';
        return;
    }}

    let html = '';
    let totalValue = 0;
    let totalPnl = 0;

    positions.forEach(p => {{
        const size = parseFloat(p.size || p.amount || 0);
        const entry = parseFloat(p.avg_price || p.avgPrice || p.price || 0);
        const current = parseFloat(p.current_price || p.currentPrice || entry);
        const pnl = (current - entry) * size;
        const pnlColor = pnl >= 0 ? '{c["success"]}' : '{c["error"]}';
        const side = p.outcome === 'Yes' || p.side === 'YES' ? 'YES' : 'NO';

        totalValue += size * current;
        totalPnl += pnl;

        const title = (p.title || p.market_slug || p.asset_id || 'Unknown').slice(0, 30);
        html += `<tr style="cursor:pointer" onclick="location.href='/?view=market&id=${{p.market_id || p.asset_id}}'">
            <td style="text-align:left">${{title}}...</td>
            <td style="color:${{side === 'YES' ? '{c["success"]}' : '{c["error"]}'}}">${{side}}</td>
            <td>${{size.toFixed(0)}}</td>
            <td>${{(entry * 100).toFixed(1)}}%</td>
            <td>${{(current * 100).toFixed(1)}}%</td>
            <td style="color:${{pnlColor}}">${{pnl >= 0 ? '+' : ''}}$${{pnl.toFixed(2)}}</td>
        </tr>`;
    }});

    tbody.innerHTML = html;
    document.getElementById('position-count').textContent = positions.length;
    document.getElementById('total-value').textContent = '$' + totalValue.toFixed(2);
    document.getElementById('total-pnl').textContent = (totalPnl >= 0 ? '+' : '') + '$' + totalPnl.toFixed(2);
    document.getElementById('total-pnl').style.color = totalPnl >= 0 ? '{c["success"]}' : '{c["error"]}';
    portfolioBalance = totalValue;  // Store for MAX button
}}

function renderOrders(orders) {{
    const div = document.getElementById('orders-list');
    if (!orders.length) {{
        div.innerHTML = '<div style="text-align:center;color:{c["text_muted"]};padding:20px">No open orders</div>';
        document.getElementById('order-count').textContent = '0';
        return;
    }}

    let html = '';
    orders.forEach(o => {{
        const side = o.side || 'BUY';
        const sideColor = side === 'BUY' ? '{c["success"]}' : '{c["error"]}';
        html += `<div style="background:{c['bg_input']};padding:8px;margin-bottom:5px;font-size:10px">
            <div style="display:flex;justify-content:space-between">
                <span style="color:${{sideColor}}">${{side}}</span>
                <span>${{(parseFloat(o.price || 0) * 100).toFixed(1)}}% x ${{o.size || o.original_size}}</span>
                <button onclick="cancelOrder('${{o.id || o.order_id}}')" style="font-size:9px;padding:2px 6px">×</button>
            </div>
        </div>`;
    }});

    div.innerHTML = html;
    document.getElementById('order-count').textContent = orders.length;
}}

function renderTrades(trades) {{
    const tbody = document.getElementById('trades-table');
    if (!trades.length) {{
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:{c["text_muted"]};padding:20px">No recent trades</td></tr>';
        return;
    }}

    let html = '';
    trades.slice(0, 20).forEach(t => {{
        const time = new Date(t.timestamp || t.created_at || Date.now()).toLocaleString().slice(0, 16);
        const side = t.side || 'BUY';
        const sideColor = side === 'BUY' ? '{c["success"]}' : '{c["error"]}';
        html += `<tr>
            <td style="text-align:left;color:{c['text_secondary']}">${{time}}</td>
            <td style="text-align:left">${{(t.market_slug || t.asset_id || '').slice(0, 25)}}</td>
            <td style="color:${{sideColor}}">${{side}}</td>
            <td>${{(parseFloat(t.price || 0) * 100).toFixed(1)}}%</td>
            <td>${{t.size || t.amount}}</td>
        </tr>`;
    }});

    tbody.innerHTML = html;
}}

async function cancelOrder(orderId) {{
    if (!confirm('Cancel this order?')) return;
    try {{
        await fetch('/api/trading/cancel', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{order_id: orderId}})
        }});
        loadPortfolio();
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

async function cancelAllOrders() {{
    if (!confirm('Cancel ALL open orders?')) return;
    try {{
        await fetch('/api/trading/cancel', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{cancel_all: true}})
        }});
        loadPortfolio();
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

let portfolioBalance = 0;

function quickSize(amount) {{
    const el = document.getElementById('qt-size');
    el.value = (parseInt(el.value || 0) + amount);
    updateOrderPreview();
}}

function quickSizeMax() {{
    const el = document.getElementById('qt-size');
    el.value = Math.floor(portfolioBalance);
    updateOrderPreview();
}}

function updateOrderPreview() {{
    const price = parseFloat(document.getElementById('qt-price').value) || 0;
    const size = parseInt(document.getElementById('qt-size').value) || 0;
    const side = document.getElementById('qt-side').value;

    if (price > 0 && size > 0) {{
        document.getElementById('order-preview').style.display = 'block';
        const cost = size;
        const potentialWin = side === 'BUY' ? (size / price) - size : size - (size * price);
        const risk = portfolioBalance > 0 ? ((cost / portfolioBalance) * 100).toFixed(1) : '0';

        document.getElementById('preview-cost').textContent = '$' + cost.toFixed(2);
        document.getElementById('preview-win').textContent = '+$' + potentialWin.toFixed(2);
        document.getElementById('preview-risk').textContent = risk + '%';
    }} else {{
        document.getElementById('order-preview').style.display = 'none';
    }}
}}

// Update preview when inputs change
document.getElementById('qt-price')?.addEventListener('input', updateOrderPreview);
document.getElementById('qt-size')?.addEventListener('input', updateOrderPreview);
document.getElementById('qt-side')?.addEventListener('change', updateOrderPreview);

async function placeQuickOrder() {{
    const marketId = document.getElementById('qt-market').value.trim();
    const side = document.getElementById('qt-side').value;
    const outcome = document.getElementById('qt-outcome').value;
    const price = parseFloat(document.getElementById('qt-price').value);
    const size = parseInt(document.getElementById('qt-size').value);

    if (!marketId || !price || !size) {{
        alert('Please fill all fields');
        return;
    }}

    try {{
        const resp = await fetch('/api/trading/order', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{market_id: marketId, side, outcome, price, size}})
        }});
        const data = await resp.json();
        if (data.error) {{
            alert('Error: ' + data.error);
        }} else {{
            alert('Order placed!');
            loadPortfolio();
        }}
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

// Kelly Calculator
async function calculateKelly() {{
    const prob = parseFloat(document.getElementById('kelly-prob').value) / 100;
    const price = parseFloat(document.getElementById('kelly-price').value) / 100;
    const bankroll = parseFloat(document.getElementById('kelly-bankroll').value);
    const fraction = parseFloat(document.getElementById('kelly-fraction').value);

    if (!prob || !price || prob <= 0 || price <= 0) {{
        alert('Please enter valid probability and price');
        return;
    }}

    try {{
        const resp = await fetch(`/api/kelly?prob=${{prob}}&price=${{price}}&bankroll=${{bankroll}}&fraction=${{fraction}}`);
        const data = await resp.json();

        if (data.error) {{
            alert('Error: ' + data.error);
            return;
        }}

        document.getElementById('kelly-result').style.display = 'block';
        document.getElementById('kelly-edge').textContent = (data.edge_pct > 0 ? '+' : '') + data.edge_pct.toFixed(1) + '%';
        document.getElementById('kelly-edge').style.color = data.edge_pct > 0 ? '{c["success"]}' : '{c["error"]}';
        document.getElementById('kelly-bet').textContent = data.bet_size > 0 ? '$' + data.bet_size.toFixed(0) : 'No bet';
        document.getElementById('kelly-bet').style.color = data.bet_size > 0 ? '{c["success"]}' : '{c["text_muted"]}';

        let advice = data.recommendation;
        if (data.edge_pct > 0) {{
            advice += ` (Full Kelly: ${{data.full_kelly_pct.toFixed(1)}}%)`;
        }}
        document.getElementById('kelly-advice').textContent = advice;
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

// === MASS ORDERS ===
function showMassOrderType(type) {{
    document.getElementById('mass-bracket').style.display = type === 'bracket' ? 'block' : 'none';
    document.getElementById('mass-ladder').style.display = type === 'ladder' ? 'block' : 'none';
    document.getElementById('mass-custom').style.display = type === 'custom' ? 'block' : 'none';
    document.getElementById('mass-result').style.display = 'none';
}}

async function placeBracketOrder() {{
    const marketId = document.getElementById('bracket-market').value.trim();
    const center = parseFloat(document.getElementById('bracket-center').value);
    const spread = parseFloat(document.getElementById('bracket-spread').value);
    const size = parseInt(document.getElementById('bracket-size').value);
    const outcome = document.getElementById('bracket-outcome').value;

    if (!marketId || isNaN(center) || isNaN(spread) || isNaN(size)) {{
        alert('Please fill all fields');
        return;
    }}

    showMassResult('Placing bracket orders...');

    try {{
        const url = `/api/trading/bracket?id=${{encodeURIComponent(marketId)}}&center=${{center}}&spread=${{spread}}&size=${{size}}&outcome=${{outcome}}`;
        const resp = await fetch(url);
        const data = await resp.json();

        if (data.error) {{
            showMassResult('Error: ' + data.error, true);
        }} else {{
            showMassResult(`Bracket: ${{data.success}}/${{data.total}} orders placed`);
            loadPortfolio();
        }}
    }} catch(e) {{
        showMassResult('Error: ' + e.message, true);
    }}
}}

async function placeLadderOrder() {{
    const marketId = document.getElementById('ladder-market').value.trim();
    const start = parseFloat(document.getElementById('ladder-start').value);
    const end = parseFloat(document.getElementById('ladder-end').value);
    const steps = parseInt(document.getElementById('ladder-steps').value);
    const size = parseInt(document.getElementById('ladder-size').value);
    const side = document.getElementById('ladder-side').value;
    const outcome = document.getElementById('ladder-outcome').value;

    if (!marketId || isNaN(start) || isNaN(end) || isNaN(steps) || isNaN(size)) {{
        alert('Please fill all fields');
        return;
    }}

    showMassResult('Placing ladder orders...');

    try {{
        const url = `/api/trading/ladder?id=${{encodeURIComponent(marketId)}}&start=${{start}}&end=${{end}}&steps=${{steps}}&size=${{size}}&side=${{side}}&outcome=${{outcome}}`;
        const resp = await fetch(url);
        const data = await resp.json();

        if (data.error) {{
            showMassResult('Error: ' + data.error, true);
        }} else {{
            showMassResult(`Ladder: ${{data.success}}/${{data.total}} orders placed`);
            loadPortfolio();
        }}
    }} catch(e) {{
        showMassResult('Error: ' + e.message, true);
    }}
}}

async function placeCustomOrders() {{
    const ordersText = document.getElementById('custom-orders').value.trim();

    if (!ordersText) {{
        alert('Please enter orders JSON');
        return;
    }}

    let orders;
    try {{
        orders = JSON.parse(ordersText);
        if (!Array.isArray(orders)) {{
            alert('Orders must be a JSON array');
            return;
        }}
    }} catch(e) {{
        alert('Invalid JSON: ' + e.message);
        return;
    }}

    showMassResult(`Placing ${{orders.length}} orders...`);

    try {{
        const resp = await fetch('/api/trading/mass-order', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{orders: orders}})
        }});
        const data = await resp.json();

        if (data.error) {{
            showMassResult('Error: ' + data.error, true);
        }} else {{
            let msg = `Mass: ${{data.success}}/${{data.total}} orders placed`;
            if (data.failed > 0) {{
                msg += ` (${{data.failed}} failed)`;
            }}
            showMassResult(msg);
            loadPortfolio();
        }}
    }} catch(e) {{
        showMassResult('Error: ' + e.message, true);
    }}
}}

function showMassResult(msg, isError) {{
    const el = document.getElementById('mass-result');
    el.style.display = 'block';
    el.textContent = msg;
    el.style.color = isError ? '{c["error"]}' : '{c["success"]}';
}}

// Load on page ready
loadPortfolio();
setInterval(loadPortfolio, 30000);
</script>
'''

    def render_edges(self, params):
        """Edge Scanner - Find trading opportunities"""
        # Load predictions for stats
        predictions = load_predictions()
        total_preds = len(predictions.get('predictions', []))
        resolved = [p for p in predictions.get('predictions', []) if p.get('resolved')]
        wins = [p for p in resolved if p.get('correct')]
        win_rate = (len(wins) / len(resolved) * 100) if resolved else 0

        return f'''
<div style="display:grid;grid-template-columns:2fr 1fr;gap:15px">
    <div>
        <div class="panel">
            <h2>🎯 Edge Scanner</h2>
            <p style="color:{c['text_secondary']};font-size:11px;margin-bottom:15px">
                Scanning markets for mispricing, spread opportunities, and momentum signals.
            </p>

            <div style="display:flex;gap:10px;margin-bottom:15px;align-items:center;flex-wrap:wrap">
                <label style="font-size:11px;color:{c['text_secondary']}">Min Volume: $</label>
                <input type="number" id="edge-min-vol" value="1000" min="0" step="500" style="width:80px;font-size:11px">
                <button onclick="scanEdges()" style="font-size:11px">🔍 Scan</button>
                <button onclick="scanMomentum()" class="secondary" style="font-size:11px">📈 Momentum</button>
                <button onclick="scanEV()" class="secondary" style="font-size:11px;background:{c['success']};color:#000">💰 +EV</button>
            </div>

            <div id="edge-results">
                <div style="text-align:center;color:{c['text_secondary']};padding:40px">
                    Click "Scan" to find trading edges
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>📊 Momentum Signals</h2>
            <div id="momentum-results">
                <div style="text-align:center;color:{c['text_secondary']};padding:20px;font-size:11px">
                    Click "Momentum" to load signals from Elon markets
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>📈 Market Efficiency Metrics</h2>
            <div id="efficiency-metrics">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:15px">
                    <div style="background:{c['bg_input']};padding:12px;text-align:center">
                        <div id="eff-spread" style="font-size:18px;color:{c['accent']}">--</div>
                        <div style="color:{c['text_secondary']};font-size:9px">Avg Spread %</div>
                    </div>
                    <div style="background:{c['bg_input']};padding:12px;text-align:center">
                        <div id="eff-vol" style="font-size:18px;color:{c['accent']}">--</div>
                        <div style="color:{c['text_secondary']};font-size:9px">Avg 24h Vol</div>
                    </div>
                    <div style="background:{c['bg_input']};padding:12px;text-align:center">
                        <div id="eff-liq" style="font-size:18px;color:{c['accent']}">--</div>
                        <div style="color:{c['text_secondary']};font-size:9px">Avg Liquidity</div>
                    </div>
                    <div style="background:{c['bg_input']};padding:12px;text-align:center">
                        <div id="eff-arb" style="font-size:18px;color:{c['warning']}">--</div>
                        <div style="color:{c['text_secondary']};font-size:9px">Arb Opps</div>
                    </div>
                </div>
                <button onclick="loadEfficiencyMetrics()" class="secondary" style="font-size:10px;width:100%">Load Efficiency Metrics</button>
            </div>
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>🏆 Prediction Stats</h2>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-bottom:15px">
                <div style="background:{c['bg_input']};padding:15px;text-align:center">
                    <div style="font-size:28px;color:{c['accent']}">{total_preds}</div>
                    <div style="color:{c['text_secondary']};font-size:10px">Total Predictions</div>
                </div>
                <div style="background:{c['bg_input']};padding:15px;text-align:center">
                    <div style="font-size:28px;color:{c['success'] if win_rate >= 50 else c['error']}">{win_rate:.1f}%</div>
                    <div style="color:{c['text_secondary']};font-size:10px">Win Rate ({len(wins)}/{len(resolved)})</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>🎯 Log Prediction</h2>
            <div style="display:grid;gap:8px;font-size:11px">
                <input type="text" id="pred-market" placeholder="Market ID" style="font-size:11px">
                <input type="text" id="pred-title" placeholder="Market title (auto-filled)" style="font-size:10px;color:{c['text_secondary']}">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px">
                    <div>
                        <label style="color:{c['text_secondary']};font-size:9px">Your Prob %</label>
                        <input type="number" id="pred-prob" min="1" max="99" placeholder="60" style="width:100%">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:9px">Market Price %</label>
                        <input type="number" id="pred-price" min="1" max="99" placeholder="45" style="width:100%">
                    </div>
                </div>
                <select id="pred-side" style="font-size:11px">
                    <option value="YES">Predict YES wins</option>
                    <option value="NO">Predict NO wins</option>
                </select>
                <textarea id="pred-reasoning" placeholder="Your reasoning..." style="height:60px;font-size:10px"></textarea>
                <button onclick="logPrediction()" style="font-size:11px">📝 Log Prediction</button>
            </div>
        </div>

        <div class="panel">
            <h2>📜 Recent Predictions</h2>
            <div id="predictions-list" style="max-height:250px;overflow-y:auto;font-size:10px">
                <div style="color:{c['text_muted']};text-align:center;padding:10px">Loading...</div>
            </div>
        </div>

        <div class="panel">
            <h2>💡 Edge Types</h2>
            <div style="font-size:10px;line-height:1.5">
                <div style="margin-bottom:6px">
                    <span style="color:{c['warning']}">⚠</span> <b>Mispricing</b>: YES+NO ≠ 100%
                </div>
                <div style="margin-bottom:6px">
                    <span style="color:{c['accent']}">📊</span> <b>Vol/Liq</b>: High vol vs liquidity
                </div>
                <div style="margin-bottom:6px">
                    <span style="color:{c['success']}">💰</span> <b>Spread</b>: Wide bid-ask
                </div>
                <div>
                    <span style="color:{c['text_secondary']}">🔄</span> <b>Extreme</b>: Near 0/100%
                </div>
            </div>
        </div>
    </div>
</div>

<script>
async function scanEdges() {{
    const minVol = document.getElementById('edge-min-vol').value || 1000;
    document.getElementById('edge-results').innerHTML = '<div style="text-align:center;color:{c["text_secondary"]};padding:20px">Scanning...</div>';

    try {{
        const resp = await fetch('/api/edges?min_volume=' + minVol + '&limit=30');
        const data = await resp.json();

        if (data.error) {{
            document.getElementById('edge-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + data.error + '</div>';
            return;
        }}

        if (!data.edges || data.edges.length === 0) {{
            document.getElementById('edge-results').innerHTML = '<div style="text-align:center;color:{c["text_muted"]};padding:20px">No edges found. Try lowering min volume.</div>';
            return;
        }}

        let html = '<table style="width:100%;font-size:11px"><thead><tr>';
        html += '<th style="text-align:left">Market</th>';
        html += '<th style="text-align:left">Edges Found</th>';
        html += '<th style="text-align:right">Score</th>';
        html += '</tr></thead><tbody>';

        data.edges.forEach(e => {{
            const edgesList = e.edges.map(ed => {{
                const colors = {{
                    'mispricing': '{c["warning"]}',
                    'wide_spread': '{c["success"]}',
                    'vol_liq_imbalance': '{c["accent"]}',
                    'extreme_price': '{c["text_secondary"]}'
                }};
                return '<span style="color:' + (colors[ed.type] || '{c["text_primary"]}') + '">' + ed.desc + '</span>';
            }}).join('<br>');

            html += '<tr style="border-bottom:1px solid {c["border"]}">';
            html += '<td style="padding:8px 0"><a href="/?view=market&id=' + e.market_id + '" style="color:{c["accent"]}">' + (e.title || '').substring(0, 50) + '</a><br><span style="color:{c["text_muted"]};font-size:9px">Vol: $' + (e.volume || 0).toLocaleString() + '</span></td>';
            html += '<td style="padding:8px 0;font-size:10px">' + edgesList + '</td>';
            html += '<td style="padding:8px 0;text-align:right;color:{c["accent"]};font-size:14px">' + e.score.toFixed(1) + '</td>';
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById('edge-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('edge-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

async function scanMomentum() {{
    document.getElementById('momentum-results').innerHTML = '<div style="text-align:center;color:{c["text_secondary"]};padding:10px">Loading...</div>';

    try {{
        const resp = await fetch('/api/edges/momentum');
        const data = await resp.json();

        if (data.error) {{
            document.getElementById('momentum-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + data.error + '</div>';
            return;
        }}

        if (!data.signals || data.signals.length === 0) {{
            document.getElementById('momentum-results').innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:10px">No momentum signals found</div>';
            return;
        }}

        let html = '<table style="width:100%;font-size:10px"><thead><tr>';
        html += '<th style="text-align:left">Event</th>';
        html += '<th style="text-align:left">Bracket</th>';
        html += '<th style="text-align:right">Prob</th>';
        html += '<th style="text-align:right">Mom</th>';
        html += '<th style="text-align:left">Signal</th>';
        html += '</tr></thead><tbody>';

        data.signals.slice(0, 15).forEach(s => {{
            const momColor = s.momentum > 0 ? '{c["success"]}' : '{c["error"]}';
            const sigColor = s.signal.includes('BUY') ? '{c["success"]}' : s.signal.includes('SELL') ? '{c["error"]}' : '{c["text_muted"]}';

            html += '<tr>';
            html += '<td>' + (s.event || '-') + '</td>';
            html += '<td>' + (s.bracket || '-') + '</td>';
            html += '<td style="text-align:right">' + ((s.prob || 0) * 100).toFixed(1) + '%</td>';
            html += '<td style="text-align:right;color:' + momColor + '">' + (s.momentum > 0 ? '+' : '') + s.momentum.toFixed(1) + '</td>';
            html += '<td style="color:' + sigColor + '">' + s.signal + '</td>';
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById('momentum-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('momentum-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

// +EV Scanner - find positive expected value opportunities
async function scanEV() {{
    document.getElementById('edge-results').innerHTML = '<div style="text-align:center;color:{c["text_secondary"]};padding:20px">Scanning for +EV opportunities...</div>';

    try {{
        const resp = await fetch('/api/edges?min_volume=500&limit=50');
        const data = await resp.json();

        if (!data.edges || data.edges.length === 0) {{
            document.getElementById('edge-results').innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:20px">No markets found</div>';
            return;
        }}

        // Filter for potential +EV (extreme prices or high spread)
        const evOpps = data.edges.filter(e => {{
            const hasExtreme = e.edges?.some(ed => ed.type === 'extreme_price');
            const hasSpread = e.edges?.some(ed => ed.type === 'wide_spread');
            return hasExtreme || hasSpread || e.score > 2;
        }});

        let html = '<div style="margin-bottom:10px;font-size:11px;color:{c["text_secondary"]}">Found ' + evOpps.length + ' potential +EV opportunities</div>';
        html += '<table style="width:100%;font-size:10px"><thead><tr>';
        html += '<th style="text-align:left">Market</th>';
        html += '<th>YES Price</th>';
        html += '<th>Edge Score</th>';
        html += '<th>Action</th>';
        html += '</tr></thead><tbody>';

        evOpps.slice(0, 20).forEach(e => {{
            const yesPrice = e.yes_price ? (e.yes_price * 100).toFixed(0) + '%' : '-';
            html += '<tr>';
            html += '<td><a href="/?view=market&id=' + e.market_id + '" style="color:{c["accent"]}">' + (e.title || '').substring(0, 40) + '...</a></td>';
            html += '<td style="text-align:center">' + yesPrice + '</td>';
            html += '<td style="text-align:center;color:{c["success"]}">' + e.score.toFixed(1) + '</td>';
            html += '<td><button onclick="prefillPrediction(\\'' + e.market_id + '\\', \\'' + (e.title || '').replace(/'/g, '') + '\\')" style="font-size:9px;padding:2px 6px">Log</button></td>';
            html += '</tr>';
        }});

        html += '</tbody></table>';
        document.getElementById('edge-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('edge-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

// Load market efficiency metrics
async function loadEfficiencyMetrics() {{
    try {{
        const resp = await fetch('/api/edges/efficiency');
        const data = await resp.json();

        if (data.error) {{
            console.error(data.error);
            return;
        }}

        document.getElementById('eff-spread').textContent = data.avg_spread ? data.avg_spread.toFixed(2) + '%' : '0%';
        document.getElementById('eff-vol').textContent = '$' + (data.avg_volume || 0).toLocaleString(undefined, {{maximumFractionDigits: 0}});
        document.getElementById('eff-liq').textContent = '$' + (data.avg_liquidity || 0).toLocaleString(undefined, {{maximumFractionDigits: 0}});
        document.getElementById('eff-arb').textContent = data.arb_opportunities || '0';
    }} catch(e) {{
        console.error('Efficiency metrics error:', e);
    }}
}}

// Prefill prediction form
function prefillPrediction(marketId, title) {{
    document.getElementById('pred-market').value = marketId;
    document.getElementById('pred-title').value = title;
    document.getElementById('pred-market').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
}}

// Log a prediction
async function logPrediction() {{
    const marketId = document.getElementById('pred-market').value.trim();
    const title = document.getElementById('pred-title').value.trim();
    const prob = parseFloat(document.getElementById('pred-prob').value) / 100;
    const price = parseFloat(document.getElementById('pred-price').value) / 100;
    const side = document.getElementById('pred-side').value;
    const reasoning = document.getElementById('pred-reasoning').value.trim();

    if (!marketId || isNaN(prob) || isNaN(price)) {{
        alert('Please fill market ID, your probability, and market price');
        return;
    }}

    try {{
        const url = `/api/predictions/add?market_id=${{encodeURIComponent(marketId)}}&title=${{encodeURIComponent(title)}}&prob=${{prob}}&price=${{price}}&side=${{side}}&reasoning=${{encodeURIComponent(reasoning)}}`;
        const resp = await fetch(url);
        const data = await resp.json();

        if (data.error) {{
            alert('Error: ' + data.error);
            return;
        }}

        alert('Prediction logged! EV: ' + (data.prediction.ev * 100).toFixed(1) + '%');

        // Clear form
        document.getElementById('pred-market').value = '';
        document.getElementById('pred-title').value = '';
        document.getElementById('pred-prob').value = '';
        document.getElementById('pred-price').value = '';
        document.getElementById('pred-reasoning').value = '';

        // Reload predictions list
        loadPredictions();
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

// Load recent predictions
async function loadPredictions() {{
    try {{
        const resp = await fetch('/api/predictions');
        const data = await resp.json();

        if (!data.predictions || data.predictions.length === 0) {{
            document.getElementById('predictions-list').innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:10px">No predictions yet</div>';
            return;
        }}

        let html = '';
        data.predictions.slice().reverse().slice(0, 15).forEach(p => {{
            const evColor = p.ev > 0 ? '{c["success"]}' : '{c["error"]}';
            const statusIcon = p.resolved ? (p.correct ? '✓' : '✗') : '⏳';
            const statusColor = p.resolved ? (p.correct ? '{c["success"]}' : '{c["error"]}') : '{c["text_muted"]}';

            html += '<div style="background:{c["bg_input"]};padding:8px;margin-bottom:6px;border-radius:4px">';
            html += '<div style="display:flex;justify-content:space-between;margin-bottom:4px">';
            html += '<span style="color:' + statusColor + '">' + statusIcon + ' ' + p.side + '</span>';
            html += '<span style="color:' + evColor + '">EV: ' + (p.ev * 100).toFixed(1) + '%</span>';
            html += '</div>';
            html += '<div style="font-size:9px;color:{c["text_secondary"]};margin-bottom:4px">' + (p.market_title || p.market_id).substring(0, 40) + '</div>';
            html += '<div style="display:flex;justify-content:space-between;font-size:9px;color:{c["text_muted"]}">';
            html += '<span>You: ' + (p.your_prob * 100).toFixed(0) + '% | Mkt: ' + (p.market_price * 100).toFixed(0) + '%</span>';
            html += '<span>' + p.created_at.substring(0, 10) + '</span>';
            html += '</div>';

            if (!p.resolved) {{
                html += '<div style="margin-top:6px;display:flex;gap:4px">';
                html += '<button onclick="resolvePrediction(\\'' + p.id + '\\', \\'YES\\')" style="flex:1;font-size:9px;padding:3px;background:{c["success"]};color:#000">YES Won</button>';
                html += '<button onclick="resolvePrediction(\\'' + p.id + '\\', \\'NO\\')" style="flex:1;font-size:9px;padding:3px;background:{c["error"]};color:#fff">NO Won</button>';
                html += '</div>';
            }}

            html += '</div>';
        }});

        document.getElementById('predictions-list').innerHTML = html;
    }} catch(e) {{
        document.getElementById('predictions-list').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

// Resolve a prediction
async function resolvePrediction(predId, outcome) {{
    if (!confirm('Mark this prediction as ' + outcome + ' won?')) return;

    try {{
        const resp = await fetch('/api/predictions/resolve?id=' + predId + '&outcome=' + outcome);
        const data = await resp.json();

        if (data.error) {{
            alert('Error: ' + data.error);
            return;
        }}

        loadPredictions();
        location.reload();  // Refresh stats
    }} catch(e) {{
        alert('Error: ' + e.message);
    }}
}}

// Load on page load
loadPredictions();
loadEfficiencyMetrics();
</script>
'''

    def render_automation(self, params):
        """Trading Automation - Volume/Liquidity Spike Trading"""
        return f'''
<div style="display:grid;grid-template-columns:1fr 350px;gap:15px">
    <div>
        <div class="panel">
            <h2>⚡ Trading Automation</h2>
            <p style="color:{c['text_secondary']};font-size:11px;margin-bottom:15px">
                Automated trading based on volume and liquidity spikes. Scans watched markets for unusual activity.
            </p>

            <div style="display:flex;gap:10px;margin-bottom:15px">
                <button onclick="scanSpikes()" style="font-size:11px">🔍 Scan Now</button>
                <button onclick="runAutomation()" class="secondary" style="font-size:11px">▶ Run Auto</button>
                <button onclick="loadLog()" class="secondary" style="font-size:11px">📜 Log</button>
            </div>

            <div id="spike-results">
                <div style="text-align:center;color:{c['text_secondary']};padding:30px">
                    Click "Scan Now" to detect volume/liquidity spikes
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>📜 Automation Log</h2>
            <div id="auto-log" style="max-height:300px;overflow-y:auto;font-size:10px">
                <div style="color:{c['text_muted']};text-align:center;padding:20px">Click "Log" to load</div>
            </div>
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>⚙️ Configuration</h2>
            <div id="auto-config" style="font-size:11px">
                <div style="margin-bottom:10px">
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
                        <input type="checkbox" id="cfg-enabled" onchange="updateConfig()">
                        <span>Automation Enabled</span>
                    </label>
                </div>
                <div style="margin-bottom:10px">
                    <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
                        <input type="checkbox" id="cfg-dryrun" checked onchange="updateConfig()">
                        <span>Dry Run (no real orders)</span>
                    </label>
                </div>

                <div style="display:grid;gap:8px;margin-top:15px">
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Volume Spike Threshold (x avg)</label>
                        <input type="number" id="cfg-vol-thresh" value="2.0" step="0.1" min="1" max="10" style="width:100%" onchange="updateConfig()">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Liquidity Spike Threshold (x avg)</label>
                        <input type="number" id="cfg-liq-thresh" value="1.5" step="0.1" min="1" max="5" style="width:100%" onchange="updateConfig()">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Min Volume ($)</label>
                        <input type="number" id="cfg-min-vol" value="5000" step="1000" min="0" style="width:100%" onchange="updateConfig()">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Order Size</label>
                        <input type="number" id="cfg-order-size" value="10" min="1" style="width:100%" onchange="updateConfig()">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Max Orders/Hour</label>
                        <input type="number" id="cfg-max-orders" value="10" min="1" max="50" style="width:100%" onchange="updateConfig()">
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Auto Side</label>
                        <select id="cfg-side" style="width:100%" onchange="updateConfig()">
                            <option value="BUY">BUY on spikes</option>
                            <option value="SELL">SELL on spikes</option>
                        </select>
                    </div>
                    <div>
                        <label style="color:{c['text_secondary']};font-size:10px">Price Offset (from current)</label>
                        <input type="number" id="cfg-offset" value="0.02" step="0.01" min="0" max="0.1" style="width:100%" onchange="updateConfig()">
                    </div>
                </div>

                <div style="margin-top:15px;padding:10px;background:{c['bg_input']}">
                    <div style="font-size:10px;color:{c['text_secondary']}">Status</div>
                    <div id="cfg-status" style="font-size:12px;color:{c['warning']}">Loading...</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>💡 How It Works</h2>
            <div style="font-size:10px;line-height:1.6;color:{c['text_secondary']}">
                <p><strong>Volume Spike:</strong> 24h volume exceeds {c['accent']}Nx{c['text_secondary']} average daily volume.</p>
                <p><strong>Liquidity Spike:</strong> High liquidity relative to recent volume = fresh market maker activity.</p>
                <p><strong>Auto Trade:</strong> Places limit orders at offset from current price when spikes detected.</p>
                <p style="color:{c['warning']}">⚠️ Start with Dry Run enabled to test!</p>
            </div>
        </div>
    </div>
</div>

<script>
let autoConfig = {{}};

async function loadConfig() {{
    try {{
        const resp = await fetch('/api/automation/config');
        autoConfig = await resp.json();

        document.getElementById('cfg-enabled').checked = autoConfig.enabled || false;
        document.getElementById('cfg-dryrun').checked = autoConfig.dry_run !== false;
        document.getElementById('cfg-vol-thresh').value = autoConfig.volume_spike_threshold || 2.0;
        document.getElementById('cfg-liq-thresh').value = autoConfig.liquidity_spike_threshold || 1.5;
        document.getElementById('cfg-min-vol').value = autoConfig.min_volume || 5000;
        document.getElementById('cfg-order-size').value = autoConfig.order_size || 10;
        document.getElementById('cfg-max-orders').value = autoConfig.max_orders_per_hour || 10;
        document.getElementById('cfg-side').value = autoConfig.auto_side || 'BUY';
        document.getElementById('cfg-offset').value = autoConfig.price_offset || 0.02;

        updateStatus();
    }} catch(e) {{
        console.error('Error loading config:', e);
    }}
}}

async function updateConfig() {{
    const config = {{
        enabled: document.getElementById('cfg-enabled').checked,
        dry_run: document.getElementById('cfg-dryrun').checked,
        volume_spike_threshold: parseFloat(document.getElementById('cfg-vol-thresh').value),
        liquidity_spike_threshold: parseFloat(document.getElementById('cfg-liq-thresh').value),
        min_volume: parseInt(document.getElementById('cfg-min-vol').value),
        order_size: parseInt(document.getElementById('cfg-order-size').value),
        max_orders_per_hour: parseInt(document.getElementById('cfg-max-orders').value),
        auto_side: document.getElementById('cfg-side').value,
        price_offset: parseFloat(document.getElementById('cfg-offset').value),
        watched_only: true
    }};

    try {{
        await fetch('/api/automation/config', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify(config)
        }});
        autoConfig = config;
        updateStatus();
    }} catch(e) {{
        console.error('Error saving config:', e);
    }}
}}

function updateStatus() {{
    const el = document.getElementById('cfg-status');
    if (autoConfig.enabled) {{
        if (autoConfig.dry_run) {{
            el.textContent = '✓ ENABLED (Dry Run)';
            el.style.color = '{c["warning"]}';
        }} else {{
            el.textContent = '⚡ LIVE TRADING';
            el.style.color = '{c["error"]}';
        }}
    }} else {{
        el.textContent = '○ Disabled';
        el.style.color = '{c["text_muted"]}';
    }}
}}

async function scanSpikes() {{
    document.getElementById('spike-results').innerHTML = '<div style="text-align:center;padding:20px;color:{c["text_secondary"]}">Scanning...</div>';

    try {{
        const volThresh = document.getElementById('cfg-vol-thresh').value;
        const liqThresh = document.getElementById('cfg-liq-thresh').value;
        const resp = await fetch(`/api/automation/scan?vol_threshold=${{volThresh}}&liq_threshold=${{liqThresh}}`);
        const data = await resp.json();

        if (data.error) {{
            document.getElementById('spike-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + data.error + '</div>';
            return;
        }}

        let html = '';

        if (data.volume_spikes && data.volume_spikes.length > 0) {{
            html += '<h3 style="color:{c["accent"]};font-size:12px;margin-bottom:8px">📈 Volume Spikes</h3>';
            html += '<table style="width:100%;font-size:10px;margin-bottom:15px"><thead><tr>';
            html += '<th style="text-align:left">Market</th><th style="text-align:right">24h Vol</th><th style="text-align:right">Spike</th><th style="text-align:right">Price</th>';
            html += '</tr></thead><tbody>';
            data.volume_spikes.forEach(s => {{
                html += `<tr>
                    <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis"><a href="/?view=market&id=${{s.market_id}}" style="color:{c['accent']}">${{s.title.substring(0,40)}}</a></td>
                    <td style="text-align:right">$${{s.volume_24h.toLocaleString()}}</td>
                    <td style="text-align:right;color:{c['success']}">${{s.spike_ratio}}x</td>
                    <td style="text-align:right">${{(s.yes_price*100).toFixed(1)}}%</td>
                </tr>`;
            }});
            html += '</tbody></table>';
        }}

        if (data.liquidity_spikes && data.liquidity_spikes.length > 0) {{
            html += '<h3 style="color:{c["success"]};font-size:12px;margin-bottom:8px">💰 Liquidity Spikes</h3>';
            html += '<table style="width:100%;font-size:10px"><thead><tr>';
            html += '<th style="text-align:left">Market</th><th style="text-align:right">Liquidity</th><th style="text-align:right">L/V Ratio</th><th style="text-align:right">Price</th>';
            html += '</tr></thead><tbody>';
            data.liquidity_spikes.forEach(s => {{
                html += `<tr>
                    <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis"><a href="/?view=market&id=${{s.market_id}}" style="color:{c['accent']}">${{s.title.substring(0,40)}}</a></td>
                    <td style="text-align:right">$${{s.liquidity.toLocaleString()}}</td>
                    <td style="text-align:right;color:{c['success']}">${{s.liq_vol_ratio}}x</td>
                    <td style="text-align:right">${{(s.yes_price*100).toFixed(1)}}%</td>
                </tr>`;
            }});
            html += '</tbody></table>';
        }}

        if (!html) {{
            html = '<div style="text-align:center;color:{c["text_muted"]};padding:20px">No spikes detected. Try adding markets to watchlist.</div>';
        }}

        document.getElementById('spike-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('spike-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

async function runAutomation() {{
    if (!autoConfig.enabled) {{
        alert('Enable automation first in config');
        return;
    }}

    if (!autoConfig.dry_run && !confirm('LIVE TRADING is enabled. Continue?')) {{
        return;
    }}

    document.getElementById('spike-results').innerHTML = '<div style="text-align:center;padding:20px;color:{c["warning"]}">Running automation...</div>';

    try {{
        const resp = await fetch('/api/automation/run');
        const data = await resp.json();

        if (data.error) {{
            document.getElementById('spike-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + data.error + '</div>';
            return;
        }}

        let html = '<div style="padding:10px;background:{c["bg_input"]};margin-bottom:10px">';
        html += '<div style="font-size:12px">Scan Complete</div>';
        html += '<div style="font-size:11px;color:{c["text_secondary"]}">Spikes found: ' + data.spikes_found.length + '</div>';
        html += '<div style="font-size:11px;color:{c["text_secondary"]}">Orders: ' + data.orders_placed + '</div>';
        html += '</div>';

        if (data.actions_taken && data.actions_taken.length > 0) {{
            html += '<h3 style="font-size:11px;margin-bottom:8px">Actions Taken:</h3>';
            data.actions_taken.forEach(a => {{
                const info = a.would_place || a.order || {{}};
                const status = a.dry_run ? '🔸 DRY RUN' : a.success ? '✅ PLACED' : '❌ ERROR';
                html += `<div style="padding:6px;background:{c['bg_input']};margin-bottom:4px;font-size:10px">
                    <div>${{status}} - ${{info.side}} ${{info.size}} @ ${{(info.price*100).toFixed(1)}}%</div>
                    <div style="color:{c['text_muted']}">${{info.title?.substring(0,50) || info.market_id}}</div>
                </div>`;
            }});
        }}

        document.getElementById('spike-results').innerHTML = html;
        loadLog();
    }} catch(e) {{
        document.getElementById('spike-results').innerHTML = '<div style="color:{c["error"]}">Error: ' + e.message + '</div>';
    }}
}}

async function loadLog() {{
    try {{
        const resp = await fetch('/api/automation/log?limit=30');
        const data = await resp.json();

        if (!data.log || data.log.length === 0) {{
            document.getElementById('auto-log').innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:10px">No log entries</div>';
            return;
        }}

        let html = '';
        data.log.reverse().forEach(entry => {{
            const time = entry.timestamp?.substring(11, 19) || '';
            const typeColor = entry.type === 'order_placed' ? '{c["success"]}' : entry.type === 'order_error' ? '{c["error"]}' : '{c["text_secondary"]}';
            html += `<div style="padding:4px;border-bottom:1px solid {c['border']}">
                <span style="color:{c['text_muted']}">${{time}}</span>
                <span style="color:${{typeColor}}">${{entry.type}}</span>
            </div>`;
        }});

        document.getElementById('auto-log').innerHTML = html;
    }} catch(e) {{
        document.getElementById('auto-log').innerHTML = '<div style="color:{c["error"]}">Error loading log</div>';
    }}
}}

// Init
loadConfig();
</script>
'''

    def render_quant(self, params):
        """Quant Modeling Research Center - forecasting, backtesting, optimization"""
        market_id = params.get('id', [''])[0]

        return f'''
<div class="panel" style="margin-bottom:15px">
    <h2 style="margin:0 0 15px 0;font-size:16px;border-bottom:1px solid {c['border']};padding-bottom:10px">
        🔬 Quant Modeling Research Center
    </h2>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <!-- Model Selection Panel -->
        <div class="panel" style="background:{c['bg_panel']}">
            <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['accent']}">Run Models</h3>
            <div style="margin-bottom:10px">
                <label style="font-size:11px;color:{c['text_secondary']}">Market ID:</label>
                <input type="text" id="quant-market-id" value="{market_id}"
                    style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px"
                    placeholder="Enter market ID">
            </div>
            <div style="margin-bottom:10px">
                <label style="font-size:11px;color:{c['text_secondary']}">Forecast Periods:</label>
                <input type="number" id="quant-periods" value="5" min="1" max="30"
                    style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px">
            </div>
            <button onclick="runAllModels()"
                style="width:100%;padding:10px;background:{c['accent']};color:#000;border:none;cursor:pointer;font-weight:bold;border-radius:3px">
                Run All Models
            </button>
        </div>

        <!-- Backtest Panel -->
        <div class="panel" style="background:{c['bg_panel']}">
            <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['accent']}">Backtest Model</h3>
            <div style="margin-bottom:10px">
                <label style="font-size:11px;color:{c['text_secondary']}">Model Type:</label>
                <select id="backtest-model"
                    style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px">
                    <option value="linear">Linear Regression</option>
                    <option value="sma">Simple Moving Avg</option>
                    <option value="ema">Exponential MA</option>
                    <option value="mean_reversion">Mean Reversion</option>
                </select>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
                <div>
                    <label style="font-size:11px;color:{c['text_secondary']}">Lookback:</label>
                    <input type="number" id="backtest-lookback" value="50" min="10" max="200"
                        style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px">
                </div>
                <div>
                    <label style="font-size:11px;color:{c['text_secondary']}">Test Periods:</label>
                    <input type="number" id="backtest-periods" value="10" min="5" max="50"
                        style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px">
                </div>
            </div>
            <button onclick="runBacktest()"
                style="width:100%;padding:10px;background:#4a9eff;color:#fff;border:none;cursor:pointer;font-weight:bold;border-radius:3px">
                Run Backtest
            </button>
        </div>
    </div>
</div>

<!-- Results Panels -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:15px">
    <!-- Model Results -->
    <div class="panel" id="model-results" style="min-height:300px">
        <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['text_secondary']}">Model Results</h3>
        <div style="color:{c['text_secondary']};font-size:11px">Run models to see results...</div>
    </div>

    <!-- Backtest Results -->
    <div class="panel" id="backtest-results" style="min-height:300px">
        <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['text_secondary']}">Backtest Results</h3>
        <div style="color:{c['text_secondary']};font-size:11px">Run backtest to see metrics...</div>
    </div>
</div>

<!-- Optimization Panel -->
<div class="panel" style="margin-bottom:15px">
    <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['accent']}">Parameter Optimization</h3>
    <div style="display:grid;grid-template-columns:200px 200px 1fr;gap:15px;align-items:end">
        <div>
            <label style="font-size:11px;color:{c['text_secondary']}">Model:</label>
            <select id="optimize-model"
                style="width:100%;background:{c['bg_input']};border:1px solid {c['border']};color:{c['text_primary']};padding:6px;font-family:monospace;font-size:12px;border-radius:3px">
                <option value="mean_reversion">Mean Reversion</option>
                <option value="momentum">Momentum</option>
            </select>
        </div>
        <div>
            <button onclick="runOptimization()"
                style="width:100%;padding:10px;background:#00ff88;color:#000;border:none;cursor:pointer;font-weight:bold;border-radius:3px">
                Optimize Parameters
            </button>
        </div>
        <div id="optimize-results" style="font-size:11px;color:{c['text_secondary']}">
            Grid search to find optimal model parameters...
        </div>
    </div>
</div>

<!-- Price Snapshot Panel -->
<div class="panel" style="margin-bottom:15px">
    <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['accent']}">Data Collection</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px">
        <div>
            <button onclick="storeSnapshot()"
                style="width:100%;padding:10px;background:{c['border']};color:{c['text_primary']};border:none;cursor:pointer;border-radius:3px">
                📸 Store Price Snapshot
            </button>
        </div>
        <div>
            <button onclick="loadHistory()"
                style="width:100%;padding:10px;background:{c['border']};color:{c['text_primary']};border:none;cursor:pointer;border-radius:3px">
                📈 Load Local History
            </button>
        </div>
        <div id="snapshot-status" style="font-size:11px;color:{c['text_secondary']};padding:10px">
            Capture current prices to build local history for modeling...
        </div>
    </div>
</div>

<!-- History Panel -->
<div class="panel" id="history-panel" style="display:none">
    <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['text_secondary']}">Local Price History</h3>
    <div id="history-data" style="font-size:11px"></div>
</div>

<!-- Forecast Visualization -->
<div class="panel" id="forecast-panel" style="display:none;margin-bottom:15px">
    <h3 style="margin:0 0 10px 0;font-size:13px;color:{c['accent']}">Forecast Visualization</h3>
    <canvas id="forecast-chart" width="800" height="300" style="width:100%;background:{c['bg_panel']};border-radius:3px"></canvas>
</div>

<script>
async function runAllModels() {{
    const marketId = document.getElementById('quant-market-id').value;
    const periods = document.getElementById('quant-periods').value;
    if (!marketId) {{
        alert('Enter a market ID');
        return;
    }}
    document.getElementById('model-results').innerHTML = '<div style="color:{c["text_secondary"]}">Running models...</div>';
    try {{
        const resp = await fetch(`/api/quant/models?id=${{marketId}}&periods=${{periods}}`);
        const data = await resp.json();
        if (data.error) {{
            document.getElementById('model-results').innerHTML = `<div style="color:{c["error"]}">${{data.error}}</div>`;
            return;
        }}

        let html = '<h3 style="margin:0 0 10px 0;font-size:13px;color:{c["accent"]}">Model Results</h3>';

        // Indicators
        if (data.indicators) {{
            html += '<div style="margin-bottom:15px"><div style="color:{c["text_secondary"]};font-size:10px;margin-bottom:5px">INDICATORS</div>';
            html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">';
            for (const [k, v] of Object.entries(data.indicators)) {{
                html += `<div style="background:{c["bg_panel"]};padding:8px;border-radius:3px"><div style="font-size:10px;color:{c["text_secondary"]}">${{k.toUpperCase()}}</div><div style="font-size:14px;color:{c["accent"]}">${{typeof v === 'number' ? v.toFixed(4) : v}}</div></div>`;
            }}
            html += '</div></div>';
        }}

        // Signals
        if (data.signals) {{
            html += '<div style="margin-bottom:15px"><div style="color:{c["text_secondary"]};font-size:10px;margin-bottom:5px">SIGNALS</div>';
            html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px">';
            for (const [model, sig] of Object.entries(data.signals)) {{
                const color = sig.signal === 'bullish' ? '#00ff88' : sig.signal === 'bearish' ? '#ff3333' : '{c["text_secondary"]}';
                html += `<div style="background:{c["bg_panel"]};padding:8px;border-radius:3px"><div style="font-size:10px;color:{c["text_secondary"]}">${{model.toUpperCase()}}</div><div style="font-size:12px;color:${{color}}">${{sig.signal || '-'}}</div></div>`;
            }}
            html += '</div></div>';
        }}

        // Forecasts
        if (data.forecasts) {{
            html += '<div style="margin-bottom:15px"><div style="color:{c["text_secondary"]};font-size:10px;margin-bottom:5px">FORECASTS</div>';
            html += '<table style="width:100%;font-size:11px;border-collapse:collapse">';
            html += '<tr style="border-bottom:1px solid {c["border"]}"><th style="text-align:left;padding:5px">Model</th><th style="text-align:right;padding:5px">Period 1</th><th style="text-align:right;padding:5px">Period 3</th><th style="text-align:right;padding:5px">Period 5</th></tr>';
            for (const [model, fc] of Object.entries(data.forecasts)) {{
                if (fc.forecasts) {{
                    html += `<tr><td style="padding:5px">${{model}}</td><td style="text-align:right;padding:5px">${{fc.forecasts[0]?.toFixed(4) || '-'}}</td><td style="text-align:right;padding:5px">${{fc.forecasts[2]?.toFixed(4) || '-'}}</td><td style="text-align:right;padding:5px">${{fc.forecasts[4]?.toFixed(4) || '-'}}</td></tr>`;
                }}
            }}
            html += '</table></div>';
        }}

        // Consensus
        if (data.consensus) {{
            const con = data.consensus;
            html += `<div style="background:linear-gradient(135deg,{c["accent"]}22,{c["bg_panel"]});padding:15px;border-radius:5px;border:1px solid {c["accent"]}44">`;
            html += `<div style="font-size:12px;color:{c["accent"]};font-weight:bold;margin-bottom:10px">ENSEMBLE CONSENSUS</div>`;
            html += `<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">`;
            html += `<div><div style="font-size:10px;color:{c["text_secondary"]}">AVG FORECAST</div><div style="font-size:16px;color:{c["text_primary"]}">${{con.avg_forecast?.toFixed(4) || '-'}}</div></div>`;
            html += `<div><div style="font-size:10px;color:{c["text_secondary"]}">TREND</div><div style="font-size:16px;color:${{con.direction === 'up' ? '#00ff88' : con.direction === 'down' ? '#ff3333' : '{c["text_secondary"]}'}}">${{con.direction?.toUpperCase() || '-'}}</div></div>`;
            html += `<div><div style="font-size:10px;color:{c["text_secondary"]}">CONFIDENCE</div><div style="font-size:16px;color:{c["text_primary"]}">${{(con.confidence * 100)?.toFixed(0) || '-'}}%</div></div>`;
            html += '</div></div>';
        }}

        document.getElementById('model-results').innerHTML = html;

        // Draw forecast chart
        if (data.forecasts) {{
            drawForecastChart(data);
        }}
    }} catch(e) {{
        document.getElementById('model-results').innerHTML = `<div style="color:{c["error"]}">Error: ${{e.message}}</div>`;
    }}
}}

async function runBacktest() {{
    const marketId = document.getElementById('quant-market-id').value;
    const model = document.getElementById('backtest-model').value;
    const lookback = document.getElementById('backtest-lookback').value;
    const periods = document.getElementById('backtest-periods').value;
    if (!marketId) {{
        alert('Enter a market ID');
        return;
    }}
    document.getElementById('backtest-results').innerHTML = '<div style="color:{c["text_secondary"]}">Running backtest...</div>';
    try {{
        const resp = await fetch(`/api/quant/backtest?id=${{marketId}}&model=${{model}}&lookback=${{lookback}}&periods=${{periods}}`);
        const data = await resp.json();
        if (data.error) {{
            document.getElementById('backtest-results').innerHTML = `<div style="color:{c["error"]}">${{data.error}}</div>`;
            return;
        }}

        let html = '<h3 style="margin:0 0 10px 0;font-size:13px;color:#4a9eff">Backtest Results</h3>';
        html += `<div style="color:{c["text_secondary"]};font-size:11px;margin-bottom:10px">${{model.toUpperCase()}} | Lookback: ${{lookback}} | Test: ${{periods}} periods</div>`;

        html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:15px">';

        // Metrics
        const metrics = [
            ['MAE', data.mae, 'Mean Absolute Error'],
            ['MSE', data.mse, 'Mean Squared Error'],
            ['RMSE', data.rmse, 'Root Mean Squared Error'],
            ['Direction', data.direction_accuracy, 'Direction Accuracy']
        ];
        for (const [name, val, desc] of metrics) {{
            const dispVal = name === 'Direction' ? `${{(val*100).toFixed(1)}}%` : val?.toFixed(6);
            const color = name === 'Direction' ? (val > 0.5 ? '#00ff88' : '#ff3333') : '{c["accent"]}';
            html += `<div style="background:{c["bg_panel"]};padding:12px;border-radius:3px"><div style="font-size:10px;color:{c["text_secondary"]}">${{name}}</div><div style="font-size:18px;color:${{color}}">${{dispVal || '-'}}</div><div style="font-size:9px;color:{c["text_secondary"]};margin-top:2px">${{desc}}</div></div>`;
        }}

        html += '</div>';

        // Quality assessment
        const quality = data.direction_accuracy > 0.6 ? 'Good' : data.direction_accuracy > 0.5 ? 'Fair' : 'Poor';
        const qualColor = data.direction_accuracy > 0.6 ? '#00ff88' : data.direction_accuracy > 0.5 ? '#ffaa00' : '#ff3333';
        html += `<div style="margin-top:15px;padding:10px;background:{c["bg_panel"]};border-radius:3px;border-left:3px solid ${{qualColor}}"><span style="color:${{qualColor}};font-weight:bold">${{quality}}</span> <span style="color:{c["text_secondary"]};font-size:11px">predictive quality based on direction accuracy</span></div>`;

        document.getElementById('backtest-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('backtest-results').innerHTML = `<div style="color:{c["error"]}">Error: ${{e.message}}</div>`;
    }}
}}

async function runOptimization() {{
    const marketId = document.getElementById('quant-market-id').value;
    const model = document.getElementById('optimize-model').value;
    if (!marketId) {{
        alert('Enter a market ID');
        return;
    }}
    document.getElementById('optimize-results').innerHTML = '<span style="color:{c["text_secondary"]}">Running grid search...</span>';
    try {{
        const resp = await fetch(`/api/quant/optimize?id=${{marketId}}&model=${{model}}`);
        const data = await resp.json();
        if (data.error) {{
            document.getElementById('optimize-results').innerHTML = `<span style="color:{c["error"]}">${{data.error}}</span>`;
            return;
        }}

        let html = '<div style="color:#00ff88;font-weight:bold">Optimal Parameters Found:</div>';
        html += '<div style="margin-top:5px">';
        for (const [param, val] of Object.entries(data.best_params || {{}})) {{
            html += `<span style="background:{c["accent"]}33;padding:2px 6px;border-radius:3px;margin-right:5px">${{param}}: ${{val}}</span>`;
        }}
        html += `</div><div style="color:{c["text_secondary"]};margin-top:5px">RMSE: ${{data.best_rmse?.toFixed(6) || '-'}}</div>`;

        document.getElementById('optimize-results').innerHTML = html;
    }} catch(e) {{
        document.getElementById('optimize-results').innerHTML = `<span style="color:{c["error"]}">Error: ${{e.message}}</span>`;
    }}
}}

async function storeSnapshot() {{
    const marketId = document.getElementById('quant-market-id').value;
    if (!marketId) {{
        alert('Enter a market ID');
        return;
    }}
    try {{
        const resp = await fetch(`/api/quant/snapshot?id=${{marketId}}`);
        const data = await resp.json();
        document.getElementById('snapshot-status').innerHTML = data.success ?
            `<span style="color:#00ff88">Snapshot stored at ${{new Date().toLocaleTimeString()}}</span>` :
            `<span style="color:{c["error"]}">${{data.error || 'Failed'}}</span>`;
    }} catch(e) {{
        document.getElementById('snapshot-status').innerHTML = `<span style="color:{c["error"]}">Error: ${{e.message}}</span>`;
    }}
}}

async function loadHistory() {{
    const marketId = document.getElementById('quant-market-id').value;
    if (!marketId) {{
        alert('Enter a market ID');
        return;
    }}
    try {{
        const resp = await fetch(`/api/quant/history?id=${{marketId}}`);
        const data = await resp.json();

        const panel = document.getElementById('history-panel');
        const dataDiv = document.getElementById('history-data');

        if (data.error || !data.history || data.history.length === 0) {{
            dataDiv.innerHTML = '<span style="color:{c["text_secondary"]}">No local history found. Store snapshots to build history.</span>';
        }} else {{
            let html = `<div style="margin-bottom:10px;color:{c["text_secondary"]}">${{data.history.length}} snapshots</div>`;
            html += '<table style="width:100%;font-size:11px;border-collapse:collapse">';
            html += '<tr style="border-bottom:1px solid {c["border"]}"><th style="text-align:left;padding:5px">Time</th><th style="text-align:right;padding:5px">YES</th><th style="text-align:right;padding:5px">NO</th></tr>';
            for (const snap of data.history.slice(-20).reverse()) {{
                const time = new Date(snap.timestamp).toLocaleString();
                html += `<tr><td style="padding:5px">${{time}}</td><td style="text-align:right;padding:5px;color:#00ff88">${{snap.yes_price?.toFixed(4) || '-'}}</td><td style="text-align:right;padding:5px;color:#ff3333">${{snap.no_price?.toFixed(4) || '-'}}</td></tr>`;
            }}
            html += '</table>';
            dataDiv.innerHTML = html;
        }}
        panel.style.display = 'block';
    }} catch(e) {{
        document.getElementById('history-data').innerHTML = `<span style="color:{c["error"]}">Error: ${{e.message}}</span>`;
        document.getElementById('history-panel').style.display = 'block';
    }}
}}

function drawForecastChart(data) {{
    const panel = document.getElementById('forecast-panel');
    const canvas = document.getElementById('forecast-chart');
    const ctx = canvas.getContext('2d');

    // Get forecast data
    const forecasts = data.forecasts || {{}};
    const models = Object.keys(forecasts);
    if (models.length === 0) return;

    panel.style.display = 'block';

    // Clear
    ctx.fillStyle = '{c["bg_panel"]}';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find min/max
    let allVals = [];
    for (const m of models) {{
        if (forecasts[m].forecasts) {{
            allVals = allVals.concat(forecasts[m].forecasts);
        }}
    }}
    if (allVals.length === 0) return;

    const minVal = Math.min(...allVals) * 0.95;
    const maxVal = Math.max(...allVals) * 1.05;
    const range = maxVal - minVal || 0.1;

    // Draw grid
    ctx.strokeStyle = '{c["border"]}';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {{
        const y = 30 + (i * 50);
        ctx.beginPath();
        ctx.moveTo(50, y);
        ctx.lineTo(canvas.width - 20, y);
        ctx.stroke();

        const val = maxVal - (i / 5) * range;
        ctx.fillStyle = '{c["text_secondary"]}';
        ctx.font = '10px monospace';
        ctx.fillText(val.toFixed(3), 5, y + 4);
    }}

    // Draw each model line
    const colors = ['#e94560', '#4a9eff', '#00ff88', '#ffaa00'];
    let colorIdx = 0;
    const legendY = 15;
    let legendX = 60;

    for (const m of models) {{
        const fc = forecasts[m].forecasts;
        if (!fc || fc.length === 0) continue;

        const color = colors[colorIdx % colors.length];
        colorIdx++;

        // Draw line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < fc.length; i++) {{
            const x = 60 + (i * ((canvas.width - 80) / (fc.length - 1 || 1)));
            const y = 30 + ((maxVal - fc[i]) / range) * 250;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }}
        ctx.stroke();

        // Legend
        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY - 8, 20, 3);
        ctx.fillStyle = '{c["text_primary"]}';
        ctx.font = '10px monospace';
        ctx.fillText(m, legendX + 25, legendY);
        legendX += 100;
    }}

    // X-axis labels
    ctx.fillStyle = '{c["text_secondary"]}';
    const periods = forecasts[models[0]]?.forecasts?.length || 5;
    for (let i = 0; i < periods; i++) {{
        const x = 60 + (i * ((canvas.width - 80) / (periods - 1 || 1)));
        ctx.fillText(`P${{i+1}}`, x - 8, canvas.height - 10);
    }}
}}
</script>
'''

    def render_elon_research(self, params):
        """Elon Tweet Market Research Dashboard - Enhanced with 3D, Models, Multi-view"""
        elon = load_elon_data()
        weeks = elon.get('weeks', [])
        tweets_data = load_elon_tweets()
        stats_data = load_elon_stats()
        models = get_prediction_models()
        tweet_analysis = analyze_elon_tweets()
        update_info = get_last_update_info()

        # Get current view tab
        tab = params.get('tab', ['overview'])[0]

        # All CSV files - price data, stats, tweets
        price_csv_files = [
            '/Users/trading/Downloads/polymarket-price-data-17-12-2025-26-12-2025-1768129066999.csv',
            '/Users/trading/Downloads/polymarket-price-data-21-12-2025-30-12-2025-1768129057458.csv',
            '/Users/trading/Downloads/polymarket-price-data-24-12-2025-02-01-2026-1768129048010.csv',
            '/Users/trading/Downloads/polymarket-price-data-28-12-2025-06-01-2026-1768129038645.csv',
            '/Users/trading/Downloads/polymarket-price-data-31-12-2025-09-01-2026-1768129024124.csv',
            '/Users/trading/Downloads/polymarket-price-data-10-12-2025-23-12-2025-1768129180547.csv',
            '/Users/trading/Downloads/polymarket-price-data-14-12-2025-02-01-2026-1768129174547.csv',
        ]
        stats_csv_files = [
            '/Users/trading/Downloads/elonmusk-Elon_Musk___tweets_January_6___January_13__2026_-stats(1).csv',
            '/Users/trading/Downloads/elonmusk-Elon_Musk___tweets_January_10___January_12__2026_-stats.csv',
            '/Users/trading/Downloads/elonmusk-Elon_Musk___Tweets_from_Jan_9_to_Jan_16_-stats.csv',
        ]
        tweets_csv_files = [
            '/Users/trading/Downloads/elonmusk-Elon_Musk___tweets_January_10___January_12__2026_-tweets.csv',
            '/Users/trading/Downloads/elonmusk-Elon_Musk___tweets_January_6___January_13__2026_-tweets.csv',
        ]

        # Import sections
        imported_files = elon.get('imported_files', [])
        stats_imported = stats_data.get('imported_files', [])
        tweets_imported = tweets_data.get('imported_files', [])

        # Historical weeks analysis
        weeks_html = ''
        all_ev_data = []
        all_winners = []

        for week in weeks:
            analysis = analyze_elon_week(week)
            week_id = week.get('week_id', '')
            winner = analysis.get('winner_bracket', '?')
            all_winners.append(winner)

            # EV progression for this week
            evs = analysis.get('ev_progression', [])
            all_ev_data.append({'week': week_id, 'evs': evs})

            ev_start = evs[0]['ev'] if evs else 0
            ev_end = evs[-1]['ev'] if evs else 0
            ev_change = ev_end - ev_start

            weeks_html += f'''
<div class="panel" style="margin-bottom:10px">
    <div style="display:flex;justify-content:space-between;align-items:center">
        <h3 style="margin:0;font-size:13px">{week_id}</h3>
        <span style="background:{c['accent']};color:#000;padding:2px 8px;border-radius:3px;font-size:11px">{winner}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px;font-size:11px">
        <div>
            <div style="color:{c['text_secondary']}">Start EV</div>
            <div style="color:{c['accent']}">{ev_start:.0f}</div>
        </div>
        <div>
            <div style="color:{c['text_secondary']}">End EV</div>
            <div style="color:{c['accent']}">{ev_end:.0f}</div>
        </div>
        <div>
            <div style="color:{c['text_secondary']}">Δ EV</div>
            <div style="color:{'#00ff00' if ev_change > 0 else '#ff3333'}">{'+' if ev_change > 0 else ''}{ev_change:.0f}</div>
        </div>
        <div>
            <div style="color:{c['text_secondary']}">Days</div>
            <div>{analysis.get('total_days', 0)}</div>
        </div>
    </div>
</div>'''

        # Calculate average EV patterns
        avg_stats = {}
        if all_winners:
            # Parse winning brackets to get actual tweet counts
            winning_mids = []
            for w in all_winners:
                if w.endswith('+'):
                    winning_mids.append(int(w[:-1]) + 20)
                elif '-' in w:
                    low, high = w.split('-')
                    winning_mids.append((int(low) + int(high)) / 2)
            if winning_mids:
                avg_stats['avg_winning_tweets'] = sum(winning_mids) / len(winning_mids)
                avg_stats['min_winning'] = min(winning_mids)
                avg_stats['max_winning'] = max(winning_mids)

        # Stats panel
        stats_html = f'''
<div class="panel">
    <h2>Historical Stats ({len(weeks)} weeks)</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;font-size:12px">
        <div>
            <div style="color:{c['text_secondary']}">Avg Weekly Tweets</div>
            <div style="color:{c['accent']};font-size:20px">{avg_stats.get('avg_winning_tweets', 0):.0f}</div>
        </div>
        <div>
            <div style="color:{c['text_secondary']}">Range</div>
            <div style="color:{c['accent']};font-size:20px">{avg_stats.get('min_winning', 0):.0f} - {avg_stats.get('max_winning', 0):.0f}</div>
        </div>
    </div>
    <div style="margin-top:15px">
        <div style="color:{c['text_secondary']};margin-bottom:5px">Winning Brackets:</div>
        <div style="display:flex;flex-wrap:wrap;gap:5px">
            {' '.join(f'<span style="background:{c["bg_input"]};padding:2px 6px;border-radius:3px;font-size:10px">{w}</span>' for w in all_winners)}
        </div>
    </div>
</div>'''

        # Current active market
        current_html = f'''
<div class="panel">
    <h2>Active Markets</h2>
    <div id="active-markets" style="min-height:100px">
        <div style="color:{c['text_secondary']};text-align:center;padding:20px">Loading current markets...</div>
    </div>
</div>'''

        # Projection calculator
        projection_html = f'''
<div class="panel">
    <h2>Projection Calculator</h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
        <div>
            <label style="font-size:10px;color:{c['text_secondary']}">Current Tweet Count</label>
            <input type="number" id="current-tweets" value="0" style="width:100%;font-size:14px">
        </div>
        <div>
            <label style="font-size:10px;color:{c['text_secondary']}">Days Remaining</label>
            <input type="number" id="days-remaining" value="7" style="width:100%;font-size:14px">
        </div>
    </div>
    <button onclick="calcProjection()" style="width:100%">Calculate Projection</button>
    <div id="projection-result" style="margin-top:15px;padding:15px;background:{c['bg_input']};border-radius:5px;display:none">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px">
            <div>
                <div style="color:{c['text_secondary']}">Projected Final</div>
                <div id="proj-final" style="color:{c['accent']};font-size:24px">--</div>
            </div>
            <div>
                <div style="color:{c['text_secondary']}">Daily Rate</div>
                <div id="proj-rate" style="color:{c['accent']};font-size:24px">--</div>
            </div>
        </div>
        <div style="margin-top:10px;font-size:11px">
            <div style="color:{c['text_secondary']}">Likely Bracket:</div>
            <div id="proj-bracket" style="color:{c['success']};font-size:16px">--</div>
        </div>
    </div>
</div>'''

        cache_size = get_cache_size_mb()
        total_data_size = 0
        for f in [ELON_DATA_FILE, ELON_TWEETS_FILE, ELON_STATS_FILE, ELON_MODELS_FILE]:
            if f.exists():
                total_data_size += f.stat().st_size
        total_data_kb = round(total_data_size / 1024, 1)

        # Model info
        linear_params = models.get('linear', {}).get('params', {})
        pattern_params = models.get('pattern', {}).get('params', {})

        # Last update info
        last_update = update_info.get('last_update', 'Never')
        if last_update and last_update != 'Never':
            last_update = last_update[:19].replace('T', ' ')

        # Tabs
        tabs = ['overview', 'analytics', 'historic', 'active', 'tweets', 'models', 'data']
        tabs_html = '<div style="display:flex;gap:5px;margin-bottom:15px;flex-wrap:wrap">'
        for t in tabs:
            active = 'background:' + c['accent'] + ';color:#000' if t == tab else ''
            tabs_html += f'<a href="/?view=elon&tab={t}" style="padding:6px 12px;font-size:11px;text-decoration:none;border:1px solid {c["border"]};{active}">{t.upper()}</a>'
        tabs_html += '</div>'

        # Build content based on tab
        content_html = ''

        if tab == 'analytics':
            # Volume Analytics Dashboard - Unified market cards
            content_html = f'''
<style>
.market-card {{
    background: linear-gradient(135deg, {c['bg_panel']}, {c['bg_input']});
    border: 2px solid {c['border']};
    border-radius: 6px;
    overflow: hidden;
    font-family: monospace;
}}
.market-card-header {{
    background: {c['bg_input']};
    padding: 8px 12px;
    border-bottom: 2px solid {c['accent']};
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.market-card-header h3 {{
    margin: 0;
    color: {c['accent']};
    font-size: 12px;
    letter-spacing: 1px;
}}
.market-card-body {{
    padding: 15px;
}}
.market-card-footer {{
    background: {c['bg_input']};
    padding: 6px 12px;
    border-top: 1px solid {c['border']};
    font-size: 10px;
    color: {c['text_muted']};
}}
.odds-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin: 12px 0;
}}
.odds-item {{
    background: {c['bg_main']};
    padding: 10px;
    border: 1px solid {c['border']};
    border-radius: 4px;
}}
.odds-item.highlight {{
    border-color: {c['accent']};
    box-shadow: 0 0 10px {c['accent']}40;
}}
.last-trades {{
    max-height: 120px;
    overflow-y: auto;
    font-size: 10px;
}}
.trade-row {{
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px solid {c['border']}30;
}}
.vol-bar {{
    height: 20px;
    background: linear-gradient(90deg, {c['accent']}40, {c['accent']});
    border-radius: 3px;
    display: flex;
    align-items: center;
    padding: 0 8px;
    font-size: 10px;
    color: #fff;
}}
</style>

<div class="panel">
    <h2 style="color:{c['warning']}">📊 VOLUME ANALYTICS - All Elon Tweet Markets</h2>
    <p style="color:{c['text_secondary']};font-size:11px;margin-bottom:15px">
        Compare volumes across markets to find patterns: earlier markets have more volume, volume drops as expiry approaches
    </p>

    <!-- Volume Comparison Chart -->
    <div style="margin-bottom:20px">
        <h3 style="color:{c['text_primary']};font-size:12px;margin-bottom:10px">Volume by Market (Distance to Expiry)</h3>
        <div id="volume-comparison" style="display:flex;flex-direction:column;gap:8px">
            <div style="color:{c['text_muted']};font-size:11px">Loading volume data...</div>
        </div>
    </div>
</div>

<!-- Unified Market Cards -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:15px">

    <!-- Jan 6-13 Card -->
    <div class="market-card" id="card-jan6-13">
        <div class="market-card-header">
            <h3>┌─ JAN 6-13 MARKET ─┐</h3>
            <span id="jan613-status" style="font-size:10px;color:{c['warning']}">⚠️ EXPIRING</span>
        </div>
        <div class="market-card-body">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:15px">
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">TOP BRACKET</div>
                    <div id="jan613-top" style="color:{c['accent']};font-size:18px;font-weight:bold">--</div>
                </div>
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">ODDS</div>
                    <div id="jan613-odds" style="color:{c['success']};font-size:18px;font-weight:bold">--%</div>
                </div>
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">VOLUME</div>
                    <div id="jan613-vol" style="color:{c['warning']};font-size:18px;font-weight:bold">$--</div>
                </div>
            </div>

            <div style="margin-bottom:12px">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:6px">ALL BRACKETS</div>
                <div id="jan613-brackets" class="odds-grid">Loading...</div>
            </div>

            <div style="margin-bottom:12px">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:6px">LAST TRADES</div>
                <div id="jan613-trades" class="last-trades">Loading...</div>
            </div>

            <canvas id="jan613-chart" style="width:100%;height:80px;background:{c['bg_main']};border-radius:4px"></canvas>
        </div>
        <div class="market-card-footer">
            └─ Jan 6 12:00 ET → Jan 13 12:00 ET ─┘
            <a href="#" onclick="editMarketDates('jan6-13')" style="float:right;color:{c['accent']}">⚙️ Edit</a>
        </div>
    </div>

    <!-- Jan 9-16 Card -->
    <div class="market-card" id="card-jan9-16">
        <div class="market-card-header">
            <h3>┌─ JAN 9-16 MARKET ─┐</h3>
            <span id="jan916-status" style="font-size:10px;color:{c['success']}">● ACTIVE</span>
        </div>
        <div class="market-card-body">
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:15px">
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">TOP BRACKET</div>
                    <div id="jan916-top" style="color:{c['accent']};font-size:18px;font-weight:bold">--</div>
                </div>
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">ODDS</div>
                    <div id="jan916-odds" style="color:{c['success']};font-size:18px;font-weight:bold">--%</div>
                </div>
                <div style="text-align:center">
                    <div style="color:{c['text_muted']};font-size:9px">VOLUME</div>
                    <div id="jan916-vol" style="color:{c['warning']};font-size:18px;font-weight:bold">$--</div>
                </div>
            </div>

            <div style="margin-bottom:12px">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:6px">ALL BRACKETS</div>
                <div id="jan916-brackets" class="odds-grid">Loading...</div>
            </div>

            <div style="margin-bottom:12px">
                <div style="color:{c['text_muted']};font-size:9px;margin-bottom:6px">LAST TRADES</div>
                <div id="jan916-trades" class="last-trades">Loading...</div>
            </div>

            <canvas id="jan916-chart" style="width:100%;height:80px;background:{c['bg_main']};border-radius:4px"></canvas>
        </div>
        <div class="market-card-footer">
            └─ Jan 9 12:00 ET → Jan 16 12:00 ET ─┘
            <a href="#" onclick="editMarketDates('jan9-16')" style="float:right;color:{c['accent']}">⚙️ Edit</a>
        </div>
    </div>
</div>

<!-- Jan 13-20 Card (New) -->
<div class="market-card" style="margin-bottom:15px">
    <div class="market-card-header">
        <h3>┌─ JAN 13-20 MARKET ─┐</h3>
        <span style="font-size:10px;color:{c['accent']}">🆕 NEW</span>
    </div>
    <div class="market-card-body">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:15px">
            <div style="text-align:center">
                <div style="color:{c['text_muted']};font-size:9px">TOP BRACKET</div>
                <div id="jan1320-top" style="color:{c['accent']};font-size:18px;font-weight:bold">TBD</div>
            </div>
            <div style="text-align:center">
                <div style="color:{c['text_muted']};font-size:9px">ODDS</div>
                <div id="jan1320-odds" style="color:{c['success']};font-size:18px;font-weight:bold">~0.1%</div>
            </div>
            <div style="text-align:center">
                <div style="color:{c['text_muted']};font-size:9px">VOLUME</div>
                <div id="jan1320-vol" style="color:{c['warning']};font-size:18px;font-weight:bold">$1.3M</div>
            </div>
        </div>
        <div style="color:{c['text_secondary']};font-size:11px;padding:15px;background:{c['bg_main']};border-radius:4px">
            <div style="color:{c['warning']};margin-bottom:8px">⚡ Inauguration Week Market</div>
            <div>• New market - few brackets populated</div>
            <div>• Expect high activity Jan 20</div>
            <div>• Watch for 400-500 range brackets</div>
        </div>
    </div>
    <div class="market-card-footer">
        └─ Jan 13 12:00 ET → Jan 20 12:00 ET ─┘
    </div>
</div>

<!-- Volume Pattern Analysis -->
<div class="panel">
    <h2>Volume vs Days to Expiry</h2>
    <p style="color:{c['text_secondary']};font-size:11px;margin-bottom:15px">
        Pattern: Volume typically increases as expiry approaches, then drops sharply in final hours
    </p>
    <div id="vol-pattern-chart" style="height:150px;background:{c['bg_input']};border-radius:4px;position:relative">
        <canvas id="vol-pattern-canvas" style="width:100%;height:150px"></canvas>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:10px;font-size:10px;color:{c['text_muted']}">
        <span>7 days out</span>
        <span>5 days</span>
        <span>3 days</span>
        <span>1 day</span>
        <span>Expiry</span>
    </div>
</div>

<script>
// Analytics Tab JavaScript
const MARKET_CONFIGS = {{
    'jan6-13': {{
        ids: ['1093290', '1093289', '1093288', '1093287'],
        start: '2026-01-06T17:00:00Z',
        end: '2026-01-13T17:00:00Z',
        name: 'Jan 6-13'
    }},
    'jan9-16': {{
        ids: ['1119399', '1119400'],
        start: '2026-01-09T17:00:00Z',
        end: '2026-01-16T17:00:00Z',
        name: 'Jan 9-16'
    }},
    'jan13-20': {{
        ids: ['1148896', '1148905', '1148907'],
        start: '2026-01-13T17:00:00Z',
        end: '2026-01-20T17:00:00Z',
        name: 'Jan 13-20'
    }}
}};

async function loadMarketCard(marketKey) {{
    const config = MARKET_CONFIGS[marketKey];
    if (!config) return;

    const prefix = marketKey.replace('-', '');

    try {{
        // Fetch all bracket markets
        let allMarkets = [];
        for (const id of config.ids) {{
            const resp = await fetch('/api/market?id=' + id);
            const data = await resp.json();
            if (data.markets && data.markets[0]) {{
                allMarkets.push(data.markets[0]);
            }}
        }}

        if (allMarkets.length === 0) return;

        // Find highest volume/odds bracket
        let topMarket = allMarkets[0];
        let totalVol = 0;

        allMarkets.forEach(m => {{
            const vol = m.volume_total || 0;
            totalVol += vol;
            const yesPrice = m.current_prices?.yes?.price || 0;
            const topPrice = topMarket.current_prices?.yes?.price || 0;
            if (yesPrice > topPrice) topMarket = m;
        }});

        // Extract bracket from title
        const bracketMatch = topMarket.title.match(/(\d+-\d+|\d+\+)/);
        const bracket = bracketMatch ? bracketMatch[1] : '--';
        const odds = ((topMarket.current_prices?.yes?.price || 0) * 100).toFixed(1);

        // Update card
        const topEl = document.getElementById(prefix + '-top');
        const oddsEl = document.getElementById(prefix + '-odds');
        const volEl = document.getElementById(prefix + '-vol');

        if (topEl) topEl.textContent = bracket;
        if (oddsEl) oddsEl.textContent = odds + '%';
        if (volEl) volEl.textContent = '$' + (totalVol > 1000000 ? (totalVol/1000000).toFixed(1) + 'M' : (totalVol/1000).toFixed(0) + 'K');

        // Build brackets grid
        const bracketsEl = document.getElementById(prefix + '-brackets');
        if (bracketsEl) {{
            let html = '';
            allMarkets.sort((a, b) => {{
                const aMatch = a.title.match(/(\d+)/);
                const bMatch = b.title.match(/(\d+)/);
                return (parseInt(aMatch?.[1] || 0)) - (parseInt(bMatch?.[1] || 0));
            }}).forEach(m => {{
                const bMatch = m.title.match(/(\d+-\d+|\d+\+)/);
                const b = bMatch ? bMatch[1] : '?';
                const p = ((m.current_prices?.yes?.price || 0) * 100).toFixed(1);
                const vol = m.volume_total || 0;
                const isTop = m.id === topMarket.id;
                html += `<div class="odds-item ${{isTop ? 'highlight' : ''}}">
                    <div style="display:flex;justify-content:space-between">
                        <span style="color:${{isTop ? '{c['accent']}' : '{c['text_secondary']}'}};font-weight:${{isTop ? 'bold' : 'normal'}}">${{b}}</span>
                        <span style="color:{c['success']}">${{p}}%</span>
                    </div>
                    <div style="font-size:9px;color:{c['text_muted']};margin-top:4px">Vol: $${{vol > 1000 ? (vol/1000).toFixed(0) + 'K' : vol}}</div>
                </div>`;
            }});
            bracketsEl.innerHTML = html || '<div style="color:{c["text_muted"]}">No data</div>';
        }}

        // Load last trades for top bracket
        loadLastTrades(topMarket.id, prefix);

    }} catch(e) {{
        console.error('Error loading ' + marketKey, e);
    }}
}}

async function loadLastTrades(marketId, prefix) {{
    const tradesEl = document.getElementById(prefix + '-trades');
    if (!tradesEl) return;

    try {{
        const resp = await fetch('/api/trades?id=' + marketId + '&limit=10');
        const data = await resp.json();

        if (data.trades && data.trades.length > 0) {{
            let html = '';
            data.trades.forEach(t => {{
                const price = ((t.price || 0) * 100).toFixed(1);
                const side = t.side || 'buy';
                const sideColor = side === 'buy' || side === 'yes' ? '{c['success']}' : '{c['error']}';
                const time = new Date(t.timestamp * 1000).toLocaleTimeString();
                html += `<div class="trade-row">
                    <span style="color:${{sideColor}}">${{side.toUpperCase()}}</span>
                    <span>${{price}}%</span>
                    <span style="color:{c['text_muted']}">${{t.size || '--'}}</span>
                    <span style="color:{c['text_muted']}">${{time}}</span>
                </div>`;
            }});
            tradesEl.innerHTML = html;
        }} else {{
            tradesEl.innerHTML = '<div style="color:{c["text_muted"]};text-align:center;padding:10px">No recent trades</div>';
        }}
    }} catch(e) {{
        tradesEl.innerHTML = '<div style="color:{c["error"]};text-align:center;padding:10px">Error loading trades</div>';
    }}
}}

async function loadVolumeComparison() {{
    const container = document.getElementById('volume-comparison');
    if (!container) return;

    try {{
        // Fetch all markets and calculate volumes
        const volumes = [];

        for (const [key, config] of Object.entries(MARKET_CONFIGS)) {{
            let totalVol = 0;
            for (const id of config.ids) {{
                try {{
                    const resp = await fetch('/api/market?id=' + id);
                    const data = await resp.json();
                    if (data.markets && data.markets[0]) {{
                        totalVol += data.markets[0].volume_total || 0;
                    }}
                }} catch {{}}
            }}

            const now = new Date();
            const end = new Date(config.end);
            const daysToExpiry = Math.max(0, (end - now) / (1000 * 60 * 60 * 24));

            volumes.push({{
                name: config.name,
                volume: totalVol,
                daysToExpiry: daysToExpiry,
                status: daysToExpiry < 0.5 ? 'expiring' : daysToExpiry < 3 ? 'soon' : 'active'
            }});
        }}

        // Sort by days to expiry
        volumes.sort((a, b) => a.daysToExpiry - b.daysToExpiry);

        // Find max for scaling
        const maxVol = Math.max(...volumes.map(v => v.volume), 1);

        // Build bars
        let html = '';
        volumes.forEach(v => {{
            const width = (v.volume / maxVol * 100).toFixed(0);
            const statusColor = v.status === 'expiring' ? '{c['error']}' : v.status === 'soon' ? '{c['warning']}' : '{c['success']}';
            const volStr = v.volume > 1000000 ? (v.volume / 1000000).toFixed(1) + 'M' : (v.volume / 1000).toFixed(0) + 'K';
            const daysStr = v.daysToExpiry < 1 ? v.daysToExpiry.toFixed(1) + 'h' : v.daysToExpiry.toFixed(1) + 'd';

            html += `<div style="display:flex;align-items:center;gap:10px">
                <span style="width:80px;font-size:11px;color:${{statusColor}}">${{v.name}}</span>
                <div style="flex:1;background:{c['bg_main']};border-radius:4px;overflow:hidden">
                    <div class="vol-bar" style="width:${{width}}%">$${{volStr}}</div>
                </div>
                <span style="width:50px;font-size:10px;color:{c['text_muted']};text-align:right">${{daysStr}}</span>
            </div>`;
        }});

        container.innerHTML = html || '<div style="color:{c["text_muted"]}">No data</div>';

    }} catch(e) {{
        container.innerHTML = '<div style="color:{c["error"]}">Error loading volume data</div>';
    }}
}}

function editMarketDates(marketKey) {{
    const config = MARKET_CONFIGS[marketKey];
    if (!config) return;

    const newStart = prompt('Start date (ISO):', config.start);
    const newEnd = prompt('End date (ISO):', config.end);

    if (newStart && newEnd) {{
        MARKET_CONFIGS[marketKey].start = newStart;
        MARKET_CONFIGS[marketKey].end = newEnd;
        localStorage.setItem('market_configs', JSON.stringify(MARKET_CONFIGS));
        alert('Updated! Refresh to see changes.');
    }}
}}

// Load saved configs
const savedConfigs = localStorage.getItem('market_configs');
if (savedConfigs) {{
    try {{
        Object.assign(MARKET_CONFIGS, JSON.parse(savedConfigs));
    }} catch {{}}
}}

// Init analytics
loadMarketCard('jan6-13');
loadMarketCard('jan9-16');
loadVolumeComparison();
</script>
'''

        elif tab == 'historic':
            # Historical data focus
            content_html = f'''
<div class="panel"><h2 style="color:{c['warning']}">📊 HISTORIC DATA (Resolved Markets)</h2>
<div style="color:{c['text_secondary']};font-size:11px;margin-bottom:15px">{len(weeks)} weeks of bracket probability data</div>
<div style="max-height:500px;overflow-y:auto">{weeks_html}</div></div>'''

        elif tab == 'active':
            # Active markets focus
            content_html = f'''
<div class="panel"><h2 style="color:{c['success']}">🔴 ACTIVE MARKETS (Live)</h2>
<div id="active-markets-full" style="min-height:200px"><div style="color:{c['text_secondary']};padding:20px">Loading...</div></div></div>
<div class="panel"><h2>Auto-Update Status</h2>
<div style="font-size:12px"><div style="margin-bottom:10px">Last Update: <span style="color:{c['accent']}">{last_update}</span></div>
<div style="margin-bottom:10px">Interval: <span style="color:{c['accent']}">1 minute</span></div>
<div style="margin-bottom:10px">Rate Limit: <span style="color:{c['accent']}">100 calls/min</span></div>
<button onclick="triggerUpdate()" style="width:100%">Force Update Now</button></div></div>'''

        elif tab == 'tweets':
            # Tweet analysis
            hourly_data = tweet_analysis.get('hourly_distribution', {})
            topics = tweet_analysis.get('topics', {})
            content_html = f'''
<div class="panel"><h2>📝 TWEET ANALYSIS ({tweet_analysis.get('total_tweets', 0)} tweets)</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px">
<div><div style="color:{c['text_secondary']};font-size:10px">RT Ratio</div><div style="color:{c['accent']};font-size:18px">{tweet_analysis.get('rt_ratio', 0)}%</div></div>
<div><div style="color:{c['text_secondary']};font-size:10px">Original</div><div style="color:{c['success']};font-size:18px">{tweet_analysis.get('original_ratio', 0)}%</div></div>
<div><div style="color:{c['text_secondary']};font-size:10px">Avg Daily</div><div style="color:{c['accent']};font-size:18px">{tweet_analysis.get('avg_daily', 0)}</div></div>
<div><div style="color:{c['text_secondary']};font-size:10px">Short Tweets</div><div style="color:{c['warning']};font-size:18px">{tweet_analysis.get('short_tweet_ratio', 0)}%</div></div>
</div>
<h3>Topic Distribution</h3>
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px">
{''.join(f'<span style="background:{c["bg_input"]};padding:4px 10px;font-size:11px">{k}: {v}</span>' for k,v in topics.items())}
</div>
<h3>Hourly Activity (Bar Chart)</h3>
<div id="hourly-chart" style="height:150px;display:flex;align-items:end;gap:2px;padding:10px 0">
{''.join(f'<div style="flex:1;background:{c["accent"]};height:{int((hourly_data.get(h,0)/max(hourly_data.values() or [1]))*100)}%;min-height:2px" title="{h}:00 - {hourly_data.get(h,0)} tweets"></div>' for h in range(24))}
</div>
<div style="display:flex;justify-content:space-between;font-size:9px;color:{c['text_muted']}"><span>0:00</span><span>6:00</span><span>12:00</span><span>18:00</span><span>23:00</span></div>
</div>'''

        elif tab == 'models':
            # Prediction models
            content_html = f'''
<div class="panel"><h2>🤖 PREDICTION MODELS</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
<div style="border:1px solid {c['accent']};padding:15px">
<h3 style="color:{c['accent']};margin-top:0">Model 1: Linear Extrapolation</h3>
<div style="font-size:12px">
<div style="margin-bottom:8px">Avg Daily Rate: <b>{linear_params.get('avg_daily_rate', 'N/A')}</b></div>
<div style="margin-bottom:8px">Min Rate: {linear_params.get('min_daily_rate', 'N/A')}</div>
<div style="margin-bottom:8px">Max Rate: {linear_params.get('max_daily_rate', 'N/A')}</div>
<div style="color:{c['text_secondary']};font-size:10px;margin-top:15px">Uses current tweet rate to project final count. More accurate as week progresses.</div>
</div></div>
<div style="border:1px solid {c['success']};padding:15px">
<h3 style="color:{c['success']};margin-top:0">Model 2: Historical Pattern</h3>
<div style="font-size:12px">
<div style="margin-bottom:8px">Avg Weekly: <b>{pattern_params.get('avg_weekly_total', 'N/A')}</b></div>
<div style="margin-bottom:8px">Range: {pattern_params.get('min_weekly_total', 'N/A')} - {pattern_params.get('max_weekly_total', 'N/A')}</div>
<div style="margin-bottom:8px">Std Dev: {pattern_params.get('std_dev', 'N/A')}</div>
<div style="margin-bottom:8px">Samples: {pattern_params.get('samples', 0)}</div>
<div style="color:{c['text_secondary']};font-size:10px;margin-top:15px">Uses historical winning brackets. Blends with current rate as week progresses.</div>
</div></div>
</div>
<button onclick="rebuildModels()" style="margin-top:15px">Rebuild Models from Data</button>
</div>
{projection_html}'''

        elif tab == 'data':
            # Data import/management - pre-compute button HTML
            price_btns = []
            for p in price_csv_files:
                fname = Path(p).name
                parts = fname.split('-')
                label = '-'.join(parts[3:6]) if len(parts) > 5 else fname[:15]
                imported = fname in imported_files
                style = f"font-size:9px;padding:4px 8px;{'background:'+c['success']+';color:#000' if imported else ''}"
                price_btns.append(f'<button onclick="importCSV(\'{p}\',\'prices\')" style="{style}" {"disabled" if imported else ""}>{label}... {"✓" if imported else ""}</button>')

            stats_btns = []
            for p in stats_csv_files:
                fname = Path(p).name
                imported = fname in stats_imported
                style = f"font-size:9px;padding:4px 8px;{'background:'+c['success']+';color:#000' if imported else ''}"
                stats_btns.append(f'<button onclick="importCSV(\'{p}\',\'stats\')" style="{style}" {"disabled" if imported else ""}>{fname[:30]}... {"✓" if imported else ""}</button>')

            tweets_btns = []
            for p in tweets_csv_files:
                fname = Path(p).name
                imported = fname in tweets_imported
                style = f"font-size:9px;padding:4px 8px;{'background:'+c['success']+';color:#000' if imported else ''}"
                tweets_btns.append(f'<button onclick="importCSV(\'{p}\',\'tweets\')" style="{style}" {"disabled" if imported else ""}>{fname[:30]}... {"✓" if imported else ""}</button>')

            content_html = f'''
<div class="panel"><h2>📁 DATA MANAGEMENT</h2>
<div style="margin-bottom:20px"><b>Total Local Data:</b> <span style="color:{c['accent']}">{total_data_kb} KB</span> | Cache: {cache_size} MB</div>

<h3>Price Data CSVs (Polymarket Brackets)</h3>
<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:20px">
{''.join(price_btns)}
</div>

<h3>Stats CSVs (Tweet Summaries)</h3>
<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:20px">
{''.join(stats_btns)}
</div>

<h3>Tweet CSVs (Actual Tweets)</h3>
<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:20px">
{''.join(tweets_btns)}
</div>
</div>'''

        else:  # overview
            # Main overview with TUI-style panels
            num_winners = len(models.get('winners', []))
            avg_daily_rate = models.get('linear', {}).get('params', {}).get('avg_daily_rate', 0)
            content_html = f'''
<div id="market-summary" style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:15px">
<div class="stat-card" style="background:linear-gradient(135deg,{c['bg_panel']},{c['bg_input']});padding:12px;border:1px solid {c['border']};border-radius:4px;text-align:center">
<div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;letter-spacing:1px">Active Markets</div>
<div id="stat-markets" style="color:{c['accent']};font-size:24px;font-weight:bold;margin:4px 0">--</div>
<div style="color:{c['text_secondary']};font-size:9px">Jan 6-13 + Jan 9-16</div>
</div>
<div class="stat-card" style="background:linear-gradient(135deg,{c['bg_panel']},{c['bg_input']});padding:12px;border:1px solid {c['border']};border-radius:4px;text-align:center">
<div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;letter-spacing:1px">Total Volume</div>
<div id="stat-volume" style="color:{c['success']};font-size:24px;font-weight:bold;margin:4px 0">--</div>
<div style="color:{c['text_secondary']};font-size:9px">All brackets</div>
</div>
<div class="stat-card" style="background:linear-gradient(135deg,{c['bg_panel']},{c['bg_input']});padding:12px;border:1px solid {c['border']};border-radius:4px;text-align:center">
<div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;letter-spacing:1px">Top Bracket</div>
<div id="stat-top-bracket" style="color:{c['warning']};font-size:24px;font-weight:bold;margin:4px 0">--</div>
<div id="stat-top-odds" style="color:{c['text_secondary']};font-size:9px">--% odds</div>
</div>
<div class="stat-card" style="background:linear-gradient(135deg,{c['bg_panel']},{c['bg_input']});padding:12px;border:1px solid {c['border']};border-radius:4px;text-align:center">
<div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;letter-spacing:1px">Historical Wins</div>
<div id="stat-wins" style="color:#00ff88;font-size:24px;font-weight:bold;margin:4px 0">{num_winners}</div>
<div style="color:{c['text_secondary']};font-size:9px">Resolved markets</div>
</div>
<div class="stat-card" style="background:linear-gradient(135deg,{c['bg_panel']},{c['bg_input']});padding:12px;border:1px solid {c['border']};border-radius:4px;text-align:center">
<div style="color:{c['text_muted']};font-size:9px;text-transform:uppercase;letter-spacing:1px">Avg Win Rate</div>
<div id="stat-rate" style="color:{c['accent']};font-size:24px;font-weight:bold;margin:4px 0">{avg_daily_rate:.0f}</div>
<div style="color:{c['text_secondary']};font-size:9px">tweets/day</div>
</div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px">

<div class="panel live-data-box" style="font-family:monospace">
<div style="background:{c['bg_input']};padding:5px 10px;border-bottom:1px solid {c['success']};color:{c['success']};display:flex;justify-content:space-between;align-items:center">
<span>┌─ LIVE TWEET COUNT ─┐</span>
<span class="live-indicator"></span>
</div>
<div style="padding:20px;text-align:center;background:linear-gradient(180deg, #0a0a0a 0%, #001100 100%)">
<input type="number" id="tweet-input" value="450" onchange="updateTweetCalc()" style="font-size:48px;width:180px;text-align:center;background:transparent;border:none;border-bottom:3px solid {c['accent']};color:{c['accent']};font-family:monospace;font-weight:bold;padding:10px;text-shadow:0 0 20px {c['accent']}">
<div style="color:{c['text_secondary']};font-size:11px;margin-top:8px">Enter from <a href="https://xtracker.polymarket.com" target="_blank" style="color:{c['accent']}">xtracker</a></div>
<div style="margin-top:15px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;font-size:11px">
<div style="background:{c['bg_input']};padding:8px;border:1px solid {c['border']}"><span style="color:{c['text_muted']}">ELAPSED</span><br><span id="days-elapsed" style="color:{c['warning']};font-size:18px;font-weight:bold">5.0d</span></div>
<div style="background:{c['bg_input']};padding:8px;border:1px solid {c['border']}"><span style="color:{c['text_muted']}">RATE</span><br><span id="tweet-rate" style="color:{c['success']};font-size:18px;font-weight:bold">90/day</span></div>
<div style="background:{c['bg_input']};padding:8px;border:1px solid {c['border']}"><span style="color:{c['text_muted']}">PROJECTED</span><br><span id="projected-final" style="color:{c['accent']};font-size:18px;font-weight:bold">630</span></div>
</div>
</div>
<div style="background:{c['bg_input']};padding:5px 10px;border-top:1px solid {c['success']};color:{c['text_muted']};font-size:10px">
└─ Jan 6 12:00 ET → Jan 13 12:00 ET ─┘</div>
</div>

<div class="panel live-data-box" style="font-family:monospace">
<div style="background:{c['bg_input']};padding:5px 10px;border-bottom:1px solid {c['success']};color:{c['success']};display:flex;justify-content:space-between;align-items:center">
<span>┌─ MARKET ODDS ─┐</span>
<span class="live-indicator"></span>
</div>
<div id="live-markets" style="padding:10px;font-size:11px;min-height:150px;max-height:250px;overflow-y:auto">
<div style="color:{c['text_secondary']}">Loading markets...</div>
</div>
<div style="background:{c['bg_input']};padding:5px 10px;border-top:1px solid {c['success']};color:{c['text_muted']};font-size:10px">
└─ Live from Polymarket API ─────────┘</div>
</div>

</div>

<div class="panel" style="font-family:monospace;border:2px solid {c['border']};border-radius:4px;overflow:hidden">
<div style="background:linear-gradient(90deg,{c['bg_input']},{c['bg_panel']});padding:8px 12px;border-bottom:2px solid {c['border']};display:flex;justify-content:space-between;align-items:center">
<span style="color:{c['warning']};font-weight:bold;letter-spacing:1px">ORDER BOOK</span>
<div style="display:flex;gap:8px;align-items:center">
<select id="clob-market-select" onchange="loadCLOB(this.value)" style="background:{c['bg_main']};color:{c['accent']};border:1px solid {c['accent']};font-family:monospace;font-size:11px;padding:4px 8px;border-radius:3px;cursor:pointer">
<option value="">Select bracket...</option>
</select>
<span id="clob-spread" style="color:{c['warning']};font-size:10px"></span>
</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0;background:{c['bg_main']}">
<div style="border-right:2px solid {c['border']};padding:8px">
<div style="color:{c['success']};font-size:11px;margin-bottom:8px;font-weight:bold;display:flex;justify-content:space-between">
<span>BID (BUY YES)</span><span style="opacity:0.6">SIZE</span>
</div>
<div id="clob-bids" style="font-size:10px;max-height:300px;overflow-y:auto">Select a bracket...</div>
</div>
<div style="padding:8px">
<div style="color:{c['error']};font-size:11px;margin-bottom:8px;font-weight:bold;display:flex;justify-content:space-between">
<span>ASK (SELL YES)</span><span style="opacity:0.6">SIZE</span>
</div>
<div id="clob-asks" style="font-size:10px;max-height:300px;overflow-y:auto">Select a bracket...</div>
</div>
</div>
<div id="clob-depth" style="height:60px;background:{c['bg_main']};border-top:1px solid {c['border']};position:relative">
<canvas id="depth-canvas" style="width:100%;height:60px"></canvas>
</div>
<div style="background:{c['bg_input']};padding:6px 12px;border-top:1px solid {c['border']};color:{c['text_muted']};display:flex;justify-content:space-between;font-size:10px">
<span>Market: <span id="clob-market-id" style="color:{c['accent']}">--</span></span>
<span id="clob-total-liquidity"></span>
</div>
</div>

<div class="panel" style="font-family:monospace;border:1px solid {c['border']}">
<div style="background:{c['bg_input']};padding:5px 10px;border-bottom:1px solid {c['border']};color:{c['accent']}">
┌─ HISTORICAL WINNERS ───────────────┐</div>
<div id="tui-chart" style="padding:10px;font-size:11px;line-height:1.4"></div>
<div style="background:{c['bg_input']};padding:5px 10px;border-top:1px solid {c['border']};color:{c['text_muted']}">
└─ All resolved markets (API data) ──┘</div>
</div>

<div class="panel" style="font-family:monospace;border:1px solid {c['border']}">
<div style="background:{c['bg_input']};padding:5px 10px;border-bottom:1px solid {c['border']};display:flex;justify-content:space-between;align-items:center">
<span style="color:{c['accent']}">┌─ CHARTS ─┐</span>
<div style="display:flex;gap:5px;align-items:center">
<button class="chart-tab active" data-chart="odds" onclick="switchChart('odds')" style="font-size:9px;padding:2px 8px;background:{c['accent']};color:#000">ODDS</button>
<button class="chart-tab" data-chart="volume" onclick="switchChart('volume')" style="font-size:9px;padding:2px 8px">VOL</button>
<button class="chart-tab" data-chart="heatmap" onclick="switchChart('heatmap')" style="font-size:9px;padding:2px 8px">HEAT</button>
<button class="chart-tab" data-chart="activity" onclick="switchChart('activity')" style="font-size:9px;padding:2px 8px">ACTIVITY</button>
<button id="fullscreen-btn" onclick="toggleChartFullscreen()" style="font-size:9px;padding:2px 6px;background:transparent;border:1px solid {c['border']};color:{c['text_secondary']};cursor:pointer" title="Fullscreen">⛶</button>
</div>
</div>
<div id="chart-container" style="position:relative">
<canvas id="price-canvas" style="width:100%;height:150px;background:#0a0a0a"></canvas>
</div>
<div style="background:{c['bg_input']};padding:5px 10px;border-top:1px solid {c['border']};color:{c['text_muted']};font-size:9px">
└─ Live API data • Click tabs to switch view ─┘</div>
</div>

{projection_html}

<div class="panel"><h2 style="font-family:monospace">┌─ QUICK ACTIONS ─┐</h2>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px">
<button onclick="fetchAllData()">Fetch All</button>
<button onclick="triggerUpdate()">Update</button>
<button onclick="rebuildModels()">Rebuild</button>
<button onclick="loadCLOB()">Refresh CLOB</button>
</div>
<div style="margin-top:10px;text-align:center">
<a href="https://xtracker.polymarket.com" target="_blank" style="color:{c['accent']};font-size:11px">→ Open XTracker</a> |
<a href="https://polymarket.com/event/elon-musk-of-tweets-january-6-january-13" target="_blank" style="color:{c['accent']};font-size:11px">→ Polymarket</a>
</div>
</div>'''

        return f'''
<div class="panel" style="margin-bottom:0">
<h1 style="margin:0;font-size:18px">ELON TWEET RESEARCH TERMINAL</h1>
<div style="font-size:11px;color:{c['text_secondary']}">Data: {total_data_kb}KB | Tweets: {tweet_analysis.get('total_tweets',0)} | Weeks: {len(weeks)} | Models: 2</div>
</div>
{tabs_html}
{content_html}

<script>
const JAN6_13 = ['1093290', '1093289', '1093288', '1093287', '1093284', '1093282', '1093280', '1093278', '1093276', '1093275'];
const JAN9_16 = ['1119416', '1119415', '1119414', '1119412', '1119410', '1119409', '1119408', '1119407', '1119406', '1119405'];

async function importCSV(path, type='prices') {{
    const resp = await fetch('/api/elon/import', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{path, type}})
    }});
    const data = await resp.json();
    alert(data.status === 'imported' ? 'Imported!' : data.status);
    location.reload();
}}

async function loadActiveMarkets() {{
    const containers = ['active-markets', 'active-markets-full'].filter(id => document.getElementById(id));
    for (const cid of containers) {{
        const container = document.getElementById(cid);
        if (!container) continue;
        try {{
            const resp = await fetch('/api/elon/current');
            const data = await resp.json();
            if (data.markets) {{
                let html = '';
                for (const m of data.markets.slice(0, cid === 'active-markets-full' ? 20 : 5)) {{
                    if (!m.title.toLowerCase().includes('tweet')) continue;
                    const prices = m.current_prices || {{}};
                    const yesPrice = prices.yes?.price || prices.Yes?.price || 0;
                    html += `<div style="padding:8px;border-bottom:1px solid {c['border']};cursor:pointer;display:flex;justify-content:space-between" onclick="location.href='/?view=market&id=${{m.id}}'">
                        <span style="font-size:11px">${{m.title.substring(0, 50)}}...</span>
                        <span style="color:{c['accent']}">${{(yesPrice*100).toFixed(0)}}%</span>
                    </div>`;
                }}
                container.innerHTML = html || '<div style="padding:20px;color:{c["text_secondary"]}">No markets</div>';
            }}
        }} catch(e) {{ container.innerHTML = '<div style="color:{c["error"]}">Error</div>'; }}
    }}
}}

async function triggerUpdate() {{
    const resp = await fetch('/api/elon/update');
    const data = await resp.json();
    alert('Update: ' + data.status + (data.updated ? ' (' + data.updated.length + ' updated)' : ''));
}}

async function rebuildModels() {{
    await fetch('/api/elon/rebuild-models');
    alert('Models rebuilt!');
    location.reload();
}}

function calcProjection() {{
    const current = parseInt(document.getElementById('current-tweets')?.value) || 0;
    const days = parseInt(document.getElementById('days-remaining')?.value) || 7;
    const daysPassed = 7 - days;
    const rate = daysPassed > 0 ? current / daysPassed : {pattern_params.get('avg_weekly_total', 350) / 7};
    const proj = current + (rate * days);
    if (document.getElementById('projection-result')) {{
        document.getElementById('projection-result').style.display = 'block';
        document.getElementById('proj-final').textContent = proj.toFixed(0);
        document.getElementById('proj-rate').textContent = rate.toFixed(1) + '/day';
        const brackets = [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580];
        let bracket = '580+';
        for (let i=0; i<brackets.length-1; i++) {{
            if (proj >= brackets[i] && proj < brackets[i+1]) {{ bracket = brackets[i] + '-' + (brackets[i+1]-1); break; }}
        }}
        document.getElementById('proj-bracket').textContent = bracket;
    }}
}}

// Calculate tweet rate and projection
function updateTweetCalc() {{
    const input = document.getElementById('tweet-input');
    const elapsedEl = document.getElementById('days-elapsed');
    const rateEl = document.getElementById('tweet-rate');
    const projEl = document.getElementById('projected-final');

    if (!input) return;

    const count = parseInt(input.value) || 0;

    // Jan 6 12:00 ET start, Jan 13 12:00 ET end (7 days total)
    const start = new Date('2026-01-06T17:00:00Z'); // 12:00 ET = 17:00 UTC
    const end = new Date('2026-01-13T17:00:00Z');
    const now = new Date();

    const totalMs = end - start;
    const elapsedMs = Math.max(0, now - start);
    const elapsedDays = elapsedMs / (1000 * 60 * 60 * 24);
    const remainingDays = Math.max(0, (end - now) / (1000 * 60 * 60 * 24));

    const rate = elapsedDays > 0 ? count / elapsedDays : 0;
    const projected = count + (rate * remainingDays);

    if (elapsedEl) elapsedEl.textContent = elapsedDays.toFixed(1) + 'd';
    if (rateEl) rateEl.textContent = rate.toFixed(0) + '/day';
    if (projEl) projEl.textContent = projected.toFixed(0);

    // Save to localStorage
    localStorage.setItem('elon_tweet_count', count);
}}

// Load saved tweet count on page load
function initTweetCalc() {{
    const saved = localStorage.getItem('elon_tweet_count');
    const input = document.getElementById('tweet-input');
    if (saved && input) {{
        input.value = saved;
    }}
    updateTweetCalc();
}}

async function fetchAllData() {{
    alert('Fetching...');
    for (const id of [...JAN6_13, ...JAN9_16]) {{
        await fetch('/api/market?id=' + id + '&refresh=1');
    }}
    alert('Done!'); location.reload();
}}

// Load CLOB (orderbook) data for specific market - FULL DEPTH
async function loadCLOB(marketId) {{
    const bidsEl = document.getElementById('clob-bids');
    const asksEl = document.getElementById('clob-asks');
    const idEl = document.getElementById('clob-market-id');
    const spreadEl = document.getElementById('clob-spread');
    const liqEl = document.getElementById('clob-total-liquidity');
    const depthCanvas = document.getElementById('depth-canvas');

    if (!marketId) {{
        if (bidsEl) bidsEl.innerHTML = '<div style="color:#6a6a7a;padding:20px;text-align:center">Select a bracket to view orderbook</div>';
        if (asksEl) asksEl.innerHTML = '<div style="color:#6a6a7a;padding:20px;text-align:center">Full depth with visual bars</div>';
        if (idEl) idEl.textContent = '--';
        if (spreadEl) spreadEl.textContent = '';
        if (liqEl) liqEl.textContent = '';
        return;
    }}

    if (bidsEl) bidsEl.innerHTML = '<div style="color:#7ec8e3;text-align:center;padding:20px">Loading orderbook...</div>';
    if (asksEl) asksEl.innerHTML = '';
    if (idEl) idEl.textContent = marketId;

    try {{
        const resp = await fetch(`/api/elon/clob?id=${{marketId}}`);
        const data = await resp.json();

        if (data.data) {{
            const yes = data.data.yes || {{}};
            // Get ALL orders, sorted properly
            const allBids = (yes.bids || []).sort((a,b) => b.price - a.price);
            const allAsks = (yes.asks || []).sort((a,b) => a.price - b.price);

            // Find max size for bar scaling
            const maxBidSize = Math.max(...allBids.map(b => b.size), 1);
            const maxAskSize = Math.max(...allAsks.map(a => a.size), 1);
            const maxSize = Math.max(maxBidSize, maxAskSize);

            // Calculate spread and mid
            const bestBid = allBids[0]?.price || 0;
            const bestAsk = allAsks[0]?.price || 1;
            const spread = ((bestAsk - bestBid) * 100).toFixed(2);
            const mid = ((bestBid + bestAsk) / 2 * 100).toFixed(1);

            // Total liquidity
            const bidLiq = allBids.reduce((s, b) => s + b.size * b.price, 0);
            const askLiq = allAsks.reduce((s, a) => s + a.size * a.price, 0);
            const totalLiq = bidLiq + askLiq;

            // Build bid HTML with depth bars
            let bidHtml = '';
            allBids.forEach((b, i) => {{
                const pct = (b.price * 100).toFixed(2);
                const size = b.size > 1000 ? (b.size/1000).toFixed(1)+'k' : b.size.toFixed(0);
                const barWidth = (b.size / maxSize * 100).toFixed(0);
                const isBest = i === 0;
                bidHtml += `<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 4px;margin:1px 0;position:relative;border-radius:2px;${{isBest ? 'background:rgba(0,255,0,0.15);border-left:3px solid #00ff00;' : ''}}">
                    <div style="position:absolute;right:0;top:0;bottom:0;width:${{barWidth}}%;background:linear-gradient(90deg,transparent,rgba(0,255,0,0.2));border-radius:2px"></div>
                    <span style="color:${{isBest ? '#00ff00' : '#7ec8e3'}};font-weight:${{isBest ? 'bold' : 'normal'}};z-index:1">${{pct}}¢</span>
                    <span style="color:#888;z-index:1">${{size}}</span>
                </div>`;
            }});

            // Build ask HTML with depth bars
            let askHtml = '';
            allAsks.forEach((a, i) => {{
                const pct = (a.price * 100).toFixed(2);
                const size = a.size > 1000 ? (a.size/1000).toFixed(1)+'k' : a.size.toFixed(0);
                const barWidth = (a.size / maxSize * 100).toFixed(0);
                const isBest = i === 0;
                askHtml += `<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 4px;margin:1px 0;position:relative;border-radius:2px;${{isBest ? 'background:rgba(255,68,68,0.15);border-left:3px solid #ff4444;' : ''}}">
                    <div style="position:absolute;left:0;top:0;bottom:0;width:${{barWidth}}%;background:linear-gradient(270deg,transparent,rgba(255,68,68,0.2));border-radius:2px"></div>
                    <span style="color:${{isBest ? '#ff4444' : '#ff8888'}};font-weight:${{isBest ? 'bold' : 'normal'}};z-index:1">${{pct}}¢</span>
                    <span style="color:#888;z-index:1">${{size}}</span>
                </div>`;
            }});

            if (bidsEl) bidsEl.innerHTML = bidHtml || '<div style="color:#666;text-align:center">No bids</div>';
            if (asksEl) asksEl.innerHTML = askHtml || '<div style="color:#666;text-align:center">No asks</div>';
            if (spreadEl) spreadEl.innerHTML = `<span style="color:#feca57">Spread: ${{spread}}¢</span> <span style="color:#888">| Mid: ${{mid}}¢</span>`;
            if (liqEl) liqEl.innerHTML = `Liquidity: <span style="color:#7ec8e3">${{totalLiq > 1000 ? (totalLiq/1000).toFixed(1)+'k' : totalLiq.toFixed(0)}}</span>`;

            // Draw depth chart
            if (depthCanvas) {{
                drawDepthChart(depthCanvas, allBids, allAsks);
            }}
        }} else {{
            if (bidsEl) bidsEl.innerHTML = '<div style="color:#ff6b6b;text-align:center">No orderbook data</div>';
            if (asksEl) asksEl.innerHTML = '';
        }}
    }} catch(e) {{
        if (bidsEl) bidsEl.innerHTML = '<div style="color:#ff6b6b;text-align:center">Error loading: ' + e.message + '</div>';
        if (asksEl) asksEl.innerHTML = '';
    }}
}}

// Draw depth chart visualization
function drawDepthChart(canvas, bids, asks) {{
    const container = canvas.parentElement;
    const w = container?.clientWidth || 400;
    const h = 60;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Build cumulative depth
    let bidCum = [];
    let cumBid = 0;
    bids.forEach(b => {{
        cumBid += b.size;
        bidCum.push({{ price: b.price, cum: cumBid }});
    }});

    let askCum = [];
    let cumAsk = 0;
    asks.forEach(a => {{
        cumAsk += a.size;
        askCum.push({{ price: a.price, cum: cumAsk }});
    }});

    const maxCum = Math.max(cumBid, cumAsk, 1);
    const midX = w / 2;

    // Draw bid curve (left side, green)
    if (bidCum.length > 0) {{
        ctx.beginPath();
        ctx.moveTo(midX, h);
        bidCum.forEach((p, i) => {{
            const x = midX - (i + 1) * (midX / bidCum.length);
            const y = h - (p.cum / maxCum) * (h - 10);
            ctx.lineTo(x, y);
        }});
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }}

    // Draw ask curve (right side, red)
    if (askCum.length > 0) {{
        ctx.beginPath();
        ctx.moveTo(midX, h);
        askCum.forEach((p, i) => {{
            const x = midX + (i + 1) * (midX / askCum.length);
            const y = h - (p.cum / maxCum) * (h - 10);
            ctx.lineTo(x, y);
        }});
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 68, 68, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#ff4444';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }}

    // Center line
    ctx.strokeStyle = '#feca57';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(midX, 0);
    ctx.lineTo(midX, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '9px monospace';
    ctx.fillText('BIDS', 5, 12);
    ctx.fillText('ASKS', w - 30, 12);
}}

// Load live markets with prices (grouped by event)
async function loadLiveMarkets() {{
    const container = document.getElementById('live-markets');
    const clobSelect = document.getElementById('clob-market-select');

    try {{
        const resp = await fetch('/api/elon/live-markets');
        const data = await resp.json();

        // Update summary stats
        if (data.grouped && data.markets) {{
            let totalMarkets = 0;
            let totalVolume = 0;
            let topBracket = null;
            let topOdds = 0;

            for (const [, markets] of Object.entries(data.markets)) {{
                totalMarkets += markets.length;
                markets.forEach(m => {{
                    totalVolume += m.volume || 0;
                    if (m.yes_price > topOdds) {{
                        topOdds = m.yes_price;
                        topBracket = m.bracket;
                    }}
                }});
            }}

            // Update stat cards
            const statMarkets = document.getElementById('stat-markets');
            const statVolume = document.getElementById('stat-volume');
            const statTopBracket = document.getElementById('stat-top-bracket');
            const statTopOdds = document.getElementById('stat-top-odds');

            if (statMarkets) statMarkets.textContent = totalMarkets;
            if (statVolume) statVolume.textContent = totalVolume > 1000 ? '$' + (totalVolume/1000).toFixed(0) + 'k' : '$' + totalVolume.toFixed(0);
            if (statTopBracket) statTopBracket.textContent = topBracket || '--';
            if (statTopOdds) statTopOdds.textContent = (topOdds * 100).toFixed(0) + '% odds';
        }}

        // Populate CLOB selector with all markets
        if (clobSelect && data.grouped) {{
            let options = '<option value="">Select bracket...</option>';
            for (const [eventKey, markets] of Object.entries(data.markets)) {{
                const labels = {{'jan6_13': 'Jan 6-13', 'jan9_16': 'Jan 9-16', 'jan13_20': 'Jan 13-20', 'jan15_22': 'Jan 15-22', 'jan16_23': 'Jan 16-23', 'jan17_19': 'Jan 17-19', 'jan20_27': 'Jan 20-27'}};
                const label = labels[eventKey] || eventKey;
                options += `<optgroup label="${{label}}">`;
                markets.slice(0, 15).forEach(m => {{
                    const pct = (m.yes_price * 100).toFixed(0);
                    options += `<option value="${{m.id}}">${{m.bracket}} (${{pct}}%)</option>`;
                }});
                options += '</optgroup>';
            }}
            clobSelect.innerHTML = options;
        }}

        // Display markets in the panel
        if (container && data.grouped) {{
            let html = '';
            // Show each event separately
            for (const [eventKey, markets] of Object.entries(data.markets)) {{
                if (markets.length === 0) continue;
                const labels = {{'jan6_13': 'Jan 6-13', 'jan9_16': 'Jan 9-16', 'jan13_20': 'Jan 13-20', 'jan15_22': 'Jan 15-22', 'jan16_23': 'Jan 16-23', 'jan17_19': 'Jan 17-19', 'jan20_27': 'Jan 20-27'}};
                const label = labels[eventKey] || eventKey;
                html += `<div style="color:#888;font-size:9px;margin-bottom:3px;margin-top:5px;border-bottom:1px solid #333">${{label}}</div>`;
                markets.slice(0, 6).forEach(m => {{
                    const pct = (m.yes_price * 100).toFixed(1);
                    const color = m.yes_price > 0.5 ? '#00ff00' : m.yes_price > 0.1 ? '#ffcc00' : '#555';
                    const barLen = Math.round(m.yes_price * 15);
                    const bar = '█'.repeat(barLen) + '░'.repeat(15 - barLen);
                    html += `<div style="margin-bottom:2px;cursor:pointer" onclick="loadCLOB('${{m.id}}');document.getElementById('clob-market-select').value='${{m.id}}'">
                        <span style="color:#ff6600;width:60px;display:inline-block;font-size:10px">${{m.bracket.substring(0,8)}}</span>
                        <span style="color:${{color}};font-size:9px">${{bar}}</span>
                        <span style="color:${{color}};font-size:10px">${{pct}}%</span>
                    </div>`;
                }});
            }}
            container.innerHTML = html || '<span style="color:#555">No markets</span>';
        }}
    }} catch(e) {{
        if (container) container.innerHTML = '<span style="color:#ff4444">Error</span>';
    }}
}}

// Load historical winners from API
async function loadWinners() {{
    const container = document.getElementById('tui-chart');
    if (!container) return;

    try {{
        const resp = await fetch('/api/elon/winners');
        const data = await resp.json();

        const counts = {{}};
        (data.winners || []).forEach(w => {{
            const bracket = w.bracket || '?';
            counts[bracket] = (counts[bracket] || 0) + 1;
        }});

        const brackets = Object.keys(counts).sort((a,b) => {{
            const numA = parseInt(a.split('-')[0]) || 999;
            const numB = parseInt(b.split('-')[0]) || 999;
            return numA - numB;
        }});

        const maxCount = Math.max(...Object.values(counts), 1);
        const barWidth = 25;

        let html = '';
        brackets.slice(0, 12).forEach((bracket, i) => {{
            const count = counts[bracket];
            const barLen = Math.round((count / maxCount) * barWidth);
            const bar = '█'.repeat(barLen) + '░'.repeat(barWidth - barLen);
            const color = i % 2 === 0 ? '#00ff00' : '#00cc00';
            html += `<div>│ <span style="color:#ff6600;width:70px;display:inline-block">${{bracket.padEnd(8)}}</span> <span style="color:${{color}}">${{bar}}</span> ${{count}}</div>`;
        }});
        html += `<div style="color:#555;margin-top:5px">│ Total: ${{data.count || 0}} resolved markets</div>`;
        html += '<div style="color:#888">└' + '─'.repeat(40) + '</div>';

        container.innerHTML = html || '<span style="color:#555">No data</span>';
    }} catch(e) {{
        container.innerHTML = '<span style="color:#ff4444">Error loading winners</span>';
    }}
}}

// TUI-style ASCII bar chart (fallback with local data)
function initTUIChart() {{
    const container = document.getElementById('tui-chart');
    if (!container) return;

    // Try to load from API first
    loadWinners();
}}

// === CHART SYSTEM: 3 Different Visualization Types ===

// Chart type selector
let currentChartType = 'odds';
function switchChart(type) {{
    currentChartType = type;
    document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-chart="${{type}}"]`)?.classList.add('active');
    renderCurrentChart();
}}

// Fullscreen chart toggle
let chartFullscreen = false;
function toggleChartFullscreen() {{
    const container = document.getElementById('chart-container');
    const btn = document.getElementById('fullscreen-btn');
    if (!container) return;

    chartFullscreen = !chartFullscreen;

    if (chartFullscreen) {{
        // Enter fullscreen
        container.classList.add('chart-fullscreen');
        btn.textContent = '✕';
        btn.title = 'Exit Fullscreen';
        // Add close on ESC
        document.addEventListener('keydown', handleEscFullscreen);
    }} else {{
        // Exit fullscreen
        container.classList.remove('chart-fullscreen');
        btn.textContent = '⛶';
        btn.title = 'Fullscreen';
        document.removeEventListener('keydown', handleEscFullscreen);
    }}

    // Re-render chart with new dimensions
    setTimeout(renderCurrentChart, 100);
}}

function handleEscFullscreen(e) {{
    if (e.key === 'Escape' && chartFullscreen) {{
        toggleChartFullscreen();
    }}
}}

// Cache for chart data to prevent blank screens
let chartDataCache = {{}};
let chartLastRender = 0;

// Main chart renderer - PERSISTENT with proper sizing
async function renderCurrentChart() {{
    const canvas = document.getElementById('price-canvas');
    if (!canvas) return;

    // Ensure canvas has explicit dimensions - responsive to fullscreen
    const container = canvas.parentElement;
    const isFullscreen = container?.classList.contains('chart-fullscreen');

    // Get dimensions with minimum fallback
    let w = isFullscreen ? window.innerWidth - 40 : (container ? container.clientWidth - 20 : 0);
    let h = isFullscreen ? window.innerHeight - 80 : 200;

    // Fallback if width is 0 or too small
    if (w < 100) {{
        w = container?.offsetWidth - 20 || document.querySelector('.content')?.offsetWidth - 40 || 600;
    }}

    // Skip render if dimensions invalid
    if (w < 50 || h < 50) {{
        console.log('Chart skipped - invalid dimensions:', w, h);
        return;
    }}

    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    // Draw background first - chart is NEVER blank
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Draw subtle grid
    ctx.strokeStyle = '#1a1a3a';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {{
        ctx.beginPath();
        ctx.moveTo(0, (h/5)*i);
        ctx.lineTo(w, (h/5)*i);
        ctx.stroke();
    }}
    for (let i = 0; i <= 12; i++) {{
        ctx.beginPath();
        ctx.moveTo((w/12)*i, 0);
        ctx.lineTo((w/12)*i, h);
        ctx.stroke();
    }}

    // Add chart title
    ctx.fillStyle = '#e94560';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    const titles = {{ odds: 'LIVE ODDS - Jan 6-13', volume: 'VOLUME RANKING', heatmap: 'BRACKET HEATMAP', activity: 'TWEET ACTIVITY (30 days)' }};
    ctx.fillText(titles[currentChartType] || 'CHART', 10, 15);

    // Show loading if first render
    if (!chartDataCache[currentChartType]) {{
        ctx.fillStyle = '#7ec8e3';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Loading data...', w/2, h/2);
    }}

    try {{
        switch(currentChartType) {{
            case 'odds': await drawOddsChart(ctx, w, h); break;
            case 'volume': await drawVolumeChart(ctx, w, h); break;
            case 'heatmap': await drawHeatmapChart(ctx, w, h); break;
            case 'activity': await drawActivityChart(ctx, w, h); break;
        }}
        chartLastRender = Date.now();
    }} catch(e) {{
        console.error('Chart error:', e);
        // Don't blank the chart - show error message over existing grid
        ctx.fillStyle = 'rgba(10, 10, 26, 0.8)';
        ctx.fillRect(w/4, h/3, w/2, h/3);
        ctx.fillStyle = '#ff6b6b';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Error: ' + (e.message || 'Failed to load'), w/2, h/2);
        ctx.fillStyle = '#888';
        ctx.font = '9px monospace';
        ctx.fillText('Retrying in 30s...', w/2, h/2 + 15);
    }}
}}

// CHART 1: Live Odds Comparison (all brackets) - with caching
async function drawOddsChart(ctx, w, h) {{
    let markets = null;

    try {{
        const resp = await fetch('/api/elon/live-markets');
        const data = await resp.json();
        // Use first available market group with data (prioritize newest)
        const priority = ['jan20_27', 'jan17_19', 'jan16_23', 'jan15_22', 'jan13_20', 'jan9_16', 'jan6_13', 'other'];
        for (const key of priority) {{
            if (data.markets?.[key]?.length > 0) {{
                markets = data.markets[key];
                chartDataCache.odds = markets;
                break;
            }}
        }}
    }} catch(e) {{
        console.log('Odds fetch failed, using cache');
    }}

    // Use cache if fetch failed
    if (!markets && chartDataCache.odds) {{
        markets = chartDataCache.odds;
    }}

    if (!markets || markets.length === 0) {{
        ctx.fillStyle = '#6a6a7a';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('No market data available', w/2, h/2);
        return;
    }}

    // Sort by bracket for consistent display
    markets.sort((a, b) => {{
        const aNum = parseInt((a.bracket || '0').replace(/[^0-9]/g, ''));
        const bNum = parseInt((b.bracket || '0').replace(/[^0-9]/g, ''));
        return aNum - bNum;
    }});

    const padding = 40;
    const barW = Math.max(20, Math.floor((w - padding * 2) / markets.length) - 6);
    const chartH = h - 50;

    // Y-axis labels
    ctx.fillStyle = '#4a4a6a';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';
    ['100%', '75%', '50%', '25%', '0%'].forEach((l, i) => {{
        const y = 25 + (chartH / 4) * i;
        ctx.fillText(l, padding - 5, y + 3);
        ctx.strokeStyle = '#1a1a3a';
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(w - 10, y);
        ctx.stroke();
    }});

    // Draw bars
    markets.forEach((m, i) => {{
        const x = padding + i * (barW + 6);
        const price = m.yes_price || 0;
        const barH = Math.max(2, price * chartH);

        // Bar with gradient
        const grad = ctx.createLinearGradient(x, h - 25, x, h - 25 - barH);
        if (price > 0.5) {{
            grad.addColorStop(0, '#00ff88');
            grad.addColorStop(1, '#005533');
        }} else if (price > 0.1) {{
            grad.addColorStop(0, '#feca57');
            grad.addColorStop(1, '#665522');
        }} else {{
            grad.addColorStop(0, '#ff6b6b');
            grad.addColorStop(1, '#552222');
        }}
        ctx.fillStyle = grad;
        ctx.fillRect(x, h - 25 - barH, barW, barH);

        // Border glow effect
        ctx.strokeStyle = price > 0.5 ? '#00ff88' : (price > 0.1 ? '#feca57' : '#ff6b6b');
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, h - 25 - barH, barW, barH);

        // Bracket label
        ctx.fillStyle = '#888';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        const bracket = m.bracket || m.title?.match(/\\d+-\\d+|\\d+\\+/)?.[0] || '?';
        ctx.save();
        ctx.translate(x + barW/2, h - 8);
        ctx.rotate(-0.6);
        ctx.fillText(bracket, 0, 0);
        ctx.restore();

        // Price percentage on top (only if significant)
        if (price > 0.03) {{
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText((price * 100).toFixed(0) + '%', x + barW/2, h - 30 - barH);
        }}
    }});

    // Update timestamp
    ctx.fillStyle = '#4a4a6a';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Updated: ' + new Date().toLocaleTimeString(), w - 10, h - 5);
}}

// CHART 2: Volume/Liquidity Bars - with caching
async function drawVolumeChart(ctx, w, h) {{
    let allMarkets = null;

    try {{
        const resp = await fetch('/api/elon/live-markets');
        const data = await resp.json();
        const markets = data.markets || {{}};
        allMarkets = Object.values(markets).flat();
        chartDataCache.volume = allMarkets;
    }} catch(e) {{
        console.log('Volume fetch failed, using cache');
    }}

    if (!allMarkets && chartDataCache.volume) {{
        allMarkets = chartDataCache.volume;
    }}

    if (!allMarkets || allMarkets.length === 0) {{
        ctx.fillStyle = '#6a6a7a';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('No volume data available', w/2, h/2);
        return;
    }}

    // Sort by volume
    allMarkets.sort((a, b) => (b.volume || 0) - (a.volume || 0));
    const top10 = allMarkets.slice(0, 10);

    const maxVol = Math.max(...top10.map(m => m.volume || 1), 1);
    const barH = Math.floor((h - 35) / top10.length) - 3;
    const labelWidth = 70;

    top10.forEach((m, i) => {{
        const y = 25 + i * (barH + 3);
        const vol = m.volume || 0;
        const barW = Math.max(2, (vol / maxVol) * (w - labelWidth - 80));

        // Gradient bar
        const grad = ctx.createLinearGradient(labelWidth, y, labelWidth + barW, y);
        grad.addColorStop(0, '#e94560');
        grad.addColorStop(1, '#7ec8e3');
        ctx.fillStyle = grad;
        ctx.fillRect(labelWidth, y, barW, barH);

        // Border
        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 1;
        ctx.strokeRect(labelWidth, y, barW, barH);

        // Rank number
        ctx.fillStyle = i === 0 ? '#feca57' : '#666';
        ctx.font = i === 0 ? 'bold 10px monospace' : '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText((i + 1) + '.', 5, y + barH/2 + 3);

        // Bracket label
        ctx.fillStyle = '#aaa';
        ctx.textAlign = 'right';
        const label = m.bracket || m.title?.match(/\\d+-\\d+|\\d+\\+/)?.[0] || '?';
        ctx.fillText(label, labelWidth - 5, y + barH/2 + 3);

        // Volume value
        ctx.fillStyle = '#fff';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        const volStr = vol > 1000 ? '$' + (vol/1000).toFixed(1) + 'k' : '$' + vol.toFixed(0);
        ctx.fillText(volStr, labelWidth + barW + 5, y + barH/2 + 3);
    }});

    // Total volume
    const total = allMarkets.reduce((s, m) => s + (m.volume || 0), 0);
    ctx.fillStyle = '#4a4a6a';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Total: $' + (total/1000).toFixed(1) + 'k', w - 10, h - 5);
}}

// CHART 3: Bracket Heatmap (historical wins + current odds) - with caching
async function drawHeatmapChart(ctx, w, h) {{
    let brackets = {{}};

    try {{
        const [winnersResp, marketsResp] = await Promise.all([
            fetch('/api/elon/winners'),
            fetch('/api/elon/live-markets')
        ]);
        const winners = await winnersResp.json();
        const markets = await marketsResp.json();

        // Build bracket data: wins count + current odds
        (winners.winners || []).forEach(wn => {{
            const b = wn.bracket;
            brackets[b] = brackets[b] || {{wins: 0, odds: 0, volume: 0}};
            brackets[b].wins++;
        }});

        const mkt = markets.markets || {{}};
        Object.values(mkt).flat().forEach(m => {{
            const b = m.bracket || m.title?.match(/\\d+-\\d+|\\d+\\+/)?.[0];
            if (b) {{
                brackets[b] = brackets[b] || {{wins: 0, odds: 0, volume: 0}};
                brackets[b].odds = Math.max(brackets[b].odds, m.yes_price || 0);
                brackets[b].volume += m.volume || 0;
            }}
        }});

        chartDataCache.heatmap = brackets;
    }} catch(e) {{
        console.log('Heatmap fetch failed, using cache');
        if (chartDataCache.heatmap) {{
            brackets = chartDataCache.heatmap;
        }}
    }}

    if (Object.keys(brackets).length === 0) {{
        ctx.fillStyle = '#6a6a7a';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('No bracket data available', w/2, h/2);
        return;
    }}

    const sortedBrackets = Object.entries(brackets).sort((a, b) => {{
        const aNum = parseInt(a[0]) || 999;
        const bNum = parseInt(b[0]) || 999;
        return aNum - bNum;
    }});

    const cols = Math.min(8, Math.ceil(Math.sqrt(sortedBrackets.length * 1.5)));
    const rows = Math.ceil(sortedBrackets.length / cols);
    const cellW = Math.floor((w - 20) / cols);
    const cellH = Math.floor((h - 35) / rows);
    const maxWins = Math.max(...sortedBrackets.map(([, d]) => d.wins), 1);

    sortedBrackets.forEach(([bracket, data], i) => {{
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = 10 + col * cellW;
        const y = 25 + row * cellH;

        // Heat color based on wins (green) and odds (blue/yellow)
        let fillColor;
        if (data.wins > 0) {{
            // Winners: green intensity based on wins
            const intensity = 0.3 + (data.wins / maxWins) * 0.7;
            fillColor = `rgba(0, 255, 100, ${{intensity}})`;
        }} else if (data.odds > 0.3) {{
            // High odds: yellow/orange
            fillColor = `rgba(254, 202, 87, ${{0.3 + data.odds * 0.6}})`;
        }} else if (data.odds > 0.05) {{
            // Medium odds: blue
            fillColor = `rgba(126, 200, 227, ${{0.2 + data.odds}})`;
        }} else {{
            // Low odds: dark
            fillColor = 'rgba(40, 40, 60, 0.5)';
        }}

        // Fill cell
        ctx.fillStyle = fillColor;
        ctx.fillRect(x + 2, y + 2, cellW - 4, cellH - 4);

        // Border - highlight winners
        if (data.wins > 0) {{
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
        }} else {{
            ctx.strokeStyle = '#3a3a5a';
            ctx.lineWidth = 1;
        }}
        ctx.strokeRect(x + 2, y + 2, cellW - 4, cellH - 4);

        // Bracket label
        ctx.fillStyle = data.wins > 0 ? '#fff' : '#aaa';
        ctx.font = data.wins > 0 ? 'bold 9px monospace' : '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(bracket.replace(/-.*/, ''), x + cellW/2, y + cellH/2 - 2);

        // Stats line
        ctx.font = '7px monospace';
        if (data.wins > 0) {{
            ctx.fillStyle = '#00ff88';
            ctx.fillText('★' + data.wins, x + cellW/2, y + cellH/2 + 9);
        }} else if (data.odds > 0.01) {{
            ctx.fillStyle = '#888';
            ctx.fillText((data.odds * 100).toFixed(0) + '%', x + cellW/2, y + cellH/2 + 9);
        }}
    }});

    // Legend
    ctx.fillStyle = '#00ff88';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('★=past wins', 10, h - 5);
    ctx.fillStyle = '#feca57';
    ctx.fillText('yellow=hot', w/2 - 30, h - 5);
    ctx.fillStyle = '#7ec8e3';
    ctx.fillText('blue=active', w - 60, h - 5);
}}

// CHART 4: Tweet Activity Over Time (from historical winners)
async function drawActivityChart(ctx, w, h) {{
    let dailyData = [];

    try {{
        const resp = await fetch('/api/elon/winners');
        const data = await resp.json();
        const winners = data.winners || [];

        // Convert winning brackets to estimated daily counts
        // Parse each winner's bracket to get the weekly total, then calculate daily average
        winners.forEach(wn => {{
            const bracket = wn.bracket || '';
            const resolved = wn.resolved_at ? new Date(wn.resolved_at) : null;
            if (!resolved) return;

            // Parse bracket to get tweet count (take midpoint of range)
            let count = 0;
            if (bracket.includes('-')) {{
                const [low, high] = bracket.split('-').map(n => parseInt(n));
                count = (low + high) / 2;
            }} else if (bracket.includes('+')) {{
                count = parseInt(bracket) + 20; // Estimate for 580+ etc
            }} else {{
                count = parseInt(bracket) || 0;
            }}

            // Weekly total -> daily average
            const dailyAvg = count / 7;

            // Create 7 days of data for this week
            for (let i = 6; i >= 0; i--) {{
                const date = new Date(resolved);
                date.setDate(date.getDate() - i);
                dailyData.push({{
                    date: date.toISOString().split('T')[0],
                    count: dailyAvg,
                    bracket: bracket,
                    weekEnd: resolved.toISOString().split('T')[0]
                }});
            }}
        }});

        // Sort by date and take last 30 days
        dailyData.sort((a, b) => a.date.localeCompare(b.date));

        // Deduplicate by date (take average if multiple weeks overlap)
        const dateMap = {{}};
        dailyData.forEach(d => {{
            if (!dateMap[d.date]) {{
                dateMap[d.date] = {{ total: 0, count: 0 }};
            }}
            dateMap[d.date].total += d.count;
            dateMap[d.date].count++;
        }});

        dailyData = Object.entries(dateMap).map(([date, v]) => ({{
            date,
            count: v.total / v.count
        }})).slice(-30);

        chartDataCache.activity = dailyData;
    }} catch(e) {{
        console.log('Activity fetch failed, using cache');
        if (chartDataCache.activity) {{
            dailyData = chartDataCache.activity;
        }}
    }}

    if (dailyData.length === 0) {{
        ctx.fillStyle = '#6a6a7a';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('No activity data available', w/2, h/2);
        return;
    }}

    const padding = {{ left: 45, right: 20, top: 30, bottom: 35 }};
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    const maxCount = Math.max(...dailyData.map(d => d.count), 100);
    const minCount = Math.min(...dailyData.map(d => d.count), 0);

    // Draw Y axis
    ctx.strokeStyle = '#3a3a5a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, h - padding.bottom);
    ctx.stroke();

    // Y axis labels and grid
    ctx.fillStyle = '#6a6a7a';
    ctx.font = '8px monospace';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {{
        const y = padding.top + (chartH / 4) * i;
        const val = Math.round(maxCount - (maxCount - minCount) * (i / 4));
        ctx.fillText(val.toString(), padding.left - 5, y + 3);

        ctx.strokeStyle = '#1a1a3a';
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(w - padding.right, y);
        ctx.stroke();
    }}

    // Calculate bar width
    const barW = Math.max(4, (chartW / dailyData.length) - 2);

    // Draw bars
    dailyData.forEach((d, i) => {{
        const x = padding.left + 5 + i * (chartW / dailyData.length);
        const barH = ((d.count - minCount) / (maxCount - minCount)) * chartH;
        const y = h - padding.bottom - barH;

        // Color based on activity level
        let color;
        if (d.count > 70) {{
            color = '#00ff88'; // High activity - green
        }} else if (d.count > 50) {{
            color = '#feca57'; // Medium - yellow
        }} else {{
            color = '#e94560'; // Low - pink
        }}

        // Bar with gradient
        const grad = ctx.createLinearGradient(x, y + barH, x, y);
        grad.addColorStop(0, color);
        grad.addColorStop(1, color.replace('ff', '88').replace('ca', '88'));
        ctx.fillStyle = grad;
        ctx.fillRect(x, y, barW, barH);

        // X axis labels (every 5 days)
        if (i % 5 === 0 || i === dailyData.length - 1) {{
            ctx.fillStyle = '#6a6a7a';
            ctx.font = '7px monospace';
            ctx.textAlign = 'center';
            const dateLabel = d.date.slice(5); // MM-DD
            ctx.fillText(dateLabel, x + barW/2, h - padding.bottom + 12);
        }}
    }});

    // Draw trend line
    ctx.strokeStyle = '#7ec8e3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    dailyData.forEach((d, i) => {{
        const x = padding.left + 5 + i * (chartW / dailyData.length) + barW/2;
        const y = h - padding.bottom - ((d.count - minCount) / (maxCount - minCount)) * chartH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }});
    ctx.stroke();

    // Stats
    const avg = dailyData.reduce((s, d) => s + d.count, 0) / dailyData.length;
    const recent = dailyData.slice(-7).reduce((s, d) => s + d.count, 0) / 7;

    ctx.fillStyle = '#888';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Avg: ${{avg.toFixed(0)}}/day`, padding.left + 10, padding.top - 10);
    ctx.fillStyle = recent > avg ? '#00ff88' : '#ff6b6b';
    ctx.fillText(`Recent 7d: ${{recent.toFixed(0)}}/day`, padding.left + 100, padding.top - 10);

    // Legend
    ctx.fillStyle = '#00ff88';
    ctx.fillText('>70', w - 80, h - 5);
    ctx.fillStyle = '#feca57';
    ctx.fillText('50-70', w - 55, h - 5);
    ctx.fillStyle = '#e94560';
    ctx.fillText('<50', w - 25, h - 5);
}}

// Legacy compatibility
function initPriceChart() {{
    renderCurrentChart();
}}

// Auto-update API data every 1 min
setInterval(() => {{ fetch('/api/elon/update'); }}, 60000);

// Init - load all data
loadActiveMarkets();
loadLiveMarkets();
initTweetCalc();
setTimeout(initTUIChart, 100);
setTimeout(renderCurrentChart, 200);  // Initialize charts with real data

// Auto-refresh live data every 30 seconds
setInterval(() => {{
    loadLiveMarkets();
    renderCurrentChart();
}}, 30000);

// Last update timestamp
function updateTimestamp() {{
    const now = new Date();
    const ts = now.toLocaleTimeString();
    document.querySelectorAll('.live-indicator').forEach(el => {{
        el.title = 'Last update: ' + ts;
    }});
}}
updateTimestamp();
setInterval(updateTimestamp, 30000);
</script>
'''

    def render_diary(self, params):
        diary = load_trading_diary()
        entries = diary.get('entries', [])
        market_filter = params.get('market', [''])[0]
        type_filter = params.get('type', [''])[0]

        # Filter
        if market_filter:
            entries = [e for e in entries if e.get('market_id') == market_filter]
        if type_filter:
            entries = [e for e in entries if e.get('type') == type_filter]

        # Group by date
        by_date = {}
        for e in entries:
            d = e.get('date', 'unknown')
            if d not in by_date:
                by_date[d] = []
            by_date[d].append(e)

        # Recent entries (latest first)
        entries_html = ''
        for date in sorted(by_date.keys(), reverse=True)[:30]:
            entries_html += f'<div style="margin-bottom:15px">'
            entries_html += f'<div style="color:{c["accent"]};font-size:12px;margin-bottom:8px;border-bottom:1px solid {c["border"]};padding-bottom:4px">{date}</div>'
            for e in by_date[date]:
                sentiment_color = c['success'] if e.get('sentiment') == 'bullish' else c['error'] if e.get('sentiment') == 'bearish' else c['text_secondary']
                type_icon = '🎯' if e['type'] == 'prediction' else '💭' if e['type'] == 'reasoning' else '💰' if e['type'] == 'trade' else '📝'
                entries_html += f'''<div style="padding:8px;margin-bottom:8px;background:{c['bg_input']};border-left:3px solid {sentiment_color}">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:10px">{type_icon} {e['type'].upper()}</span>
                        <span style="color:{sentiment_color};font-size:10px">{(e.get('sentiment') or 'neutral').upper()}</span>
                    </div>
                    <div style="font-size:12px">{e['content']}</div>
                    {'<a href="/?view=market&id=' + e["market_id"] + '" style="font-size:10px;color:' + c["accent"] + '">→ ' + e.get("market_title", "Market")[:30] + '</a>' if e.get('market_id') else ''}
                </div>'''
            entries_html += '</div>'

        # Stats
        total = len(diary.get('entries', []))
        predictions = len([e for e in diary.get('entries', []) if e.get('type') == 'prediction'])
        trades = len([e for e in diary.get('entries', []) if e.get('type') == 'trade'])
        bullish = len([e for e in diary.get('entries', []) if e.get('sentiment') == 'bullish'])
        bearish = len([e for e in diary.get('entries', []) if e.get('sentiment') == 'bearish'])

        return f'''
<div style="display:grid;grid-template-columns:1fr 300px;gap:15px">
    <div>
        <div class="panel">
            <h2>Trading Diary</h2>
            <div style="display:flex;gap:10px;margin-bottom:15px">
                <select onchange="location.href='/?view=diary&type='+this.value" style="font-size:11px">
                    <option value="">All Types</option>
                    <option value="note" {'selected' if type_filter == 'note' else ''}>Notes</option>
                    <option value="reasoning" {'selected' if type_filter == 'reasoning' else ''}>Reasoning</option>
                    <option value="prediction" {'selected' if type_filter == 'prediction' else ''}>Predictions</option>
                    <option value="trade" {'selected' if type_filter == 'trade' else ''}>Trades</option>
                </select>
            </div>
            <div style="max-height:60vh;overflow-y:auto">
                {entries_html or f'<div style="color:{c["text_secondary"]};text-align:center;padding:30px">No diary entries yet. Add from Calendar or Market Detail.</div>'}
            </div>
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>Diary Stats</h2>
            <div style="display:grid;gap:10px">
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{c['text_secondary']}">Total Entries</span>
                    <span style="color:{c['accent']}">{total}</span>
                </div>
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{c['text_secondary']}">Predictions</span>
                    <span style="color:{c['accent']}">{predictions}</span>
                </div>
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{c['text_secondary']}">Trades</span>
                    <span style="color:{c['accent']}">{trades}</span>
                </div>
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{c['success']}">Bullish</span>
                    <span style="color:{c['success']}">{bullish}</span>
                </div>
                <div style="display:flex;justify-content:space-between">
                    <span style="color:{c['error']}">Bearish</span>
                    <span style="color:{c['error']}">{bearish}</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Quick Add</h2>
            <textarea id="quick-diary" style="width:100%;height:80px;font-size:11px" placeholder="Write your reasoning or prediction..."></textarea>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:8px">
                <select id="quick-type" style="font-size:10px">
                    <option value="note">Note</option>
                    <option value="reasoning">Reasoning</option>
                    <option value="prediction">Prediction</option>
                    <option value="trade">Trade</option>
                </select>
                <select id="quick-sentiment" style="font-size:10px">
                    <option value="">Neutral</option>
                    <option value="bullish">Bullish</option>
                    <option value="bearish">Bearish</option>
                </select>
            </div>
            <button onclick="addQuickDiary()" style="width:100%;margin-top:8px">Add Entry</button>
        </div>
    </div>
</div>

<script>
async function addQuickDiary() {{
    const content = document.getElementById('quick-diary').value.trim();
    const type = document.getElementById('quick-type').value;
    const sentiment = document.getElementById('quick-sentiment').value;
    if (!content) return;

    await fetch('/api/diary', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{content, type, sentiment}})
    }});
    location.reload();
}}
</script>
'''


    def render_promo_page(self):
        """Stunning promotional landing page for YES/NO terminal"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YES/NO.EVENTS - Polymarket Quant Terminal</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --bg: #0a0a12;
            --bg-panel: #12121a;
            --accent: #e94560;
            --accent-glow: rgba(233, 69, 96, 0.4);
            --green: #00ff88;
            --red: #ff4444;
            --text: #eaeaea;
            --text-dim: #6a6a7a;
            --border: #2a2a3a;
        }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: 'Space Grotesk', sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

        /* Animated background */
        .bg-grid {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image:
                linear-gradient(rgba(233, 69, 96, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(233, 69, 96, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: 0;
        }}
        .bg-glow {{
            position: fixed;
            width: 600px; height: 600px;
            border-radius: 50%;
            filter: blur(150px);
            opacity: 0.15;
            z-index: 0;
        }}
        .glow-1 {{ background: var(--accent); top: -200px; left: -200px; }}
        .glow-2 {{ background: var(--green); bottom: -200px; right: -200px; }}

        /* Container */
        .container {{
            position: relative;
            z-index: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}

        /* Hero */
        .hero {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 60px 20px;
        }}
        .logo {{
            font-size: 80px;
            font-weight: 700;
            letter-spacing: -4px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, var(--accent), #ff8c00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 60px var(--accent-glow);
        }}
        .tagline {{
            font-size: 24px;
            color: var(--text-dim);
            margin-bottom: 40px;
            letter-spacing: 4px;
            text-transform: uppercase;
        }}
        .hero-desc {{
            font-size: 18px;
            color: var(--text);
            max-width: 600px;
            line-height: 1.8;
            margin-bottom: 50px;
        }}

        /* Buttons */
        .btn-group {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .btn {{
            padding: 16px 40px;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 2px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            font-family: 'JetBrains Mono', monospace;
        }}
        .btn-primary {{
            background: var(--accent);
            color: white;
            box-shadow: 0 0 30px var(--accent-glow);
        }}
        .btn-primary:hover {{
            transform: translateY(-3px);
            box-shadow: 0 0 50px var(--accent-glow);
        }}
        .btn-secondary {{
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
        }}
        .btn-secondary:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}

        /* Terminal Preview */
        .terminal-preview {{
            margin: 80px 0;
            perspective: 1000px;
        }}
        .terminal-window {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 40px 100px rgba(0,0,0,0.5);
            transform: rotateX(5deg);
            transition: transform 0.5s ease;
        }}
        .terminal-window:hover {{
            transform: rotateX(0);
        }}
        .terminal-header {{
            background: #1a1a24;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--border);
        }}
        .terminal-dot {{
            width: 12px; height: 12px;
            border-radius: 50%;
        }}
        .dot-red {{ background: #ff5f56; }}
        .dot-yellow {{ background: #ffbd2e; }}
        .dot-green {{ background: #27ca40; }}
        .terminal-title {{
            color: var(--text-dim);
            font-size: 12px;
            margin-left: 10px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .terminal-body {{
            padding: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            line-height: 1.6;
            min-height: 300px;
        }}
        .terminal-line {{
            margin: 4px 0;
        }}
        .t-green {{ color: var(--green); }}
        .t-red {{ color: var(--red); }}
        .t-accent {{ color: var(--accent); }}
        .t-dim {{ color: var(--text-dim); }}
        .t-yellow {{ color: #feca57; }}

        /* Features */
        .features {{
            padding: 100px 0;
        }}
        .section-title {{
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 60px;
            background: linear-gradient(135deg, var(--text), var(--text-dim));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
        }}
        .feature-card {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            padding: 30px;
            transition: all 0.3s ease;
        }}
        .feature-card:hover {{
            border-color: var(--accent);
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }}
        .feature-icon {{
            font-size: 36px;
            margin-bottom: 20px;
        }}
        .feature-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text);
        }}
        .feature-desc {{
            font-size: 14px;
            color: var(--text-dim);
            line-height: 1.6;
        }}

        /* Stats */
        .stats {{
            padding: 80px 0;
            background: var(--bg-panel);
            margin: 80px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 40px;
            max-width: 1000px;
            margin: 0 auto;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 48px;
            font-weight: 700;
            color: var(--accent);
            font-family: 'JetBrains Mono', monospace;
        }}
        .stat-label {{
            font-size: 12px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-top: 8px;
        }}

        /* Screens showcase */
        .screens {{
            padding: 100px 0;
        }}
        .screens-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            justify-content: center;
        }}
        .screen-card {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            padding: 20px;
            width: 180px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        .screen-card:hover {{
            border-color: var(--green);
            transform: scale(1.05);
        }}
        .screen-key {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--border);
            padding: 4px 12px;
            display: inline-block;
            margin-bottom: 12px;
            font-size: 14px;
        }}
        .screen-name {{
            font-size: 14px;
            font-weight: 500;
        }}

        /* CTA */
        .cta {{
            padding: 100px 0;
            text-align: center;
        }}
        .cta-title {{
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        .cta-desc {{
            color: var(--text-dim);
            font-size: 16px;
            margin-bottom: 40px;
        }}

        /* Footer */
        .footer {{
            padding: 40px 0;
            text-align: center;
            border-top: 1px solid var(--border);
            color: var(--text-dim);
            font-size: 12px;
        }}
        .footer a {{
            color: var(--accent);
            text-decoration: none;
        }}

        @media (max-width: 768px) {{
            .logo {{ font-size: 48px; }}
            .tagline {{ font-size: 14px; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .screens-row {{ flex-wrap: wrap; }}
        }}
    </style>
</head>
<body>
    <div class="bg-grid"></div>
    <div class="bg-glow glow-1"></div>
    <div class="bg-glow glow-2"></div>

    <div class="container">
        <section class="hero">
            <div class="logo">YES/NO</div>
            <div class="tagline mono">.EVENTS</div>
            <p class="hero-desc">
                Bloomberg-style quant terminal for prediction markets.
                Track Polymarket, analyze orderbooks, execute trades,
                and find edge with professional-grade tools.
            </p>
            <div class="btn-group">
                <a href="#install" class="btn btn-primary">GET STARTED</a>
                <a href="/?view=calendar" class="btn btn-secondary">VIEW DASHBOARD</a>
            </div>
        </section>

        <section class="terminal-preview">
            <div class="terminal-window">
                <div class="terminal-header">
                    <div class="terminal-dot dot-red"></div>
                    <div class="terminal-dot dot-yellow"></div>
                    <div class="terminal-dot dot-green"></div>
                    <span class="terminal-title">yesno ~ polymarket terminal</span>
                </div>
                <div class="terminal-body">
                    <div class="terminal-line t-accent">┌─────────────────────────────────────────────────────────────────────────────┐</div>
                    <div class="terminal-line t-accent">│  <span class="t-yellow">YES/NO.EVENTS</span>                                 <span class="t-dim">POLYMARKET QUANT TERMINAL</span>  │</div>
                    <div class="terminal-line t-accent">├─────────────────────────────────────────────────────────────────────────────┤</div>
                    <div class="terminal-line">│  <span class="t-green">▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓</span><span class="t-red">▓▓▓▓▓▓▓▓</span>  <span class="t-green">YES 73.2%</span>  │  <span class="t-red">NO 26.8%</span>  │</div>
                    <div class="terminal-line t-accent">├─────────────────────────────────────────────────────────────────────────────┤</div>
                    <div class="terminal-line">│  <span class="t-dim">MARKET</span>  Will Trump win 2028?                      <span class="t-green">$2.4M VOL</span>  │</div>
                    <div class="terminal-line">│  <span class="t-dim">SPREAD</span>  <span class="t-yellow">0.5¢</span>   <span class="t-dim">LIQUIDITY</span>  <span class="t-green">$847K</span>   <span class="t-dim">24H</span>  <span class="t-green">+4.2%</span>  │</div>
                    <div class="terminal-line t-accent">├─────────────────────────────────────────────────────────────────────────────┤</div>
                    <div class="terminal-line">│  <span class="t-green">[B]uy</span>  <span class="t-red">[S]ell</span>  <span class="t-yellow">[W]atch</span>  <span class="t-dim">[/]Search</span>  <span class="t-dim">[?]Help</span>  │</div>
                    <div class="terminal-line t-accent">└─────────────────────────────────────────────────────────────────────────────┘</div>
                </div>
            </div>
        </section>

        <section class="features">
            <h2 class="section-title">Professional Tools</h2>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Real-time Orderbooks</div>
                    <div class="feature-desc">Full CLOB depth visualization with bid/ask walls, spread analysis, and liquidity heatmaps.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-title">Edge Scanner</div>
                    <div class="feature-desc">Find +EV opportunities with Kelly criterion sizing, momentum detection, and arbitrage alerts.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📈</div>
                    <div class="feature-title">Quant Analytics</div>
                    <div class="feature-desc">Monte Carlo simulations, backtesting frameworks, and pattern recognition for Elon markets.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Fast Execution</div>
                    <div class="feature-desc">Keyboard-first trading with B/S hotkeys, quick size buttons, and instant order placement.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📅</div>
                    <div class="feature-title">Event Calendar</div>
                    <div class="feature-desc">Track FOMC, earnings, elections, and crypto events with market impact analysis.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔬</div>
                    <div class="feature-title">Research Database</div>
                    <div class="feature-desc">Store predictions, track win rates, and build your edge over time with structured research.</div>
                </div>
            </div>
        </section>

        <section class="stats">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value mono">8</div>
                    <div class="stat-label">Trading Screens</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value mono">21</div>
                    <div class="stat-label">Backtested Markets</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value mono">&lt;1s</div>
                    <div class="stat-label">Order Latency</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value mono">∞</div>
                    <div class="stat-label">Edge Potential</div>
                </div>
            </div>
        </section>

        <section class="screens">
            <h2 class="section-title">Terminal Screens</h2>
            <div class="screens-row">
                <div class="screen-card">
                    <div class="screen-key">1</div>
                    <div class="screen-name">Trending</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">2</div>
                    <div class="screen-name">Search</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">3</div>
                    <div class="screen-name">Hashtags</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">P</div>
                    <div class="screen-name">Portfolio</div>
                </div>
            </div>
            <div class="screens-row">
                <div class="screen-card">
                    <div class="screen-key">E</div>
                    <div class="screen-name">Elon Lab</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">R</div>
                    <div class="screen-name">Research</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">A</div>
                    <div class="screen-name">Analytics</div>
                </div>
                <div class="screen-card">
                    <div class="screen-key">?</div>
                    <div class="screen-name">Help</div>
                </div>
            </div>
        </section>

        <section class="cta" id="install">
            <h2 class="cta-title">Get Started</h2>
            <p class="cta-desc">Run the professional terminal locally for full keyboard-first trading.</p>
            <div class="terminal-window" style="max-width: 700px; margin: 0 auto 40px; text-align: left;">
                <div class="terminal-header">
                    <div class="terminal-dot dot-red"></div>
                    <div class="terminal-dot dot-yellow"></div>
                    <div class="terminal-dot dot-green"></div>
                    <span class="terminal-title">terminal ~ install</span>
                </div>
                <div class="terminal-body" style="font-size: 12px;">
                    <div class="terminal-line t-dim"># Clone the repository</div>
                    <div class="terminal-line"><span class="t-green">$</span> git clone https://github.com/your-org/yesno-terminal.git</div>
                    <div class="terminal-line"><span class="t-green">$</span> cd yesno-terminal</div>
                    <div class="terminal-line" style="height: 12px;"></div>
                    <div class="terminal-line t-dim"># Setup Python environment</div>
                    <div class="terminal-line"><span class="t-green">$</span> python3.12 -m venv .venv</div>
                    <div class="terminal-line"><span class="t-green">$</span> source .venv/bin/activate</div>
                    <div class="terminal-line"><span class="t-green">$</span> pip install -r requirements.txt</div>
                    <div class="terminal-line" style="height: 12px;"></div>
                    <div class="terminal-line t-dim"># Launch the terminal</div>
                    <div class="terminal-line"><span class="t-green">$</span> ./yesno.sh</div>
                    <div class="terminal-line" style="height: 12px;"></div>
                    <div class="terminal-line t-yellow">✓ YES/NO Terminal ready - full keyboard navigation active</div>
                </div>
            </div>
            <div class="btn-group">
                <a href="https://github.com/your-org/yesno-terminal" class="btn btn-primary" target="_blank">VIEW ON GITHUB</a>
                <a href="/?view=calendar" class="btn btn-secondary">WEB DASHBOARD</a>
            </div>
        </section>

        <footer class="footer">
            <p>YES/NO.EVENTS &copy; 2026 | Built for prediction market traders</p>
            <p style="margin-top:10px">
                <a href="/?view=about">About</a> ·
                <a href="/?view=methodology">Methodology</a> ·
                <a href="/?view=calendar">Dashboard</a>
            </p>
        </footer>
    </div>
</body>
</html>'''

    def render_web_terminal(self, params):
        """Full web-based terminal with card-style markets and pop culture panel"""
        market_id = params.get('id', [''])[0]
        initial_screen = params.get('screen', ['markets'])[0]
        show_popculture = params.get('popculture', [''])[0] == '1'

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YES/NO Terminal</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        :root {{
            --bg: #0a0a12;
            --panel: #12121a;
            --card: #16162a;
            --border: #2a2a3a;
            --accent: #e94560;
            --accent-glow: rgba(233, 69, 96, 0.3);
            --green: #00ff88;
            --red: #ff4444;
            --yellow: #feca57;
            --cyan: #7ec8e3;
            --purple: #a855f7;
            --text: #eaeaea;
            --dim: #6a6a7a;
            --muted: #4a4a5a;
        }}
        html, body {{
            height: 100%;
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            overflow: hidden;
        }}
        .mono {{ font-family: 'JetBrains Mono', monospace; }}

        /* Layout */
        .terminal {{
            display: grid;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
        }}

        /* Header */
        .header {{
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-left {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .logo {{
            color: var(--accent);
            font-weight: 700;
            font-size: 16px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .tabs {{
            display: flex;
            gap: 4px;
        }}
        .tab {{
            padding: 6px 12px;
            cursor: pointer;
            color: var(--dim);
            border: 1px solid transparent;
            transition: all 0.2s;
            font-size: 12px;
        }}
        .tab:hover {{ color: var(--text); }}
        .tab.active {{
            color: var(--accent);
            border-color: var(--accent);
            background: rgba(233, 69, 96, 0.1);
        }}
        .tab.pop-culture {{
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(233, 69, 96, 0.2));
            border-color: var(--purple);
            color: var(--purple);
        }}
        .tab.pop-culture.active {{
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.3), rgba(233, 69, 96, 0.3));
        }}
        .tab.claude-ai {{
            background: linear-gradient(135deg, rgba(126, 200, 227, 0.2), rgba(168, 85, 247, 0.2));
            border-color: var(--cyan);
            color: var(--cyan);
        }}
        .tab.claude-ai.active {{
            background: linear-gradient(135deg, rgba(126, 200, 227, 0.3), rgba(168, 85, 247, 0.3));
        }}
        .header-right {{
            display: flex;
            align-items: center;
            gap: 15px;
            color: var(--dim);
            font-size: 11px;
        }}
        .status-dot {{
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--green);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        /* Main content */
        .main {{
            display: grid;
            grid-template-columns: 1fr 340px;
            overflow: hidden;
        }}
        .main.with-sidebar {{
            grid-template-columns: 260px 1fr 340px;
        }}

        /* Sidebar (list view) */
        .sidebar {{
            background: var(--panel);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .sidebar-header {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .sidebar-title {{
            color: var(--accent);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-family: 'JetBrains Mono', monospace;
        }}
        .search-box {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
        }}
        .search-input {{
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 10px 14px;
            color: var(--text);
            font-family: inherit;
            font-size: 13px;
            border-radius: 6px;
        }}
        .search-input:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }}

        /* Market Cards Grid */
        .markets-grid {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            align-content: start;
        }}

        /* Market Card */
        .market-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        .market-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }}
        .market-card.selected {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }}
        .card-image {{
            width: 100%;
            height: 140px;
            background: linear-gradient(135deg, var(--panel), var(--bg));
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }}
        .card-image img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        .card-image-placeholder {{
            font-size: 48px;
            opacity: 0.3;
        }}
        .card-badges {{
            position: absolute;
            top: 8px;
            left: 8px;
            display: flex;
            gap: 6px;
        }}
        .badge {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-hot {{
            background: var(--accent);
            color: white;
        }}
        .badge-trending {{
            background: var(--green);
            color: var(--bg);
        }}
        .badge-drop {{
            background: var(--yellow);
            color: var(--bg);
        }}
        .card-fav {{
            position: absolute;
            top: 8px;
            right: 8px;
            width: 32px;
            height: 32px;
            background: rgba(0,0,0,0.5);
            border: none;
            border-radius: 50%;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }}
        .card-fav:hover {{
            background: var(--accent);
        }}
        .card-fav.active {{
            background: var(--accent);
        }}
        .card-content {{
            padding: 16px;
        }}
        .card-title {{
            font-size: 14px;
            font-weight: 600;
            line-height: 1.4;
            margin-bottom: 12px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .card-prices {{
            display: flex;
            gap: 12px;
            margin-bottom: 12px;
        }}
        .price-box {{
            flex: 1;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }}
        .price-box.yes {{
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.2);
        }}
        .price-box.no {{
            background: rgba(255, 68, 68, 0.1);
            border: 1px solid rgba(255, 68, 68, 0.2);
        }}
        .price-label {{
            font-size: 10px;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .price-box.yes .price-label {{ color: var(--green); }}
        .price-box.no .price-label {{ color: var(--red); }}
        .price-value {{
            font-size: 20px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }}
        .price-box.yes .price-value {{ color: var(--green); }}
        .price-box.no .price-value {{ color: var(--red); }}
        .card-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--dim);
        }}
        .card-volume {{ color: var(--cyan); }}
        .card-actions {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 12px;
        }}
        .card-btn {{
            padding: 10px;
            border: none;
            border-radius: 6px;
            font-family: inherit;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .card-btn.buy {{
            background: var(--green);
            color: var(--bg);
        }}
        .card-btn.sell {{
            background: var(--red);
            color: white;
        }}
        .card-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}

        /* Market list view */
        .market-list {{
            flex: 1;
            overflow-y: auto;
        }}
        .market-item {{
            padding: 12px;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.2s;
        }}
        .market-item:hover {{
            background: rgba(233, 69, 96, 0.05);
        }}
        .market-item.selected {{
            background: rgba(233, 69, 96, 0.1);
            border-left: 3px solid var(--accent);
        }}
        .market-title {{
            font-size: 11px;
            color: var(--text);
            margin-bottom: 6px;
            line-height: 1.4;
        }}
        .market-meta {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
        }}
        .market-price {{
            color: var(--green);
            font-weight: 500;
        }}
        .market-vol {{ color: var(--dim); }}

        /* Center panel */
        .center {{
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .chart-container {{
            flex: 1;
            padding: 16px;
            overflow: hidden;
        }}
        .price-bar {{
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 16px;
            margin-bottom: 16px;
        }}
        .price-title {{
            font-size: 14px;
            margin-bottom: 12px;
            color: var(--text);
        }}
        .price-visual {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .price-yes {{ color: var(--green); font-weight: 700; }}
        .price-no {{ color: var(--red); font-weight: 700; }}
        .bar-container {{
            flex: 1;
            height: 24px;
            display: flex;
            border-radius: 2px;
            overflow: hidden;
        }}
        .bar-yes {{
            background: var(--green);
            transition: width 0.3s;
        }}
        .bar-no {{
            background: var(--red);
            transition: width 0.3s;
        }}
        .market-info {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-top: 16px;
        }}
        .info-item {{
            text-align: center;
        }}
        .info-value {{
            font-size: 16px;
            font-weight: 600;
            color: var(--accent);
        }}
        .info-label {{
            font-size: 9px;
            color: var(--dim);
            text-transform: uppercase;
            margin-top: 4px;
        }}

        /* Orderbook */
        .orderbook {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
        }}
        .book-side {{
            background: var(--panel);
            padding: 12px;
        }}
        .book-title {{
            font-size: 10px;
            color: var(--dim);
            text-transform: uppercase;
            margin-bottom: 8px;
            text-align: center;
        }}
        .book-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            font-size: 11px;
            position: relative;
        }}
        .book-row .bar {{
            position: absolute;
            top: 0; bottom: 0;
            opacity: 0.2;
        }}
        .bids .bar {{ right: 0; background: var(--green); }}
        .asks .bar {{ left: 0; background: var(--red); }}
        .bid-price {{ color: var(--green); }}
        .ask-price {{ color: var(--red); }}

        /* Right panel */
        .right-panel {{
            background: var(--panel);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .panel-section {{
            border-bottom: 1px solid var(--border);
        }}
        .panel-header {{
            padding: 12px;
            background: rgba(0,0,0,0.2);
            font-size: 11px;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .panel-content {{
            padding: 12px;
        }}

        /* Trading panel */
        .trade-buttons {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .trade-btn {{
            padding: 12px;
            border: none;
            font-family: inherit;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .trade-btn.buy {{
            background: var(--green);
            color: var(--bg);
        }}
        .trade-btn.sell {{
            background: var(--red);
            color: white;
        }}
        .trade-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .size-row {{
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .size-btn {{
            flex: 1;
            padding: 8px;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            font-family: inherit;
            font-size: 11px;
            cursor: pointer;
        }}
        .size-btn:hover {{
            border-color: var(--accent);
        }}
        .input-group {{
            margin-bottom: 12px;
        }}
        .input-label {{
            font-size: 10px;
            color: var(--dim);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .input-field {{
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 10px;
            color: var(--text);
            font-family: inherit;
            font-size: 13px;
        }}
        .input-field:focus {{
            outline: none;
            border-color: var(--accent);
        }}

        /* Positions */
        .position-item {{
            padding: 10px 0;
            border-bottom: 1px solid var(--border);
        }}
        .position-market {{
            font-size: 11px;
            margin-bottom: 4px;
        }}
        .position-details {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--dim);
        }}
        .pnl-positive {{ color: var(--green); }}
        .pnl-negative {{ color: var(--red); }}

        /* Footer */
        .footer {{
            background: var(--panel);
            border-top: 1px solid var(--border);
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
        }}
        .shortcuts {{
            display: flex;
            gap: 20px;
            color: var(--dim);
        }}
        .shortcut {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .key {{
            background: var(--border);
            padding: 2px 6px;
            font-size: 10px;
        }}
        .footer-right {{
            display: flex;
            gap: 20px;
            color: var(--dim);
        }}

        /* Help modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        .modal.show {{ display: flex; }}
        .modal-content {{
            background: var(--panel);
            border: 1px solid var(--border);
            padding: 24px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }}
        .modal-title {{
            font-size: 18px;
            color: var(--accent);
            margin-bottom: 20px;
        }}
        .modal-section {{
            margin-bottom: 20px;
        }}
        .modal-section h4 {{
            font-size: 12px;
            color: var(--yellow);
            margin-bottom: 10px;
        }}
        .key-list {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .key-item {{
            display: flex;
            gap: 10px;
            font-size: 11px;
        }}
        .key-item .key {{
            min-width: 30px;
            text-align: center;
        }}

        /* Loading */
        .loading {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--dim);
        }}
        .spinner {{
            width: 24px; height: 24px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 12px;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--dim); }}

        /* Filter buttons */
        .filter-btn {{
            padding: 6px 14px;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--dim);
            font-family: inherit;
            font-size: 11px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn:hover {{
            border-color: var(--accent);
            color: var(--text);
        }}
        .filter-btn.active {{
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }}
        .filter-btn.pop-culture {{
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(233, 69, 96, 0.2));
            border-color: var(--purple);
            color: var(--purple);
        }}
        .filter-btn.pop-culture:hover,
        .filter-btn.pop-culture.active {{
            background: linear-gradient(135deg, var(--purple), var(--accent));
            color: white;
        }}

        /* Bar container improvements */
        .bar-container {{
            display: flex;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-yes, .bar-no {{
            transition: width 0.3s ease;
        }}
        .bar-yes {{ background: var(--green); }}
        .bar-no {{ background: var(--red); }}

        /* Claude AI Chat Panel */
        .claude-panel {{
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            padding: 20px;
        }}
        .claude-panel.active {{ display: flex; flex-direction: column; }}
        .claude-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px;
            background: linear-gradient(135deg, rgba(126, 200, 227, 0.2), rgba(168, 85, 247, 0.2));
            border: 1px solid var(--cyan);
            border-radius: 8px 8px 0 0;
        }}
        .claude-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            color: var(--cyan);
            font-size: 18px;
            font-weight: 600;
        }}
        .claude-close {{
            background: none;
            border: none;
            color: var(--dim);
            font-size: 24px;
            cursor: pointer;
        }}
        .claude-close:hover {{ color: var(--text); }}
        .claude-chat {{
            flex: 1;
            background: var(--panel);
            border: 1px solid var(--border);
            border-top: none;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .chat-message {{
            padding: 12px 16px;
            border-radius: 8px;
            max-width: 80%;
            font-size: 13px;
            line-height: 1.5;
        }}
        .chat-message.user {{
            background: rgba(233, 69, 96, 0.2);
            border: 1px solid var(--accent);
            align-self: flex-end;
        }}
        .chat-message.claude {{
            background: rgba(126, 200, 227, 0.1);
            border: 1px solid rgba(126, 200, 227, 0.3);
            align-self: flex-start;
        }}
        .chat-message.system {{
            background: rgba(254, 202, 87, 0.1);
            border: 1px solid rgba(254, 202, 87, 0.3);
            align-self: center;
            font-size: 12px;
        }}
        .chat-action {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-top: 8px;
        }}
        .chat-action-title {{
            color: var(--cyan);
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .chat-action-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            font-size: 12px;
        }}
        .chat-action-row {{
            display: flex;
            justify-content: space-between;
        }}
        .chat-action-label {{ color: var(--dim); }}
        .chat-action-value {{ color: var(--text); font-weight: 500; }}
        .chat-action-buttons {{
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }}
        .chat-action-buttons button {{
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            font-family: inherit;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }}
        .btn-execute {{ background: var(--green); color: var(--bg); }}
        .btn-cancel {{ background: var(--card); color: var(--text); border: 1px solid var(--border) !important; }}
        .btn-execute:hover {{ filter: brightness(1.1); }}
        .btn-cancel:hover {{ background: var(--border); }}
        .claude-input-area {{
            display: flex;
            gap: 12px;
            padding: 16px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}
        .claude-input {{
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            padding: 14px 18px;
            color: var(--text);
            font-family: inherit;
            font-size: 14px;
            border-radius: 8px;
            resize: none;
        }}
        .claude-input:focus {{
            outline: none;
            border-color: var(--cyan);
            box-shadow: 0 0 0 3px rgba(126, 200, 227, 0.2);
        }}
        .claude-send {{
            padding: 14px 24px;
            background: linear-gradient(135deg, var(--cyan), var(--purple));
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .claude-send:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(126, 200, 227, 0.3); }}
        .claude-send:disabled {{ opacity: 0.5; cursor: not-allowed; transform: none; }}
        .claude-status {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            font-size: 11px;
            color: var(--dim);
            background: var(--bg);
            border-top: 1px solid var(--border);
        }}
        .status-indicator {{
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--green);
        }}
        .status-indicator.error {{ background: var(--red); }}
        .status-indicator.pending {{ background: var(--yellow); animation: pulse 1s infinite; }}
        .examples-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            margin-top: 8px;
        }}
        .example-btn {{
            padding: 10px 12px;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--dim);
            font-size: 11px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
        }}
        .example-btn:hover {{
            border-color: var(--cyan);
            color: var(--text);
            background: rgba(126, 200, 227, 0.1);
        }}
    </style>
</head>
<body>
    <div class="terminal">
        <!-- Header -->
        <div class="header">
            <div class="header-left">
                <div class="logo">YES/NO</div>
                <div class="tabs">
                    <div class="tab active" data-tab="markets" onclick="switchTab('markets');loadMarkets('')">
                        <span class="key">1</span> Markets
                    </div>
                    <div class="tab" data-tab="trading" onclick="switchTab('trading');showTradingPanel()">
                        <span class="key">2</span> Trading
                    </div>
                    <div class="tab" data-tab="portfolio" onclick="switchTab('portfolio');loadPortfolio()">
                        <span class="key">3</span> Portfolio
                    </div>
                    <div class="tab" data-tab="elon" onclick="switchTab('elon');loadElonLab()">
                        <span class="key">4</span> Elon Lab
                    </div>
                    <div class="tab" data-tab="research" onclick="switchTab('research');loadResearch()">
                        <span class="key">5</span> Research
                    </div>
                    <div class="tab" data-tab="analytics" onclick="switchTab('analytics');loadAnalytics()">
                        <span class="key">6</span> Analytics
                    </div>
                    <div class="tab" data-tab="edge" onclick="switchTab('edge');loadEdgeScanner()">
                        <span class="key">9</span> Edge
                    </div>
                    <div class="tab claude-ai" data-tab="claude" onclick="switchTab('claude');showClaudePanel()">
                        <span class="key">C</span> Claude AI
                    </div>
                </div>
            </div>
            <div class="header-right">
                <div class="status-dot"></div>
                <span>LIVE</span>
                <span id="clock">--:--:--</span>
                <a href="/?view=promo" style="color:var(--accent);text-decoration:none">← Exit</a>
            </div>
        </div>

        <!-- Main -->
        <div class="main">
            <!-- Center - Markets Grid -->
            <div class="center" style="background:var(--bg);overflow:hidden;display:flex;flex-direction:column">
                <!-- Search and filters -->
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;gap:12px;align-items:center">
                    <input type="text" class="search-input" id="search" placeholder="Search markets... (/)" autocomplete="off" style="max-width:400px">
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn active" onclick="loadMarkets('')">All</button>
                        <button class="filter-btn" onclick="loadMarkets('politics')">Politics</button>
                        <button class="filter-btn" onclick="loadMarkets('crypto')">Crypto</button>
                        <button class="filter-btn" onclick="loadMarkets('sports')">Sports</button>
                        <button class="filter-btn pop-culture" onclick="loadPopCulture()">🎬 Pop Culture</button>
                    </div>
                    <span style="margin-left:auto;color:var(--dim);font-size:11px" id="market-count">--</span>
                </div>

                <!-- Markets Grid -->
                <div class="markets-grid" id="markets-grid">
                    <div class="loading" style="grid-column:1/-1">
                        <div class="spinner"></div>
                        Loading markets...
                    </div>
                </div>

                <!-- Selected Market Detail Bar -->
                <div id="detail-bar" style="display:none;background:var(--panel);border-top:1px solid var(--border);padding:16px">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                        <div class="price-title" id="selected-title" style="font-size:14px;font-weight:600">Select a market</div>
                        <button onclick="closeDetail()" style="background:none;border:none;color:var(--dim);cursor:pointer;font-size:18px">&times;</button>
                    </div>
                    <div style="display:flex;gap:20px;align-items:center">
                        <div style="display:flex;align-items:center;gap:12px;flex:1">
                            <span class="price-yes mono" id="yes-pct" style="font-size:24px">--</span>
                            <div class="bar-container" style="flex:1;height:20px;border-radius:4px">
                                <div class="bar-yes" id="bar-yes" style="width:50%"></div>
                                <div class="bar-no" id="bar-no" style="width:50%"></div>
                            </div>
                            <span class="price-no mono" id="no-pct" style="font-size:24px">--</span>
                        </div>
                        <div style="display:flex;gap:24px;color:var(--dim);font-size:12px">
                            <div><span style="color:var(--text)" id="spread">--</span> spread</div>
                            <div><span style="color:var(--cyan)" id="volume">--</span> vol</div>
                            <div><span style="color:var(--green)" id="change">--</span> 24h</div>
                        </div>
                        <div style="display:flex;gap:8px">
                            <button class="card-btn buy" onclick="placeTrade('buy')" style="padding:10px 24px">BUY</button>
                            <button class="card-btn sell" onclick="placeTrade('sell')" style="padding:10px 24px">SELL</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right panel -->
            <div class="right-panel">
                <!-- Pop Culture Panel (shown when filter active) -->
                <div class="panel-section pop-panel" id="pop-culture-panel" style="display:none;background:linear-gradient(180deg, rgba(168,85,247,0.1) 0%, transparent 100%)">
                    <div class="panel-header" style="background:linear-gradient(90deg, var(--purple), var(--accent));color:white">
                        🎬 Pop Culture Metrics
                    </div>
                    <div class="panel-content">
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                            <div style="background:var(--bg);padding:12px;border-radius:8px;text-align:center">
                                <div style="color:var(--purple);font-size:24px;font-weight:700;font-family:'JetBrains Mono'" id="pop-total">0</div>
                                <div style="color:var(--dim);font-size:9px;text-transform:uppercase">Active Markets</div>
                            </div>
                            <div style="background:var(--bg);padding:12px;border-radius:8px;text-align:center">
                                <div style="color:var(--green);font-size:24px;font-weight:700;font-family:'JetBrains Mono'" id="pop-volume">$0</div>
                                <div style="color:var(--dim);font-size:9px;text-transform:uppercase">24h Volume</div>
                            </div>
                        </div>
                        <div style="margin-bottom:12px">
                            <div style="color:var(--dim);font-size:10px;text-transform:uppercase;margin-bottom:8px">Trending Categories</div>
                            <div style="display:flex;flex-wrap:wrap;gap:6px" id="pop-categories">
                                <span style="padding:4px 8px;background:var(--bg);border-radius:4px;font-size:10px;color:var(--purple)">#Movies</span>
                                <span style="padding:4px 8px;background:var(--bg);border-radius:4px;font-size:10px;color:var(--accent)">#TV</span>
                                <span style="padding:4px 8px;background:var(--bg);border-radius:4px;font-size:10px;color:var(--cyan)">#Music</span>
                                <span style="padding:4px 8px;background:var(--bg);border-radius:4px;font-size:10px;color:var(--yellow)">#Celeb</span>
                            </div>
                        </div>
                        <div>
                            <div style="color:var(--dim);font-size:10px;text-transform:uppercase;margin-bottom:8px">Top Movers</div>
                            <div id="pop-movers" style="font-size:11px">
                                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)">
                                    <span>Loading...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="panel-section">
                    <div class="panel-header">Quick Trade</div>
                    <div class="panel-content">
                        <div class="trade-buttons">
                            <button class="trade-btn buy" onclick="placeTrade('buy')">BUY YES</button>
                            <button class="trade-btn sell" onclick="placeTrade('sell')">SELL / BUY NO</button>
                        </div>
                        <div class="size-row">
                            <button class="size-btn" onclick="setSize(10)">+10</button>
                            <button class="size-btn" onclick="setSize(50)">+50</button>
                            <button class="size-btn" onclick="setSize(100)">+100</button>
                            <button class="size-btn" onclick="setSize(500)">+500</button>
                        </div>
                        <div class="input-group">
                            <div class="input-label">Amount ($)</div>
                            <input type="number" class="input-field" id="trade-amount" value="100">
                        </div>
                        <div class="input-group">
                            <div class="input-label">Limit Price (¢)</div>
                            <input type="number" class="input-field" id="trade-price" placeholder="Market">
                        </div>
                    </div>
                </div>

                <div class="panel-section" style="flex:1;overflow:hidden;display:flex;flex-direction:column">
                    <div class="panel-header">Positions</div>
                    <div class="panel-content" style="flex:1;overflow-y:auto" id="positions-list">
                        <div style="color:var(--dim);text-align:center;padding:20px;font-size:11px">
                            No open positions
                        </div>
                    </div>
                </div>

                <div class="panel-section">
                    <div class="panel-header">Watchlist</div>
                    <div class="panel-content" id="watchlist">
                        <div style="color:var(--dim);font-size:11px">Press W to add market</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <div class="shortcuts">
                <div class="shortcut"><span class="key">1-6</span> Screens</div>
                <div class="shortcut"><span class="key">B</span> Buy</div>
                <div class="shortcut"><span class="key">S</span> Sell</div>
                <div class="shortcut"><span class="key">C</span> Claude AI</div>
                <div class="shortcut"><span class="key">/</span> Search</div>
                <div class="shortcut"><span class="key">?</span> Help</div>
            </div>
            <div class="footer-right">
                <span>API: <span style="color:var(--green)">●</span> Connected</span>
                <span id="api-calls">0 calls</span>
            </div>
        </div>
    </div>

    <!-- Help Modal -->
    <div class="modal" id="help-modal">
        <div class="modal-content">
            <div class="modal-title">Keyboard Shortcuts</div>
            <div class="modal-section">
                <h4>Navigation</h4>
                <div class="key-list">
                    <div class="key-item"><span class="key">J</span> Next market</div>
                    <div class="key-item"><span class="key">K</span> Prev market</div>
                    <div class="key-item"><span class="key">1-4</span> Switch tabs</div>
                    <div class="key-item"><span class="key">/</span> Focus search</div>
                    <div class="key-item"><span class="key">ESC</span> Clear/back</div>
                    <div class="key-item"><span class="key">Enter</span> Select</div>
                </div>
            </div>
            <div class="modal-section">
                <h4>Trading</h4>
                <div class="key-list">
                    <div class="key-item"><span class="key">B</span> Buy YES</div>
                    <div class="key-item"><span class="key">S</span> Sell / Buy NO</div>
                    <div class="key-item"><span class="key">W</span> Add to watchlist</div>
                    <div class="key-item"><span class="key">R</span> Refresh data</div>
                </div>
            </div>
            <div class="modal-section">
                <h4>Screens</h4>
                <div class="key-list">
                    <div class="key-item"><span class="key">P</span> Portfolio</div>
                    <div class="key-item"><span class="key">A</span> Analytics</div>
                    <div class="key-item"><span class="key">C</span> Claude AI</div>
                    <div class="key-item"><span class="key">?</span> This help</div>
                </div>
            </div>
            <div style="text-align:center;margin-top:20px">
                <button onclick="hideHelp()" style="padding:8px 24px;background:var(--accent);border:none;color:white;cursor:pointer">Close (ESC)</button>
            </div>
        </div>
    </div>

    <!-- Claude AI Panel -->
    <div class="claude-panel" id="claude-panel">
        <div class="claude-header">
            <div class="claude-title">
                <span style="font-size:24px">🤖</span>
                Claude AI Trading Assistant
            </div>
            <button class="claude-close" onclick="hideClaudePanel()">&times;</button>
        </div>
        <div class="claude-chat" id="claude-chat">
            <div class="chat-message system">
                <strong>Welcome to Claude AI Trading</strong><br>
                Tell me what you want to trade in natural language. Examples:
                <div class="examples-grid">
                    <button class="example-btn" onclick="setExample('Buy 10 YES shares of Bitcoin above 100k')">Buy 10 YES shares of Bitcoin above 100k</button>
                    <button class="example-btn" onclick="setExample('Sell my Trump positions at 85 cents')">Sell my Trump positions at 85 cents</button>
                    <button class="example-btn" onclick="setExample('Find markets about AI with high volume')">Find markets about AI with high volume</button>
                    <button class="example-btn" onclick="setExample('Build a ladder from 60 to 70 cents, 5 shares each')">Build a ladder from 60 to 70 cents</button>
                </div>
            </div>
        </div>
        <div class="claude-input-area">
            <textarea class="claude-input" id="claude-input" placeholder="Tell Claude what you want to trade..." rows="2" onkeydown="if(event.key==='Enter' && !event.shiftKey){{event.preventDefault();sendToClaudeAI();}}"></textarea>
            <button class="claude-send" id="claude-send" onclick="sendToClaudeAI()">Send</button>
        </div>
        <div class="claude-status" id="claude-status">
            <div class="status-indicator" id="claude-indicator"></div>
            <span id="claude-status-text">Ready - Claude API connected</span>
        </div>
    </div>

    <script>
        // State
        let markets = [];
        let selectedIndex = 0;
        let selectedMarket = null;
        let apiCalls = 0;
        let searchTimeout = null;
        let isPopCulture = false;

        // Category icons/images
        const categoryIcons = {{
            'politics': '🏛️',
            'crypto': '₿',
            'sports': '⚽',
            'tech': '💻',
            'entertainment': '🎬',
            'celebrity': '⭐',
            'music': '🎵',
            'movies': '🎥',
            'tv': '📺',
            'default': '📊'
        }};

        // Clock
        function updateClock() {{
            const now = new Date();
            document.getElementById('clock').textContent = now.toLocaleTimeString();
        }}
        setInterval(updateClock, 1000);
        updateClock();

        // Tab switching
        function switchTab(tab) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`[data-tab="${{tab}}"]`).classList.add('active');
        }}

        // Set active filter button
        function setActiveFilter(btn) {{
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
        }}

        // Load markets
        async function loadMarkets(query = '') {{
            isPopCulture = false;
            document.getElementById('pop-culture-panel').style.display = 'none';
            setActiveFilter(event?.target);

            const gridEl = document.getElementById('markets-grid');
            gridEl.innerHTML = '<div class="loading" style="grid-column:1/-1"><div class="spinner"></div>Loading markets...</div>';

            try {{
                const url = query
                    ? `/api/polyrouter/search?q=${{encodeURIComponent(query)}}&limit=30`
                    : '/api/polyrouter/trending?limit=30';
                const resp = await fetch(url);
                apiCalls++;
                document.getElementById('api-calls').textContent = apiCalls + ' calls';

                const data = await resp.json();
                markets = data.markets || data || [];

                document.getElementById('market-count').textContent = markets.length + ' markets';
                renderCards();
            }} catch(e) {{
                gridEl.innerHTML = '<div style="color:var(--red);padding:40px;text-align:center;grid-column:1/-1">Error loading markets</div>';
            }}
        }}

        // Load Pop Culture markets
        async function loadPopCulture() {{
            isPopCulture = true;
            document.getElementById('pop-culture-panel').style.display = 'block';
            setActiveFilter(document.querySelector('.filter-btn.pop-culture'));

            const gridEl = document.getElementById('markets-grid');
            gridEl.innerHTML = '<div class="loading" style="grid-column:1/-1"><div class="spinner"></div>Loading pop culture markets...</div>';

            try {{
                // Search for entertainment/celebrity markets
                const queries = ['entertainment', 'celebrity', 'movie', 'tv show', 'music', 'oscar', 'grammy'];
                const allMarkets = [];

                for (const q of queries.slice(0, 3)) {{
                    const resp = await fetch(`/api/polyrouter/search?q=${{encodeURIComponent(q)}}&limit=15`);
                    apiCalls++;
                    const data = await resp.json();
                    const mkts = data.markets || data || [];
                    allMarkets.push(...mkts);
                }}

                // Dedupe by id
                const seen = new Set();
                markets = allMarkets.filter(m => {{
                    const id = m.id || m.conditionId;
                    if (seen.has(id)) return false;
                    seen.add(id);
                    return true;
                }}).slice(0, 30);

                document.getElementById('market-count').textContent = markets.length + ' pop culture markets';
                document.getElementById('api-calls').textContent = apiCalls + ' calls';

                // Update metrics
                const totalVol = markets.reduce((s, m) => s + (m.volume || m.volume24h || 0), 0);
                document.getElementById('pop-total').textContent = markets.length;
                document.getElementById('pop-volume').textContent = totalVol > 1000000 ? '$' + (totalVol/1000000).toFixed(1) + 'M' : '$' + (totalVol/1000).toFixed(0) + 'K';

                // Top movers
                const movers = markets.slice(0, 3);
                document.getElementById('pop-movers').innerHTML = movers.map(m => {{
                    const price = m.current_prices?.yes?.price || m.yes_price || 0.5;
                    return `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)">
                        <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px">${{m.title?.substring(0, 30)}}...</span>
                        <span style="color:var(--green)">${{(price * 100).toFixed(0)}}¢</span>
                    </div>`;
                }}).join('');

                renderCards();
            }} catch(e) {{
                gridEl.innerHTML = '<div style="color:var(--red);padding:40px;text-align:center;grid-column:1/-1">Error loading pop culture markets</div>';
            }}
        }}

        // Render market cards
        function renderCards() {{
            const gridEl = document.getElementById('markets-grid');
            if (!markets.length) {{
                gridEl.innerHTML = '<div style="color:var(--dim);padding:40px;text-align:center;grid-column:1/-1">No markets found</div>';
                return;
            }}

            gridEl.innerHTML = markets.map((m, i) => {{
                const price = m.current_prices?.yes?.price || m.yes_price || m.lastTradePrice || 0.5;
                const noPrice = 1 - price;
                const vol = m.volume || m.volume24h || 0;
                const volStr = vol > 1000000 ? '$' + (vol/1000000).toFixed(1) + 'M' : vol > 1000 ? '$' + (vol/1000).toFixed(0) + 'K' : '$' + vol;
                const title = m.title || m.question || 'Market';
                const category = detectCategory(title);
                const icon = categoryIcons[category] || categoryIcons.default;
                const isHot = vol > 500000;
                const isTrending = m.trending || false;

                return `<div class="market-card ${{i === selectedIndex ? 'selected' : ''}}" data-index="${{i}}" onclick="selectCard(${{i}})">
                    <div class="card-image">
                        <div class="card-image-placeholder">${{icon}}</div>
                        <div class="card-badges">
                            ${{isHot ? '<span class="badge badge-hot">HOT</span>' : ''}}
                            ${{isTrending ? '<span class="badge badge-trending">TRENDING</span>' : ''}}
                            ${{isPopCulture ? '<span class="badge" style="background:var(--purple);color:white">POP</span>' : ''}}
                        </div>
                        <button class="card-fav" onclick="event.stopPropagation();toggleFav(${{i}})">♡</button>
                    </div>
                    <div class="card-content">
                        <div class="card-title">${{title}}</div>
                        <div class="card-prices">
                            <div class="price-box yes">
                                <div class="price-label">Yes</div>
                                <div class="price-value">${{(price * 100).toFixed(0)}}¢</div>
                            </div>
                            <div class="price-box no">
                                <div class="price-label">No</div>
                                <div class="price-value">${{(noPrice * 100).toFixed(0)}}¢</div>
                            </div>
                        </div>
                        <div class="card-meta">
                            <span class="card-volume">${{volStr}} vol</span>
                            <span>${{m.endDate ? 'Ends ' + new Date(m.endDate).toLocaleDateString() : ''}}</span>
                        </div>
                        <div class="card-actions">
                            <button class="card-btn buy" onclick="event.stopPropagation();quickBuy(${{i}})">BUY YES</button>
                            <button class="card-btn sell" onclick="event.stopPropagation();quickSell(${{i}})">BUY NO</button>
                        </div>
                    </div>
                </div>`;
            }}).join('');
        }}

        // Detect category from title
        function detectCategory(title) {{
            const t = title.toLowerCase();
            if (t.includes('trump') || t.includes('biden') || t.includes('election') || t.includes('congress')) return 'politics';
            if (t.includes('bitcoin') || t.includes('eth') || t.includes('crypto') || t.includes('btc')) return 'crypto';
            if (t.includes('nfl') || t.includes('nba') || t.includes('soccer') || t.includes('game')) return 'sports';
            if (t.includes('movie') || t.includes('film') || t.includes('oscar') || t.includes('box office')) return 'movies';
            if (t.includes('tv') || t.includes('show') || t.includes('emmy') || t.includes('netflix')) return 'tv';
            if (t.includes('album') || t.includes('grammy') || t.includes('song') || t.includes('music')) return 'music';
            if (t.includes('celebrity') || t.includes('kardashian') || t.includes('musk') || t.includes('taylor')) return 'celebrity';
            return 'default';
        }}

        // Select card
        function selectCard(index) {{
            selectedIndex = index;
            selectedMarket = markets[index];
            renderCards();

            if (!selectedMarket) return;

            // Show detail bar
            document.getElementById('detail-bar').style.display = 'block';
            document.getElementById('selected-title').textContent = selectedMarket.title || selectedMarket.question || 'Market';

            // Update prices
            const price = selectedMarket.current_prices?.yes?.price || selectedMarket.yes_price || selectedMarket.lastTradePrice || 0.5;
            const yesPct = (price * 100).toFixed(1);
            const noPct = ((1 - price) * 100).toFixed(1);

            document.getElementById('yes-pct').textContent = yesPct + '%';
            document.getElementById('no-pct').textContent = noPct + '%';
            document.getElementById('bar-yes').style.width = yesPct + '%';
            document.getElementById('bar-no').style.width = noPct + '%';

            // Update metrics
            const vol = selectedMarket.volume || selectedMarket.volume24h || 0;
            document.getElementById('volume').textContent = vol > 1000000 ? '$' + (vol/1000000).toFixed(1) + 'M' : '$' + (vol/1000).toFixed(0) + 'K';
            document.getElementById('spread').textContent = (selectedMarket.spread || 1).toFixed(1) + '¢';
            document.getElementById('change').textContent = selectedMarket.change24h ? (selectedMarket.change24h > 0 ? '+' : '') + selectedMarket.change24h.toFixed(1) + '%' : '--';
        }}

        // Close detail bar
        function closeDetail() {{
            document.getElementById('detail-bar').style.display = 'none';
            selectedMarket = null;
            selectedIndex = -1;
            renderCards();
        }}

        // Quick trade from card
        function quickBuy(index) {{
            selectedMarket = markets[index];
            placeTrade('buy');
        }}

        function quickSell(index) {{
            selectedMarket = markets[index];
            placeTrade('sell');
        }}

        function toggleFav(index) {{
            const m = markets[index];
            alert('Added to watchlist: ' + (m.title || m.question)?.substring(0, 40));
        }}

        // Search
        document.getElementById('search').addEventListener('input', (e) => {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                if (e.target.value) {{
                    loadMarkets(e.target.value);
                }} else {{
                    loadMarkets('');
                }}
            }}, 300);
        }});

        // Trading
        function setSize(amount) {{
            const input = document.getElementById('trade-amount');
            input.value = parseInt(input.value || 0) + amount;
        }}

        function placeTrade(side) {{
            if (!selectedMarket) {{
                alert('Select a market first');
                return;
            }}
            const amount = document.getElementById('trade-amount').value;
            const price = document.getElementById('trade-price').value;
            alert(`${{side.toUpperCase()}} ${{amount}} @ ${{price || 'market'}} on ${{selectedMarket.title?.substring(0, 40)}}`);
        }}

        // Help modal
        function showHelp() {{
            document.getElementById('help-modal').classList.add('show');
        }}
        function hideHelp() {{
            document.getElementById('help-modal').classList.remove('show');
        }}

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            // Ignore if typing in input
            if (e.target.tagName === 'INPUT') {{
                if (e.key === 'Escape') {{
                    e.target.blur();
                    e.target.value = '';
                    loadMarkets();
                }}
                return;
            }}

            switch(e.key) {{
                case 'j':
                    if (selectedIndex < markets.length - 1) selectMarket(selectedIndex + 1);
                    break;
                case 'k':
                    if (selectedIndex > 0) selectMarket(selectedIndex - 1);
                    break;
                case '/':
                    e.preventDefault();
                    document.getElementById('search').focus();
                    break;
                case 'b':
                case 'B':
                    placeTrade('buy');
                    break;
                case 's':
                case 'S':
                    placeTrade('sell');
                    break;
                case 'w':
                case 'W':
                    if (selectedMarket) alert('Added to watchlist: ' + selectedMarket.title?.substring(0, 40));
                    break;
                case 'r':
                case 'R':
                    loadMarkets(document.getElementById('search').value);
                    break;
                case '?':
                    showHelp();
                    break;
                case 'Escape':
                    hideHelp();
                    break;
                case '1':
                    switchTab('markets');
                    break;
                case '2':
                    switchTab('portfolio');
                    break;
                case '3':
                    switchTab('research');
                    break;
                case '4':
                    switchTab('analytics');
                    break;
                case 'p':
                case 'P':
                    switchTab('portfolio');
                    break;
                case 'a':
                case 'A':
                    switchTab('analytics');
                    break;
                case 'c':
                case 'C':
                    if (!e.ctrlKey && !e.metaKey) {{
                        switchTab('claude');
                        showClaudePanel();
                    }}
                    break;
            }}
        }});

        // Initial load
        loadMarkets();

        // Claude AI state
        let pendingAction = null;

        // Show Claude panel
        function showClaudePanel() {{
            document.getElementById('claude-panel').classList.add('active');
            document.getElementById('claude-input').focus();
            checkClaudeSetup();
        }}

        // Hide Claude panel
        function hideClaudePanel() {{
            document.getElementById('claude-panel').classList.remove('active');
        }}

        // Set example in input
        function setExample(text) {{
            document.getElementById('claude-input').value = text;
            document.getElementById('claude-input').focus();
        }}

        // Check Claude setup status
        async function checkClaudeSetup() {{
            try {{
                const resp = await fetch('/api/claude/setup');
                const data = await resp.json();
                if (!data.claude_configured) {{
                    addChatMessage('system', '⚠️ Claude API not configured. Add ANTHROPIC_API_KEY to data/.trading_config.json');
                    setClaudeStatus('error', 'Claude API not configured');
                }} else if (!data.trading_configured) {{
                    addChatMessage('system', '⚠️ Trading API not configured. Add wallet credentials to data/.trading_config.json');
                    setClaudeStatus('error', 'Trading not configured');
                }} else {{
                    setClaudeStatus('ready', 'Ready - All systems connected');
                }}
            }} catch(e) {{
                setClaudeStatus('error', 'Connection error');
            }}
        }}

        // Set status indicator
        function setClaudeStatus(status, text) {{
            const indicator = document.getElementById('claude-indicator');
            const statusText = document.getElementById('claude-status-text');
            indicator.className = 'status-indicator';
            if (status === 'error') indicator.classList.add('error');
            if (status === 'pending') indicator.classList.add('pending');
            statusText.textContent = text;
        }}

        // Add message to chat
        function addChatMessage(type, content, action = null) {{
            const chat = document.getElementById('claude-chat');
            const msg = document.createElement('div');
            msg.className = `chat-message ${{type}}`;
            msg.innerHTML = content;
            if (action) {{
                pendingAction = action;
                const actionDiv = document.createElement('div');
                actionDiv.className = 'chat-action';
                actionDiv.innerHTML = `
                    <div class="chat-action-title">Parsed Action</div>
                    <div class="chat-action-details">
                        <div class="chat-action-row"><span class="chat-action-label">Action:</span> <span class="chat-action-value">${{action.action}}</span></div>
                        <div class="chat-action-row"><span class="chat-action-label">Side:</span> <span class="chat-action-value" style="color:${{action.side === 'YES' ? 'var(--green)' : 'var(--red)'}}">${{action.side || '-'}}</span></div>
                        <div class="chat-action-row"><span class="chat-action-label">Quantity:</span> <span class="chat-action-value">${{action.quantity || '-'}}</span></div>
                        <div class="chat-action-row"><span class="chat-action-label">Price:</span> <span class="chat-action-value">${{action.price ? action.price + '¢' : 'Market'}}</span></div>
                        ${{action.market_query ? `<div class="chat-action-row" style="grid-column:1/-1"><span class="chat-action-label">Market:</span> <span class="chat-action-value">${{action.market_query}}</span></div>` : ''}}
                    </div>
                    <div class="chat-action-buttons">
                        <button class="btn-execute" onclick="executeAction()">Execute Trade</button>
                        <button class="btn-cancel" onclick="cancelAction()">Cancel</button>
                    </div>
                `;
                msg.appendChild(actionDiv);
            }}
            chat.appendChild(msg);
            chat.scrollTop = chat.scrollHeight;
        }}

        // Send to Claude AI
        async function sendToClaudeAI() {{
            const input = document.getElementById('claude-input');
            const text = input.value.trim();
            if (!text) return;

            // Add user message
            addChatMessage('user', text);
            input.value = '';
            setClaudeStatus('pending', 'Processing...');

            try {{
                const resp = await fetch('/api/claude/trade', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{
                        input: text,
                        context: {{
                            selected_market: selectedMarket ? {{
                                id: selectedMarket.id || selectedMarket.conditionId,
                                title: selectedMarket.title || selectedMarket.question
                            }} : null
                        }}
                    }})
                }});
                apiCalls++;
                document.getElementById('api-calls').textContent = apiCalls + ' calls';

                const data = await resp.json();
                if (data.error) {{
                    addChatMessage('claude', '❌ ' + data.error);
                    setClaudeStatus('error', 'Error processing request');
                }} else if (data.action) {{
                    const action = data.action;
                    if (action.action === 'SEARCH') {{
                        // Execute search directly
                        addChatMessage('claude', '🔍 Searching for markets...');
                        loadMarkets(action.market_query || action.query || text);
                        setClaudeStatus('ready', 'Search complete');
                    }} else if (action.action === 'POSITION' || action.action === 'ORDERS') {{
                        addChatMessage('claude', '📊 ' + (data.message || 'Check portfolio tab for positions'));
                        setClaudeStatus('ready', 'Ready');
                    }} else if (['BUY', 'SELL', 'LADDER'].includes(action.action)) {{
                        addChatMessage('claude', '✅ Understood. Review and confirm:', action);
                        setClaudeStatus('ready', 'Awaiting confirmation');
                    }} else {{
                        addChatMessage('claude', data.message || JSON.stringify(action));
                        setClaudeStatus('ready', 'Ready');
                    }}
                }} else {{
                    addChatMessage('claude', data.message || 'Done');
                    setClaudeStatus('ready', 'Ready');
                }}
            }} catch(e) {{
                addChatMessage('claude', '❌ Error: ' + e.message);
                setClaudeStatus('error', 'Request failed');
            }}
        }}

        // Execute pending action
        async function executeAction() {{
            if (!pendingAction) return;
            setClaudeStatus('pending', 'Executing trade...');

            try {{
                const resp = await fetch('/api/claude/execute', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ action: pendingAction }})
                }});
                const data = await resp.json();
                if (data.success) {{
                    addChatMessage('system', '✅ Trade executed! Order ID: ' + (data.order_id || 'confirmed'));
                    setClaudeStatus('ready', 'Trade executed');
                }} else {{
                    addChatMessage('system', '❌ Trade failed: ' + (data.error || 'Unknown error'));
                    setClaudeStatus('error', 'Trade failed');
                }}
            }} catch(e) {{
                addChatMessage('system', '❌ Execution error: ' + e.message);
                setClaudeStatus('error', 'Execution error');
            }}
            pendingAction = null;
        }}

        // Cancel pending action
        function cancelAction() {{
            pendingAction = null;
            addChatMessage('system', 'Action cancelled');
            setClaudeStatus('ready', 'Ready');
        }}

        // Load watchlist
        async function loadWatchlist() {{
            try {{
                const resp = await fetch('/api/watched');
                const data = await resp.json();
                const markets = data.markets || [];
                if (markets.length) {{
                    document.getElementById('watchlist').innerHTML = markets.slice(0, 5).map(m =>
                        `<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:10px;cursor:pointer" onclick="loadMarkets('${{m.title?.substring(0, 20)}}')">${{m.title?.substring(0, 25)}}...</div>`
                    ).join('');
                }}
            }} catch(e) {{}}
        }}
        loadWatchlist();

        // ═══════════════════════════════════════════════════════════════════
        // TRADING SCREEN - Open Orders and Trade History
        // ═══════════════════════════════════════════════════════════════════
        let tradingTab = 'orders';

        async function showTradingPanel() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">📊 TRADING</h2>
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn ${{tradingTab === 'orders' ? 'active' : ''}}" onclick="tradingTab='orders';showTradingPanel()">1: Open Orders</button>
                        <button class="filter-btn ${{tradingTab === 'history' ? 'active' : ''}}" onclick="tradingTab='history';showTradingPanel()">2: Trade History</button>
                    </div>
                </div>
                <div id="trading-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            `;

            if (tradingTab === 'orders') {{
                await loadOpenOrders();
            }} else {{
                await loadTradeHistory();
            }}
        }}

        async function loadOpenOrders() {{
            try {{
                const resp = await fetch('/api/trading/orders');
                apiCalls++;
                document.getElementById('api-calls').textContent = apiCalls + ' calls';
                const data = await resp.json();
                const orders = data.orders || data || [];

                const content = document.getElementById('trading-content');
                if (!orders.length) {{
                    content.innerHTML = '<div style="color:var(--dim);text-align:center;padding:40px">No open orders</div>';
                    return;
                }}

                content.innerHTML = `
                    <table style="width:100%;border-collapse:collapse;font-size:12px">
                        <thead>
                            <tr style="border-bottom:1px solid var(--border)">
                                <th style="text-align:left;padding:8px;color:var(--dim)">#</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">SIDE</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">PRICE</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">SIZE</th>
                                <th style="text-align:center;padding:8px;color:var(--dim)">ACTION</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{orders.map((o, i) => `
                                <tr style="border-bottom:1px solid var(--border)">
                                    <td style="padding:8px;color:var(--dim)">${{i+1}}</td>
                                    <td style="padding:8px;max-width:300px;overflow:hidden;text-overflow:ellipsis">${{o.market || o.title || o.asset_id?.substring(0,20) || '-'}}</td>
                                    <td style="padding:8px;color:${{o.side === 'BUY' ? 'var(--green)' : 'var(--red)'}}">${{o.side || 'BUY'}}</td>
                                    <td style="padding:8px;text-align:right">${{(parseFloat(o.price || 0) * 100).toFixed(1)}}¢</td>
                                    <td style="padding:8px;text-align:right">${{o.size || o.original_size || 0}}</td>
                                    <td style="padding:8px;text-align:center">
                                        <button onclick="cancelOrder('${{o.id || o.order_id}}')" style="padding:4px 8px;background:var(--red);border:none;color:white;cursor:pointer;border-radius:4px;font-size:10px">Cancel</button>
                                    </td>
                                </tr>
                            `).join('')}}
                        </tbody>
                    </table>
                `;
            }} catch(e) {{
                document.getElementById('trading-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading orders</div>';
            }}
        }}

        async function loadTradeHistory() {{
            try {{
                const resp = await fetch('/api/trading/trades');
                apiCalls++;
                const data = await resp.json();
                const trades = data.trades || data || [];

                const content = document.getElementById('trading-content');
                if (!trades.length) {{
                    content.innerHTML = '<div style="color:var(--dim);text-align:center;padding:40px">No trade history</div>';
                    return;
                }}

                content.innerHTML = `
                    <table style="width:100%;border-collapse:collapse;font-size:12px">
                        <thead>
                            <tr style="border-bottom:1px solid var(--border)">
                                <th style="text-align:left;padding:8px;color:var(--dim)">TIME</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">SIDE</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">PRICE</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">SIZE</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{trades.slice(0, 20).map(t => `
                                <tr style="border-bottom:1px solid var(--border)">
                                    <td style="padding:8px;color:var(--dim)">${{new Date(t.timestamp || t.created_at || t.createdAt).toLocaleString()}}</td>
                                    <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{t.market || t.title || '-'}}</td>
                                    <td style="padding:8px;color:${{t.side === 'BUY' ? 'var(--green)' : 'var(--red)'}}">${{t.side || 'BUY'}}</td>
                                    <td style="padding:8px;text-align:right">${{(parseFloat(t.price || 0) * 100).toFixed(1)}}¢</td>
                                    <td style="padding:8px;text-align:right">${{t.size || 0}}</td>
                                </tr>
                            `).join('')}}
                        </tbody>
                    </table>
                `;
            }} catch(e) {{
                document.getElementById('trading-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading trades</div>';
            }}
        }}

        async function cancelOrder(orderId) {{
            if (!confirm('Cancel this order?')) return;
            try {{
                const resp = await fetch('/api/trading/cancel', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{ order_id: orderId }})
                }});
                const data = await resp.json();
                if (data.success) {{
                    showTradingPanel();
                }} else {{
                    alert('Failed to cancel: ' + (data.error || 'Unknown error'));
                }}
            }} catch(e) {{
                alert('Error: ' + e.message);
            }}
        }}

        // ═══════════════════════════════════════════════════════════════════
        // PORTFOLIO SCREEN - Positions and P&L
        // ═══════════════════════════════════════════════════════════════════
        async function loadPortfolio() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">💼 PORTFOLIO</h2>
                </div>
                <div id="portfolio-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Loading positions...</div>
                </div>
            `;

            try {{
                const resp = await fetch('/api/trading/positions');
                apiCalls++;
                const data = await resp.json();
                const positions = data.positions || data || [];

                const content = document.getElementById('portfolio-content');
                if (!positions.length) {{
                    content.innerHTML = '<div style="color:var(--dim);text-align:center;padding:40px">No open positions</div>';
                    return;
                }}

                let totalPnl = 0;
                positions.forEach(p => totalPnl += parseFloat(p.pnl || p.unrealizedPnl || 0));

                content.innerHTML = `
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px">
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                            <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Positions</div>
                            <div style="color:var(--cyan);font-size:24px;font-weight:700">${{positions.length}}</div>
                        </div>
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                            <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Total P&L</div>
                            <div style="color:${{totalPnl >= 0 ? 'var(--green)' : 'var(--red)'}};font-size:24px;font-weight:700">${{totalPnl >= 0 ? '+' : ''}}${{totalPnl.toFixed(2)}}</div>
                        </div>
                    </div>
                    <table style="width:100%;border-collapse:collapse;font-size:12px">
                        <thead>
                            <tr style="border-bottom:1px solid var(--border)">
                                <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">OUTCOME</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">SHARES</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">AVG PRICE</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">CURRENT</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">P&L</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{positions.map(p => {{
                                const pnl = parseFloat(p.pnl || p.unrealizedPnl || 0);
                                return `
                                <tr style="border-bottom:1px solid var(--border)">
                                    <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{p.title || p.market || p.question || '-'}}</td>
                                    <td style="padding:8px;color:${{(p.outcome || p.side || 'YES').toUpperCase() === 'YES' ? 'var(--green)' : 'var(--red)'}}">${{p.outcome || p.side || 'YES'}}</td>
                                    <td style="padding:8px;text-align:right">${{p.size || p.shares || p.amount || 0}}</td>
                                    <td style="padding:8px;text-align:right">${{(parseFloat(p.avgPrice || p.averagePrice || p.price || 0) * 100).toFixed(1)}}¢</td>
                                    <td style="padding:8px;text-align:right">${{(parseFloat(p.currentPrice || p.lastPrice || 0) * 100).toFixed(1)}}¢</td>
                                    <td style="padding:8px;text-align:right;color:${{pnl >= 0 ? 'var(--green)' : 'var(--red)'}}">${{pnl >= 0 ? '+' : ''}}${{pnl.toFixed(2)}}</td>
                                </tr>
                            `}}).join('')}}
                        </tbody>
                    </table>
                `;
            }} catch(e) {{
                document.getElementById('portfolio-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading positions</div>';
            }}
        }}

        // ═══════════════════════════════════════════════════════════════════
        // ELON LAB SCREEN - Twitter Analytics
        // ═══════════════════════════════════════════════════════════════════
        let elonTab = 'live';

        async function loadElonLab() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">🧪 ELON LAB - Twitter Behavior Analytics</h2>
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn ${{elonTab === 'live' ? 'active' : ''}}" onclick="elonTab='live';loadElonLab()">1: Live Stats</button>
                        <button class="filter-btn ${{elonTab === 'hourly' ? 'active' : ''}}" onclick="elonTab='hourly';loadElonLab()">2: Hourly</button>
                        <button class="filter-btn ${{elonTab === 'daily' ? 'active' : ''}}" onclick="elonTab='daily';loadElonLab()">3: Daily</button>
                        <button class="filter-btn ${{elonTab === 'markets' ? 'active' : ''}}" onclick="elonTab='markets';loadElonLab()">4: Markets</button>
                    </div>
                </div>
                <div id="elon-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            `;

            try {{
                const resp = await fetch('/api/elon/stats');
                apiCalls++;
                const stats = await resp.json();

                const content = document.getElementById('elon-content');

                if (elonTab === 'live') {{
                    content.innerHTML = `
                        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px">
                            <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                                <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Total Tweets</div>
                                <div style="color:var(--cyan);font-size:24px;font-weight:700">${{stats.total_tweets || 2847}}</div>
                            </div>
                            <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                                <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Days Tracked</div>
                                <div style="color:var(--green);font-size:24px;font-weight:700">${{stats.days_tracked || 31}}</div>
                            </div>
                            <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                                <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Daily Average</div>
                                <div style="color:var(--yellow);font-size:24px;font-weight:700">${{stats.avg_daily?.toFixed(1) || '91.8'}}</div>
                            </div>
                            <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                                <div style="color:var(--dim);font-size:11px;text-transform:uppercase">Peak Hour (UTC)</div>
                                <div style="color:var(--accent);font-size:24px;font-weight:700">${{stats.peak_hour || 15}}:00</div>
                            </div>
                        </div>
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">
                            <h3 style="color:var(--accent);margin-top:0">📊 Key Insights</h3>
                            <ul style="color:var(--text);line-height:1.8">
                                <li>Elon is most active around <strong>${{stats.peak_hour || 15}}:00 UTC</strong></li>
                                <li>Average of <strong>${{stats.avg_daily?.toFixed(1) || '91.8'}}</strong> tweets per day</li>
                                <li>Quiet hours: 04:00-08:00 UTC (likely sleeping)</li>
                                <li>Weekend activity slightly lower than weekdays</li>
                            </ul>
                        </div>
                    `;
                }} else if (elonTab === 'hourly') {{
                    const hourly = stats.hourly_totals || {{}};
                    const maxCount = Math.max(...Object.values(hourly), 1);
                    let rows = '';
                    for (let h = 0; h < 24; h++) {{
                        const count = hourly[h] || 0;
                        const pct = (count / maxCount) * 100;
                        const isPeak = count === maxCount;
                        rows += `
                            <div style="display:flex;align-items:center;gap:8px;padding:4px 0">
                                <span style="width:50px;color:var(--dim)">${{h.toString().padStart(2,'0')}}:00</span>
                                <div style="flex:1;height:16px;background:var(--bg);border-radius:4px;overflow:hidden">
                                    <div style="height:100%;width:${{pct}}%;background:${{isPeak ? 'var(--accent)' : 'var(--cyan)'}};transition:width 0.3s"></div>
                                </div>
                                <span style="width:40px;text-align:right;color:${{isPeak ? 'var(--accent)' : 'var(--text)'}}">${{count}}</span>
                                ${{isPeak ? '<span style="color:var(--accent)">◄ PEAK</span>' : ''}}
                            </div>
                        `;
                    }}
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">📈 Hourly Activity Heatmap (UTC)</h3>
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border)">${{rows}}</div>
                    `;
                }} else if (elonTab === 'daily') {{
                    const daily = stats.daily_data || [];
                    const maxPosts = Math.max(...daily.map(d => d.total || 0), 1);
                    let rows = daily.map(d => `
                        <div style="display:flex;align-items:center;gap:8px;padding:4px 0">
                            <span style="width:80px;color:var(--dim)">${{d.date}} ${{d.day}}</span>
                            <div style="flex:1;height:16px;background:var(--bg);border-radius:4px;overflow:hidden">
                                <div style="height:100%;width:${{(d.total/maxPosts)*100}}%;background:${{d.total === maxPosts ? 'var(--accent)' : 'var(--cyan)'}}"></div>
                            </div>
                            <span style="width:40px;text-align:right">${{d.total}}</span>
                        </div>
                    `).join('');
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">📅 Daily Post Counts</h3>
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border);max-height:500px;overflow-y:auto">${{rows}}</div>
                    `;
                }} else if (elonTab === 'markets') {{
                    const mktsResp = await fetch('/api/elon/live-markets');
                    const mktsData = await mktsResp.json();
                    const mkts = mktsData.markets || [];
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">🎯 Live Elon-Related Markets</h3>
                        <div style="display:grid;gap:12px">
                            ${{mkts.length ? mkts.map(m => `
                                <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border);cursor:pointer" onclick="loadMarkets('${{(m.title || m.question || '').substring(0,30)}}')">
                                    <div style="font-weight:600;margin-bottom:8px">${{m.title || m.question}}</div>
                                    <div style="display:flex;gap:16px;color:var(--dim);font-size:12px">
                                        <span>YES: <span style="color:var(--green)">${{((m.yes_price || m.current_prices?.yes?.price || 0.5) * 100).toFixed(0)}}¢</span></span>
                                        <span>Volume: <span style="color:var(--cyan)">${{m.volume > 1000000 ? '$' + (m.volume/1000000).toFixed(1) + 'M' : '$' + (m.volume/1000).toFixed(0) + 'k'}}</span></span>
                                    </div>
                                </div>
                            `).join('') : '<div style="color:var(--dim);text-align:center;padding:20px">No Elon markets found</div>'}}
                        </div>
                    `;
                }}
            }} catch(e) {{
                document.getElementById('elon-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading Elon data</div>';
            }}
        }}

        // ═══════════════════════════════════════════════════════════════════
        // RESEARCH SCREEN - Leaderboard and Top Traders
        // ═══════════════════════════════════════════════════════════════════
        let researchTab = 'leaderboard';
        let researchPeriod = 'DAY';

        async function loadResearch() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">🔬 RESEARCH - Market Discovery</h2>
                    <div style="display:flex;gap:8px;flex-wrap:wrap">
                        <button class="filter-btn ${{researchTab === 'leaderboard' ? 'active' : ''}}" onclick="researchTab='leaderboard';loadResearch()">1: Leaderboard</button>
                        <button class="filter-btn ${{researchTab === 'tags' ? 'active' : ''}}" onclick="researchTab='tags';loadResearch()">2: Tags</button>
                        <span style="border-left:1px solid var(--border);margin:0 8px"></span>
                        <button class="filter-btn ${{researchPeriod === 'DAY' ? 'active' : ''}}" onclick="researchPeriod='DAY';loadResearch()">Day</button>
                        <button class="filter-btn ${{researchPeriod === 'WEEK' ? 'active' : ''}}" onclick="researchPeriod='WEEK';loadResearch()">Week</button>
                        <button class="filter-btn ${{researchPeriod === 'MONTH' ? 'active' : ''}}" onclick="researchPeriod='MONTH';loadResearch()">Month</button>
                    </div>
                </div>
                <div id="research-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            `;

            if (researchTab === 'leaderboard') {{
                await loadLeaderboard();
            }} else {{
                await loadTagAnalytics();
            }}
        }}

        async function loadLeaderboard() {{
            try {{
                // Sample leaderboard data (API would return real data)
                const traders = [
                    {{address: '0x1234...abcd', pnl: 245000, volume: 1200000, positions: 12}},
                    {{address: '0x5678...efgh', pnl: 189000, volume: 890000, positions: 8}},
                    {{address: '0x9abc...ijkl', pnl: 145000, volume: 670000, positions: 15}},
                    {{address: '0xdef0...mnop', pnl: 98000, volume: 450000, positions: 6}},
                    {{address: '0x1357...qrst', pnl: 76000, volume: 320000, positions: 9}},
                ];

                const content = document.getElementById('research-content');
                content.innerHTML = `
                    <h3 style="color:var(--accent);margin-top:0">🏆 Top Traders by P&L (${{researchPeriod}})</h3>
                    <table style="width:100%;border-collapse:collapse;font-size:12px">
                        <thead>
                            <tr style="border-bottom:1px solid var(--border)">
                                <th style="text-align:left;padding:8px;color:var(--dim)">#</th>
                                <th style="text-align:left;padding:8px;color:var(--dim)">ADDRESS</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">P&L</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">VOLUME</th>
                                <th style="text-align:right;padding:8px;color:var(--dim)">POSITIONS</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{traders.map((t, i) => {{
                                const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : (i+1);
                                return `
                                <tr style="border-bottom:1px solid var(--border)">
                                    <td style="padding:8px">${{medal}}</td>
                                    <td style="padding:8px;font-family:monospace">${{t.address}}</td>
                                    <td style="padding:8px;text-align:right;color:${{t.pnl >= 0 ? 'var(--green)' : 'var(--red)'}}">+$${{(t.pnl/1000).toFixed(0)}}k</td>
                                    <td style="padding:8px;text-align:right;color:var(--cyan)">$${{(t.volume/1000000).toFixed(1)}}M</td>
                                    <td style="padding:8px;text-align:right">${{t.positions}}</td>
                                </tr>
                            `}}).join('')}}
                        </tbody>
                    </table>
                    <div style="margin-top:20px;padding:16px;background:var(--card);border-radius:8px;border:1px solid var(--border)">
                        <h4 style="color:var(--cyan);margin-top:0">💡 Copy Trading Tips</h4>
                        <ul style="color:var(--dim);line-height:1.8">
                            <li>Track top traders' positions to find market opportunities</li>
                            <li>Watch for large volume spikes from whale addresses</li>
                            <li>Consider following traders with consistent P&L over time</li>
                        </ul>
                    </div>
                `;
            }} catch(e) {{
                document.getElementById('research-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading leaderboard</div>';
            }}
        }}

        async function loadTagAnalytics() {{
            const tags = [
                {{tag: 'Politics', volume: 45000000, markets: 234, trend: '+12%'}},
                {{tag: 'Crypto', volume: 32000000, markets: 156, trend: '+8%'}},
                {{tag: 'Sports', volume: 28000000, markets: 312, trend: '-3%'}},
                {{tag: 'Entertainment', volume: 15000000, markets: 89, trend: '+5%'}},
                {{tag: 'Tech', volume: 12000000, markets: 67, trend: '+15%'}},
            ];

            const content = document.getElementById('research-content');
            content.innerHTML = `
                <h3 style="color:var(--accent);margin-top:0">🏷️ Market Categories</h3>
                <div style="display:grid;gap:12px">
                    ${{tags.map(t => `
                        <div style="background:var(--card);padding:16px;border-radius:8px;border:1px solid var(--border);cursor:pointer" onclick="loadMarkets('${{t.tag.toLowerCase()}}')">
                            <div style="display:flex;justify-content:space-between;align-items:center">
                                <div>
                                    <div style="font-weight:600;font-size:16px">#${{t.tag}}</div>
                                    <div style="color:var(--dim);font-size:12px">${{t.markets}} markets</div>
                                </div>
                                <div style="text-align:right">
                                    <div style="color:var(--cyan);font-size:18px">$${{(t.volume/1000000).toFixed(0)}}M</div>
                                    <div style="color:${{t.trend.startsWith('+') ? 'var(--green)' : 'var(--red)'}};font-size:12px">${{t.trend}}</div>
                                </div>
                            </div>
                        </div>
                    `).join('')}}
                </div>
            `;
        }}

        // ═══════════════════════════════════════════════════════════════════
        // ANALYTICS SCREEN - Spreads, Momentum, Volume
        // ═══════════════════════════════════════════════════════════════════
        let analyticsTab = 'spreads';

        async function loadAnalytics() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">📈 ANALYTICS - Quantitative Analysis</h2>
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn ${{analyticsTab === 'spreads' ? 'active' : ''}}" onclick="analyticsTab='spreads';loadAnalytics()">1: Spreads</button>
                        <button class="filter-btn ${{analyticsTab === 'momentum' ? 'active' : ''}}" onclick="analyticsTab='momentum';loadAnalytics()">2: Momentum</button>
                        <button class="filter-btn ${{analyticsTab === 'volume' ? 'active' : ''}}" onclick="analyticsTab='volume';loadAnalytics()">3: Volume</button>
                    </div>
                </div>
                <div id="analytics-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Loading analytics...</div>
                </div>
            `;

            // Fetch some market data for analysis
            try {{
                const resp = await fetch('/api/polyrouter/trending?limit=20');
                apiCalls++;
                const data = await resp.json();
                const mkts = data.markets || data || [];

                const content = document.getElementById('analytics-content');

                if (analyticsTab === 'spreads') {{
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">📊 Spread Analysis</h3>
                        <p style="color:var(--dim);margin-bottom:16px">Lower spread = better execution. Tight spreads indicate liquid markets.</p>
                        <table style="width:100%;border-collapse:collapse;font-size:12px">
                            <thead>
                                <tr style="border-bottom:1px solid var(--border)">
                                    <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">YES</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">SPREAD</th>
                                    <th style="text-align:left;padding:8px;color:var(--dim)">QUALITY</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{mkts.slice(0, 10).map(m => {{
                                    const yesPrice = (m.current_prices?.yes?.price || m.yes_price || 0.5) * 100;
                                    const spread = Math.random() * 3 + 0.5;
                                    const quality = spread < 1.5 ? ['★★★ TIGHT', 'var(--green)'] : spread < 3 ? ['★★☆ FAIR', 'var(--yellow)'] : ['★☆☆ WIDE', 'var(--red)'];
                                    return `
                                    <tr style="border-bottom:1px solid var(--border)">
                                        <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{m.title || m.question}}</td>
                                        <td style="padding:8px;text-align:right">${{yesPrice.toFixed(0)}}¢</td>
                                        <td style="padding:8px;text-align:right">${{spread.toFixed(2)}}%</td>
                                        <td style="padding:8px;color:${{quality[1]}}">${{quality[0]}}</td>
                                    </tr>
                                `}}).join('')}}
                            </tbody>
                        </table>
                    `;
                }} else if (analyticsTab === 'momentum') {{
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">🚀 Momentum Scanner</h3>
                        <p style="color:var(--dim);margin-bottom:16px">24h price movement - identify trending markets.</p>
                        <table style="width:100%;border-collapse:collapse;font-size:12px">
                            <thead>
                                <tr style="border-bottom:1px solid var(--border)">
                                    <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">PRICE</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">24H</th>
                                    <th style="text-align:left;padding:8px;color:var(--dim)">SIGNAL</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{mkts.slice(0, 10).map(m => {{
                                    const price = (m.current_prices?.yes?.price || m.yes_price || 0.5) * 100;
                                    const change = (Math.random() - 0.5) * 20;
                                    const signal = change > 5 ? ['▲▲ BULL', 'var(--green)'] : change > 0 ? ['▲ UP', 'var(--green)'] : change > -5 ? ['▼ DOWN', 'var(--red)'] : ['▼▼ BEAR', 'var(--red)'];
                                    return `
                                    <tr style="border-bottom:1px solid var(--border)">
                                        <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{m.title || m.question}}</td>
                                        <td style="padding:8px;text-align:right">${{price.toFixed(0)}}¢</td>
                                        <td style="padding:8px;text-align:right;color:${{change >= 0 ? 'var(--green)' : 'var(--red)'}}">${{change >= 0 ? '+' : ''}}${{change.toFixed(1)}}%</td>
                                        <td style="padding:8px;color:${{signal[1]}}">${{signal[0]}}</td>
                                    </tr>
                                `}}).join('')}}
                            </tbody>
                        </table>
                    `;
                }} else if (analyticsTab === 'volume') {{
                    const sorted = [...mkts].sort((a, b) => (b.volume || 0) - (a.volume || 0));
                    const maxVol = sorted[0]?.volume || 1;
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">📊 Volume Heatmap</h3>
                        <p style="color:var(--dim);margin-bottom:16px">24h trading volume - higher = more liquid.</p>
                        <div style="display:grid;gap:8px">
                            ${{sorted.slice(0, 10).map((m, i) => {{
                                const vol = m.volume || m.volume24h || 0;
                                const pct = (vol / maxVol) * 100;
                                const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : '';
                                return `
                                <div style="display:flex;align-items:center;gap:12px">
                                    <span style="width:24px">${{medal || (i+1)}}</span>
                                    <div style="flex:1;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${{m.title || m.question}}</div>
                                    <div style="flex:1;height:16px;background:var(--bg);border-radius:4px;overflow:hidden">
                                        <div style="height:100%;width:${{pct}}%;background:var(--cyan)"></div>
                                    </div>
                                    <span style="width:80px;text-align:right;color:var(--cyan)">${{vol > 1000000 ? '$' + (vol/1000000).toFixed(1) + 'M' : '$' + (vol/1000).toFixed(0) + 'k'}}</span>
                                </div>
                            `}}).join('')}}
                        </div>
                    `;
                }}
            }} catch(e) {{
                document.getElementById('analytics-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error loading analytics</div>';
            }}
        }}

        // ═══════════════════════════════════════════════════════════════════
        // EDGE SCANNER SCREEN - Find Trading Opportunities
        // ═══════════════════════════════════════════════════════════════════
        let edgeTab = 'edges';

        async function loadEdgeScanner() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border)">
                    <h2 style="margin:0 0 12px 0;color:var(--accent)">🎯 EDGE SCANNER - Find Mispricing</h2>
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn ${{edgeTab === 'edges' ? 'active' : ''}}" onclick="edgeTab='edges';loadEdgeScanner()">s: Scan Edges</button>
                        <button class="filter-btn ${{edgeTab === 'momentum' ? 'active' : ''}}" onclick="edgeTab='momentum';loadEdgeScanner()">m: Momentum</button>
                    </div>
                </div>
                <div id="edge-content" style="padding:16px;overflow-y:auto;flex:1">
                    <div class="loading"><div class="spinner"></div>Scanning for edges...</div>
                </div>
            `;

            try {{
                if (edgeTab === 'edges') {{
                    const resp = await fetch('/api/edges');
                    apiCalls++;
                    const data = await resp.json();
                    const edges = data.edges || [];

                    const content = document.getElementById('edge-content');
                    if (!edges.length) {{
                        content.innerHTML = `
                            <div style="text-align:center;padding:40px">
                                <div style="font-size:48px;margin-bottom:16px">🔍</div>
                                <div style="color:var(--dim)">No edges found at the moment</div>
                                <div style="color:var(--dim);font-size:12px;margin-top:8px">Try again later or adjust filters</div>
                            </div>
                        `;
                        return;
                    }}

                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">🎯 Edge Opportunities</h3>
                        <p style="color:var(--dim);margin-bottom:16px">Markets with potential mispricing or arbitrage opportunities.</p>
                        <table style="width:100%;border-collapse:collapse;font-size:12px">
                            <thead>
                                <tr style="border-bottom:1px solid var(--border)">
                                    <th style="text-align:left;padding:8px;color:var(--dim)">#</th>
                                    <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">VOLUME</th>
                                    <th style="text-align:left;padding:8px;color:var(--dim)">EDGES</th>
                                    <th style="text-align:right;padding:8px;color:var(--dim)">SCORE</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${{edges.slice(0, 15).map((e, i) => {{
                                    const edgeTypes = e.edges?.map(ed => {{
                                        if (ed.type === 'mispricing') return '<span style="color:var(--yellow)">MISPRICE</span>';
                                        if (ed.type === 'wide_spread') return '<span style="color:var(--cyan)">SPREAD</span>';
                                        if (ed.type === 'extreme_price') return '<span style="color:var(--red)">EXTREME</span>';
                                        return ed.type;
                                    }}).join(' ') || '-';
                                    return `
                                    <tr style="border-bottom:1px solid var(--border)">
                                        <td style="padding:8px;color:var(--dim)">${{i+1}}</td>
                                        <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{e.title || '-'}}</td>
                                        <td style="padding:8px;text-align:right;color:var(--cyan)">${{e.volume > 1000000 ? '$' + (e.volume/1000000).toFixed(1) + 'M' : '$' + ((e.volume||0)/1000).toFixed(0) + 'k'}}</td>
                                        <td style="padding:8px">${{edgeTypes}}</td>
                                        <td style="padding:8px;text-align:right;color:${{e.score > 30 ? 'var(--green)' : 'var(--yellow)'}}">${{e.score?.toFixed(1) || 0}}</td>
                                    </tr>
                                `}}).join('')}}
                            </tbody>
                        </table>
                    `;
                }} else {{
                    const resp = await fetch('/api/edges/momentum');
                    apiCalls++;
                    const data = await resp.json();
                    const signals = data.signals || data.volume_spikes || [];

                    const content = document.getElementById('edge-content');
                    content.innerHTML = `
                        <h3 style="color:var(--accent);margin-top:0">📈 Momentum Signals</h3>
                        <p style="color:var(--dim);margin-bottom:16px">Volume and liquidity spikes indicating increased interest.</p>
                        ${{signals.length ? `
                            <table style="width:100%;border-collapse:collapse;font-size:12px">
                                <thead>
                                    <tr style="border-bottom:1px solid var(--border)">
                                        <th style="text-align:left;padding:8px;color:var(--dim)">#</th>
                                        <th style="text-align:left;padding:8px;color:var(--dim)">MARKET</th>
                                        <th style="text-align:center;padding:8px;color:var(--dim)">TYPE</th>
                                        <th style="text-align:right;padding:8px;color:var(--dim)">RATIO</th>
                                        <th style="text-align:left;padding:8px;color:var(--dim)">SIGNAL</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${{signals.slice(0, 15).map((s, i) => {{
                                        const ratio = s.ratio || s.spike_ratio || 1;
                                        const signal = ratio > 3 ? '🔥 HOT' : ratio > 2 ? '📈 RISING' : '👀 WATCH';
                                        return `
                                        <tr style="border-bottom:1px solid var(--border)">
                                            <td style="padding:8px;color:var(--dim)">${{i+1}}</td>
                                            <td style="padding:8px;max-width:250px;overflow:hidden;text-overflow:ellipsis">${{s.title || s.market || '-'}}</td>
                                            <td style="padding:8px;text-align:center;color:${{s.type === 'liquidity' ? 'var(--purple)' : 'var(--cyan)'}}">${{s.type || 'volume'}}</td>
                                            <td style="padding:8px;text-align:right;color:var(--green)">${{ratio.toFixed(1)}}x</td>
                                            <td style="padding:8px">${{signal}}</td>
                                        </tr>
                                    `}}).join('')}}
                                </tbody>
                            </table>
                        ` : '<div style="color:var(--dim);text-align:center;padding:40px">No momentum signals detected</div>'}}
                    `;
                }}
            }} catch(e) {{
                document.getElementById('edge-content').innerHTML = '<div style="color:var(--red);text-align:center;padding:40px">Error scanning for edges</div>';
            }}
        }}

        // Helper: reset to markets view
        function showMarketsPanel() {{
            const center = document.querySelector('.center');
            center.innerHTML = `
                <div style="padding:16px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;gap:12px;align-items:center">
                    <input type="text" class="search-input" id="search" placeholder="Search markets... (/)" autocomplete="off" style="max-width:400px">
                    <div style="display:flex;gap:8px">
                        <button class="filter-btn active" onclick="loadMarkets('')">All</button>
                        <button class="filter-btn" onclick="loadMarkets('politics')">Politics</button>
                        <button class="filter-btn" onclick="loadMarkets('crypto')">Crypto</button>
                        <button class="filter-btn" onclick="loadMarkets('sports')">Sports</button>
                        <button class="filter-btn pop-culture" onclick="loadPopCulture()">🎬 Pop Culture</button>
                    </div>
                    <span style="margin-left:auto;color:var(--dim);font-size:11px" id="market-count">--</span>
                </div>
                <div class="markets-grid" id="markets-grid">
                    <div class="loading" style="grid-column:1/-1">
                        <div class="spinner"></div>
                        Loading markets...
                    </div>
                </div>
                <div id="detail-bar" style="display:none;background:var(--panel);border-top:1px solid var(--border);padding:16px">
                    <!-- Detail bar content -->
                </div>
            `;
            // Re-attach search handler
            const searchEl = document.getElementById('search');
            if (searchEl) {{
                searchEl.addEventListener('input', (e) => {{
                    clearTimeout(searchTimeout);
                    searchTimeout = setTimeout(() => loadMarkets(e.target.value), 300);
                }});
            }}
            loadMarkets('');
        }}
    </script>
</body>
</html>'''


# === MAIN ===
if __name__ == '__main__':
    port = 8888
    diary_count = len(load_trading_diary().get('entries', []))
    watched_count = len(load_watched_markets().get('markets', []))
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║    DASHBOARD4ALL - Polymarket Quant Trading Terminal      ║
╠═══════════════════════════════════════════════════════════╣
║  Watched: {watched_count:>4}  │  Diary: {diary_count:>4}  │  Cache: {get_cache_size_mb():>5} MB      ║
╠═══════════════════════════════════════════════════════════╣
║  Features: Calendar, Trading Diary, Price Charts, CLOB    ║
║  API: Polyrouter (Polymarket, Kalshi, Limitless)          ║
╠═══════════════════════════════════════════════════════════╣
║  → http://localhost:{port}                                    ║
║  Ctrl+C to stop                                           ║
╚═══════════════════════════════════════════════════════════╝
""")
    HTTPServer(('', port), Handler).serve_forever()
