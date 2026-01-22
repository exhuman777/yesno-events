#!/usr/bin/env python3
"""
YES/NO.EVENTS Quant Models
Monte Carlo, Kelly, Momentum, and visualization tools
"""
import sys
import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta

# Activate venv
VENV_PATH = Path(__file__).parent / ".venv"
if VENV_PATH.exists():
    for pyver in ["python3.12", "python3.11", "python3.10"]:
        venv_site = VENV_PATH / "lib" / pyver / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
            break

from trading import scan_elon_markets, get_price, analyze_distribution

# ============================================
# ASCII Chart Utilities
# ============================================

def ascii_bar(value, max_val, width=40, char='█'):
    """Create ASCII bar"""
    if max_val == 0:
        return ''
    filled = int((value / max_val) * width)
    return char * filled

def ascii_histogram(data, labels=None, width=50, title=None):
    """Create ASCII histogram"""
    if not data:
        return "No data"

    max_val = max(data)
    lines = []

    if title:
        lines.append(f"\n{'═' * (width + 20)}")
        lines.append(f" {title}")
        lines.append(f"{'═' * (width + 20)}")

    for i, val in enumerate(data):
        label = labels[i] if labels and i < len(labels) else f"{i}"
        bar = ascii_bar(val, max_val, width)
        pct = val * 100 if val <= 1 else val
        lines.append(f"{label:>10} │{bar} {pct:.1f}%")

    return '\n'.join(lines)

def ascii_line_chart(data, width=60, height=15, title=None):
    """Create ASCII line chart"""
    if not data or len(data) < 2:
        return "Insufficient data"

    min_val = min(data)
    max_val = max(data)
    val_range = max_val - min_val or 1

    lines = []
    if title:
        lines.append(f"\n {title}")
        lines.append(f" {'─' * (width + 5)}")

    # Y-axis labels
    for row in range(height, -1, -1):
        y_val = min_val + (row / height) * val_range

        line = f"{y_val:>6.1f} │"
        for col in range(width):
            idx = int(col * len(data) / width)
            val = data[idx]
            normalized = (val - min_val) / val_range * height

            if abs(normalized - row) < 0.5:
                line += '●'
            elif row == 0:
                line += '─'
            else:
                line += ' '

        lines.append(line)

    # X-axis
    lines.append(f"       └{'─' * width}")
    lines.append(f"        0{' ' * (width//2 - 2)}t{' ' * (width//2 - 2)}{len(data)}")

    return '\n'.join(lines)

def ascii_distribution(brackets, probs, width=50):
    """Create ASCII probability distribution chart"""
    if not brackets or not probs:
        return "No data"

    max_prob = max(probs)
    lines = []
    lines.append(f"\n{'═' * 65}")
    lines.append(" PROBABILITY DISTRIBUTION")
    lines.append(f"{'═' * 65}")
    lines.append(f"{'Bracket':>12} {'Prob':>7} {'Distribution':<{width}}")
    lines.append(f"{'─' * 12} {'─' * 7} {'─' * width}")

    for bracket, prob in zip(brackets, probs):
        bar_len = int((prob / max_prob) * width) if max_prob > 0 else 0
        bar = '█' * bar_len + '░' * (width - bar_len)
        pct = prob * 100

        # Color indicator
        if pct > 10:
            indicator = '◆'
        elif pct > 5:
            indicator = '◇'
        else:
            indicator = '·'

        lines.append(f"{bracket:>12} {pct:>6.1f}% {indicator}{bar}")

    return '\n'.join(lines)

# ============================================
# Monte Carlo Simulation
# ============================================

def monte_carlo_tweets(current_count, hours_elapsed, total_hours=168,
                       simulations=10000, volatility=0.15):
    """
    Monte Carlo simulation for tweet count projections.

    Args:
        current_count: Current tweet count
        hours_elapsed: Hours since event start
        total_hours: Total event duration (168 = 1 week)
        simulations: Number of simulations
        volatility: Rate volatility factor

    Returns:
        dict with projection stats and distribution
    """
    if hours_elapsed <= 0:
        hours_elapsed = 1

    current_rate = current_count / hours_elapsed
    hours_remaining = total_hours - hours_elapsed

    if hours_remaining <= 0:
        return {"final": current_count, "distribution": {current_count: 1.0}}

    results = []

    for _ in range(simulations):
        # Simulate remaining hours with variable rate
        projected = current_count
        rate = current_rate

        for h in range(int(hours_remaining)):
            # Random walk on rate with mean reversion
            rate_change = random.gauss(0, volatility * current_rate)
            rate = max(0.1, rate + rate_change)
            rate = rate * 0.95 + current_rate * 0.05  # Mean reversion
            projected += rate

        results.append(int(projected))

    # Bucket into ranges
    buckets = {}
    bucket_size = 20
    for r in results:
        bucket = (r // bucket_size) * bucket_size
        bucket_key = f"{bucket}-{bucket + bucket_size - 1}"
        buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

    # Normalize
    for k in buckets:
        buckets[k] /= simulations

    # Stats
    results.sort()
    mean = sum(results) / len(results)
    median = results[len(results) // 2]
    p10 = results[int(len(results) * 0.1)]
    p90 = results[int(len(results) * 0.9)]
    std_dev = (sum((x - mean) ** 2 for x in results) / len(results)) ** 0.5

    return {
        "current_count": current_count,
        "hours_elapsed": hours_elapsed,
        "current_rate": current_rate,
        "hours_remaining": hours_remaining,
        "simulations": simulations,
        "mean": mean,
        "median": median,
        "std_dev": std_dev,
        "p10": p10,
        "p90": p90,
        "range_80": f"{p10}-{p90}",
        "distribution": dict(sorted(buckets.items(), key=lambda x: int(x[0].split('-')[0])))
    }

def visualize_monte_carlo(mc_result):
    """Visualize Monte Carlo results"""
    lines = []
    lines.append(f"\n{'═' * 60}")
    lines.append(" MONTE CARLO TWEET PROJECTION")
    lines.append(f"{'═' * 60}")
    lines.append(f" Current: {mc_result['current_count']} tweets in {mc_result['hours_elapsed']:.0f}h")
    lines.append(f" Rate: {mc_result['current_rate']:.1f} tweets/hour")
    lines.append(f" Remaining: {mc_result['hours_remaining']:.0f} hours")
    lines.append(f"{'─' * 60}")
    lines.append(f" Projections ({mc_result['simulations']:,} simulations):")
    lines.append(f"   Mean:   {mc_result['mean']:.0f}")
    lines.append(f"   Median: {mc_result['median']:.0f}")
    lines.append(f"   StdDev: {mc_result['std_dev']:.1f}")
    lines.append(f"   80% CI: {mc_result['range_80']}")
    lines.append(f"{'─' * 60}")

    # Distribution chart
    dist = mc_result['distribution']
    if dist:
        brackets = list(dist.keys())
        probs = list(dist.values())
        max_prob = max(probs)

        lines.append(" Distribution:")
        for bracket, prob in zip(brackets, probs):
            bar = '█' * int(prob / max_prob * 30)
            lines.append(f"   {bracket:>12}: {bar} {prob*100:.1f}%")

    return '\n'.join(lines)

# ============================================
# Kelly Criterion Calculator
# ============================================

def kelly_criterion(prob_win, odds, fraction=0.25):
    """
    Calculate Kelly Criterion bet size.

    Args:
        prob_win: Probability of winning (0-1)
        odds: Decimal odds (payout per $1 bet, including stake)
        fraction: Kelly fraction (0.25 = quarter Kelly, safer)

    Returns:
        dict with kelly calculations
    """
    if prob_win <= 0 or prob_win >= 1 or odds <= 1:
        return {"kelly": 0, "edge": 0, "recommendation": "No edge"}

    # b = odds - 1 (net profit per $1)
    b = odds - 1
    p = prob_win
    q = 1 - p

    # Kelly formula: f* = (bp - q) / b
    full_kelly = (b * p - q) / b

    # Edge = expected value per $1
    edge = p * b - q

    # Fractional Kelly (safer)
    recommended = max(0, full_kelly * fraction)

    return {
        "prob_win": prob_win,
        "odds": odds,
        "edge": edge,
        "edge_pct": edge * 100,
        "full_kelly": full_kelly,
        "full_kelly_pct": full_kelly * 100,
        "fraction": fraction,
        "recommended": recommended,
        "recommended_pct": recommended * 100,
        "recommendation": f"Bet {recommended*100:.1f}% of bankroll" if recommended > 0 else "No bet"
    }

def kelly_for_market(market_id, your_prob, bankroll=1000):
    """
    Calculate Kelly bet for a specific market.

    Args:
        market_id: Market ID
        your_prob: Your estimated probability (0-1)
        bankroll: Your bankroll in $
    """
    try:
        price = get_price(market_id)
        market_prob = price['yes']
        odds = 1 / market_prob  # Decimal odds

        kelly = kelly_criterion(your_prob, odds)
        bet_size = kelly['recommended'] * bankroll

        return {
            **kelly,
            "market_id": market_id,
            "market_prob": market_prob,
            "your_prob": your_prob,
            "bankroll": bankroll,
            "bet_size": bet_size,
            "potential_profit": bet_size * (odds - 1) if kelly['recommended'] > 0 else 0
        }
    except Exception as e:
        return {"error": str(e)}

def visualize_kelly(kelly_result):
    """Visualize Kelly calculation"""
    lines = []
    lines.append(f"\n{'═' * 50}")
    lines.append(" KELLY CRITERION CALCULATOR")
    lines.append(f"{'═' * 50}")

    if 'error' in kelly_result:
        lines.append(f" Error: {kelly_result['error']}")
        return '\n'.join(lines)

    lines.append(f" Market: {kelly_result.get('market_id', 'N/A')}")
    lines.append(f" Market Prob: {kelly_result.get('market_prob', 0)*100:.1f}%")
    lines.append(f" Your Prob:   {kelly_result['your_prob']*100:.1f}%")
    lines.append(f"{'─' * 50}")
    lines.append(f" Edge: {kelly_result['edge_pct']:+.2f}%")
    lines.append(f" Full Kelly: {kelly_result['full_kelly_pct']:.1f}%")
    lines.append(f" {kelly_result['fraction']*100:.0f}% Kelly: {kelly_result['recommended_pct']:.1f}%")
    lines.append(f"{'─' * 50}")

    if kelly_result['recommended'] > 0:
        lines.append(f" Bankroll: ${kelly_result.get('bankroll', 0):,.0f}")
        lines.append(f" Bet Size: ${kelly_result.get('bet_size', 0):,.2f}")
        lines.append(f" Potential Profit: ${kelly_result.get('potential_profit', 0):,.2f}")

        # Visual gauge
        kelly_pct = min(kelly_result['recommended_pct'], 25)
        gauge = '█' * int(kelly_pct) + '░' * (25 - int(kelly_pct))
        lines.append(f"\n Risk: [{gauge}] {kelly_pct:.1f}%")
    else:
        lines.append(f" ⚠ NO EDGE - Don't bet")

    return '\n'.join(lines)

# ============================================
# Momentum Scorer
# ============================================

def momentum_score(event=None):
    """
    Calculate momentum scores for brackets.
    Compares current odds to normalized distribution.
    """
    analysis = analyze_distribution(event)
    scores = []

    for evt, data in analysis.items():
        brackets = data['brackets']
        if len(brackets) < 3:
            continue

        # Calculate momentum as deviation from smooth curve
        for i, b in enumerate(brackets):
            prob = b['market_prob']

            # Expected based on neighbors (smoothed)
            if i == 0:
                expected = brackets[1]['market_prob'] * 0.7
            elif i == len(brackets) - 1:
                expected = brackets[-2]['market_prob'] * 0.7
            else:
                expected = (brackets[i-1]['market_prob'] + brackets[i+1]['market_prob']) / 2

            deviation = prob - expected

            # Volume momentum
            avg_vol = sum(x['volume'] for x in brackets) / len(brackets)
            vol_ratio = b['volume'] / avg_vol if avg_vol > 0 else 1

            # Combined score
            score = deviation * 100 + (vol_ratio - 1) * 5

            scores.append({
                'event': evt,
                'bracket': b['bracket'],
                'prob': prob,
                'expected': expected,
                'deviation': deviation,
                'volume': b['volume'],
                'vol_ratio': vol_ratio,
                'momentum': score,
                'signal': 'OVERBOUGHT' if score > 3 else 'OVERSOLD' if score < -3 else 'NEUTRAL',
                'id': b['id']
            })

    return sorted(scores, key=lambda x: abs(x['momentum']), reverse=True)

def visualize_momentum(scores, top_n=15):
    """Visualize momentum scores"""
    lines = []
    lines.append(f"\n{'═' * 70}")
    lines.append(" MOMENTUM SCANNER")
    lines.append(f"{'═' * 70}")
    lines.append(f" {'Event':<12} {'Bracket':<12} {'Prob':>7} {'Exp':>7} {'Mom':>7} {'Signal':<10}")
    lines.append(f" {'─'*12} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

    for s in scores[:top_n]:
        mom_bar = ''
        if s['momentum'] > 0:
            mom_bar = '+' * min(int(s['momentum']), 10)
        else:
            mom_bar = '-' * min(int(abs(s['momentum'])), 10)

        lines.append(f" {s['event']:<12} {s['bracket']:<12} {s['prob']*100:>6.1f}% {s['expected']*100:>6.1f}% {s['momentum']:>+6.1f} {s['signal']:<10}")

    return '\n'.join(lines)

# ============================================
# Expected Value Calculator
# ============================================

def calculate_ev(market_prob, your_prob, position_size=100):
    """Calculate expected value of a trade"""
    # Buying YES
    cost = position_size * market_prob
    payout_if_win = position_size
    payout_if_lose = 0

    ev = your_prob * payout_if_win + (1 - your_prob) * payout_if_lose - cost
    ev_pct = ev / cost * 100 if cost > 0 else 0

    return {
        'market_prob': market_prob,
        'your_prob': your_prob,
        'position_size': position_size,
        'cost': cost,
        'ev': ev,
        'ev_pct': ev_pct,
        'profitable': ev > 0
    }

# ============================================
# CLI Interface
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="YES/NO.EVENTS Quant Models")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Monte Carlo
    mc_parser = subparsers.add_parser("mc", help="Monte Carlo tweet projection")
    mc_parser.add_argument("current", type=int, help="Current tweet count")
    mc_parser.add_argument("hours", type=float, help="Hours elapsed")
    mc_parser.add_argument("--total", type=int, default=168, help="Total hours (default 168)")
    mc_parser.add_argument("--sims", type=int, default=10000, help="Simulations")
    mc_parser.add_argument("--vol", type=float, default=0.15, help="Volatility")

    # Kelly
    kelly_parser = subparsers.add_parser("kelly", help="Kelly criterion calculator")
    kelly_parser.add_argument("market_id", help="Market ID")
    kelly_parser.add_argument("your_prob", type=float, help="Your probability (0-1)")
    kelly_parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll")
    kelly_parser.add_argument("--fraction", type=float, default=0.25, help="Kelly fraction")

    # Momentum
    mom_parser = subparsers.add_parser("momentum", help="Momentum scanner")
    mom_parser.add_argument("--event", help="Filter by event")
    mom_parser.add_argument("--top", type=int, default=15, help="Top N results")

    # Distribution chart
    dist_parser = subparsers.add_parser("chart", help="Distribution chart")
    dist_parser.add_argument("--event", help="Event to chart")

    # EV calculator
    ev_parser = subparsers.add_parser("ev-calc", help="EV calculator")
    ev_parser.add_argument("market_prob", type=float, help="Market probability")
    ev_parser.add_argument("your_prob", type=float, help="Your probability")
    ev_parser.add_argument("--size", type=float, default=100, help="Position size")

    args = parser.parse_args()

    if args.command == "mc":
        result = monte_carlo_tweets(
            args.current, args.hours, args.total, args.sims, args.vol
        )
        print(visualize_monte_carlo(result))

    elif args.command == "kelly":
        result = kelly_for_market(args.market_id, args.your_prob, args.bankroll)
        print(visualize_kelly(result))

    elif args.command == "momentum":
        scores = momentum_score(args.event)
        print(visualize_momentum(scores, args.top))

    elif args.command == "chart":
        analysis = analyze_distribution(args.event)
        for evt, data in analysis.items():
            brackets = [b['bracket'] for b in data['brackets']]
            probs = [b['market_prob'] for b in data['brackets']]
            print(ascii_distribution(brackets, probs))

    elif args.command == "ev-calc":
        result = calculate_ev(args.market_prob, args.your_prob, args.size)
        print(f"\n{'═' * 40}")
        print(" EV CALCULATOR")
        print(f"{'═' * 40}")
        print(f" Market Prob: {result['market_prob']*100:.1f}%")
        print(f" Your Prob:   {result['your_prob']*100:.1f}%")
        print(f" Position:    ${result['position_size']:.0f}")
        print(f"{'─' * 40}")
        print(f" Cost:        ${result['cost']:.2f}")
        print(f" EV:          ${result['ev']:+.2f} ({result['ev_pct']:+.1f}%)")
        print(f" Verdict:     {'✓ PROFITABLE' if result['profitable'] else '✗ NOT PROFITABLE'}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
