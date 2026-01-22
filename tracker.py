#!/usr/bin/env python3
"""
YES/NO.EVENTS Tweet Tracker
Calendar view, historical data, live tracking
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Activate venv
VENV_PATH = Path(__file__).parent / ".venv"
if VENV_PATH.exists():
    for pyver in ["python3.12", "python3.11", "python3.10"]:
        venv_site = VENV_PATH / "lib" / pyver / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
            break

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
TWEETS_FILE = DATA_DIR / "elon_tweets.json"

# ============================================
# Data Storage
# ============================================

def load_tweet_data():
    """Load tweet data from storage"""
    if TWEETS_FILE.exists():
        return json.loads(TWEETS_FILE.read_text())
    return {"events": {}, "hourly": {}}

def save_tweet_data(data):
    """Save tweet data to storage"""
    DATA_DIR.mkdir(exist_ok=True)
    TWEETS_FILE.write_text(json.dumps(data, indent=2))

def add_hourly_data(date_str, hour, count, event=None):
    """Add hourly tweet count

    Args:
        date_str: Date in YYYY-MM-DD format
        hour: Hour (0-23)
        count: Number of tweets
        event: Event name (e.g., 'jan9_16')
    """
    data = load_tweet_data()

    if "hourly" not in data:
        data["hourly"] = {}

    key = f"{date_str}_{hour:02d}"
    data["hourly"][key] = {
        "date": date_str,
        "hour": hour,
        "count": count,
        "event": event,
        "updated": datetime.now().isoformat()
    }

    save_tweet_data(data)
    return data

def add_daily_data(date_str, hourly_counts, event=None):
    """Add full day of hourly data

    Args:
        date_str: Date in YYYY-MM-DD format
        hourly_counts: List of 24 counts (index = hour)
        event: Event name
    """
    data = load_tweet_data()

    if "hourly" not in data:
        data["hourly"] = {}

    for hour, count in enumerate(hourly_counts):
        if count is not None:
            key = f"{date_str}_{hour:02d}"
            data["hourly"][key] = {
                "date": date_str,
                "hour": hour,
                "count": count,
                "event": event,
                "updated": datetime.now().isoformat()
            }

    save_tweet_data(data)
    return data

def get_hourly_count(date_str, hour):
    """Get tweet count for specific hour"""
    data = load_tweet_data()
    key = f"{date_str}_{hour:02d}"
    entry = data.get("hourly", {}).get(key)
    return entry["count"] if entry else None

def get_day_totals(date_str):
    """Get all hourly counts for a day"""
    data = load_tweet_data()
    counts = []
    for hour in range(24):
        key = f"{date_str}_{hour:02d}"
        entry = data.get("hourly", {}).get(key)
        counts.append(entry["count"] if entry else None)
    return counts

def get_date_range_data(start_date, end_date):
    """Get data for date range"""
    data = load_tweet_data()
    result = {}

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        counts = get_day_totals(date_str)
        if any(c is not None for c in counts):
            result[date_str] = counts
        current += timedelta(days=1)

    return result

# ============================================
# Calendar View
# ============================================

def color_cell(count, max_count=15):
    """Get ANSI color for cell based on count"""
    if count is None or count == 0:
        return "\033[90m"  # Gray

    # Green intensity based on count
    intensity = min(count / max_count, 1.0)

    if intensity > 0.8:
        return "\033[92m"  # Bright green
    elif intensity > 0.5:
        return "\033[32m"  # Green
    elif intensity > 0.3:
        return "\033[33m"  # Yellow
    else:
        return "\033[37m"  # White

def reset_color():
    return "\033[0m"

def calendar_view(start_date, days=7, event_name=None):
    """Generate ASCII calendar view like the screenshot

    Args:
        start_date: Start date (YYYY-MM-DD)
        days: Number of days to show
        event_name: Optional event name for title
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)) for i in range(days)]

    lines = []

    # Header
    title = f" ELON TWEET CALENDAR"
    if event_name:
        title += f" - {event_name}"
    lines.append(f"\033[1;36m{'═' * 75}\033[0m")
    lines.append(f"\033[1;36m{title}\033[0m")
    lines.append(f"\033[1;36m{'═' * 75}\033[0m")

    # Day headers
    header = "      │"
    for d in dates:
        day_name = d.strftime("%a")
        day_num = d.strftime("%d")
        header += f" {day_name:^3} │"
    lines.append(header)

    # Date numbers
    date_row = "      │"
    for d in dates:
        day_num = d.strftime("%d")
        date_row += f"  {day_num} │"
    lines.append(date_row)
    lines.append("──────┼" + "─────┼" * days)

    # Hourly rows
    totals = [0] * days
    for hour in range(24):
        row = f" {hour:02d}:00│"
        for i, d in enumerate(dates):
            date_str = d.strftime("%Y-%m-%d")
            count = get_hourly_count(date_str, hour)

            if count is not None:
                totals[i] += count
                color = color_cell(count)
                row += f"{color} {count:>3} {reset_color()}│"
            else:
                row += f"\033[90m  ·  \033[0m│"

        lines.append(row)

    # Totals row
    lines.append("──────┼" + "─────┼" * days)
    total_row = "Total │"
    for t in totals:
        if t > 0:
            total_row += f"\033[1;32m {t:>3} \033[0m│"
        else:
            total_row += f"\033[90m  ·  \033[0m│"
    lines.append(total_row)

    # Grand total
    grand_total = sum(totals)
    lines.append(f"\n\033[1mGrand Total: {grand_total} tweets\033[0m")

    return '\n'.join(lines)

def mini_calendar(start_date, days=7):
    """Compact calendar view"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)) for i in range(days)]

    lines = []

    # Header with day names
    header = "     "
    for d in dates:
        header += f" {d.strftime('%a')[:2]}"
    lines.append(header)

    # Compact hourly view (6-hour blocks)
    for block in range(4):
        start_hour = block * 6
        end_hour = start_hour + 5
        row = f"{start_hour:02d}-{end_hour:02d}"

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            block_total = 0
            for h in range(start_hour, end_hour + 1):
                count = get_hourly_count(date_str, h)
                if count:
                    block_total += count

            if block_total > 0:
                color = color_cell(block_total, max_count=30)
                row += f"{color}{block_total:>3}{reset_color()}"
            else:
                row += f"\033[90m  ·\033[0m"

        lines.append(row)

    return '\n'.join(lines)

# ============================================
# Live Tweet Fetching (xtracker.polymarket.com)
# ============================================

def fetch_live_count():
    """Fetch current tweet count from xtracker.polymarket.com"""
    import urllib.request
    import ssl
    import re

    url = "https://xtracker.polymarket.com/"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) YesNoEvents/1.0'
        })
        ctx = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            html = resp.read().decode('utf-8')

            # Try to extract count from page
            # Look for patterns like "Current count: 123" or JSON data

            # Try JSON pattern
            json_match = re.search(r'"count"\s*:\s*(\d+)', html)
            if json_match:
                return int(json_match.group(1))

            # Try text pattern
            count_match = re.search(r'(\d{2,4})\s*(?:tweets?|posts?)', html, re.I)
            if count_match:
                return int(count_match.group(1))

            # Return None if not found
            return None

    except Exception as e:
        print(f"Error fetching: {e}")
        return None

def fetch_from_api():
    """Fetch from Polymarket API for current event data"""
    import urllib.request
    import ssl

    # Use gamma API to get event info
    url = "https://gamma-api.polymarket.com/events?slug=elon-musk-tweet-tracker"

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'YesNoEvents/1.0'
        })
        ctx = ssl.create_default_context()

        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read())
            return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================
# Dashboard View
# ============================================

def dashboard_view():
    """Main CLI dashboard"""
    from trading import scan_elon_markets, find_ev_opportunities
    from quant import momentum_score

    lines = []

    # Header
    lines.append("\033[1;36m" + "═" * 80 + "\033[0m")
    lines.append("\033[1;36m YES/NO.EVENTS DASHBOARD                    " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\033[0m")
    lines.append("\033[1;36m" + "═" * 80 + "\033[0m")

    # Fetch markets
    try:
        markets = scan_elon_markets()
    except:
        markets = {}

    # Current Events Summary
    lines.append("\n\033[1;33m▶ ACTIVE EVENTS\033[0m")
    lines.append("─" * 50)

    for event, mlist in markets.items():
        if not mlist:
            continue

        total_vol = sum(m['volume'] for m in mlist) / 1000
        peak = max(mlist, key=lambda x: x['yes'])
        peak_pct = peak['yes'] * 100

        lines.append(f"  {event:<12} {len(mlist):>2} brackets  ${total_vol:>6.0f}k  Peak: {peak['bracket']} @ {peak_pct:.0f}%")

    # EV Opportunities
    lines.append("\n\033[1;33m▶ TOP OPPORTUNITIES\033[0m")
    lines.append("─" * 50)

    try:
        opps = find_ev_opportunities(0.03)
        for o in opps[:5]:
            signal_color = "\033[31m" if o['signal'] == 'SHORT' else "\033[32m"
            lines.append(f"  {o['event']:<12} {o['bracket']:<12} {signal_color}{o['signal']:<6}\033[0m edge:{o['edge']*100:+.1f}%")
    except:
        lines.append("  (Unable to calculate)")

    # Momentum Signals
    lines.append("\n\033[1;33m▶ MOMENTUM SIGNALS\033[0m")
    lines.append("─" * 50)

    try:
        scores = momentum_score()
        oversold = [s for s in scores if s['signal'] == 'OVERSOLD'][:3]
        overbought = [s for s in scores if s['signal'] == 'OVERBOUGHT'][:3]

        if oversold:
            lines.append("  \033[32mOVERSOLD (potential longs):\033[0m")
            for s in oversold:
                lines.append(f"    {s['event']:<10} {s['bracket']:<12} mom:{s['momentum']:+.1f}")

        if overbought:
            lines.append("  \033[31mOVERBOUGHT (potential shorts):\033[0m")
            for s in overbought:
                lines.append(f"    {s['event']:<10} {s['bracket']:<12} mom:{s['momentum']:+.1f}")
    except:
        lines.append("  (Unable to calculate)")

    # Recent Calendar (last 3 days)
    lines.append("\n\033[1;33m▶ RECENT TWEET ACTIVITY\033[0m")
    lines.append("─" * 50)

    today = datetime.now()
    for i in range(3):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        counts = get_day_totals(date_str)
        total = sum(c for c in counts if c is not None)

        if total > 0:
            # Mini sparkline
            spark = ""
            for c in counts:
                if c is None or c == 0:
                    spark += "·"
                elif c < 3:
                    spark += "▁"
                elif c < 6:
                    spark += "▃"
                elif c < 10:
                    spark += "▅"
                else:
                    spark += "█"
            lines.append(f"  {d.strftime('%a %d')}: {spark} = {total}")
        else:
            lines.append(f"  {d.strftime('%a %d')}: \033[90mno data\033[0m")

    lines.append("\n" + "═" * 80)
    lines.append("\033[90mCommands: ./tracker.sh cal | add | live | dash\033[0m")

    return '\n'.join(lines)

# ============================================
# CLI Interface
# ============================================

def main():
    parser = argparse.ArgumentParser(description="YES/NO.EVENTS Tweet Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Calendar view
    cal_parser = subparsers.add_parser("cal", help="Show tweet calendar")
    cal_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    cal_parser.add_argument("--days", type=int, default=7, help="Days to show")
    cal_parser.add_argument("--event", help="Event name")
    cal_parser.add_argument("--mini", action="store_true", help="Mini view")

    # Add data
    add_parser = subparsers.add_parser("add", help="Add tweet data")
    add_parser.add_argument("date", help="Date (YYYY-MM-DD)")
    add_parser.add_argument("--hour", type=int, help="Hour (0-23)")
    add_parser.add_argument("--count", type=int, help="Tweet count")
    add_parser.add_argument("--counts", help="24 comma-separated counts for full day")
    add_parser.add_argument("--event", help="Event name")

    # Bulk add
    bulk_parser = subparsers.add_parser("bulk", help="Bulk add from JSON")
    bulk_parser.add_argument("file", help="JSON file with data")

    # Live fetch
    live_parser = subparsers.add_parser("live", help="Fetch live tweet count")
    live_parser.add_argument("--watch", action="store_true", help="Watch mode (refresh every minute)")

    # Dashboard
    dash_parser = subparsers.add_parser("dash", help="Show dashboard")

    # Export
    export_parser = subparsers.add_parser("export", help="Export data")
    export_parser.add_argument("--format", choices=['json', 'csv'], default='json')

    # Import from screenshot data
    import_parser = subparsers.add_parser("import-week", help="Import week of data")
    import_parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    import_parser.add_argument("--event", help="Event name")

    args = parser.parse_args()

    if args.command == "cal":
        start = args.start or (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d")
        if args.mini:
            print(mini_calendar(start, args.days))
        else:
            print(calendar_view(start, args.days, args.event))

    elif args.command == "add":
        if args.counts:
            # Full day
            counts = [int(x.strip()) if x.strip() and x.strip() != '.' else None
                     for x in args.counts.split(',')]
            add_daily_data(args.date, counts, args.event)
            total = sum(c for c in counts if c is not None)
            print(f"Added {total} tweets for {args.date}")
        elif args.hour is not None and args.count is not None:
            # Single hour
            add_hourly_data(args.date, args.hour, args.count, args.event)
            print(f"Added {args.count} tweets for {args.date} {args.hour:02d}:00")
        else:
            print("Specify --hour and --count, or --counts for full day")

    elif args.command == "bulk":
        with open(args.file) as f:
            bulk_data = json.load(f)

        for date_str, counts in bulk_data.items():
            if isinstance(counts, list):
                add_daily_data(date_str, counts)
                print(f"Added {date_str}")
            elif isinstance(counts, dict):
                for hour, count in counts.items():
                    add_hourly_data(date_str, int(hour), count)
                print(f"Added {date_str}")

    elif args.command == "live":
        if args.watch:
            import time
            print("Watching for updates (Ctrl+C to stop)...")
            while True:
                count = fetch_live_count()
                now = datetime.now().strftime("%H:%M:%S")
                if count:
                    print(f"\r[{now}] Current count: {count}    ", end="", flush=True)
                else:
                    print(f"\r[{now}] Unable to fetch    ", end="", flush=True)
                time.sleep(60)
        else:
            count = fetch_live_count()
            if count:
                print(f"Current tweet count: {count}")
            else:
                print("Unable to fetch current count")

    elif args.command == "dash":
        print(dashboard_view())

    elif args.command == "export":
        data = load_tweet_data()
        if args.format == 'json':
            print(json.dumps(data, indent=2))
        else:
            # CSV
            print("date,hour,count,event")
            for key, entry in data.get("hourly", {}).items():
                print(f"{entry['date']},{entry['hour']},{entry['count']},{entry.get('event', '')}")

    elif args.command == "import-week":
        print(f"Interactive import for week starting {args.start_date}")
        print("Enter 24 comma-separated counts for each day (or 'skip' to skip):")
        print("Example: 0,0,0,0,1,2,3,5,4,3,2,1,0,1,2,4,5,3,2,1,0,0,0,0")
        print()

        start = datetime.strptime(args.start_date, "%Y-%m-%d")
        for i in range(7):
            d = start + timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            day_name = d.strftime("%a %b %d")

            inp = input(f"{day_name}: ").strip()
            if inp.lower() == 'skip' or not inp:
                continue

            try:
                counts = [int(x.strip()) if x.strip() and x.strip() != '.' else None
                         for x in inp.split(',')]
                if len(counts) != 24:
                    print(f"  Warning: expected 24 values, got {len(counts)}")
                    counts = counts[:24] + [None] * (24 - len(counts))

                add_daily_data(date_str, counts, args.event)
                total = sum(c for c in counts if c is not None)
                print(f"  ✓ Added {total} tweets")
            except Exception as e:
                print(f"  Error: {e}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
