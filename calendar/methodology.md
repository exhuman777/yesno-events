---
updated: 2026-01-11
auto_update: true
sources:
  - xtracker (elon tweets)
  - polyrouter (market odds)
  - manual (events, insights)
---

# Auto-Update Methodology

## Data Sources

### 1. XTracker (Elon Tweets)
- Source: Twitter/X API via xtracker
- Update frequency: 5 min
- Dashboard endpoint: `/?view=elon`
- Storage: In-memory + cache

### 2. Polyrouter (Polymarket)
- Source: api-v2.polyrouter.io
- Update frequency: On request
- Dashboard endpoint: `/?view=markets`
- Storage: `data/cache/markets/*.json`

### 3. Manual Files (Calendar)
- Source: Local .md files
- Update: Edit files directly
- Location: `calendar/` directory

## File Update Patterns

### Daily News (`calendar/news/YYYY-MM-DD.md`)

Auto-generated section at bottom:
```markdown
## Auto-Generated (API)

### Polymarket Activity
```json
{
  "last_sync": "2026-01-11T18:00:00Z",
  "elon_markets": {
    "jan6_13": {
      "top_bracket": "580+",
      "top_odds": 0.78
    }
  }
}
```
```

Manual sections:
- High Impact events
- Personal observations
- Market inefficiencies

### Monthly Events (`calendar/events/YYYY-MM.md`)

Structure per day:
```yaml
### 2026-01-14
- type: ECONOMIC
  title: CPI Release
  time: 08:30
  impact: HIGH
  markets: [cpi-jan-2026]
```

Add events manually. Types auto-styled.

### Watched Markets (`calendar/markets/watched.md`)

Track positions:
```yaml
### market-slug
- status: WATCHING | OPEN | CLOSED
- entry_price: 0.XX
- current_price: 0.XX
- thesis: Why you entered
- exit_plan: When to exit
```

Update `current_price` and `status` as positions change.

## Automation Options

### Cron Jobs

```bash
# Update elon count every 5 min
*/5 * * * * curl -s "localhost:5556/api/xtracker/sync"

# Daily news file creation
0 6 * * * python3 scripts/create_daily_news.py
```

### RSS Feeds

Configure in `feeds/sources.md`:
```yaml
## Sources
- url: https://feeds.reuters.com/reuters/topNews
  category: news
  check_interval: 1h
```

### Webhooks

Dashboard accepts POST to `/webhook`:
```json
{
  "type": "market_update",
  "market_id": "1093290",
  "new_odds": 0.82
}
```

## Update Frequency Guidelines

| Data Type | Frequency | Method |
|-----------|-----------|--------|
| Elon count | 5 min | Auto (xtracker) |
| Market odds | On demand | API call |
| Events | Manual | Edit .md |
| News | Daily | Create file |
| Positions | As traded | Edit watched.md |

## File Naming

| Type | Pattern | Example |
|------|---------|---------|
| Events | `events/YYYY-MM.md` | `events/2026-01.md` |
| News | `news/YYYY-MM-DD.md` | `news/2026-01-11.md` |
| Insights | `insights/*.md` | `insights/elon-tweets.md` |

## Dashboard Refresh

Dashboard reads files on each HTTP request:
1. Open browser to `localhost:5556`
2. Edit .md file in calendar/
3. Refresh browser - changes appear

No restart needed for file changes.

## Data Validation

Files parsed with fallbacks:
- Missing frontmatter: Use defaults
- Invalid YAML: Skip entry, continue
- Missing file: Show placeholder

Check console for parse errors.

## Performance

- File reads cached per-request
- API calls cached in `data/cache/`
- Cache TTL: 5 min for prices, 1h for markets
- Clear cache: Delete files in `data/cache/`

---

## Quick Reference

```bash
# Start dashboard
python3 dashboard4all.py

# Add event
vim calendar/events/2026-01.md

# Add position
vim calendar/markets/watched.md

# Check elon count
curl localhost:5556/?view=elon

# Search markets
open "localhost:5556/?view=markets&q=trump"
```
