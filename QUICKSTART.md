# Quick Start Guide

## Installation

```bash
cd bingo-market
npm install
```

## Development Mode

```bash
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000)

## How to Use

### 1. Start a Round
- Click the "Start Round (30s)" button
- A random Bingo card with 10 words will be auto-selected
- The 30-second timer will begin

### 2. Watch the Action
- **Live Feed (right)**: Synthetic news headlines stream every 2-5 seconds
- **Bingo Card (left)**: Your 10 selected words
  - Words glow green when they get a hit
  - Progress bars fill up with each mention
  - Anomaly scores show trending words

### 3. Monitor Insights
- **Dashboard Header**: Prize pool, time remaining, top 3 trending words
- **Insight Footer**: Alerts for spiking words and underperforming picks

### 4. Round Complete
- After 30 seconds, see your total hits
- View your payout (based on hit distribution)
- Click "New Round" to play again

## Tips

- Words with +150% anomaly score are "heating up" 🔥
- Watch for words that spike early in the round
- Multiple occurrences of a word in one headline count separately
- The longer a word goes without hits, the more it's underperforming

## Testing Different Intervals

Try different round durations:
- **30s**: Fast-paced testing
- **1m**: Standard gameplay
- **1d**: Long-term trend analysis (mock data only)
- **7d**: Weekly trends (mock data only)

## Troubleshooting

### No news events appearing
- Make sure you clicked "Start Round"
- Check browser console for errors
- Refresh the page and try again

### Round won't start
- Wait for any existing round to complete
- Click "Stop Round" to reset
- Ensure JavaScript is enabled

### Performance issues
- Close other browser tabs
- Reduce round duration to 30s
- Clear browser cache

## Production Build

```bash
npm run build
npm start
```

## Architecture Notes

- **State Management**: Zustand (real-time updates)
- **News Generation**: 2-5 second intervals
- **Anomaly Detection**: 60s baseline, 10s velocity window
- **Word Pool**: 50 keywords across 5 categories

Enjoy the Bingo Mention Market! 🎯
