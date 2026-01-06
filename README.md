# yesno.events - Mention Markets

A real-time prediction market platform where users track word frequencies in news cycles.

## 🎯 Concept

Players manually select 10 keywords from a pool of 50 trending words to form their **Market Positions**. As synthetic news headlines stream in, the system tracks which words are being mentioned and calculates anomaly scores to identify words "heating up" above their baseline frequency.

## 🏗️ Architecture

### Core Components

- **MockNewsProvider**: Generates synthetic news headlines every 2-5 seconds
- **MatchingEngine**: Scans headlines for keyword matches
- **AnomalyScorer**: Calculates velocity and anomaly scores using Z-score methodology
- **Zustand Store**: Real-time state management for fast-paced 30s game loops

### Data Flow

```
MockNewsProvider → NewsEvent → MatchingEngine → Detect Mentions
                                                      ↓
                                              Update Market Positions
                                                      ↓
                                              AnomalyScorer → Calculate Velocity/Anomaly
                                                      ↓
                                              Live Signal Feed
```

## 📊 Key Features

### Manual Word Selection

- Interactive word picker with search functionality
- Organized by 5 categories: Finance, Politics, Tech, Climate, Health
- Visual feedback for selected words
- Confirmation required before starting market

### Anomaly Detection

```
anomalyScore = (currentVelocity - baseline) / baseline

- Baseline: Average mentions/sec over 60 seconds
- Velocity: Current mentions/sec over 10 seconds
- Anomaly > 1.5 = "Heating" trend
```

### Game Mechanics

- **Market Durations**: 30s (testing), 1m, 1d, 7d
- **Word Pool**: 50 trending keywords across Finance, Politics, Tech, Climate, Health
- **Market Positions**: 10 words manually selected by user
- **Prize Pool**: Proportional distribution based on total mentions


## 🎮 How to Play

1. **Select Your Words**: Click 10 keywords from the available pool
   - Use search to filter words
   - Click on words to select/deselect
   - Confirm your selection

2. **Start Market**: Choose duration and click "Start Market"
   - 30s recommended for testing
   - Market begins immediately

3. **Watch Live Feed**: Synthetic news streams in real-time
   - Your selected words glow when mentioned
   - Progress bars fill with each mention
   - Anomaly scores show trending words

4. **Monitor Insights**: Track market signals
   - Spiking words (+150% above baseline)
   - Underperforming picks (0 mentions)
   - Time-sensitive notifications

5. **Market Close**: View your performance
   - Total mentions counted
   - Payout calculated
   - Start new market

## 🧩 Project Structure

```
yesno-events/
├── app/
│   ├── page.tsx          # Main dashboard
│   ├── layout.tsx        # Root layout
│   └── globals.css       # Global styles
├── components/
│   ├── MentionCard.tsx      # Market positions with mention tracking
│   ├── WordSelector.tsx     # Manual word selection interface
│   ├── LiveFeed.tsx         # Terminal-style news feed
│   ├── DashboardHeader.tsx  # Prize pool, timer, top anomalies
│   ├── InsightFooter.tsx    # Market insights and alerts
│   └── RoundControl.tsx     # Market start/stop controls
├── logic/
│   ├── matching.ts       # Keyword matching engine
│   ├── anomalyScore.ts   # Anomaly detection algorithm
│   └── payout.ts         # Payout calculation
├── services/
│   ├── newsProvider.ts   # Abstract base provider
│   ├── mockNewsProvider.ts  # Synthetic news generator
│   └── rssProvider.ts       # RSS adapter (stub)
├── store/
│   └── marketStore.ts    # Zustand state management
├── types/
│   ├── market.ts         # Market/round types
│   └── news.ts           # News event types
└── lib/
    ├── constants.ts      # Word pool, durations, thresholds
    └── utils.ts          # Helper functions
```

## 🔧 Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State**: Zustand
- **Icons**: Lucide React

## 📈 Future Enhancements

### Near-term
- [ ] Multi-player support
- [ ] Leaderboard with historical rankings
- [ ] RSS feed integration (replace mock provider)
- [ ] Custom word pools per market

### Advanced
- [ ] WebSocket for real-time multi-player sync
- [ ] Historical analytics dashboard
- [ ] Machine learning for trend prediction
- [ ] Integration with real news APIs (NewsAPI, Guardian, etc.)
- [ ] Mobile app version

## 🧪 Technical Highlights

### Performance
- Event processing: < 10ms per news event
- UI updates: 60 FPS target
- Memory efficient: Sliding window for mention history

### Scalability
- Modular provider pattern for easy data source swapping
- Zustand for optimized re-renders
- Separation of concerns (logic/services/components)

## 📝 Configuration

Edit `lib/constants.ts` to customize:

```typescript
WORD_POOL           // 50 trending keywords
ROUND_DURATIONS     // Time intervals
BASELINE_WINDOW     // 60s for anomaly baseline
VELOCITY_WINDOW     // 10s for current velocity
ANOMALY_THRESHOLD   // 1.5 = 150% above baseline
```

## 🎨 UI/UX Design

- **Dark Theme**: Optimized for extended viewing
- **Terminal Aesthetics**: Monospace feed for technical vibe
- **Real-time Animations**: Glow effects on mentions, pulse on active words
- **Minimal Footprint**: One-screen dashboard with all key info
- **Interactive Selection**: Search, filter, and visual feedback

## 🧠 Anomaly Score Deep Dive

The anomaly detection uses a **moving baseline** approach:

1. **Track mention timestamps** for each word
2. **Calculate baseline**: Average frequency over last 60 seconds
3. **Calculate velocity**: Current frequency over last 10 seconds
4. **Compute Z-score**: `(velocity - baseline) / baseline`
5. **Classify trend**:
   - `> +0.5`: Heating 🔥
   - `< -0.5`: Cooling ❄️
   - Otherwise: Stable

This allows the system to detect both **absolute spikes** (high velocity) and **relative anomalies** (sudden changes from baseline).

## 📄 License

MIT

## 👤 Author

Built for yesno.events - Mention Markets platform

---

**Note**: This MVP uses synthetic data. For production, integrate with live RSS feeds or news APIs.
