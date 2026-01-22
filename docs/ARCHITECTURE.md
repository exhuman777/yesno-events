# YES/NO.EVENTS - Project Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YES/NO.EVENTS TERMINAL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Markets   │  │   Trading   │  │  Portfolio  │  │  Elon Lab   │        │
│  │   Screen    │  │   Screen    │  │   Screen    │  │   Screen    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐        │
│  │  Research   │  │  Analytics  │  │ API Explorer│  │  Settings   │        │
│  │   Screen    │  │   Screen    │  │   Screen    │  │   Screen    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         └────────────────┴────────────────┴────────────────┘               │
│                                    │                                        │
│                          ┌─────────┴─────────┐                             │
│                          │   App State (Global)   │                        │
│                          │   - logged_in         │                        │
│                          │   - wallet            │                        │
│                          │   - theme             │                        │
│                          └─────────┬─────────┘                             │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────┐
│                           TRADING MODULE                                    │
├────────────────────────────────────┼────────────────────────────────────────┤
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐ │
│  │                          API LAYER                                     │ │
│  ├───────────────────┬───────────────────┬───────────────────────────────┤ │
│  │                   │                   │                               │ │
│  │  ┌─────────────┐  │  ┌─────────────┐  │  ┌─────────────┐             │ │
│  │  │  Gamma API  │  │  │  CLOB API   │  │  │  Data API   │             │ │
│  │  │  (Markets)  │  │  │ (Orderbook) │  │  │ (Positions) │             │ │
│  │  └─────────────┘  │  └─────────────┘  │  └─────────────┘             │ │
│  │                   │                   │                               │ │
│  │  ┌─────────────┐  │  ┌─────────────┐  │  ┌─────────────┐             │ │
│  │  │  XTracker   │  │  │ Polyrouter  │  │  │    RTDS     │             │ │
│  │  │  (Users)    │  │  │  (Search)   │  │  │ (WebSocket) │             │ │
│  │  └─────────────┘  │  └─────────────┘  │  └─────────────┘             │ │
│  │                   │                   │                               │ │
│  └───────────────────┴───────────────────┴───────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│   │ gamma-api.      │  │ clob.           │  │ data-api.       │           │
│   │ polymarket.com  │  │ polymarket.com  │  │ polymarket.com  │           │
│   │                 │  │                 │  │                 │           │
│   │ • Markets       │  │ • Orderbooks    │  │ • Positions     │           │
│   │ • Events        │  │ • Orders        │  │ • Trades        │           │
│   │ • Comments      │  │ • Trades        │  │ • Leaderboard   │           │
│   │ • Profiles      │  │ • Spreads       │  │ • Activity      │           │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐                                │
│   │ xtracker.       │  │ ws-live-data.   │                                │
│   │ polymarket.com  │  │ polymarket.com  │                                │
│   │                 │  │                 │                                │
│   │ • Users         │  │ • Crypto prices │                                │
│   │ • Trackings     │  │ • Comments      │                                │
│   │ • Metrics       │  │ • Real-time     │                                │
│   └─────────────────┘  └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
dashboard4all/
├── app.py                 # Main terminal application (Textual TUI)
├── trading.py             # All API functions and trading logic
├── web_app.py            # Flask web application (REST API)
├── requirements.txt       # Python dependencies
│
├── .venv/                # Python virtual environment
│   └── ...
│
├── data/                 # Data storage
│   ├── cache/           # API response cache
│   │   ├── markets/     # Market data cache
│   │   ├── trades/      # Trade data cache
│   │   └── prices/      # Price history cache
│   ├── .last_api_update.json
│   ├── .research_history.json
│   └── .yesno_config.json
│
├── templates/            # Flask HTML templates
│   └── ...
│
├── tests/               # Test suite
│   ├── test_trading_api.py   # API tests
│   ├── test_app_ui.py        # UI tests
│   └── __init__.py
│
├── docs/                # Documentation
│   ├── USER_STORIES.md  # User stories & examples
│   ├── ARCHITECTURE.md  # This file
│   └── API_REFERENCE.md # API documentation
│
├── index.html           # Promotional website
├── README.md            # GitHub readme
└── ABOUT.md            # Project description
```

---

## Component Details

### 1. Terminal Application (app.py)

```
app.py
├── AppState (Global State)
│   ├── logged_in: bool
│   ├── wallet: str
│   ├── private_key: str
│   ├── theme: str
│   └── THEMES: dict
│
├── Helper Functions
│   ├── sparkline()      # ASCII sparkline chart
│   ├── progress_bar()   # ASCII progress bar
│   ├── fmt_pct()        # Format percentage
│   ├── fmt_money()      # Format currency
│   ├── fmt_price()      # Format price in cents
│   └── depth_bar()      # Orderbook depth bar
│
├── Screens
│   ├── MainMenuScreen
│   │   └── Hub for all navigation
│   │
│   ├── MarketsScreen
│   │   ├── Market browsing
│   │   ├── Search functionality
│   │   └── Category filters
│   │
│   ├── MarketDetailScreen (Modal)
│   │   ├── Price display
│   │   ├── Orderbook depth
│   │   ├── Recent trades
│   │   ├── Top holders
│   │   ├── Comments
│   │   └── Trading form
│   │
│   ├── TradingScreen
│   │   ├── Order placement
│   │   ├── Order management
│   │   └── Execution feedback
│   │
│   ├── PortfolioScreen
│   │   ├── Positions tab
│   │   ├── Trades tab
│   │   ├── Activity tab
│   │   └── Orders tab
│   │
│   ├── ElonLabScreen
│   │   ├── Live stats
│   │   ├── Hourly heatmap
│   │   ├── Daily patterns
│   │   └── Behavior analysis
│   │
│   ├── ResearchScreen
│   │   ├── Leaderboard tab
│   │   ├── Tags tab
│   │   ├── Patterns tab
│   │   └── Correlations tab
│   │
│   ├── AnalyticsScreen
│   │   ├── Spreads analysis
│   │   ├── Momentum indicators
│   │   └── Volume analysis
│   │
│   ├── APIExplorerScreen
│   │   └── All 50+ endpoints
│   │
│   └── SettingsScreen
│       ├── Wallet tab
│       ├── Theme tab
│       └── Display tab
│
└── YesNoApp (Main Application)
    └── Screen management
```

### 2. Trading Module (trading.py)

```
trading.py
├── Configuration
│   ├── GAMMA_API = "https://gamma-api.polymarket.com"
│   ├── CLOB_API = "https://clob.polymarket.com"
│   ├── DATA_API = "https://data-api.polymarket.com"
│   ├── RTDS_WS = "wss://ws-live-data.polymarket.com"
│   └── XTRACKER_API = "https://xtracker.polymarket.com"
│
├── Gamma API Functions
│   ├── gamma_search(query)
│   ├── gamma_get_market(id)
│   ├── gamma_get_event(id)
│   ├── gamma_list_tags()
│   ├── gamma_get_profile(address)
│   ├── gamma_get_public_profile(address)
│   ├── gamma_get_comments(entity_type, id)
│   └── gamma_get_market_resolution(id)
│
├── CLOB API Functions
│   ├── get_orderbook(market_id, outcome)
│   ├── get_spread(market_id)
│   ├── get_orderbook_depth(market_id)
│   ├── get_market_trades(market_id)
│   ├── clob_get_spreads(market_ids)
│   └── place_order(market, side, price, size)
│
├── Data API v2 Functions
│   ├── dataapi_get_positions(wallet)
│   ├── dataapi_get_trades(wallet, market)
│   ├── dataapi_get_activity(wallet)
│   ├── dataapi_get_value(wallet)
│   ├── dataapi_get_holders(market_id)
│   └── dataapi_get_leaderboard(category, period)
│
├── XTracker Functions
│   ├── xtracker_get_users()
│   ├── xtracker_get_user(id)
│   ├── xtracker_get_trackings(user_id)
│   ├── xtracker_get_tracking(id)
│   ├── xtracker_get_metrics()
│   └── xtracker_get_all_trackings()
│
├── Analysis Functions
│   ├── get_price_history(market_id, interval)
│   ├── find_ev_opportunities(min_volume)
│   ├── get_live_volume(market_id)
│   └── get_open_interest(market_id)
│
├── Elon Functions
│   ├── load_elon_historic_tweets()
│   ├── load_elon_daily_metrics()
│   ├── analyze_elon_patterns()
│   └── scan_elon_markets()
│
├── Account Functions
│   ├── get_positions(wallet)
│   ├── get_balances(wallet)
│   ├── get_open_orders(wallet)
│   └── get_user_trades(wallet)
│
└── RTDSClient (WebSocket)
    ├── connect()
    ├── disconnect()
    ├── subscribe_crypto_prices(symbols)
    ├── subscribe_comments(auth)
    └── receive(timeout)
```

### 3. Web Application (web_app.py)

```
web_app.py
├── Flask Application
│   └── Routes
│       ├── / (index)
│       ├── /api/markets
│       ├── /api/market/<id>
│       ├── /api/elon/stats
│       └── /api/portfolio/<wallet>
│
└── Templates
    └── index.html
```

---

## Data Flow

### Market Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Gamma     │───▶│   Cache     │───▶│   Screen    │───▶│   Display   │
│   API       │    │   (JSON)    │    │   Logic     │    │   (Rich)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Trading Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│   Order     │───▶│   CLOB      │───▶│   Response  │
│   Input     │    │   Params    │    │   API       │    │   Handler   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Real-Time Data Flow
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   RTDS      │───▶│   WebSocket │───▶│   Event     │───▶│   UI        │
│   Server    │    │   Client    │    │   Handler   │    │   Update    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## API Endpoint Map

### Gamma API (gamma-api.polymarket.com)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /markets | GET | List/search markets |
| /markets/{id} | GET | Get market details |
| /events | GET | List events |
| /events/{id} | GET | Get event details |
| /tags | GET | List all tags |
| /profiles/{address} | GET | Get user profile |
| /public-profile | GET | Get public profile |
| /comments | GET | Get comments |

### CLOB API (clob.polymarket.com)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /book | GET | Get orderbook |
| /spread | GET | Get market spread |
| /trades | GET | Get recent trades |
| /orders | POST | Place order |
| /orders/{id} | DELETE | Cancel order |
| /user/orders | GET | Get user orders |

### Data API v2 (data-api.polymarket.com)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /positions | GET | Get user positions |
| /trades | GET | Get trades |
| /activity | GET | Get on-chain activity |
| /value | GET | Get portfolio value |
| /holders | GET | Get market holders |
| /v1/leaderboard | GET | Get trader leaderboard |

### XTracker API (xtracker.polymarket.com)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /users | GET | List tracked users |
| /users/{id} | GET | Get user details |
| /trackings | GET | Get all trackings |
| /trackings/{id} | GET | Get tracking details |
| /metrics | GET | Get platform metrics |

---

## Technology Stack

### Frontend (Terminal)
- **Python 3.10+**: Core language
- **Textual**: TUI framework
- **Rich**: Terminal formatting

### Backend
- **Flask**: Web framework (optional)
- **urllib**: HTTP requests
- **websocket-client**: RTDS connection

### Data Storage
- **JSON files**: Local cache
- **File-based config**: User settings

### Testing
- **pytest**: Test framework
- **unittest.mock**: Mocking

---

## Performance Considerations

### Caching Strategy
- Market data: 5-minute TTL
- User data: 1-minute TTL
- Static data: 1-hour TTL

### Rate Limiting
- Gamma API: 100 req/min
- CLOB API: 50 req/min
- Data API: 100 req/min

### Optimization
- Lazy loading for screens
- Background data fetching
- Pagination for large lists

---

## Security

### Wallet Security
- Private keys stored in memory only
- No logging of sensitive data
- Secure environment variable handling

### API Security
- HTTPS for all connections
- API key rotation (when applicable)
- Request signing for orders

---

## Future Architecture (Paid Tier)

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│   │  WebSocket  │    │    ML       │    │  Automation │       │
│   │   Server    │    │   Models    │    │   Engine    │       │
│   └─────────────┘    └─────────────┘    └─────────────┘       │
│          │                 │                  │                │
│          └─────────────────┼──────────────────┘                │
│                            │                                    │
│                   ┌────────┴────────┐                          │
│                   │   API Gateway   │                          │
│                   └────────┬────────┘                          │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Terminal App   │
                    │   (Enhanced)    │
                    └─────────────────┘
```
