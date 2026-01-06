export const WORD_POOL: string[] = [
  // Finance (10)
  'BITCOIN', 'STOCKS', 'INFLATION', 'FEDERAL', 'MARKET',
  'BANKING', 'CRYPTO', 'DEBT', 'NASDAQ', 'YIELD',

  // Politics (10)
  'ELECTION', 'CONGRESS', 'SENATE', 'PRESIDENT', 'VOTE',
  'POLL', 'CAMPAIGN', 'REFORM', 'BILL', 'POLICY',

  // Tech (10)
  'AI', 'OPENAI', 'GOOGLE', 'APPLE', 'META',
  'TESLA', 'CHIP', 'QUANTUM', 'ROBOT', 'CYBERSECURITY',

  // Climate (10)
  'CLIMATE', 'CARBON', 'EMISSION', 'RENEWABLE', 'SOLAR',
  'WIND', 'TEMPERATURE', 'DROUGHT', 'FLOOD', 'WILDFIRE',

  // Health (10)
  'VACCINE', 'PANDEMIC', 'FDA', 'DRUG', 'HOSPITAL',
  'MEDICARE', 'CANCER', 'TRIAL', 'OUTBREAK', 'VIRUS',
];

export const ROUND_DURATIONS: Record<string, number> = {
  '30s': 30 * 1000,
  '1m': 60 * 1000,
  '1d': 24 * 60 * 60 * 1000,
  '7d': 7 * 24 * 60 * 60 * 1000,
};

export const DEFAULT_BUY_IN = 100;
export const BINGO_CARD_SIZE = 10;
export const BASELINE_WINDOW = 60000; // 1 minute
export const VELOCITY_WINDOW = 10000; // 10 seconds
export const ANOMALY_THRESHOLD = 1.5; // 150% above baseline

export const MOCK_NEWS_MIN_INTERVAL = 2000; // 2 seconds
export const MOCK_NEWS_MAX_INTERVAL = 5000; // 5 seconds
