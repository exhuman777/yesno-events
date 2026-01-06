export type NewsSource = 'mock' | 'rss';

export interface NewsEvent {
  id: string;
  timestamp: number;
  headline: string;
  content: string;
  matchedWords: string[];
  source: NewsSource;
}

export interface NewsProvider {
  start(): void;
  stop(): void;
  subscribe(callback: (event: NewsEvent) => void): () => void;
}

export interface MockNewsConfig {
  minInterval: number; // milliseconds
  maxInterval: number; // milliseconds
  wordPool: string[];
}

export interface RSSFeedConfig {
  url: string;
  refreshInterval: number; // milliseconds
}
