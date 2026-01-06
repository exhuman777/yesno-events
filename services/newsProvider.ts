import type { NewsEvent, NewsProvider } from '@/types/news';

/**
 * Abstract base class for news providers
 * Implements the adapter pattern for different news sources
 */
export abstract class BaseNewsProvider implements NewsProvider {
  protected subscribers: Set<(event: NewsEvent) => void> = new Set();
  protected isRunning: boolean = false;

  /**
   * Subscribe to news events
   */
  subscribe(callback: (event: NewsEvent) => void): () => void {
    this.subscribers.add(callback);

    // Return unsubscribe function
    return () => {
      this.subscribers.delete(callback);
    };
  }

  /**
   * Notify all subscribers of a new event
   */
  protected notify(event: NewsEvent): void {
    this.subscribers.forEach(callback => callback(event));
  }

  /**
   * Start the news provider
   */
  abstract start(): void;

  /**
   * Stop the news provider
   */
  abstract stop(): void;
}
