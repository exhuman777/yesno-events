import { BaseNewsProvider } from './newsProvider';
import type { NewsEvent, MockNewsConfig } from '@/types/news';
import { generateId } from '@/lib/utils';

/**
 * Mock news provider that generates synthetic headlines
 */
export class MockNewsProvider extends BaseNewsProvider {
  private config: MockNewsConfig;
  private intervalId: NodeJS.Timeout | null = null;

  // Sentence templates with {WORD} placeholders
  private templates = [
    'Breaking: {WORD} announcement shakes global markets',
    'Experts predict {WORD} will dominate headlines this week',
    'New study reveals shocking {WORD} statistics',
    'Major breakthrough in {WORD} technology unveiled',
    'Debate intensifies over {WORD} policy changes',
    'Industry leaders discuss future of {WORD}',
    '{WORD} impact exceeds all expectations',
    'Reports indicate {WORD} reaching record levels',
    'Analysts warn of potential {WORD} disruption',
    'Government announces new {WORD} initiative',
    '{WORD} concerns spark international response',
    'Latest {WORD} data shows surprising trends',
    'Scientists discover link between {WORD} and climate',
    '{WORD} market sees unprecedented growth',
    'Public opinion shifts on {WORD} issues',
    'Tech giants invest heavily in {WORD} research',
    '{WORD} regulation faces fierce opposition',
    'Emergency summit called to address {WORD} crisis',
    'Innovative {WORD} solution gains traction',
    '{WORD} becomes central to election campaign',
    'Experts divided on {WORD} implications',
    'New {WORD} standards set to transform industry',
    '{WORD} scandal rocks financial sector',
    'Breakthrough {WORD} treatment shows promise',
    'Global {WORD} summit concludes with agreement',
  ];

  constructor(config: MockNewsConfig) {
    super();
    this.config = config;
  }

  /**
   * Generate a random headline using words from the pool
   */
  private generateHeadline(): string {
    const template = this.templates[Math.floor(Math.random() * this.templates.length)];

    // Replace placeholders with random words from pool
    let headline = template;
    while (headline.includes('{WORD}')) {
      const word = this.config.wordPool[Math.floor(Math.random() * this.config.wordPool.length)];
      headline = headline.replace('{WORD}', word);
    }

    return headline;
  }

  /**
   * Generate synthetic content with multiple keyword mentions
   */
  private generateContent(): string {
    const sentences = Math.floor(Math.random() * 3) + 2; // 2-4 sentences
    const content: string[] = [];

    for (let i = 0; i < sentences; i++) {
      const template = this.templates[Math.floor(Math.random() * this.templates.length)];
      let sentence = template;

      while (sentence.includes('{WORD}')) {
        const word = this.config.wordPool[Math.floor(Math.random() * this.config.wordPool.length)];
        sentence = sentence.replace('{WORD}', word);
      }

      content.push(sentence);
    }

    return content.join('. ') + '.';
  }

  /**
   * Generate a complete news event
   */
  private generateEvent(): NewsEvent {
    const headline = this.generateHeadline();
    const content = this.generateContent();

    return {
      id: generateId(),
      timestamp: Date.now(),
      headline,
      content,
      matchedWords: [], // Will be filled by matching engine
      source: 'mock',
    };
  }

  /**
   * Schedule next event generation
   */
  private scheduleNext(): void {
    if (!this.isRunning) return;

    const delay = Math.random() *
      (this.config.maxInterval - this.config.minInterval) +
      this.config.minInterval;

    this.intervalId = setTimeout(() => {
      const event = this.generateEvent();
      this.notify(event);
      this.scheduleNext();
    }, delay);
  }

  /**
   * Start generating mock news
   */
  start(): void {
    if (this.isRunning) return;

    this.isRunning = true;
    this.scheduleNext();
  }

  /**
   * Stop generating mock news
   */
  stop(): void {
    this.isRunning = false;

    if (this.intervalId) {
      clearTimeout(this.intervalId);
      this.intervalId = null;
    }
  }
}
