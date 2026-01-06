import type { NewsEvent } from '@/types/news';

/**
 * Hit detection engine - scans text for keywords from word pool
 */
export class MatchingEngine {
  private wordPool: Set<string>;

  constructor(wordPool: string[]) {
    this.wordPool = new Set(wordPool.map(w => w.toUpperCase()));
  }

  /**
   * Extract matched words from text
   */
  extractMatches(text: string): string[] {
    const upperText = text.toUpperCase();
    const matches: string[] = [];

    // Create word boundaries regex for each word in pool
    for (const word of this.wordPool) {
      const regex = new RegExp(`\\b${word}\\b`, 'g');
      const found = upperText.match(regex);

      if (found && found.length > 0) {
        // Add each occurrence (word can match multiple times)
        for (let i = 0; i < found.length; i++) {
          matches.push(word);
        }
      }
    }

    return matches;
  }

  /**
   * Process news event and return matched words
   */
  processNewsEvent(event: NewsEvent): string[] {
    const fullText = `${event.headline} ${event.content}`;
    return this.extractMatches(fullText);
  }

  /**
   * Count word frequency in text
   */
  countWordFrequency(text: string, word: string): number {
    const upperText = text.toUpperCase();
    const upperWord = word.toUpperCase();
    const regex = new RegExp(`\\b${upperWord}\\b`, 'g');
    const matches = upperText.match(regex);
    return matches ? matches.length : 0;
  }

  /**
   * Update word pool
   */
  updateWordPool(newPool: string[]): void {
    this.wordPool = new Set(newPool.map(w => w.toUpperCase()));
  }
}
