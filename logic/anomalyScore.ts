import { BASELINE_WINDOW, VELOCITY_WINDOW } from '@/lib/constants';
import type { TrendDirection } from '@/types/market';

export interface HitRecord {
  word: string;
  timestamp: number;
}

/**
 * Anomaly scoring engine - detects unusual word frequency patterns
 */
export class AnomalyScorer {
  private hitHistory: Map<string, number[]> = new Map();
  private baselineWindow: number;
  private velocityWindow: number;

  constructor(
    baselineWindow: number = BASELINE_WINDOW,
    velocityWindow: number = VELOCITY_WINDOW
  ) {
    this.baselineWindow = baselineWindow;
    this.velocityWindow = velocityWindow;
  }

  /**
   * Record a hit for a word
   */
  recordHit(word: string, timestamp: number): void {
    const history = this.hitHistory.get(word) || [];
    history.push(timestamp);
    this.hitHistory.set(word, history);
  }

  /**
   * Calculate baseline hit rate (hits per second over baseline window)
   */
  calculateBaseline(word: string, currentTime: number): number {
    const history = this.hitHistory.get(word) || [];
    const baselineHits = history.filter(
      t => currentTime - t <= this.baselineWindow
    );

    return baselineHits.length / (this.baselineWindow / 1000);
  }

  /**
   * Calculate current velocity (hits per second over velocity window)
   */
  calculateVelocity(word: string, currentTime: number): number {
    const history = this.hitHistory.get(word) || [];
    const recentHits = history.filter(
      t => currentTime - t <= this.velocityWindow
    );

    return recentHits.length / (this.velocityWindow / 1000);
  }

  /**
   * Calculate anomaly score (standard deviations from baseline)
   * Returns: (velocity - baseline) / baseline
   */
  calculateAnomaly(word: string, currentTime: number): number {
    const baseline = this.calculateBaseline(word, currentTime);
    const velocity = this.calculateVelocity(word, currentTime);

    if (baseline === 0) {
      return velocity > 0 ? Infinity : 0;
    }

    return (velocity - baseline) / baseline;
  }

  /**
   * Determine trend direction based on velocity changes
   */
  getTrend(word: string, currentTime: number): TrendDirection {
    const anomaly = this.calculateAnomaly(word, currentTime);

    if (anomaly > 0.5) return 'heating';
    if (anomaly < -0.5) return 'cooling';
    return 'stable';
  }

  /**
   * Get total hits for a word
   */
  getTotalHits(word: string): number {
    const history = this.hitHistory.get(word) || [];
    return history.length;
  }

  /**
   * Get hits in current window
   */
  getHitsInWindow(word: string, currentTime: number, windowSize: number): number {
    const history = this.hitHistory.get(word) || [];
    return history.filter(t => currentTime - t <= windowSize).length;
  }

  /**
   * Get last hit time for a word
   */
  getLastHitTime(word: string): number {
    const history = this.hitHistory.get(word) || [];
    return history.length > 0 ? history[history.length - 1] : 0;
  }

  /**
   * Clean up old history beyond baseline window
   */
  cleanupHistory(currentTime: number): void {
    for (const [word, history] of this.hitHistory.entries()) {
      const filtered = history.filter(
        t => currentTime - t <= this.baselineWindow
      );

      if (filtered.length === 0) {
        this.hitHistory.delete(word);
      } else {
        this.hitHistory.set(word, filtered);
      }
    }
  }

  /**
   * Reset all history
   */
  reset(): void {
    this.hitHistory.clear();
  }

  /**
   * Get all words with their statistics
   */
  getAllStats(currentTime: number): Map<string, {
    velocity: number;
    anomaly: number;
    baseline: number;
    totalHits: number;
    trend: TrendDirection;
  }> {
    const stats = new Map();

    for (const word of this.hitHistory.keys()) {
      stats.set(word, {
        velocity: this.calculateVelocity(word, currentTime),
        anomaly: this.calculateAnomaly(word, currentTime),
        baseline: this.calculateBaseline(word, currentTime),
        totalHits: this.getTotalHits(word),
        trend: this.getTrend(word, currentTime),
      });
    }

    return stats;
  }
}
