import { create } from 'zustand';
import type {
  MarketRound,
  RoundInterval,
  WordStats,
  LiveSignal,
  Participant,
  BingoCard,
} from '@/types/market';
import type { NewsEvent } from '@/types/news';
import { MatchingEngine } from '@/logic/matching';
import { AnomalyScorer } from '@/logic/anomalyScore';
import { calculatePayouts } from '@/logic/payout';
import { MockNewsProvider } from '@/services/mockNewsProvider';
import {
  WORD_POOL,
  ROUND_DURATIONS,
  DEFAULT_BUY_IN,
  BINGO_CARD_SIZE,
  MOCK_NEWS_MIN_INTERVAL,
  MOCK_NEWS_MAX_INTERVAL,
} from '@/lib/constants';
import { generateId, selectRandom } from '@/lib/utils';

interface MarketStore {
  // State
  currentRound: MarketRound | null;
  wordStats: Map<string, WordStats>;
  liveSignals: LiveSignal[];
  newsEvents: NewsEvent[];
  timeRemaining: number;
  currentPlayer: Participant | null;

  // Services
  newsProvider: MockNewsProvider | null;
  matchingEngine: MatchingEngine;
  anomalyScorer: AnomalyScorer;

  // Actions
  startRound: (interval: RoundInterval) => void;
  processNewsEvent: (event: NewsEvent) => void;
  selectBingoCard: (words: string[]) => void;
  endRound: () => void;
  updateTimer: () => void;
  reset: () => void;
  initializeServices: () => void;
}

export const useMarketStore = create<MarketStore>((set, get) => ({
  // Initial state
  currentRound: null,
  wordStats: new Map(),
  liveSignals: [],
  newsEvents: [],
  timeRemaining: 0,
  currentPlayer: null,

  // Initialize services
  newsProvider: null,
  matchingEngine: new MatchingEngine(WORD_POOL),
  anomalyScorer: new AnomalyScorer(),

  initializeServices: () => {
    const { newsProvider } = get();

    if (!newsProvider) {
      const provider = new MockNewsProvider({
        minInterval: MOCK_NEWS_MIN_INTERVAL,
        maxInterval: MOCK_NEWS_MAX_INTERVAL,
        wordPool: WORD_POOL,
      });

      // Subscribe to news events
      provider.subscribe((event) => {
        get().processNewsEvent(event);
      });

      set({ newsProvider: provider });
    }
  },

  startRound: (interval: RoundInterval) => {
    const { newsProvider, matchingEngine, anomalyScorer, currentPlayer } = get();

    if (!currentPlayer) {
      console.error('No player selected');
      return;
    }

    // Reset anomaly scorer
    anomalyScorer.reset();

    // Seed baseline data with synthetic history for realistic anomaly detection
    const startTime = Date.now();
    anomalyScorer.seedBaseline(WORD_POOL, startTime);

    // Create new round
    const duration = ROUND_DURATIONS[interval];
    const endTime = startTime + duration;

    const round: MarketRound = {
      id: generateId(),
      interval,
      startTime,
      endTime,
      prizePool: DEFAULT_BUY_IN, // Single player for MVP
      status: 'active',
      wordPool: WORD_POOL,
      participants: [currentPlayer],
    };

    set({
      currentRound: round,
      wordStats: new Map(),
      liveSignals: [],
      newsEvents: [],
      timeRemaining: Math.floor(duration / 1000),
    });

    // Start news provider
    if (newsProvider) {
      newsProvider.start();
    }

    // Start timer
    const timerInterval = setInterval(() => {
      const { currentRound } = get();

      if (!currentRound || currentRound.status !== 'active') {
        clearInterval(timerInterval);
        return;
      }

      const remaining = Math.max(0, Math.floor((currentRound.endTime - Date.now()) / 1000));

      set({ timeRemaining: remaining });

      if (remaining === 0) {
        clearInterval(timerInterval);
        get().endRound();
      }
    }, 1000);
  },

  processNewsEvent: (event: NewsEvent) => {
    const {
      currentRound,
      matchingEngine,
      anomalyScorer,
      wordStats,
      liveSignals,
      newsEvents,
      currentPlayer,
    } = get();

    if (!currentRound || currentRound.status !== 'active' || !currentPlayer) {
      return;
    }

    // Extract matched words
    const matches = matchingEngine.processNewsEvent(event);
    event.matchedWords = matches;

    // Process each match
    const currentTime = Date.now();
    const newStats = new Map(wordStats);
    const newSignals: LiveSignal[] = [];

    matches.forEach((word) => {
      // Record hit in anomaly scorer
      anomalyScorer.recordHit(word, currentTime);

      // Update player's bingo card if word is in their card
      if (currentPlayer.bingoCard.words.includes(word)) {
        currentPlayer.bingoCard.hits[word] = (currentPlayer.bingoCard.hits[word] || 0) + 1;
        currentPlayer.bingoCard.lastHitTime[word] = currentTime;
        currentPlayer.totalHits++;
      }

      // Calculate stats
      const velocity = anomalyScorer.calculateVelocity(word, currentTime);
      const anomaly = anomalyScorer.calculateAnomaly(word, currentTime);
      const baseline = anomalyScorer.calculateBaseline(word, currentTime);
      const trend = anomalyScorer.getTrend(word, currentTime);

      const stats: WordStats = {
        word,
        totalHits: anomalyScorer.getTotalHits(word),
        hitsInWindow: anomalyScorer.getHitsInWindow(word, currentTime, 10000),
        velocity,
        anomalyScore: anomaly,
        baseline,
        lastHitTime: currentTime,
        trend,
      };

      newStats.set(word, stats);

      // Create live signal for significant anomalies
      if (anomaly > 0.5) {
        newSignals.push({
          timestamp: currentTime,
          word,
          velocity,
          anomalyScore: anomaly,
          trend,
        });
      }
    });

    // Update state
    set({
      wordStats: newStats,
      liveSignals: [...newSignals, ...liveSignals].slice(0, 10), // Keep last 10
      newsEvents: [event, ...newsEvents].slice(0, 50), // Keep last 50
      currentPlayer,
    });
  },

  selectBingoCard: (words: string[]) => {
    if (words.length !== BINGO_CARD_SIZE) {
      console.error(`Must select exactly ${BINGO_CARD_SIZE} words`);
      return;
    }

    const bingoCard: BingoCard = {
      words,
      hits: {},
      lastHitTime: {},
    };

    const player: Participant = {
      id: 'player-1',
      name: 'You',
      buyIn: DEFAULT_BUY_IN,
      bingoCard,
      totalHits: 0,
      payout: 0,
    };

    set({ currentPlayer: player });
  },

  endRound: () => {
    const { currentRound, newsProvider } = get();

    if (!currentRound) return;

    // Stop news provider
    if (newsProvider) {
      newsProvider.stop();
    }

    // Calculate payouts
    const payoutResults = calculatePayouts(
      currentRound.participants,
      currentRound.prizePool
    );

    // Update participants with payouts
    currentRound.participants.forEach((p) => {
      const result = payoutResults.find((r) => r.participantId === p.id);
      if (result) {
        p.payout = result.payout;
      }
    });

    currentRound.status = 'completed';

    set({ currentRound });
  },

  updateTimer: () => {
    const { currentRound } = get();

    if (!currentRound || currentRound.status !== 'active') {
      return;
    }

    const remaining = Math.max(0, Math.floor((currentRound.endTime - Date.now()) / 1000));
    set({ timeRemaining: remaining });
  },

  reset: () => {
    const { newsProvider, anomalyScorer } = get();

    if (newsProvider) {
      newsProvider.stop();
    }

    anomalyScorer.reset();

    set({
      currentRound: null,
      wordStats: new Map(),
      liveSignals: [],
      newsEvents: [],
      timeRemaining: 0,
      currentPlayer: null,
    });
  },
}));

// Helper to auto-select random bingo card (for MVP)
export function autoSelectBingoCard(): void {
  const words = selectRandom(WORD_POOL, BINGO_CARD_SIZE);
  useMarketStore.getState().selectBingoCard(words);
}
