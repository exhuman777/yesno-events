export type RoundInterval = '30s' | '1m' | '1d' | '7d';
export type RoundStatus = 'waiting' | 'active' | 'completed';
export type TrendDirection = 'heating' | 'cooling' | 'stable';

export interface MarketRound {
  id: string;
  interval: RoundInterval;
  startTime: number;
  endTime: number;
  prizePool: number;
  status: RoundStatus;
  wordPool: string[];
  participants: Participant[];
}

export interface Participant {
  id: string;
  name: string;
  buyIn: number;
  bingoCard: BingoCard;
  totalHits: number;
  payout: number;
}

export interface BingoCard {
  words: string[];
  hits: Record<string, number>;
  lastHitTime: Record<string, number>;
}

export interface WordStats {
  word: string;
  totalHits: number;
  hitsInWindow: number;
  velocity: number;
  anomalyScore: number;
  baseline: number;
  lastHitTime: number;
  trend: TrendDirection;
}

export interface LiveSignal {
  timestamp: number;
  word: string;
  velocity: number;
  anomalyScore: number;
  trend: TrendDirection;
}

export interface PayoutResult {
  participantId: string;
  payout: number;
  rank: number;
  hitShare: number;
}
