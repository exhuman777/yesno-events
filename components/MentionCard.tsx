'use client';

import { useMarketStore } from '@/store/marketStore';
import { formatAnomalyScore } from '@/lib/utils';
import { TrendingUp, Circle } from 'lucide-react';

export function MentionCard() {
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const wordStats = useMarketStore((state) => state.wordStats);

  if (!currentPlayer) {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Your Market Positions</h2>
        <p className="text-zinc-500">Select your words to begin tracking</p>
      </div>
    );
  }

  const { bingoCard, totalHits } = currentPlayer;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Your Market Positions</h2>
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-green-500" />
          <span className="text-sm font-medium">{totalHits} mentions</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {bingoCard.words.map((word) => {
          const hits = bingoCard.hits[word] || 0;
          const stats = wordStats.get(word);
          const isActive = hits > 0;

          return (
            <div
              key={word}
              className={`
                relative p-4 rounded-lg border-2 transition-all duration-300
                ${
                  isActive
                    ? 'border-green-500 bg-green-500/10 word-hit'
                    : 'border-zinc-700 bg-zinc-800/50'
                }
              `}
            >
              <div className="flex items-start justify-between mb-2">
                <span className="font-bold text-sm">{word}</span>
                {stats?.trend === 'heating' && (
                  <TrendingUp className="w-3 h-3 text-orange-500" />
                )}
              </div>

              <div className="flex items-center gap-1 mb-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Circle
                    key={i}
                    className={`w-2 h-2 ${
                      i < hits
                        ? 'fill-green-500 text-green-500'
                        : 'text-zinc-600'
                    }`}
                  />
                ))}
              </div>

              <div className="text-xs text-zinc-500">
                {hits} {hits === 1 ? 'mention' : 'mentions'}
                {stats && stats.anomalyScore > 0 && (
                  <span className="ml-2 text-orange-500 font-semibold">
                    {formatAnomalyScore(stats.anomalyScore)}
                  </span>
                )}
              </div>

              {isActive && (
                <div className="absolute inset-0 rounded-lg bg-green-500/20 animate-pulse pointer-events-none" />
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-zinc-800">
        <div className="flex items-center justify-between text-sm">
          <span className="text-zinc-400">Total Mentions</span>
          <span className="font-bold text-green-500">{totalHits}</span>
        </div>
        <div className="mt-2 h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-500"
            style={{ width: `${Math.min((totalHits / 20) * 100, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
