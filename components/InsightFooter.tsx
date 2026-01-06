'use client';

import { useMarketStore } from '@/store/marketStore';
import { AlertTriangle, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { ANOMALY_THRESHOLD } from '@/lib/constants';

export function InsightFooter() {
  const currentRound = useMarketStore((state) => state.currentRound);
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const wordStats = useMarketStore((state) => state.wordStats);
  const timeRemaining = useMarketStore((state) => state.timeRemaining);

  if (!currentRound || !currentPlayer || currentRound.status !== 'active') {
    return null;
  }

  const insights: Array<{
    type: 'spike' | 'underperform' | 'info';
    message: string;
  }> = [];

  // Check for spiking words in player's card
  currentPlayer.bingoCard.words.forEach((word) => {
    const stats = wordStats.get(word);
    if (stats && stats.anomalyScore > ANOMALY_THRESHOLD) {
      insights.push({
        type: 'spike',
        message: `"${word}" spiking: +${Math.round(stats.anomalyScore * 100)}% above baseline`,
      });
    }
  });

  // Check for underperforming words
  const underperformingWords = currentPlayer.bingoCard.words.filter((word) => {
    const hits = currentPlayer.bingoCard.hits[word] || 0;
    return hits === 0 && timeRemaining < 20;
  });

  if (underperformingWords.length > 0) {
    insights.push({
      type: 'underperform',
      message: `${underperformingWords.length} word(s) with 0 hits: ${underperformingWords.slice(0, 2).join(', ')}${underperformingWords.length > 2 ? '...' : ''}`,
    });
  }

  // Time-based insights
  if (timeRemaining < 10 && timeRemaining > 0) {
    insights.push({
      type: 'info',
      message: `${timeRemaining}s to resolution - final moments!`,
    });
  }

  // Show only top 3 insights
  const displayInsights = insights.slice(0, 3);

  if (displayInsights.length === 0) {
    return null;
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-yellow-500 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-sm font-bold mb-2">Market Insights</h3>
          <ul className="space-y-2">
            {displayInsights.map((insight, index) => (
              <li key={index} className="flex items-start gap-2 text-sm">
                {insight.type === 'spike' && (
                  <TrendingUp className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                )}
                {insight.type === 'underperform' && (
                  <TrendingDown className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                )}
                {insight.type === 'info' && (
                  <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                )}
                <span className="text-zinc-300">{insight.message}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
