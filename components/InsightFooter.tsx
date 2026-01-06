'use client';

import { useMarketStore } from '@/store/marketStore';
import { formatAnomalyScore } from '@/lib/utils';
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
        message: `"${word}" SPIKING: ${formatAnomalyScore(stats.anomalyScore)} ABOVE BASELINE`,
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
      message: `${underperformingWords.length} WORD(S) WITH 0 HITS: ${underperformingWords.slice(0, 2).join(', ')}${underperformingWords.length > 2 ? '...' : ''}`,
    });
  }

  // Time-based insights
  if (timeRemaining < 10 && timeRemaining > 0) {
    insights.push({
      type: 'info',
      message: `${timeRemaining}S TO RESOLUTION - FINAL MOMENTS!`,
    });
  }

  // Show only top 3 insights
  const displayInsights = insights.slice(0, 3);

  if (displayInsights.length === 0) {
    return null;
  }

  return (
    <div className="term-box">
      <div className="term-box-title">[ MARKET INSIGHTS ]</div>
      <div className="space-y-2 font-mono text-xs">
        {displayInsights.map((insight, index) => (
          <div key={index} className="flex items-start gap-2">
            {insight.type === 'spike' && (
              <span className="text-[#00ff00]">[▲]</span>
            )}
            {insight.type === 'underperform' && (
              <span className="text-[#ff0000]">[▼]</span>
            )}
            {insight.type === 'info' && (
              <span className="text-[#00ffff]">[i]</span>
            )}
            <span className={
              insight.type === 'spike' ? 'text-[#00ff00]' :
              insight.type === 'underperform' ? 'text-[#ff0000]' :
              'text-[#00ffff]'
            }>
              {insight.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
