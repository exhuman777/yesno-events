'use client';

import { useMarketStore } from '@/store/marketStore';
import { formatCurrency, formatTime, formatAnomalyScore } from '@/lib/utils';
import { Trophy, Clock, Flame } from 'lucide-react';

export function DashboardHeader() {
  const currentRound = useMarketStore((state) => state.currentRound);
  const timeRemaining = useMarketStore((state) => state.timeRemaining);
  const wordStats = useMarketStore((state) => state.wordStats);

  // Get top 3 anomalies
  const topAnomalies = Array.from(wordStats.values())
    .filter((stat) => stat.anomalyScore > 0)
    .sort((a, b) => b.anomalyScore - a.anomalyScore)
    .slice(0, 3);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
      <div className="grid grid-cols-3 gap-6">
        {/* Prize Pool */}
        <div className="flex items-start gap-3">
          <div className="p-2 bg-yellow-500/10 rounded-lg">
            <Trophy className="w-5 h-5 text-yellow-500" />
          </div>
          <div>
            <p className="text-xs text-zinc-500 mb-1">Prize Pool</p>
            <p className="text-2xl font-bold text-yellow-500">
              {currentRound ? formatCurrency(currentRound.prizePool) : '$0'}
            </p>
          </div>
        </div>

        {/* Timer */}
        <div className="flex items-start gap-3">
          <div className="p-2 bg-blue-500/10 rounded-lg">
            <Clock className="w-5 h-5 text-blue-500" />
          </div>
          <div>
            <p className="text-xs text-zinc-500 mb-1">Time Remaining</p>
            <p className="text-2xl font-bold font-mono text-blue-500">
              {currentRound?.status === 'active'
                ? formatTime(timeRemaining)
                : '--:--'}
            </p>
          </div>
        </div>

        {/* Top Anomalies */}
        <div className="flex items-start gap-3">
          <div className="p-2 bg-orange-500/10 rounded-lg">
            <Flame className="w-5 h-5 text-orange-500" />
          </div>
          <div className="flex-1">
            <p className="text-xs text-zinc-500 mb-2">Top Trending</p>
            {topAnomalies.length === 0 ? (
              <p className="text-sm text-zinc-600">No trends yet</p>
            ) : (
              <div className="space-y-1">
                {topAnomalies.map((stat, index) => (
                  <div key={stat.word} className="flex items-center gap-2">
                    <span className="text-xs text-zinc-600">#{index + 1}</span>
                    <span className="text-sm font-bold text-orange-500">
                      {stat.word}
                    </span>
                    <span className="text-xs text-zinc-500 font-semibold">
                      {formatAnomalyScore(stat.anomalyScore)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
