'use client';

import { useMarketStore } from '@/store/marketStore';
import { formatCurrency, formatTime, formatAnomalyScore } from '@/lib/utils';

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
    <div className="term-box">
      <div className="term-box-title">[ STATUS ]</div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-sm">
        {/* Prize Pool */}
        <div className="border-2 border-[#ffff00] p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[#ffff00]">[$]</span>
            <span className="text-[#ffff00] font-bold">PRIZE POOL</span>
          </div>
          <div className="text-2xl font-bold text-[#ffff00] tracking-wider">
            {currentRound ? formatCurrency(currentRound.prizePool) : '$0'}
          </div>
        </div>

        {/* Timer */}
        <div className="border-2 border-[#00ffff] p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[#00ffff]">[⏱]</span>
            <span className="text-[#00ffff] font-bold">TIME LEFT</span>
          </div>
          <div className="text-2xl font-bold text-[#00ffff] tracking-widest">
            {currentRound?.status === 'active'
              ? formatTime(timeRemaining)
              : '--:--'}
          </div>
        </div>

        {/* Top Anomalies */}
        <div className="border-2 border-[#ff00ff] p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[#ff00ff]">[▲]</span>
            <span className="text-[#ff00ff] font-bold">TOP TRENDING</span>
          </div>
          {topAnomalies.length === 0 ? (
            <p className="text-xs text-[#008800]">[ NO TRENDS DETECTED ]</p>
          ) : (
            <div className="space-y-1">
              {topAnomalies.map((stat, index) => (
                <div key={stat.word} className="text-xs text-[#00ff00]">
                  <span className="text-[#ff00ff]">{index + 1}.</span>{' '}
                  <span className="font-bold">{stat.word}</span>{' '}
                  <span className="text-[#ffff00]">{formatAnomalyScore(stat.anomalyScore)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
