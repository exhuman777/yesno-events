'use client';

import { useMarketStore } from '@/store/marketStore';
import { formatAnomalyScore } from '@/lib/utils';

export function MentionCard() {
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const wordStats = useMarketStore((state) => state.wordStats);

  if (!currentPlayer) {
    return (
      <div className="term-box">
        <div className="term-box-title">[ YOUR POSITIONS ]</div>
        <p className="text-[#008800] font-mono text-sm">[ SELECT WORDS TO BEGIN TRACKING ]</p>
      </div>
    );
  }

  const { bingoCard, totalHits } = currentPlayer;

  return (
    <div className="term-box">
      <div className="term-box-title">[ YOUR POSITIONS ]</div>
      <div className="flex items-center justify-between mb-4 font-mono">
        <span className="text-[#00ff00] text-xs">[▲] {totalHits} TOTAL MENTIONS</span>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {bingoCard.words.map((word) => {
          const hits = bingoCard.hits[word] || 0;
          const stats = wordStats.get(word);
          const isActive = hits > 0;

          return (
            <div
              key={word}
              className={`
                relative p-2 border-2 transition-all font-mono text-xs
                ${
                  isActive
                    ? 'border-[#00ff00] bg-[#00ff00]/10 word-hit'
                    : 'border-[#008800] bg-black'
                }
              `}
            >
              <div className="flex items-start justify-between mb-1">
                <span className={`font-bold ${isActive ? 'text-[#00ff00]' : 'text-[#008800]'}`}>
                  {word}
                </span>
                {stats?.trend === 'heating' && (
                  <span className="text-[#ff00ff]">▲</span>
                )}
              </div>

              <div className="flex items-center gap-0.5 mb-1">
                {Array.from({ length: 5 }).map((_, i) => (
                  <span
                    key={i}
                    className={`${
                      i < hits ? 'text-[#00ff00]' : 'text-[#003300]'
                    }`}
                  >
                    {i < hits ? '█' : '░'}
                  </span>
                ))}
              </div>

              <div className="text-[10px]">
                <span className={isActive ? 'text-[#00ff00]' : 'text-[#008800]'}>
                  {hits} HIT{hits === 1 ? '' : 'S'}
                </span>
                {stats && stats.anomalyScore > 0 && (
                  <span className="ml-2 text-[#ffff00] font-bold">
                    {formatAnomalyScore(stats.anomalyScore)}
                  </span>
                )}
              </div>

              {isActive && (
                <div className="absolute inset-0 border-2 border-[#00ff00] animate-pulse pointer-events-none" />
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-3 border-t-2 border-[#00ff00]">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-[#00ff00]">[ TOTAL MENTIONS ]</span>
          <span className="font-bold text-[#00ff00]">{totalHits}</span>
        </div>
        <div className="mt-2 h-2 bg-black border border-[#00ff00] overflow-hidden">
          <div
            className="h-full bg-[#00ff00]"
            style={{ width: `${Math.min((totalHits / 20) * 100, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
