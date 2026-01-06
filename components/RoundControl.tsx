'use client';

import { useState } from 'react';
import { useMarketStore } from '@/store/marketStore';
import type { RoundInterval } from '@/types/market';
import { Play, RotateCcw, Trophy } from 'lucide-react';
import { formatCurrency } from '@/lib/utils';

export function RoundControl() {
  const currentRound = useMarketStore((state) => state.currentRound);
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const startRound = useMarketStore((state) => state.startRound);
  const endRound = useMarketStore((state) => state.endRound);
  const reset = useMarketStore((state) => state.reset);
  const initializeServices = useMarketStore((state) => state.initializeServices);

  const [selectedInterval, setSelectedInterval] = useState<RoundInterval>('30s');

  const handleStartRound = () => {
    if (!currentPlayer) {
      alert('Please select your words first');
      return;
    }

    initializeServices();
    startRound(selectedInterval);
  };

  const handleEndRound = () => {
    endRound();
  };

  const handleReset = () => {
    reset();
  };

  // Show results if round is completed
  if (currentRound?.status === 'completed') {
    const player = currentRound.participants[0];

    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
        <div className="text-center">
          <Trophy className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">Market Closed!</h2>

          <div className="bg-zinc-800 rounded-lg p-4 mb-4">
            <p className="text-sm text-zinc-500 mb-1">Your Performance</p>
            <p className="text-3xl font-bold text-green-500 mb-2">
              {player.totalHits} mentions
            </p>
            <p className="text-sm text-zinc-400">
              Payout: {formatCurrency(player.payout)}
            </p>
          </div>

          <button
            onClick={handleReset}
            className="flex items-center gap-2 mx-auto px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            New Market
          </button>
        </div>
      </div>
    );
  }

  // Show start controls if no round or waiting
  if (!currentRound || currentRound.status === 'waiting') {
    return (
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
        <h2 className="text-xl font-bold mb-4">Start Market</h2>

        {!currentPlayer && (
          <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <p className="text-sm text-yellow-500">
              ⚠ Select your words below before starting the market
            </p>
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-2">
              Market Duration
            </label>
            <div className="grid grid-cols-4 gap-2">
              {(['30s', '1m', '1d', '7d'] as RoundInterval[]).map((interval) => (
                <button
                  key={interval}
                  onClick={() => setSelectedInterval(interval)}
                  className={`
                    px-4 py-2 rounded-lg font-semibold transition-all
                    ${
                      selectedInterval === interval
                        ? 'bg-blue-600 text-white'
                        : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                    }
                  `}
                >
                  {interval}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleStartRound}
            disabled={!currentPlayer}
            className="w-full flex items-center justify-center gap-2 px-6 py-4 bg-green-600 hover:bg-green-700 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed rounded-lg font-bold text-lg transition-colors"
          >
            <Play className="w-5 h-5" />
            Start Market ({selectedInterval})
          </button>
        </div>
      </div>
    );
  }

  // Round is active - show status
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-zinc-500 mb-1">Market Status</p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <p className="text-lg font-bold text-green-500">ACTIVE</p>
          </div>
        </div>
        <button
          onClick={handleEndRound}
          className="px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-500 rounded-lg text-sm font-semibold transition-colors"
        >
          Close Market
        </button>
      </div>
    </div>
  );
}
