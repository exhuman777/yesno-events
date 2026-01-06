'use client';

import { useState } from 'react';
import { useMarketStore } from '@/store/marketStore';
import type { RoundInterval } from '@/types/market';
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
      <div className="term-box">
        <div className="term-box-title">[ MARKET CLOSED ]</div>
        <div className="text-center font-mono">
          <pre className="text-[#ffff00] text-4xl mb-4">
{`   ╔═══╗
   ║ ★ ║
   ╚═══╝`}
          </pre>
          <div className="border-2 border-[#00ff00] p-4 mb-4 inline-block">
            <p className="text-xs text-[#00ff00] mb-2">[ FINAL RESULTS ]</p>
            <p className="text-3xl font-bold text-[#00ff00] mb-2">
              {player.totalHits} MENTIONS
            </p>
            <p className="text-sm text-[#ffff00]">
              PAYOUT: {formatCurrency(player.payout)}
            </p>
          </div>

          <button
            onClick={handleReset}
            className="term-button"
          >
            [↻] NEW MARKET
          </button>
        </div>
      </div>
    );
  }

  // Show start controls if no round or waiting
  if (!currentRound || currentRound.status === 'waiting') {
    return (
      <div className="term-box">
        <div className="term-box-title">[ MARKET CONTROL ]</div>

        {!currentPlayer && (
          <div className="mb-4 border-2 border-[#ffff00] p-3 bg-[#ffff00]/10">
            <p className="text-sm text-[#ffff00] font-mono">
              [!] SELECT YOUR WORDS BELOW BEFORE STARTING
            </p>
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-[#00ff00] mb-2 font-mono font-bold">
              [ DURATION ]
            </label>
            <div className="grid grid-cols-4 gap-2">
              {(['30s', '1m', '1d', '7d'] as RoundInterval[]).map((interval) => (
                <button
                  key={interval}
                  onClick={() => setSelectedInterval(interval)}
                  className={`
                    px-4 py-2 font-mono font-bold transition-all border-2
                    ${
                      selectedInterval === interval
                        ? 'bg-[#00ff00] text-black border-[#00ff00]'
                        : 'bg-black text-[#00ff00] border-[#00ff00] hover:bg-[#00ff00]/20'
                    }
                  `}
                >
                  {interval.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleStartRound}
            disabled={!currentPlayer}
            className="w-full term-button text-lg py-4"
          >
            [▶] START MARKET ({selectedInterval.toUpperCase()})
          </button>
        </div>
      </div>
    );
  }

  // Round is active - show status
  return (
    <div className="term-box">
      <div className="term-box-title">[ MARKET CONTROL ]</div>
      <div className="flex items-center justify-between font-mono">
        <div>
          <p className="text-xs text-[#00ff00] mb-1">[ STATUS ]</p>
          <div className="flex items-center gap-2">
            <span className="text-[#00ff00] term-blink text-xl">●</span>
            <p className="text-lg font-bold text-[#00ff00]">ACTIVE</p>
          </div>
        </div>
        <button
          onClick={handleEndRound}
          className="border-2 border-[#ff0000] bg-black text-[#ff0000] px-4 py-2 font-bold hover:bg-[#ff0000] hover:text-black transition-all"
        >
          [■] CLOSE MARKET
        </button>
      </div>
    </div>
  );
}
