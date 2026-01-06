'use client';

import { useEffect } from 'react';
import { DashboardHeader } from '@/components/DashboardHeader';
import { MentionCard } from '@/components/MentionCard';
import { LiveFeed } from '@/components/LiveFeed';
import { InsightFooter } from '@/components/InsightFooter';
import { RoundControl } from '@/components/RoundControl';
import { WordSelector } from '@/components/WordSelector';
import { useMarketStore } from '@/store/marketStore';
import { TrendingUp } from 'lucide-react';

export default function Home() {
  const initializeServices = useMarketStore((state) => state.initializeServices);
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const currentRound = useMarketStore((state) => state.currentRound);

  useEffect(() => {
    initializeServices();
  }, [initializeServices]);

  const showWordSelector = !currentPlayer || (currentRound?.status === 'completed');

  return (
    <main className="min-h-screen p-4" style={{background: '#000'}}>
      <div className="max-w-7xl mx-auto">
        {/* Terminal Header */}
        <div className="mb-6">
          <pre className="text-[#00ff00] text-xs sm:text-sm leading-tight">
{`╔══════════════════════════════════════════════════════════════════════════╗
║                           YESNO.EVENTS v2.0                             ║
║                     Mention Markets Terminal Interface                  ║
║           Real-time word frequency prediction and anomaly detection     ║
╚══════════════════════════════════════════════════════════════════════════╝`}
          </pre>
        </div>

        {/* Dashboard Header */}
        <div className="mb-6">
          <DashboardHeader />
        </div>

        {/* Round Control */}
        <div className="mb-6">
          <RoundControl />
        </div>

        {/* Word Selector (shown before round starts or after completion) */}
        {showWordSelector && (
          <div className="mb-6">
            <WordSelector />
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Mention Card */}
          <div className="lg:col-span-1">
            <MentionCard />
          </div>

          {/* Live Feed */}
          <div className="lg:col-span-2 h-[600px]">
            <LiveFeed />
          </div>
        </div>

        {/* Insights Footer */}
        <InsightFooter />

        {/* Terminal Footer */}
        <div className="mt-6">
          <pre className="text-[#00ff00] text-xs text-center opacity-60">
{`────────────────────────────────────────────────────────────────────────────
[SYSTEM] yesno.events terminal v2.0 | Type 'help' for commands
────────────────────────────────────────────────────────────────────────────`}
          </pre>
        </div>
      </div>
    </main>
  );
}
