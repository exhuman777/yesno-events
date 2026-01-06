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
    <main className="min-h-screen bg-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-8 h-8 text-green-500" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-green-500 to-blue-500 bg-clip-text text-transparent">
              yesno.events
            </h1>
          </div>
          <p className="text-zinc-500">
            Mention Markets • Real-time word frequency prediction
          </p>
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

        {/* Info Footer */}
        <div className="mt-6 text-center text-xs text-zinc-600">
          <p>
            yesno.events • Mention Markets MVP • Real-time Anomaly Detection
          </p>
        </div>
      </div>
    </main>
  );
}
