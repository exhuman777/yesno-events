'use client';

import { useMarketStore } from '@/store/marketStore';
import { Activity, Zap } from 'lucide-react';
import { useEffect, useRef } from 'react';

export function LiveFeed() {
  const newsEvents = useMarketStore((state) => state.newsEvents);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [newsEvents]);

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-blue-500" />
        <h2 className="text-xl font-bold">Live News Feed</h2>
        <div className="ml-auto flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-xs text-zinc-500">LIVE</span>
        </div>
      </div>

      <div
        ref={feedRef}
        className="terminal flex-1 rounded-lg p-4 overflow-y-auto custom-scrollbar space-y-3"
      >
        {newsEvents.length === 0 ? (
          <div className="text-center text-zinc-600 py-8">
            <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">Waiting for news events...</p>
          </div>
        ) : (
          newsEvents.map((event) => (
            <div
              key={event.id}
              className="border-l-2 border-blue-500 pl-3 py-2 animate-slide-up"
            >
              <div className="flex items-start gap-2 mb-1">
                <span className="text-xs text-zinc-500 font-mono">
                  {new Date(event.timestamp).toLocaleTimeString()}
                </span>
                {event.matchedWords.length > 0 && (
                  <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3 text-yellow-500" />
                    <span className="text-xs text-yellow-500">
                      {event.matchedWords.length} {event.matchedWords.length === 1 ? 'match' : 'matches'}
                    </span>
                  </div>
                )}
              </div>

              <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                {event.headline}
              </p>

              {event.matchedWords.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {Array.from(new Set(event.matchedWords)).map((word) => (
                    <span
                      key={word}
                      className="text-xs px-2 py-0.5 bg-green-500/20 text-green-500 rounded-full border border-green-500/30"
                    >
                      {word}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
