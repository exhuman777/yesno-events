'use client';

import { useMarketStore } from '@/store/marketStore';
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
    <div className="term-box h-full flex flex-col">
      <div className="term-box-title">[ LIVE NEWS FEED ]</div>
      <div className="flex items-center gap-2 mb-3 font-mono text-xs">
        <span className="text-[#00ff00] term-blink">●</span>
        <span className="text-[#00ff00]">LIVE</span>
        <span className="text-[#008800]">|</span>
        <span className="text-[#008800]">{newsEvents.length} EVENTS</span>
      </div>

      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto custom-scrollbar space-y-2 pr-2"
      >
        {newsEvents.length === 0 ? (
          <div className="text-center text-[#008800] py-8 font-mono">
            <pre className="text-xs">
{`   ┌─────────┐
   │ WAITING │
   └─────────┘`}
            </pre>
            <p className="text-xs mt-2">[ NO EVENTS YET ]</p>
          </div>
        ) : (
          newsEvents.map((event) => (
            <div
              key={event.id}
              className="border-l-2 border-[#00ff00] pl-3 py-2 animate-slide-up bg-black/50"
            >
              <div className="flex items-start gap-2 mb-1 font-mono text-[10px]">
                <span className="text-[#00ffff]">
                  [{new Date(event.timestamp).toLocaleTimeString()}]
                </span>
                {event.matchedWords.length > 0 && (
                  <span className="text-[#ffff00]">
                    [⚡ {event.matchedWords.length} {event.matchedWords.length === 1 ? 'MATCH' : 'MATCHES'}]
                  </span>
                )}
              </div>

              <p className="text-xs text-[#00ff00] leading-relaxed mb-2 font-mono">
                {event.headline}
              </p>

              {event.matchedWords.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {Array.from(new Set(event.matchedWords)).map((word) => (
                    <span
                      key={word}
                      className="text-[10px] px-2 py-0.5 bg-[#00ff00]/20 text-[#00ff00] border border-[#00ff00] font-mono font-bold"
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
