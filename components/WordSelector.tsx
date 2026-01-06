'use client';

import { useState } from 'react';
import { WORD_POOL, BINGO_CARD_SIZE } from '@/lib/constants';
import { useMarketStore } from '@/store/marketStore';

export function WordSelector() {
  const selectBingoCard = useMarketStore((state) => state.selectBingoCard);
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const currentRound = useMarketStore((state) => state.currentRound);

  const [selectedWords, setSelectedWords] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  const isRoundActive = currentRound?.status === 'active';
  const isSelectionComplete = selectedWords.length === BINGO_CARD_SIZE;

  const filteredWords = WORD_POOL.filter(word =>
    word.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleWord = (word: string) => {
    if (isRoundActive) return;

    if (selectedWords.includes(word)) {
      setSelectedWords(selectedWords.filter(w => w !== word));
    } else if (selectedWords.length < BINGO_CARD_SIZE) {
      setSelectedWords([...selectedWords, word]);
    }
  };

  const handleConfirm = () => {
    if (isSelectionComplete) {
      selectBingoCard(selectedWords);
    }
  };

  const handleClear = () => {
    if (!isRoundActive) {
      setSelectedWords([]);
    }
  };

  const categories = {
    'FINANCE': WORD_POOL.slice(0, 10),
    'POLITICS': WORD_POOL.slice(10, 20),
    'TECH': WORD_POOL.slice(20, 30),
    'CLIMATE': WORD_POOL.slice(30, 40),
    'HEALTH': WORD_POOL.slice(40, 50),
  };

  if (currentPlayer && !isRoundActive) {
    return null;
  }

  return (
    <div className="term-box">
      <div className="term-box-title">[ WORD SELECTION ]</div>
      <div className="flex items-center justify-between mb-4 font-mono text-xs">
        <span className="text-[#00ff00]">
          [{selectedWords.length}/{BINGO_CARD_SIZE} SELECTED]
        </span>
        {selectedWords.length > 0 && (
          <button
            onClick={handleClear}
            disabled={isRoundActive}
            className="text-[#ff0000] hover:text-[#ff0000]/70 disabled:opacity-50"
          >
            [X] CLEAR ALL
          </button>
        )}
      </div>

      {selectedWords.length > 0 && (
        <div className="mb-4 border-2 border-[#00ff00] p-2">
          <div className="flex flex-wrap gap-1">
            {selectedWords.map((word) => (
              <span
                key={word}
                onClick={() => !isRoundActive && toggleWord(word)}
                className="text-[10px] px-2 py-1 bg-[#00ff00]/20 text-[#00ff00] border border-[#00ff00] font-mono font-bold cursor-pointer hover:bg-[#00ff00]/30"
              >
                {word}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mb-3">
        <input
          type="text"
          placeholder="[SEARCH KEYWORDS...]"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          disabled={isRoundActive}
          className="term-input w-full text-xs"
        />
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
        {searchTerm ? (
          <div>
            <h3 className="text-xs font-bold text-[#00ff00] mb-2 font-mono">[ SEARCH RESULTS ]</h3>
            <div className="grid grid-cols-3 gap-1">
              {filteredWords.map((word) => {
                const isSelected = selectedWords.includes(word);
                const canSelect = selectedWords.length < BINGO_CARD_SIZE || isSelected;

                return (
                  <button
                    key={word}
                    onClick={() => toggleWord(word)}
                    disabled={isRoundActive || (!canSelect && !isSelected)}
                    className={`
                      px-2 py-1 text-xs font-mono font-bold transition-all border-2
                      ${isSelected
                        ? 'bg-[#00ff00] text-black border-[#00ff00]'
                        : canSelect
                        ? 'bg-black text-[#00ff00] border-[#00ff00] hover:bg-[#00ff00]/20'
                        : 'bg-black text-[#008800] border-[#008800] opacity-50 cursor-not-allowed'
                      }
                      disabled:opacity-50
                    `}
                  >
                    {isSelected && '[✓] '}{word}
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          Object.entries(categories).map(([category, words]) => (
            <div key={category}>
              <h3 className="text-xs font-bold text-[#ffff00] mb-2 font-mono">[ {category} ]</h3>
              <div className="grid grid-cols-3 gap-1">
                {words.map((word) => {
                  const isSelected = selectedWords.includes(word);
                  const canSelect = selectedWords.length < BINGO_CARD_SIZE || isSelected;

                  return (
                    <button
                      key={word}
                      onClick={() => toggleWord(word)}
                      disabled={isRoundActive || (!canSelect && !isSelected)}
                      className={`
                        px-2 py-1 text-xs font-mono font-bold transition-all border-2
                        ${isSelected
                          ? 'bg-[#00ff00] text-black border-[#00ff00]'
                          : canSelect
                          ? 'bg-black text-[#00ff00] border-[#00ff00] hover:bg-[#00ff00]/20'
                          : 'bg-black text-[#008800] border-[#008800] opacity-50 cursor-not-allowed'
                        }
                        disabled:opacity-50
                      `}
                    >
                      {isSelected && '[✓] '}{word}
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </div>

      {isSelectionComplete && !currentPlayer && (
        <button
          onClick={handleConfirm}
          disabled={isRoundActive}
          className="w-full mt-4 term-button text-sm py-3"
        >
          [✓] CONFIRM SELECTION
        </button>
      )}
    </div>
  );
}
