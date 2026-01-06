'use client';

import { useState } from 'react';
import { WORD_POOL, BINGO_CARD_SIZE } from '@/lib/constants';
import { useMarketStore } from '@/store/marketStore';
import { Search, X, Check } from 'lucide-react';

export function WordSelector() {
  const selectBingoCard = useMarketStore((state) => state.selectBingoCard);
  const currentPlayer = useMarketStore((state) => state.currentPlayer);
  const currentRound = useMarketStore((state) => state.currentRound);

  const [selectedWords, setSelectedWords] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  const isRoundActive = currentRound?.status === 'active';
  const isSelectionComplete = selectedWords.length === BINGO_CARD_SIZE;

  // Filter words based on search
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

  // Group words by category
  const categories = {
    'Finance': WORD_POOL.slice(0, 10),
    'Politics': WORD_POOL.slice(10, 20),
    'Tech': WORD_POOL.slice(20, 30),
    'Climate': WORD_POOL.slice(30, 40),
    'Health': WORD_POOL.slice(40, 50),
  };

  if (currentPlayer && !isRoundActive) {
    return null; // Hide selector when round is not active but player exists
  }

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-bold">Select Your Words</h2>
          <p className="text-sm text-zinc-500">
            Choose {BINGO_CARD_SIZE} words to track ({selectedWords.length}/{BINGO_CARD_SIZE} selected)
          </p>
        </div>
        {selectedWords.length > 0 && (
          <button
            onClick={handleClear}
            disabled={isRoundActive}
            className="text-sm text-red-500 hover:text-red-400 disabled:opacity-50"
          >
            Clear All
          </button>
        )}
      </div>

      {/* Selected Words */}
      {selectedWords.length > 0 && (
        <div className="mb-4 p-3 bg-zinc-800/50 rounded-lg">
          <p className="text-xs text-zinc-400 mb-2">Your Selections:</p>
          <div className="flex flex-wrap gap-2">
            {selectedWords.map((word) => (
              <div
                key={word}
                className="flex items-center gap-1 px-3 py-1 bg-green-500/20 text-green-500 rounded-full border border-green-500/30"
              >
                <span className="text-sm font-medium">{word}</span>
                {!isRoundActive && (
                  <button
                    onClick={() => toggleWord(word)}
                    className="hover:bg-green-500/30 rounded-full p-0.5"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search */}
      <div className="mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            placeholder="Search keywords..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            disabled={isRoundActive}
            className="w-full pl-10 pr-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
        </div>
      </div>

      {/* Word Categories */}
      <div className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar">
        {searchTerm ? (
          <div>
            <h3 className="text-sm font-bold text-zinc-400 mb-2">Search Results</h3>
            <div className="grid grid-cols-3 gap-2">
              {filteredWords.map((word) => {
                const isSelected = selectedWords.includes(word);
                const canSelect = selectedWords.length < BINGO_CARD_SIZE || isSelected;

                return (
                  <button
                    key={word}
                    onClick={() => toggleWord(word)}
                    disabled={isRoundActive || (!canSelect && !isSelected)}
                    className={`
                      px-3 py-2 rounded-lg text-sm font-medium transition-all
                      ${isSelected
                        ? 'bg-green-500/20 text-green-500 border-2 border-green-500'
                        : canSelect
                        ? 'bg-zinc-800 text-zinc-300 border border-zinc-700 hover:border-zinc-600'
                        : 'bg-zinc-800/50 text-zinc-600 border border-zinc-800 cursor-not-allowed'
                      }
                      disabled:opacity-50
                    `}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span>{word}</span>
                      {isSelected && <Check className="w-3 h-3" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          Object.entries(categories).map(([category, words]) => (
            <div key={category}>
              <h3 className="text-sm font-bold text-zinc-400 mb-2">{category}</h3>
              <div className="grid grid-cols-3 gap-2">
                {words.map((word) => {
                  const isSelected = selectedWords.includes(word);
                  const canSelect = selectedWords.length < BINGO_CARD_SIZE || isSelected;

                  return (
                    <button
                      key={word}
                      onClick={() => toggleWord(word)}
                      disabled={isRoundActive || (!canSelect && !isSelected)}
                      className={`
                        px-3 py-2 rounded-lg text-sm font-medium transition-all
                        ${isSelected
                          ? 'bg-green-500/20 text-green-500 border-2 border-green-500'
                          : canSelect
                          ? 'bg-zinc-800 text-zinc-300 border border-zinc-700 hover:border-zinc-600'
                          : 'bg-zinc-800/50 text-zinc-600 border border-zinc-800 cursor-not-allowed'
                        }
                        disabled:opacity-50
                      `}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span>{word}</span>
                        {isSelected && <Check className="w-3 h-3" />}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Confirm Button */}
      {isSelectionComplete && !currentPlayer && (
        <button
          onClick={handleConfirm}
          disabled={isRoundActive}
          className="w-full mt-4 px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-zinc-700 disabled:text-zinc-500 rounded-lg font-bold transition-colors"
        >
          Confirm Selection
        </button>
      )}
    </div>
  );
}
