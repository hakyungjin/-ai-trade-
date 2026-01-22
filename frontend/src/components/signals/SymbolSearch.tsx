/**
 * 심볼 검색 및 등록 컴포넌트
 */

import React, { useState, useEffect } from 'react';
import { Search, Plus, X } from 'lucide-react';
import { apiClient } from '@/api/client';

interface Symbol {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
}

interface SymbolSearchProps {
  onSymbolAdd: (symbol: string) => void;
}

export const SymbolSearch: React.FC<SymbolSearchProps> = ({ onSymbolAdd }) => {
  const [query, setQuery] = useState('');
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeSymbols, setActiveSymbols] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);

  // 활성 심볼 로드
  useEffect(() => {
    loadActiveSymbols();
  }, []);

  const loadActiveSymbols = async () => {
    try {
      const response = await apiClient.get('/api/signals/symbols/active');
      if (response.data.success) {
        setActiveSymbols(response.data.symbols);
      }
    } catch (error) {
      console.error('Failed to load active symbols:', error);
    }
  };

  // 심볼 검색
  const searchSymbols = async (searchQuery: string) => {
    if (searchQuery.length < 1) {
      setSymbols([]);
      setShowResults(false);
      return;
    }

    setLoading(true);
    try {
      const response = await apiClient.get('/api/signals/symbols/search', {
        params: { query: searchQuery, limit: 20 }
      });

      if (response.data.success) {
        setSymbols(response.data.symbols);
        setShowResults(true);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setLoading(false);
    }
  };

  // 디바운스 검색
  useEffect(() => {
    const timer = setTimeout(() => {
      searchSymbols(query);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  // 심볼 추가
  const handleAddSymbol = async (symbol: string) => {
    try {
      const response = await apiClient.post('/api/signals/symbols/add', null, {
        params: { symbol }
      });

      if (response.data.success) {
        setActiveSymbols([...activeSymbols, symbol]);
        onSymbolAdd(symbol);
        setQuery('');
        setShowResults(false);
      }
    } catch (error) {
      console.error('Failed to add symbol:', error);
      alert('심볼 추가 실패');
    }
  };

  // 심볼 제거
  const handleRemoveSymbol = async (symbol: string) => {
    try {
      await apiClient.delete(`/api/signals/symbols/${symbol}`);
      setActiveSymbols(activeSymbols.filter(s => s !== symbol));
    } catch (error) {
      console.error('Failed to remove symbol:', error);
    }
  };

  return (
    <div className="space-y-4">
      {/* 검색 입력 */}
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value.toUpperCase())}
            placeholder="심볼 검색 (예: BTC, ETH)"
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* 검색 결과 */}
        {showResults && (
          <div className="absolute z-10 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-96 overflow-y-auto">
            {loading ? (
              <div className="p-4 text-center text-gray-500">
                검색 중...
              </div>
            ) : symbols.length === 0 ? (
              <div className="p-4 text-center text-gray-500">
                검색 결과가 없습니다
              </div>
            ) : (
              <ul className="divide-y divide-gray-100">
                {symbols.map((symbol) => (
                  <li
                    key={symbol.symbol}
                    className="p-3 hover:bg-gray-50 cursor-pointer flex items-center justify-between"
                    onClick={() => handleAddSymbol(symbol.symbol)}
                  >
                    <div>
                      <div className="font-semibold text-gray-900">
                        {symbol.symbol}
                      </div>
                      <div className="text-sm text-gray-500">
                        {symbol.baseAsset} / {symbol.quoteAsset}
                      </div>
                    </div>
                    <Plus className="w-5 h-5 text-blue-500" />
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* 활성 심볼 목록 */}
      {activeSymbols.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2">
            모니터링 중인 심볼
          </h3>
          <div className="flex flex-wrap gap-2">
            {activeSymbols.map((symbol) => (
              <div
                key={symbol}
                className="flex items-center gap-2 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                <span className="font-semibold">{symbol}</span>
                <button
                  onClick={() => handleRemoveSymbol(symbol)}
                  className="hover:text-blue-900"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
