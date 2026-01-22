/**
 * 심볼 검색 및 등록 컴포넌트
 */

import React, { useState, useEffect } from 'react';
import { Search, Plus, X, Flame, TrendingUp } from 'lucide-react';
import { apiClient } from '@/api/client';

interface Symbol {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  price?: number;
  priceChangePercent?: number;
  volume?: number;
  trend?: 'up' | 'down' | 'neutral';
}

interface SymbolSearchProps {
  onSymbolAdd: (symbol: string) => void;
}

export const SymbolSearch: React.FC<SymbolSearchProps> = ({ onSymbolAdd }) => {
  const [query, setQuery] = useState('');
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [trendingSymbols, setTrendingSymbols] = useState<Symbol[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeSymbols, setActiveSymbols] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [searchMode, setSearchMode] = useState<'all' | 'trending' | 'altcoins'>('all');

  // 활성 심볼 로드
  useEffect(() => {
    loadActiveSymbols();
    loadTrendingSymbols();
  }, []);

  const loadActiveSymbols = async () => {
    try {
      const response = await apiClient.get('/signals/symbols/active');
      if (response.data.success) {
        setActiveSymbols(response.data.symbols);
      }
    } catch (error) {
      console.error('Failed to load active symbols:', error);
    }
  };

  const loadTrendingSymbols = async () => {
    try {
      const response = await apiClient.get('/market/search/trending', {
        params: { limit: 10 }
      });
      if (response.data.success) {
        setTrendingSymbols(response.data.data);
      }
    } catch (error) {
      console.error('Failed to load trending symbols:', error);
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
      let response;
      
      if (searchMode === 'altcoins') {
        response = await apiClient.get('/market/search/altcoins', {
          params: { query: searchQuery, limit: 100, exclude_major: true }
        });
      } else if (searchMode === 'trending') {
        response = await apiClient.get('/market/search/trending', {
          params: { limit: 100 }
        });
      } else {
        // 일반 검색 - 신 API 사용
        response = await apiClient.get('/market/search/all', {
          params: { query: searchQuery, limit: 100 }
        });
      }

      if (response.data.success && response.data.data) {
        setSymbols(response.data.data);
        setShowResults(true);
      } else {
        setSymbols([]);
        setShowResults(true);
      }
    } catch (error) {
      console.error('Search error:', error);
      setSymbols([]);
      setShowResults(true);
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
  }, [query, searchMode]);

  // 심볼 추가
  const handleAddSymbol = async (symbol: string) => {
    try {
      const response = await apiClient.post('/signals/symbols/add', null, {
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
      await apiClient.delete(`/signals/symbols/${symbol}`);
      setActiveSymbols(activeSymbols.filter(s => s !== symbol));
    } catch (error) {
      console.error('Failed to remove symbol:', error);
    }
  };

  const formatPrice = (price: number) => {
    if (price < 0.01) return `$${price.toFixed(8)}`;
    if (price < 1) return `$${price.toFixed(4)}`;
    if (price < 1000) return `$${price.toFixed(2)}`;
    return `$${(price / 1000).toFixed(2)}K`;
  };

  const formatVolume = (volume: number) => {
    if (volume > 1000000) return `$${(volume / 1000000).toFixed(2)}M`;
    if (volume > 1000) return `$${(volume / 1000).toFixed(2)}K`;
    return `$${volume.toFixed(0)}`;
  };

  return (
    <div className="space-y-4">
      {/* 검색 모드 선택 */}
      <div className="flex gap-2">
        <button
          onClick={() => setSearchMode('all')}
          className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
            searchMode === 'all'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          모든 코인
        </button>
        <button
          onClick={() => setSearchMode('altcoins')}
          className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
            searchMode === 'altcoins'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          알트코인
        </button>
        <button
          onClick={() => setSearchMode('trending')}
          className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${
            searchMode === 'trending'
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          <Flame className="w-4 h-4" />
          트렌딩
        </button>
      </div>

      {/* 검색 입력 */}
      <div className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value.toUpperCase())}
            placeholder="심볼 검색 (예: BTC, ETH, DOGE)"
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 placeholder-gray-500"
          />
        </div>

        {/* 검색 결과 */}
        {showResults && (
          <div className="absolute z-10 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-xl max-h-96 overflow-y-auto">
            {loading ? (
              <div className="p-8 text-center">
                <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mb-2"></div>
                <p className="text-gray-600">검색 중...</p>
              </div>
            ) : symbols.length === 0 ? (
              <div className="p-8 text-center text-gray-500">
                <Search className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p>검색 결과가 없습니다</p>
              </div>
            ) : (
              <div>
                <div className="p-2 bg-gray-50 border-b border-gray-100 text-xs text-gray-600">
                  {symbols.length}개의 결과
                </div>
                <ul className="divide-y divide-gray-100">
                  {symbols.map((symbol) => (
                    <li
                      key={symbol.symbol}
                      className="p-3 hover:bg-blue-50 cursor-pointer transition-colors"
                      onClick={() => handleAddSymbol(symbol.symbol)}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold text-gray-900">
                            {symbol.symbol}
                          </div>
                          <div className="text-xs text-gray-500">
                            {symbol.baseAsset} / {symbol.quoteAsset}
                          </div>
                        </div>
                        <div className="text-right flex-shrink-0">
                          {symbol.price && (
                            <div className="text-sm font-semibold text-gray-900">
                              {formatPrice(symbol.price)}
                            </div>
                          )}
                          {symbol.priceChangePercent !== undefined && (
                            <div
                              className={`text-xs font-semibold ${
                                symbol.priceChangePercent > 0
                                  ? 'text-green-600'
                                  : symbol.priceChangePercent < 0
                                  ? 'text-red-600'
                                  : 'text-gray-600'
                              }`}
                            >
                              {symbol.priceChangePercent > 0 ? '▲ ' : symbol.priceChangePercent < 0 ? '▼ ' : ''}
                              {Math.abs(symbol.priceChangePercent).toFixed(2)}%
                            </div>
                          )}
                        </div>
                        <Plus className="w-5 h-5 text-blue-500 flex-shrink-0" />
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 트렌딩 심볼 (검색 비활성시) */}
      {!showResults && searchMode === 'trending' && trendingSymbols.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
            <Flame className="w-4 h-4 text-orange-500" />
            거래량 상위 심볼
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {trendingSymbols.map((symbol) => (
              <button
                key={symbol.symbol}
                onClick={() => handleAddSymbol(symbol.symbol)}
                className="p-2 bg-gradient-to-r from-orange-50 to-yellow-50 border border-orange-200 rounded-lg hover:border-orange-400 transition-colors text-left"
              >
                <div className="font-semibold text-gray-900">{symbol.symbol}</div>
                <div className="text-xs text-gray-600">
                  {formatPrice(symbol.price || 0)}
                  {symbol.priceChangePercent !== undefined && (
                    <span
                      className={
                        symbol.priceChangePercent > 0
                          ? ' text-green-600'
                          : ' text-red-600'
                      }
                    >
                      {' '}
                      {symbol.priceChangePercent > 0 ? '+' : ''}
                      {symbol.priceChangePercent.toFixed(2)}%
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 활성 심볼 목록 */}
      {activeSymbols.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-500" />
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
