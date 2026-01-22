/**
 * 코인 모니터링 페이지
 * - 모니터링 중인 코인 목록 조회
 * - 새 코인 추가
 * - 코인 정보 및 설정 관리
 * - 통계 조회
 */

import React, { useState, useEffect } from 'react';
import {
  Plus,
  Trash2,
  Settings,
  TrendingUp,
  TrendingDown,
  Loader,
  AlertCircle,
  Check,
  X,
  Eye,
} from 'lucide-react';
import { apiClient } from '@/api/client';

interface Coin {
  id: number;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  is_monitoring: boolean;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  market_cap: number;
  candle_count: number;
  monitoring_timeframes: string[];
}

interface CoinStats {
  coin_id: number;
  total_candles: number;
  candles_1h: number;
  candles_4h: number;
  candles_1d: number;
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  neutral_signals: number;
  average_confidence: number;
  win_rate: number;
}

interface CoinConfig {
  coin_id: number;
  use_rsi: boolean;
  use_macd: boolean;
  use_bollinger: boolean;
  use_ema_cross: boolean;
  use_stochastic: boolean;
  use_volume: boolean;
  ai_model: string;
  buy_threshold: number;
  strong_buy_threshold: number;
  sell_threshold: number;
  strong_sell_threshold: number;
}

export const CoinMonitoring: React.FC = () => {
  const [coins, setCoins] = useState<Coin[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  
  // 검색/추가 상태
  const [searchSymbol, setSearchSymbol] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);
  const [addLoading, setAddLoading] = useState(false);
  
  // 상세 정보 모달
  const [selectedCoin, setSelectedCoin] = useState<Coin | null>(null);
  const [coinStats, setCoinStats] = useState<CoinStats | null>(null);
  const [coinConfig, setCoinConfig] = useState<CoinConfig | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // 초기 데이터 로드
  useEffect(() => {
    loadCoins();
  }, []);

  const loadCoins = async () => {
    setLoading(true);
    try {
      const response = await apiClient.get('/api/v1/coins/monitoring');
      if (response.data.success) {
        setCoins(response.data.data || []);
      }
    } catch (error) {
      console.error('Failed to load coins:', error);
      setMessage({ type: 'error', text: '코인 목록 로드 실패' });
    } finally {
      setLoading(false);
    }
  };

  // 코인 추가
  const handleAddCoin = async () => {
    if (!searchSymbol.trim()) {
      setMessage({ type: 'error', text: '심볼을 입력하세요' });
      return;
    }

    setAddLoading(true);
    try {
      const response = await apiClient.post(
        `/api/v1/coins/add-monitoring/${searchSymbol.toUpperCase()}`
      );

      if (response.data.success) {
        setMessage({ type: 'success', text: `${searchSymbol} 추가 완료!` });
        setSearchSymbol('');
        setShowAddForm(false);
        loadCoins();
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || '코인 추가 실패';
      setMessage({ type: 'error', text: errorMsg });
    } finally {
      setAddLoading(false);
    }
  };

  // 코인 삭제
  const handleDeleteCoin = async (coinId: number, symbol: string) => {
    if (!confirm(`${symbol}을 삭제하시겠습니까?`)) {
      return;
    }

    try {
      const response = await apiClient.delete(`/api/v1/coins/${coinId}`);
      if (response.data.success) {
        setMessage({ type: 'success', text: `${symbol} 삭제 완료` });
        loadCoins();
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: '코인 삭제 실패' });
    }
  };

  // 코인 상세 정보 로드
  const handleViewDetails = async (coin: Coin) => {
    setSelectedCoin(coin);
    setShowDetails(true);

    try {
      // 통계 로드
      const statsResponse = await apiClient.get(`/api/v1/coins/${coin.id}/stats`);
      if (statsResponse.data.success) {
        setCoinStats(statsResponse.data.data);
      }

      // 설정 로드
      const configResponse = await apiClient.get(`/api/v1/coins/${coin.id}/config`);
      if (configResponse.data.success) {
        setCoinConfig(configResponse.data.data);
      }
    } catch (error) {
      console.error('Failed to load coin details:', error);
      setMessage({ type: 'error', text: '코인 상세 정보 로드 실패' });
    }
  };

  // 설정 저장
  const handleSaveConfig = async () => {
    if (!selectedCoin || !coinConfig) return;

    try {
      const response = await apiClient.put(
        `/api/v1/coins/${selectedCoin.id}/config`,
        coinConfig
      );

      if (response.data.success) {
        setMessage({ type: 'success', text: '설정 저장 완료' });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error: any) {
      setMessage({ type: 'error', text: '설정 저장 실패' });
    }
  };

  // 포맷팅 함수들
  const formatPrice = (price: number) => {
    if (price >= 1000) return `$${(price / 1000).toFixed(1)}K`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(4)}`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000000) return `${(volume / 1000000000).toFixed(1)}B`;
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`;
    return volume.toFixed(0);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 헤더 */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <TrendingUp className="w-8 h-8 text-blue-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                코인 모니터링
              </h1>
            </div>
            <p className="text-gray-600">
              모니터링 중인 암호화폐를 관리하고 분석하세요
            </p>
          </div>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-5 h-5" />
            코인 추가
          </button>
        </div>

        {/* 메시지 */}
        {message && (
          <div
            className={`mb-6 p-4 rounded-lg flex items-center gap-2 ${
              message.type === 'success'
                ? 'bg-green-50 text-green-800'
                : 'bg-red-50 text-red-800'
            }`}
          >
            {message.type === 'success' ? (
              <Check className="w-5 h-5" />
            ) : (
              <AlertCircle className="w-5 h-5" />
            )}
            <span>{message.text}</span>
          </div>
        )}

        {/* 추가 폼 */}
        {showAddForm && (
          <div className="mb-6 bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              새 코인 추가
            </h3>
            <div className="flex gap-3">
              <input
                type="text"
                placeholder="심볼 입력 (예: BTCUSDT)"
                value={searchSymbol}
                onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') handleAddCoin();
                }}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={handleAddCoin}
                disabled={addLoading}
                className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors flex items-center gap-2"
              >
                {addLoading ? (
                  <Loader className="w-5 h-5 animate-spin" />
                ) : (
                  <Plus className="w-5 h-5" />
                )}
                추가
              </button>
              <button
                onClick={() => setShowAddForm(false)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>
        )}

        {/* 로딩 */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader className="w-8 h-8 text-blue-600 animate-spin" />
          </div>
        ) : coins.length === 0 ? (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <TrendingUp className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600">모니터링 중인 코인이 없습니다</p>
            <p className="text-sm text-gray-500 mt-2">
              "코인 추가" 버튼을 클릭하여 첫 코인을 추가하세요
            </p>
          </div>
        ) : (
          /* 코인 테이블 */
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-100 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    심볼
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    가격
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    24h 변화
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    24h 거래량
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    시간봉
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    수집 캔들
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-900">
                    작업
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {coins.map((coin) => (
                  <tr
                    key={coin.id}
                    className="hover:bg-gray-50 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="font-semibold text-gray-900">
                        {coin.symbol}
                      </div>
                      <div className="text-sm text-gray-500">
                        {coin.base_asset}/{coin.quote_asset}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="font-mono text-gray-900">
                        {formatPrice(coin.current_price)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div
                        className={`flex items-center gap-2 ${
                          coin.price_change_24h >= 0
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}
                      >
                        {coin.price_change_24h >= 0 ? (
                          <TrendingUp className="w-4 h-4" />
                        ) : (
                          <TrendingDown className="w-4 h-4" />
                        )}
                        <span>
                          {coin.price_change_24h.toFixed(2)}%
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-gray-600">
                        ${formatVolume(coin.volume_24h)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-wrap gap-2">
                        {coin.monitoring_timeframes?.map((tf) => (
                          <span
                            key={tf}
                            className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"
                          >
                            {tf}
                          </span>
                        )) || (
                          <span className="text-gray-500 text-sm">
                            없음
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-semibold ${
                          coin.candle_count >= 100
                            ? 'bg-green-100 text-green-700'
                            : coin.candle_count >= 50
                            ? 'bg-yellow-100 text-yellow-700'
                            : 'bg-gray-100 text-gray-700'
                        }`}
                      >
                        {coin.candle_count}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleViewDetails(coin)}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="상세 정보"
                        >
                          <Eye className="w-5 h-5" />
                        </button>
                        <button
                          onClick={() =>
                            handleDeleteCoin(coin.id, coin.symbol)
                          }
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="삭제"
                        >
                          <Trash2 className="w-5 h-5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* 상세 정보 모달 */}
        {showDetails && selectedCoin && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-lg shadow-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              {/* 모달 헤더 */}
              <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">
                  {selectedCoin.symbol} - 상세 정보
                </h2>
                <button
                  onClick={() => {
                    setShowDetails(false);
                    setSelectedCoin(null);
                    setCoinStats(null);
                    setCoinConfig(null);
                  }}
                  className="text-gray-500 hover:text-gray-700"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* 모달 내용 */}
              <div className="p-6 space-y-6">
                {/* 기본 정보 */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    기본 정보
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-gray-600">
                        현재 가격
                      </label>
                      <p className="text-lg font-semibold text-gray-900">
                        {formatPrice(selectedCoin.current_price)}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-600">
                        시장 지분
                      </label>
                      <p className="text-lg font-semibold text-gray-900">
                        ${formatVolume(selectedCoin.market_cap)}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-600">
                        24h 변화
                      </label>
                      <p
                        className={`text-lg font-semibold ${
                          selectedCoin.price_change_24h >= 0
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}
                      >
                        {selectedCoin.price_change_24h.toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-600">
                        24h 거래량
                      </label>
                      <p className="text-lg font-semibold text-gray-900">
                        ${formatVolume(selectedCoin.volume_24h)}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 통계 */}
                {coinStats && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      📊 통계
                    </h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm text-gray-600">
                          전체 캔들
                        </label>
                        <p className="text-lg font-semibold text-gray-900">
                          {coinStats.total_candles}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          1시간봉
                        </label>
                        <p className="text-lg font-semibold text-gray-900">
                          {coinStats.candles_1h}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          4시간봉
                        </label>
                        <p className="text-lg font-semibold text-gray-900">
                          {coinStats.candles_4h}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          일봉
                        </label>
                        <p className="text-lg font-semibold text-gray-900">
                          {coinStats.candles_1d}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          총 신호
                        </label>
                        <p className="text-lg font-semibold text-gray-900">
                          {coinStats.total_signals}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          매수 신호
                        </label>
                        <p className="text-lg font-semibold text-green-600">
                          {coinStats.buy_signals}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          매도 신호
                        </label>
                        <p className="text-lg font-semibold text-red-600">
                          {coinStats.sell_signals}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          중립 신호
                        </label>
                        <p className="text-lg font-semibold text-gray-600">
                          {coinStats.neutral_signals}
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          평균 확신도
                        </label>
                        <p className="text-lg font-semibold text-blue-600">
                          {(coinStats.average_confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <label className="text-sm text-gray-600">
                          승률
                        </label>
                        <p className="text-lg font-semibold text-purple-600">
                          {(coinStats.win_rate * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* 설정 */}
                {coinConfig && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      ⚙️ 분석 설정
                    </h3>

                    {/* 지표 선택 */}
                    <div className="bg-gray-50 rounded-lg p-4 mb-4">
                      <h4 className="font-semibold text-gray-900 mb-3">
                        사용 지표
                      </h4>
                      <div className="grid grid-cols-2 gap-3">
                        {[
                          { key: 'use_rsi', label: 'RSI' },
                          { key: 'use_macd', label: 'MACD' },
                          { key: 'use_bollinger', label: 'Bollinger Bands' },
                          { key: 'use_ema_cross', label: 'EMA Cross' },
                          { key: 'use_stochastic', label: 'Stochastic' },
                          { key: 'use_volume', label: 'Volume' },
                        ].map(({ key, label }) => (
                          <label
                            key={key}
                            className="flex items-center gap-2 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={
                                coinConfig[
                                  key as keyof CoinConfig
                                ] as boolean
                              }
                              onChange={(e) => {
                                setCoinConfig({
                                  ...coinConfig,
                                  [key]: e.target.checked,
                                });
                              }}
                              className="w-4 h-4 rounded border-gray-300"
                            />
                            <span className="text-sm text-gray-700">
                              {label}
                            </span>
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* AI 모델 선택 */}
                    <div className="mb-4">
                      <label className="block text-sm font-semibold text-gray-900 mb-2">
                        AI 모델
                      </label>
                      <select
                        value={coinConfig.ai_model}
                        onChange={(e) => {
                          setCoinConfig({
                            ...coinConfig,
                            ai_model: e.target.value,
                          });
                        }}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="gemini">Gemini AI</option>
                        <option value="local">Local Model</option>
                        <option value="vector_patterns">
                          Vector Patterns
                        </option>
                      </select>
                    </div>

                    {/* 신호 임계값 */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-semibold text-gray-900 mb-3">
                        신호 임계값
                      </h4>
                      <div className="space-y-3">
                        <div>
                          <label className="block text-sm text-gray-700 mb-1">
                            매수 (Buy):
                            <span className="font-semibold ml-2">
                              {coinConfig.buy_threshold.toFixed(2)}
                            </span>
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={coinConfig.buy_threshold}
                            onChange={(e) => {
                              setCoinConfig({
                                ...coinConfig,
                                buy_threshold: parseFloat(
                                  e.target.value
                                ),
                              });
                            }}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-gray-700 mb-1">
                            강한 매수 (Strong Buy):
                            <span className="font-semibold ml-2">
                              {coinConfig.strong_buy_threshold.toFixed(2)}
                            </span>
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={coinConfig.strong_buy_threshold}
                            onChange={(e) => {
                              setCoinConfig({
                                ...coinConfig,
                                strong_buy_threshold: parseFloat(
                                  e.target.value
                                ),
                              });
                            }}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-gray-700 mb-1">
                            매도 (Sell):
                            <span className="font-semibold ml-2">
                              {coinConfig.sell_threshold.toFixed(2)}
                            </span>
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={coinConfig.sell_threshold}
                            onChange={(e) => {
                              setCoinConfig({
                                ...coinConfig,
                                sell_threshold: parseFloat(
                                  e.target.value
                                ),
                              });
                            }}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-sm text-gray-700 mb-1">
                            강한 매도 (Strong Sell):
                            <span className="font-semibold ml-2">
                              {coinConfig.strong_sell_threshold.toFixed(2)}
                            </span>
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={coinConfig.strong_sell_threshold}
                            onChange={(e) => {
                              setCoinConfig({
                                ...coinConfig,
                                strong_sell_threshold: parseFloat(
                                  e.target.value
                                ),
                              });
                            }}
                            className="w-full"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* 액션 버튼 */}
                <div className="flex gap-3 pt-4 border-t border-gray-200">
                  <button
                    onClick={handleSaveConfig}
                    className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <Settings className="w-5 h-5" />
                    설정 저장
                  </button>
                  <button
                    onClick={() => {
                      setShowDetails(false);
                      setSelectedCoin(null);
                      setCoinStats(null);
                      setCoinConfig(null);
                    }}
                    className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                  >
                    닫기
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
