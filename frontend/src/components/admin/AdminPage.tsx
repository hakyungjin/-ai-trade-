/**
 * Admin 페이지 - 가중치 관리 & 코인 모니터링
 */

import React, { useState, useEffect } from 'react';
import { Settings, Save, RotateCcw, TrendingUp, AlertCircle } from 'lucide-react';
import { apiClient } from '@/api/client';
import { CoinMonitoring } from './CoinMonitoring';

interface Weights {
  rsi: number;
  macd: number;
  bollinger: number;
  ema_cross: number;
  stochastic: number;
  volume: number;
}

interface Preset {
  name: string;
  description: string;
  weights: Weights;
}

export const AdminPage: React.FC = () => {
  const [weights, setWeights] = useState<Weights>({
    rsi: 0.20,
    macd: 0.25,
    bollinger: 0.15,
    ema_cross: 0.20,
    stochastic: 0.10,
    volume: 0.10
  });

  const [presets, setPresets] = useState<Record<string, Preset>>({});
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'weights' | 'coins'>('weights');

  // 초기 데이터 로드
  useEffect(() => {
    loadWeights();
    loadPresets();
  }, []);

  const loadWeights = async () => {
    try {
      const response = await apiClient.get('/api/admin/weights');
      if (response.data.success) {
        setWeights(response.data.weights);
      }
    } catch (error) {
      console.error('Failed to load weights:', error);
    }
  };

  const loadPresets = async () => {
    try {
      const response = await apiClient.get('/api/admin/presets');
      if (response.data.success) {
        setPresets(response.data.presets);
      }
    } catch (error) {
      console.error('Failed to load presets:', error);
    }
  };

  // 가중치 변경
  const handleWeightChange = (key: keyof Weights, value: number) => {
    setWeights({
      ...weights,
      [key]: value
    });
  };

  // 가중치 저장
  const handleSave = async () => {
    setLoading(true);
    setMessage(null);

    try {
      const response = await apiClient.put('/api/admin/weights', weights);

      if (response.data.success) {
        setMessage({ type: 'success', text: '가중치가 저장되었습니다!' });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail?.message || '저장에 실패했습니다';
      setMessage({ type: 'error', text: errorMsg });
    } finally {
      setLoading(false);
    }
  };

  // 프리셋 로드
  const handleLoadPreset = async (presetName: string) => {
    setLoading(true);
    setMessage(null);

    try {
      const response = await apiClient.post(`/api/admin/presets/${presetName}/load`);

      if (response.data.success) {
        setWeights(response.data.weights);
        setMessage({ type: 'success', text: `프리셋 "${response.data.preset.name}" 로드 완료!` });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error) {
      setMessage({ type: 'error', text: '프리셋 로드에 실패했습니다' });
    } finally {
      setLoading(false);
    }
  };

  // 기본값으로 리셋
  const handleReset = async () => {
    if (!confirm('기본 가중치로 리셋하시겠습니까?')) {
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const response = await apiClient.post('/api/admin/weights/reset');

      if (response.data.success) {
        setWeights(response.data.weights);
        setMessage({ type: 'success', text: '기본값으로 리셋되었습니다!' });
        setTimeout(() => setMessage(null), 3000);
      }
    } catch (error) {
      setMessage({ type: 'error', text: '리셋에 실패했습니다' });
    } finally {
      setLoading(false);
    }
  };

  // 가중치 합계
  const total = Object.values(weights).reduce((sum, val) => sum + val, 0);
  const isValid = total >= 0.95 && total <= 1.05;

  // 지표 정보
  const indicatorInfo: Record<keyof Weights, { name: string; description: string; color: string }> = {
    rsi: {
      name: 'RSI',
      description: '과매수/과매도 판단',
      color: 'bg-purple-500'
    },
    macd: {
      name: 'MACD',
      description: '추세 전환 감지',
      color: 'bg-blue-500'
    },
    bollinger: {
      name: 'Bollinger Bands',
      description: '변동성 분석',
      color: 'bg-green-500'
    },
    ema_cross: {
      name: 'EMA Cross',
      description: '이동평균선 교차',
      color: 'bg-yellow-500'
    },
    stochastic: {
      name: 'Stochastic',
      description: '모멘텀 지표',
      color: 'bg-red-500'
    },
    volume: {
      name: 'Volume',
      description: '거래량 분석',
      color: 'bg-indigo-500'
    }
  };

  // 코인 모니터링 탭이면 해당 컴포넌트 반환
  if (activeTab === 'coins') {
    return (
      <div className="min-h-screen bg-gray-50">
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />
        <CoinMonitoring />
      </div>
    );
  }

  // 가중치 관리 탭
  return (
    <div className="min-h-screen bg-gray-50">
      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      {/* 가중치 관리 콘텐츠 */}
      <div className="p-6">
        <div className="max-w-6xl mx-auto">
          {/* 헤더 */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-2">
              <Settings className="w-8 h-8 text-blue-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                Admin - 가중치 관리
              </h1>
            </div>
            <p className="text-gray-600">
              전략 지표의 가중치를 조정하여 트레이딩 신호를 커스터마이징하세요
            </p>
          </div>

          {/* 메시지 */}
          {message && (
            <div
              className={`mb-6 p-4 rounded-lg ${
                message.type === 'success'
                  ? 'bg-green-50 text-green-800'
                  : 'bg-red-50 text-red-800'
              }`}
            >
              <div className="flex items-center gap-2">
                {message.type === 'success' ? (
                  <TrendingUp className="w-5 h-5" />
                ) : (
                  <AlertCircle className="w-5 h-5" />
                )}
                <span>{message.text}</span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 가중치 조정 */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">
                  지표 가중치 조정
                </h2>

                <div className="space-y-6">
                  {(Object.keys(weights) as Array<keyof Weights>).map((key) => {
                    const info = indicatorInfo[key];
                    return (
                      <div key={key}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className={`w-3 h-3 rounded-full ${info.color}`} />
                            <span className="font-semibold text-gray-900">
                              {info.name}
                            </span>
                            <span className="text-sm text-gray-500">
                              ({info.description})
                            </span>
                          </div>
                          <span className="text-lg font-bold text-blue-600">
                            {(weights[key] * 100).toFixed(0)}%
                          </span>
                        </div>

                        <input
                          type="range"
                          min="0"
                          max="100"
                          step="1"
                          value={weights[key] * 100}
                          onChange={(e) =>
                            handleWeightChange(
                              key,
                              parseFloat(e.target.value) / 100
                            )
                          }
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                          style={{
                            background: `linear-gradient(to right, ${info.color.replace(
                              'bg-',
                              '#'
                            )} ${weights[key] * 100}%, #e5e7eb ${
                              weights[key] * 100
                            }%)`
                          }}
                        />

                        <div className="flex justify-between text-xs text-gray-400 mt-1">
                          <span>0%</span>
                          <span>50%</span>
                          <span>100%</span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* 합계 표시 */}
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-semibold text-gray-900">
                      가중치 합계:
                    </span>
                    <span
                      className={`text-2xl font-bold ${
                        isValid ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {(total * 100).toFixed(1)}%
                    </span>
                  </div>
                  {!isValid && (
                    <p className="text-sm text-red-600 mt-2">
                      ⚠️ 가중치 합계는 95-105% 범위여야 합니다
                    </p>
                  )}
                </div>

                {/* 액션 버튼 */}
                <div className="mt-6 flex gap-3">
                  <button
                    onClick={handleSave}
                    disabled={loading || !isValid}
                    className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Save className="w-5 h-5" />
                    저장
                  </button>
                  <button
                    onClick={handleReset}
                    disabled={loading}
                    className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <RotateCcw className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* 프리셋 */}
            <div>
              <div className="bg-white rounded-lg shadow-md p-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  프리셋
                </h2>

                <div className="space-y-3">
                  {Object.entries(presets).map(([key, preset]) => (
                    <button
                      key={key}
                      onClick={() => handleLoadPreset(key)}
                      disabled={loading}
                      className="w-full text-left p-4 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <div className="font-semibold text-gray-900 mb-1">
                        {preset.name}
                      </div>
                      <div className="text-sm text-gray-600">
                        {preset.description}
                      </div>
                    </button>
                  ))}
                </div>

                {/* 프리셋 정보 */}
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h3 className="text-sm font-semibold text-blue-900 mb-2">
                    💡 프리셋 정보
                  </h3>
                  <ul className="text-xs text-blue-800 space-y-1">
                    <li>• 균형: 모든 지표 균형있게</li>
                    <li>• 추세 추종: MACD와 EMA 중심</li>
                    <li>• 모멘텀: RSI와 Stochastic 중심</li>
                    <li>• 변동성: Bollinger Bands 중심</li>
                    <li>• 거래량: Volume 분석 중심</li>
                  </ul>
                </div>
              </div>

              {/* 가이드 */}
              <div className="mt-6 bg-yellow-50 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-yellow-900 mb-2">
                  ⚠️ 주의사항
                </h3>
                <ul className="text-xs text-yellow-800 space-y-1">
                  <li>• 가중치 변경은 즉시 적용됩니다</li>
                  <li>• 백테스팅으로 검증 후 사용하세요</li>
                  <li>• 극단적인 가중치는 위험할 수 있습니다</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// 탭 네비게이션 컴포넌트
const TabNavigation: React.FC<{
  activeTab: 'weights' | 'coins';
  onTabChange: (tab: 'weights' | 'coins') => void;
}> = ({ activeTab, onTabChange }) => (
  <div className="border-b border-gray-200 bg-white sticky top-0 z-10">
    <div className="max-w-7xl mx-auto px-6">
      <div className="flex gap-8">
        <button
          onClick={() => onTabChange('weights')}
          className={`py-4 px-2 border-b-2 font-semibold transition-colors flex items-center gap-2 ${
            activeTab === 'weights'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          <Settings className="w-5 h-5" />
          가중치 관리
        </button>
        <button
          onClick={() => onTabChange('coins')}
          className={`py-4 px-2 border-b-2 font-semibold transition-colors flex items-center gap-2 ${
            activeTab === 'coins'
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          <TrendingUp className="w-5 h-5" />
          코인 모니터링
        </button>
      </div>
    </div>
  </div>
);
