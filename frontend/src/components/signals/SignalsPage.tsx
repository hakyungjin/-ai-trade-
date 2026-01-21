/**
 * 실시간 신호 페이지
 */

import React, { useState, useEffect } from 'react';
import { SymbolSearch } from './SymbolSearch';
import { SignalDisplay } from './SignalDisplay';
import { RefreshCw, AlertCircle } from 'lucide-react';
import { apiClient } from '@/api/client';

interface SignalData {
  symbol: string;
  timestamp: string;
  price: number;
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  strength: 'very_strong' | 'strong' | 'moderate' | 'weak' | 'very_weak';
  confidence: number;
  score: number;
  recommendation: any;
  indicators?: any;
}

export const SignalsPage: React.FC = () => {
  const [signals, setSignals] = useState<Record<string, SignalData>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ws, setWs] = useState<WebSocket | null>(null);

  // WebSocket 연결
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
      const socket = new WebSocket(`${wsUrl}/api/signals/ws/signals`);

      socket.onopen = () => {
        console.log('WebSocket connected');
        setError(null);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'signals_update' && data.signals) {
            setSignals(data.signals);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('실시간 연결에 문제가 발생했습니다');
      };

      socket.onclose = () => {
        console.log('WebSocket disconnected');
        // 재연결 시도
        setTimeout(connectWebSocket, 5000);
      };

      setWs(socket);
    } catch (err) {
      console.error('Failed to connect WebSocket:', err);
      setError('실시간 연결을 시작할 수 없습니다');
    }
  };

  // 수동 새로고침
  const handleRefresh = async () => {
    setLoading(true);
    try {
      const response = await apiClient.post('/api/signals/signals/update');
      if (response.data.success) {
        setSignals(response.data.signals);
      }
    } catch (err) {
      console.error('Failed to refresh signals:', err);
      setError('신호 업데이트 실패');
    } finally {
      setLoading(false);
    }
  };

  // 심볼 추가 핸들러
  const handleSymbolAdd = async (symbol: string) => {
    // 즉시 신호 가져오기
    try {
      const response = await apiClient.get(`/api/signals/signal/${symbol}`);
      if (response.data.success) {
        setSignals({
          ...signals,
          [symbol]: response.data.signal
        });
      }
    } catch (err) {
      console.error('Failed to get initial signal:', err);
    }
  };

  const signalList = Object.values(signals);

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 헤더 */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                실시간 트레이딩 신호
              </h1>
              <p className="text-gray-600 mt-2">
                AI와 기술적 분석을 결합한 실시간 매매 신호
              </p>
            </div>
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
              새로고침
            </button>
          </div>

          {/* 오류 메시지 */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <span className="text-red-800">{error}</span>
            </div>
          )}
        </div>

        {/* 심볼 검색 */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            심볼 검색 및 등록
          </h2>
          <SymbolSearch onSymbolAdd={handleSymbolAdd} />
        </div>

        {/* 신호 목록 */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              실시간 신호 ({signalList.length})
            </h2>
            {ws && ws.readyState === WebSocket.OPEN && (
              <div className="flex items-center gap-2 text-green-600">
                <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse" />
                <span className="text-sm">실시간 연결됨</span>
              </div>
            )}
          </div>

          {signalList.length === 0 ? (
            <div className="bg-white rounded-lg shadow-md p-12 text-center">
              <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">
                검색하여 심볼을 추가하면 실시간 신호를 받을 수 있습니다
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {signalList.map((signal) => (
                <SignalDisplay
                  key={signal.symbol}
                  signal={signal}
                  onClick={() => {
                    // 차트 페이지로 이동 또는 모달 열기
                    console.log('Signal clicked:', signal.symbol);
                  }}
                />
              ))}
            </div>
          )}
        </div>

        {/* 사용 가이드 */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">
            📊 신호 해석 가이드
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
            <div>
              <strong>🚀 강한 매수:</strong> 여러 지표가 강한 매수 신호. 적극적 진입 고려
            </div>
            <div>
              <strong>📈 매수:</strong> 매수 신호 포착. 신중한 진입 권장
            </div>
            <div>
              <strong>⏸️ 횡보:</strong> 명확한 방향성 없음. 관망 권장
            </div>
            <div>
              <strong>📉 매도:</strong> 매도 신호 포착. 포지션 정리 고려
            </div>
            <div>
              <strong>🔴 강한 매도:</strong> 여러 지표가 강한 매도 신호. 즉시 청산 고려
            </div>
            <div>
              <strong>💪 신호 강도:</strong> 신뢰도와 여러 지표의 일치도를 나타냄
            </div>
          </div>
          <div className="mt-4 text-xs text-blue-700">
            ⚠️ 이 신호는 참고용이며, 투자 결정은 본인의 판단과 책임하에 진행하세요.
          </div>
        </div>
      </div>
    </div>
  );
};
