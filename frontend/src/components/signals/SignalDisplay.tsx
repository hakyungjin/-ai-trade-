/**
 * 실시간 신호 표시 컴포넌트
 */

import React, { useState } from 'react';
import { TrendingUp, TrendingDown, Minus, Activity, ChevronDown, ChevronUp } from 'lucide-react';
import { SignalComparison } from './SignalComparison';

interface SignalData {
  symbol: string;
  timestamp: string;
  price: number;
  signal: 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';
  strength: 'very_strong' | 'strong' | 'moderate' | 'weak' | 'very_weak';
  confidence: number;
  score: number;
  weighted_signal?: string;
  weighted_confidence?: number;
  ai_signal?: string | null;
  ai_confidence?: number | null;
  recommendation: {
    action: string;
    strength: string;
    message: string;
    strength_message: string;
    action_text: string;
  };
  indicators?: {
    rsi?: { value: number; signal: string };
    macd?: { value: number; signal: string };
    bollinger?: { signal: string };
    ema_cross?: { signal: string };
  };
}

interface SignalDisplayProps {
  signal: SignalData;
  onClick?: () => void;
}

export const SignalDisplay: React.FC<SignalDisplayProps> = ({ signal, onClick }) => {
  const [showComparison, setShowComparison] = useState(false);

  // 신호 색상
  const getSignalColor = (signalType: string) => {
    switch (signalType) {
      case 'strong_buy':
        return 'bg-green-600 text-white';
      case 'buy':
        return 'bg-green-400 text-white';
      case 'neutral':
        return 'bg-gray-400 text-white';
      case 'sell':
        return 'bg-red-400 text-white';
      case 'strong_sell':
        return 'bg-red-600 text-white';
      default:
        return 'bg-gray-300 text-gray-800';
    }
  };

  // 신호 아이콘
  const getSignalIcon = (signalType: string) => {
    switch (signalType) {
      case 'strong_buy':
      case 'buy':
        return <TrendingUp className="w-6 h-6" />;
      case 'sell':
      case 'strong_sell':
        return <TrendingDown className="w-6 h-6" />;
      case 'neutral':
        return <Minus className="w-6 h-6" />;
      default:
        return <Activity className="w-6 h-6" />;
    }
  };

  // 강도 색상
  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'very_strong':
        return 'text-purple-600 font-bold';
      case 'strong':
        return 'text-blue-600 font-semibold';
      case 'moderate':
        return 'text-yellow-600';
      case 'weak':
        return 'text-orange-500';
      case 'very_weak':
        return 'text-gray-500';
      default:
        return 'text-gray-600';
    }
  };

  // 신호 한글 변환
  const getSignalText = (signalType: string) => {
    switch (signalType) {
      case 'strong_buy':
        return '강한 매수';
      case 'buy':
        return '매수';
      case 'neutral':
        return '횡보';
      case 'sell':
        return '매도';
      case 'strong_sell':
        return '강한 매도';
      default:
        return '알 수 없음';
    }
  };

  // 강도 한글 변환
  const getStrengthText = (strength: string) => {
    switch (strength) {
      case 'very_strong':
        return '매우 강함';
      case 'strong':
        return '강함';
      case 'moderate':
        return '보통';
      case 'weak':
        return '약함';
      case 'very_weak':
        return '매우 약함';
      default:
        return '-';
    }
  };

  return (
    <div
      className="bg-white rounded-lg shadow-md border border-gray-200 p-4 hover:shadow-lg transition-shadow"
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-xl font-bold text-gray-900">{signal.symbol}</h3>
          <span className="text-lg text-gray-600">${signal.price.toLocaleString()}</span>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${getSignalColor(signal.signal)}`}>
          {getSignalIcon(signal.signal)}
          <span className="font-semibold">{getSignalText(signal.signal)}</span>
        </div>
      </div>

      {/* 신호 상세 */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div className="text-sm text-gray-500">신호 강도</div>
          <div className={`text-lg font-semibold ${getStrengthColor(signal.strength)}`}>
            {getStrengthText(signal.strength)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-500">신뢰도</div>
          <div className="text-lg font-semibold text-blue-600">
            {(signal.confidence * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* 신호 점수 게이지 */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-gray-500">신호 점수</span>
          <span className="font-semibold">{signal.score.toFixed(2)}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all ${
              signal.score > 0 ? 'bg-green-500' : signal.score < 0 ? 'bg-red-500' : 'bg-gray-400'
            }`}
            style={{
              width: `${Math.abs(signal.score) * 50}%`,
              marginLeft: signal.score < 0 ? `${50 - Math.abs(signal.score) * 50}%` : '50%'
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>매도</span>
          <span>중립</span>
          <span>매수</span>
        </div>
      </div>

      {/* 권장사항 */}
      <div className="bg-blue-50 rounded-lg p-3 mb-4">
        <div className="text-sm font-semibold text-blue-900 mb-1">
          {signal.recommendation.message}
        </div>
        <div className="text-xs text-blue-700">
          {signal.recommendation.strength_message}
        </div>
      </div>

      {/* 기술적 지표 요약 */}
      {signal.indicators && (
        <div className="grid grid-cols-2 gap-2 text-sm">
          {signal.indicators.rsi && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">RSI:</span>
              <span className={`font-semibold ${
                signal.indicators.rsi.signal === 'oversold' ? 'text-green-600' :
                signal.indicators.rsi.signal === 'overbought' ? 'text-red-600' :
                'text-gray-600'
              }`}>
                {signal.indicators.rsi.value.toFixed(1)}
              </span>
            </div>
          )}
          {signal.indicators.macd && (
            <div className="flex items-center justify-between">
              <span className="text-gray-600">MACD:</span>
              <span className={`font-semibold ${
                signal.indicators.macd.signal === 'bullish' ? 'text-green-600' :
                signal.indicators.macd.signal === 'bearish' ? 'text-red-600' :
                'text-gray-600'
              }`}>
                {signal.indicators.macd.signal}
              </span>
            </div>
          )}
        </div>
      )}

      {/* 전략 비교 버튼 */}
      {signal.weighted_signal && (
        <div className="mt-4">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowComparison(!showComparison);
            }}
            className="w-full flex items-center justify-center gap-2 py-2 px-4 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-sm font-semibold text-gray-700"
          >
            {showComparison ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            전략 비교 {showComparison ? '접기' : '펼치기'}
          </button>

          {/* 비교 섹션 */}
          {showComparison && (
            <div className="mt-4 pt-4 border-t border-gray-200" onClick={(e) => e.stopPropagation()}>
              <SignalComparison
                weightedSignal={signal.weighted_signal}
                weightedConfidence={signal.weighted_confidence || 0}
                aiSignal={signal.ai_signal}
                aiConfidence={signal.ai_confidence}
                finalSignal={signal.signal}
                finalConfidence={signal.confidence}
                finalScore={signal.score}
              />
            </div>
          )}
        </div>
      )}

      {/* 타임스탬프 */}
      <div className="text-xs text-gray-400 mt-4">
        업데이트: {new Date(signal.timestamp).toLocaleString('ko-KR')}
      </div>
    </div>
  );
};
