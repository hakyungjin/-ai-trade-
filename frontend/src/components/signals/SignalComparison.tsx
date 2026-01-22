/**
 * 신호 비교 컴포넌트 - AI vs 가중치 전략
 */

import React from 'react';
import { TrendingUp, TrendingDown, Brain, BarChart3, Minus } from 'lucide-react';

interface SignalComparisonProps {
  weightedSignal: string;
  weightedConfidence: number;
  aiSignal: string | null;
  aiConfidence: number | null;
  finalSignal: string;
  finalConfidence: number;
  finalScore: number;
}

export const SignalComparison: React.FC<SignalComparisonProps> = ({
  weightedSignal,
  weightedConfidence,
  aiSignal,
  aiConfidence,
  finalSignal,
  finalConfidence,
  finalScore
}) => {
  // 신호 한글 변환
  const getSignalText = (signal: string | null) => {
    if (!signal) return '-';
    switch (signal) {
      case 'strong_buy': return '강한 매수';
      case 'buy': return '매수';
      case 'neutral': return '횡보';
      case 'sell': return '매도';
      case 'strong_sell': return '강한 매도';
      default: return signal;
    }
  };

  // 신호 색상
  const getSignalColor = (signal: string | null) => {
    if (!signal) return 'text-gray-400';
    switch (signal) {
      case 'strong_buy': return 'text-green-600 font-bold';
      case 'buy': return 'text-green-500';
      case 'neutral': return 'text-gray-600';
      case 'sell': return 'text-red-500';
      case 'strong_sell': return 'text-red-600 font-bold';
      default: return 'text-gray-600';
    }
  };

  // 신호 아이콘
  const getSignalIcon = (signal: string | null) => {
    if (!signal) return <Minus className="w-5 h-5" />;
    switch (signal) {
      case 'strong_buy':
      case 'buy':
        return <TrendingUp className="w-5 h-5" />;
      case 'sell':
      case 'strong_sell':
        return <TrendingDown className="w-5 h-5" />;
      default:
        return <Minus className="w-5 h-5" />;
    }
  };

  // 배경 색상
  const getBackgroundColor = (signal: string | null) => {
    if (!signal) return 'bg-gray-50';
    switch (signal) {
      case 'strong_buy': return 'bg-green-100';
      case 'buy': return 'bg-green-50';
      case 'neutral': return 'bg-gray-50';
      case 'sell': return 'bg-red-50';
      case 'strong_sell': return 'bg-red-100';
      default: return 'bg-gray-50';
    }
  };

  return (
    <div className="space-y-4">
      {/* 제목 */}
      <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
        <BarChart3 className="w-5 h-5" />
        전략 비교
      </h3>

      {/* 비교 카드들 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* 가중치 기반 전략 */}
        <div className={`p-4 rounded-lg border-2 border-blue-200 ${getBackgroundColor(weightedSignal)}`}>
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-5 h-5 text-blue-600" />
            <span className="font-semibold text-gray-900">가중치 기반</span>
            <span className="text-xs text-gray-500">(60%)</span>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">신호:</span>
              <div className={`flex items-center gap-1 ${getSignalColor(weightedSignal)}`}>
                {getSignalIcon(weightedSignal)}
                <span className="font-semibold">{getSignalText(weightedSignal)}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">신뢰도:</span>
              <span className="font-semibold text-blue-600">
                {(weightedConfidence * 100).toFixed(1)}%
              </span>
            </div>

            {/* 신뢰도 바 */}
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all"
                style={{ width: `${weightedConfidence * 100}%` }}
              />
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-xs text-gray-600">
              RSI, MACD, Bollinger Bands 등 6개 기술적 지표 종합
            </p>
          </div>
        </div>

        {/* AI 기반 전략 */}
        <div className={`p-4 rounded-lg border-2 border-purple-200 ${getBackgroundColor(aiSignal || 'neutral')}`}>
          <div className="flex items-center gap-2 mb-3">
            <Brain className="w-5 h-5 text-purple-600" />
            <span className="font-semibold text-gray-900">AI 기반</span>
            <span className="text-xs text-gray-500">(40%)</span>
          </div>

          {aiSignal && aiConfidence ? (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">신호:</span>
                  <div className={`flex items-center gap-1 ${getSignalColor(aiSignal)}`}>
                    {getSignalIcon(aiSignal)}
                    <span className="font-semibold">{getSignalText(aiSignal)}</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">신뢰도:</span>
                  <span className="font-semibold text-purple-600">
                    {(aiConfidence * 100).toFixed(1)}%
                  </span>
                </div>

                {/* 신뢰도 바 */}
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all"
                    style={{ width: `${aiConfidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-xs text-gray-600">
                  LSTM 딥러닝 모델 기반 가격 예측
                </p>
              </div>
            </>
          ) : (
            <div className="text-center py-6">
              <p className="text-sm text-gray-500">AI 신호 없음</p>
              <p className="text-xs text-gray-400 mt-1">
                AI 모델 학습 필요
              </p>
            </div>
          )}
        </div>

        {/* 최종 결합 신호 */}
        <div className={`p-4 rounded-lg border-2 border-green-200 ${getBackgroundColor(finalSignal)}`}>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="font-semibold text-gray-900">최종 신호</span>
            <span className="text-xs text-gray-500">(결합)</span>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">신호:</span>
              <div className={`flex items-center gap-1 ${getSignalColor(finalSignal)}`}>
                {getSignalIcon(finalSignal)}
                <span className="font-semibold text-lg">{getSignalText(finalSignal)}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">신뢰도:</span>
              <span className="font-semibold text-green-600 text-lg">
                {(finalConfidence * 100).toFixed(1)}%
              </span>
            </div>

            {/* 신뢰도 바 */}
            <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all"
                style={{ width: `${finalConfidence * 100}%` }}
              />
            </div>

            {/* 점수 게이지 */}
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-500">신호 점수</span>
                <span className="font-semibold">{finalScore.toFixed(2)}</span>
              </div>
              <div className="relative w-full bg-gray-200 rounded-full h-3">
                {/* 중심선 */}
                <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-400" />
                {/* 점수 바 */}
                <div
                  className={`absolute h-3 rounded-full transition-all ${
                    finalScore > 0 ? 'bg-green-500' : finalScore < 0 ? 'bg-red-500' : 'bg-gray-400'
                  }`}
                  style={{
                    width: `${Math.abs(finalScore) * 50}%`,
                    left: finalScore < 0 ? `${50 - Math.abs(finalScore) * 50}%` : '50%'
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>-1.0 (매도)</span>
                <span>0</span>
                <span>+1.0 (매수)</span>
              </div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-xs text-gray-600">
              가중치 60% + AI 40% 결합
            </p>
          </div>
        </div>
      </div>

      {/* 차이점 설명 */}
      <div className="bg-blue-50 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-blue-900 mb-2">
          📊 전략 차이점
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-blue-800">
          <div>
            <strong>가중치 기반 (60%):</strong>
            <ul className="list-disc list-inside text-xs mt-1 space-y-1">
              <li>여러 기술적 지표 종합 분석</li>
              <li>검증된 전통적 방법</li>
              <li>Admin에서 가중치 조정 가능</li>
              <li>신뢰성이 높고 안정적</li>
            </ul>
          </div>
          <div>
            <strong>AI 기반 (40%):</strong>
            <ul className="list-disc list-inside text-xs mt-1 space-y-1">
              <li>딥러닝으로 가격 패턴 학습</li>
              <li>복잡한 패턴 인식</li>
              <li>데이터 많을수록 정확</li>
              <li>학습 필요 (초기 신뢰도 낮음)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* 신호 일치도 */}
      {aiSignal && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-900 mb-2">
            🎯 신호 일치도
          </h4>
          <div className="flex items-center gap-4">
            <div className="flex-1">
              {weightedSignal === aiSignal ? (
                <div className="flex items-center gap-2 text-green-600">
                  <div className="w-3 h-3 bg-green-500 rounded-full" />
                  <span className="font-semibold">일치</span>
                  <span className="text-sm">
                    두 전략이 같은 방향을 제시합니다
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-yellow-600">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full" />
                  <span className="font-semibold">불일치</span>
                  <span className="text-sm">
                    두 전략의 의견이 다릅니다
                  </span>
                </div>
              )}
            </div>
            <div className="text-xs text-gray-500">
              {weightedSignal === aiSignal
                ? '신뢰도가 높아집니다'
                : '신중한 판단이 필요합니다'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
