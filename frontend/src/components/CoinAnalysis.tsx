import { useState, useEffect, useRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  BarChart3,
  Sparkles,
  Plus,
  Search,
  Loader,
  X,
  ChevronRight,
  Coins,
  LineChart,
  Brain,
  Target,
  Activity,
  Award,
  AlertCircle,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { marketApi, aiApi, apiClient, feedbackApi, modelApi } from '@/api/client';
import { PriceChart } from './market';
import { useTradingStore } from '@/store/tradingStore';

type MarketType = 'spot' | 'futures';

interface MonitoredCoin {
  id: number;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  market_type: MarketType;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  candle_count: number;
}

interface WeightedAnalysis {
  final_signal: 'BUY' | 'SELL' | 'HOLD';
  final_confidence: number;
  weighted_signal?: {
    signal: string;
    score: number;
    confidence: number;
    indicators?: Record<string, any>;
    recommendation?: string;
  };
  ai_prediction?: {
    signal: string;
    confidence: number;
    analysis?: string;
  };
}

// interface ModelInfo {
//   name: string;
//   type: 'XGBoost' | 'LSTM' | 'Ensemble';
//   classes: number;
//   features: number;
//   lastUpdated?: string;
// }

interface ModelAccuracy {
  period_days: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  ai_predictions: number;
  ai_correct: number;
  ai_accuracy: number;
  total_pnl: number;
  avg_pnl_percent: number;
}

interface SignalStats {
  BUY: { count: number; wins: number; win_rate: number; total_pnl: number; avg_pnl: number };
  SELL: { count: number; wins: number; win_rate: number; total_pnl: number; avg_pnl: number };
  HOLD: { count: number; wins: number; win_rate: number; total_pnl: number; avg_pnl: number };
}

export function CoinAnalysis() {
  // 전역 상태에서 선택된 심볼과 마켓 타입 가져오기
  const { 
    selectedSymbol, 
    setSelectedSymbol, 
    selectedMarketType: marketType, 
    setSelectedMarketType: setMarketType 
  } = useTradingStore();
  
  // 기본 메이저 코인
  const defaultSpotSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  const defaultFuturesSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  
  const [monitoredCoins, setMonitoredCoins] = useState<MonitoredCoin[]>([]);
  const [spotSymbols, setSpotSymbols] = useState<string[]>(defaultSpotSymbols);
  const [futuresSymbols, setFuturesSymbols] = useState<string[]>(defaultFuturesSymbols);
  
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  
  const [weightedAnalysis, setWeightedAnalysis] = useState<WeightedAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  
  // 코인 추가 모달
  const [showAddModal, setShowAddModal] = useState(false);
  const [searchSymbol, setSearchSymbol] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [addLoading, setAddLoading] = useState(false);

  // 모델 성능 탭
  const [activeTab, setActiveTab] = useState<'analysis' | 'model'>('analysis');
  const [modelAccuracy, setModelAccuracy] = useState<ModelAccuracy | null>(null);
  const [signalStats, setSignalStats] = useState<SignalStats | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [modelLoading, setModelLoading] = useState(false);
  // const [modelExists, setModelExists] = useState<boolean | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<{status: string; step?: string; progress?: number} | null>(null);
  const [isTraining, setIsTraining] = useState(false);

  // 모니터링 코인 로드
  const loadMonitoredCoins = async () => {
    try {
      const response = await apiClient.get('/v1/coins/monitoring');
      const coinsData = response.data.coins || response.data.data || [];
      setMonitoredCoins(coinsData);
      
      // 심볼 목록 업데이트 (마켓 타입별 분리)
      const spotCoins = coinsData.filter((c: MonitoredCoin) => c.market_type === 'spot').map((c: MonitoredCoin) => c.symbol);
      const futuresCoins = coinsData.filter((c: MonitoredCoin) => c.market_type === 'futures').map((c: MonitoredCoin) => c.symbol);
      
      setSpotSymbols([...new Set([...defaultSpotSymbols, ...spotCoins])]);
      setFuturesSymbols([...new Set([...defaultFuturesSymbols, ...futuresCoins])]);
    } catch (error) {
      console.error('Failed to load monitored coins:', error);
    }
  };

  // 모델 학습 시작
  const handleStartTraining = async () => {
    setIsTraining(true);
    try {
      await modelApi.autoTrain(selectedSymbol, '5m', 10000, marketType);
      // 상태 폴링 시작
      pollTrainingStatus();
    } catch (error) {
      console.error('Failed to start training:', error);
      setIsTraining(false);
    }
  };

  // 학습 상태 폴링
  const pollTrainingStatus = () => {
    const interval = setInterval(async () => {
      try {
        const res = await modelApi.getStatus(selectedSymbol);
        setTrainingStatus(res.data);
        
        if (res.data.status === 'completed' || res.data.status === 'error' || res.data.status === 'idle') {
          clearInterval(interval);
          setIsTraining(false);
          if (res.data.status === 'completed') {
            loadModelPerformance(); // 완료 시 새로고침
          }
        }
      } catch {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 3000);
  };

  // 모델 성능 데이터 로드
  const loadModelPerformance = async () => {
    setModelLoading(true);
    try {
      // 모델 존재 여부 확인
      // try {
      //   const modelCheck = await modelApi.checkModel(selectedSymbol, '5m');
      //   setModelExists(modelCheck.data?.exists || false);
      // } catch {
      //   setModelExists(false);
      // }

      // 사용 가능한 모델 목록
      const modelsRes = await aiApi.getModels();
      setAvailableModels(modelsRes.data.models || []);
      
      // 피드백 기반 정확도
      try {
        const accuracyRes = await feedbackApi.getAccuracy(selectedSymbol, 30);
        // error 필드가 있거나 total_trades가 0이면 null 처리
        if (accuracyRes.data?.error || accuracyRes.data?.total_trades === 0) {
          setModelAccuracy(null);
        } else {
          setModelAccuracy(accuracyRes.data);
        }
      } catch {
        setModelAccuracy(null);
      }
      
      // 신호별 성과
      try {
        const statsRes = await feedbackApi.getStatsBySignal(selectedSymbol, 30);
        setSignalStats(statsRes.data);
      } catch {
        setSignalStats(null);
      }
    } catch (error) {
      console.error('Failed to load model performance:', error);
    } finally {
      setModelLoading(false);
    }
  };

  // 심볼 변경 시 모델 성능 데이터 리셋
  useEffect(() => {
    setModelAccuracy(null);
    setSignalStats(null);
    setAvailableModels([]);
    // setModelExists(null);
    setTrainingStatus(null);
    setIsTraining(false);
    
    // 모델 탭이 열려있으면 바로 로드
    if (activeTab === 'model') {
      loadModelPerformance();
    }
  }, [selectedSymbol]);

  // 탭 변경 시 모델 데이터 로드
  useEffect(() => {
    if (activeTab === 'model') {
      loadModelPerformance();
    }
  }, [activeTab]);

  useEffect(() => {
    loadMonitoredCoins();
  }, []);

  // 마켓 타입 변경 시 심볼 초기화
  useEffect(() => {
    const symbols = marketType === 'spot' ? spotSymbols : futuresSymbols;
    if (!symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0] || 'BTCUSDT');
    }
  }, [marketType, spotSymbols, futuresSymbols]);

  // 요청 ID를 추적해서 레이스 컨디션 방지
  const requestIdRef = useRef(0);

  // 현재가 + 분석 조회 (마켓 타입 반영)
  const fetchData = async (symbol: string, market: MarketType) => {
    const currentRequestId = ++requestIdRef.current;
    
    setLoading(true);
    setAnalysisLoading(true);
    
    try {
      const [tickerRes, analysisRes] = await Promise.all([
        marketApi.getTicker(symbol, market),
        aiApi.combinedAnalysis(symbol, '5m', market).catch((e) => {
          console.warn('분석 조회 실패:', e?.message);
          return null;
        }),
      ]);

      // 요청 ID가 변경되었으면 (다른 심볼이 선택됨) 결과 무시
      if (currentRequestId !== requestIdRef.current) {
        console.log(`[${symbol}] 요청 무시 - 다른 심볼이 선택됨`);
        return;
      }

      // 현재가
      if (tickerRes?.data) {
        const tickerData = tickerRes.data.data || tickerRes.data;
        if (tickerData?.price !== undefined) {
          setCurrentPrice(tickerData.price);
          setPriceChange(tickerData.priceChangePercent || 0);
        }
      }

      // 분석
      if (analysisRes?.data) {
        setWeightedAnalysis(analysisRes.data);
      }
    } catch (error) {
      // 요청 ID가 변경되었으면 에러도 무시
      if (currentRequestId !== requestIdRef.current) return;
      console.error('데이터 조회 실패:', error);
    } finally {
      // 요청 ID가 같을 때만 로딩 상태 해제
      if (currentRequestId === requestIdRef.current) {
        setLoading(false);
        setAnalysisLoading(false);
      }
    }
  };

  useEffect(() => {
    setCurrentPrice(0);
    setPriceChange(0);
    setWeightedAnalysis(null);
    fetchData(selectedSymbol, marketType);
    
    const interval = setInterval(() => fetchData(selectedSymbol, marketType), 15000);
    return () => clearInterval(interval);
  }, [selectedSymbol, marketType]);

  // 심볼 검색
  const searchSymbols = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    setSearchLoading(true);
    try {
      const endpoint = marketType === 'futures' 
        ? `/v1/coins/search/futures?query=${query}&limit=20`
        : `/v1/coins/search/spot?query=${query}&limit=20`;
      
      const response = await apiClient.get(endpoint);
      if (response.data.success) {
        setSearchResults(response.data.symbols || []);
      }
    } catch (error) {
      console.error('심볼 검색 실패:', error);
    } finally {
      setSearchLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (showAddModal && searchSymbol) {
        searchSymbols(searchSymbol);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchSymbol, showAddModal, marketType]);

  // 코인 추가
  const handleAddCoin = async (symbol: string) => {
    setAddLoading(true);
    try {
      await apiClient.post(`/v1/coins/add-monitoring/${symbol}?market_type=${marketType}`);
      setShowAddModal(false);
      setSearchSymbol('');
      setSearchResults([]);
      loadMonitoredCoins();
    } catch (error: any) {
      console.error('코인 추가 실패:', error);
      alert(error.response?.data?.detail || '코인 추가 실패');
    } finally {
      setAddLoading(false);
    }
  };

  // 코인 삭제
  const handleDeleteCoin = async (coinId: number, symbol: string) => {
    if (!confirm(`${symbol}을 삭제하시겠습니까?`)) return;
    
    try {
      await apiClient.delete(`/v1/coins/${coinId}`);
      loadMonitoredCoins();
    } catch (error) {
      console.error('코인 삭제 실패:', error);
    }
  };

  // 해당 심볼이 모니터링 코인인지 확인
  const getMonitoredCoin = (symbol: string) => {
    return monitoredCoins.find((c) => c.symbol === symbol && c.market_type === marketType);
  };

  const isDefaultSymbol = (symbol: string) => {
    return marketType === 'spot' 
      ? defaultSpotSymbols.includes(symbol)
      : defaultFuturesSymbols.includes(symbol);
  };

  const currentSymbols = marketType === 'spot' ? spotSymbols : futuresSymbols;
  const currentMarketCoins = monitoredCoins.filter((c) => c.market_type === marketType);

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="w-7 h-7 text-blue-600" />
            코인 분석
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            기술적 지표와 AI 분석으로 암호화폐를 분석하세요
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => setShowAddModal(true)}
            variant="default"
            size="sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            코인 추가
          </Button>
          <Button
            onClick={() => fetchData(selectedSymbol, marketType)}
            variant="outline"
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={cn('w-4 h-4 mr-2', loading && 'animate-spin')} />
            새로고침
          </Button>
        </div>
      </div>

      {/* 마켓 타입 선택 */}
      <div className="flex gap-2">
        <Button
          variant={marketType === 'spot' ? 'default' : 'outline'}
          onClick={() => setMarketType('spot')}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'spot' && 'bg-blue-600 hover:bg-blue-700'
          )}
        >
          <Coins className="w-5 h-5 mr-2" />
          현물 (Spot)
        </Button>
        <Button
          variant={marketType === 'futures' ? 'default' : 'outline'}
          onClick={() => setMarketType('futures')}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'futures' && 'bg-orange-500 hover:bg-orange-600'
          )}
        >
          <LineChart className="w-5 h-5 mr-2" />
          선물 (Futures)
        </Button>
      </div>

      {/* 심볼 선택 */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-2">
            {currentSymbols.map((symbol) => {
              const monitoredCoin = getMonitoredCoin(symbol);
              const isDefault = isDefaultSymbol(symbol);
              
              return (
                <div
                  key={symbol}
                  className={cn(
                    'flex items-center gap-1 rounded-lg border transition-colors',
                    selectedSymbol === symbol 
                      ? marketType === 'spot'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-orange-500 text-white border-orange-500'
                      : 'bg-background hover:bg-accent border-border'
                  )}
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedSymbol(symbol)}
                    className={cn(
                      'min-w-[70px] h-9',
                      selectedSymbol === symbol && 'text-white hover:bg-transparent hover:text-white'
                    )}
                  >
                    {symbol.replace('USDT', '')}
                  </Button>
                  {monitoredCoin && !isDefault && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        'h-7 w-7 mr-1',
                        selectedSymbol === symbol 
                          ? 'hover:bg-white/20 text-white' 
                          : 'hover:bg-red-100 text-red-600'
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCoin(monitoredCoin.id, symbol);
                      }}
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* 탭 선택 */}
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'analysis' | 'model')}>
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="analysis" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            분석
          </TabsTrigger>
          <TabsTrigger value="model" className="flex items-center gap-2">
            <Brain className="w-4 h-4" />
            모델 성능
          </TabsTrigger>
        </TabsList>

        {/* 분석 탭 */}
        <TabsContent value="analysis" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* 메인 컨텐츠 */}
            <div className="lg:col-span-2 space-y-6">
              {/* 현재가 */}
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm text-muted-foreground flex items-center gap-2">
                        {selectedSymbol}
                        <Badge variant="outline" className={marketType === 'spot' ? 'text-blue-600' : 'text-orange-500'}>
                          {marketType === 'spot' ? '현물' : '선물'}
                        </Badge>
                      </div>
                      <div className="text-3xl font-bold">
                        ${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                      </div>
                    </div>
                    <Badge
                      variant={priceChange >= 0 ? 'default' : 'destructive'}
                      className="text-lg py-1 px-3"
                    >
                      {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* 차트 */}
              <PriceChart symbol={selectedSymbol} marketType={marketType} />
            </div>

        {/* 분석 패널 */}
        <div className="space-y-6">
          {/* 기술적 지표 분석 */}
          <Card className={cn(
            'border-2',
            marketType === 'spot' ? 'border-blue-500/30' : 'border-orange-500/30'
          )}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Sparkles className={marketType === 'spot' ? 'w-4 h-4 text-blue-600' : 'w-4 h-4 text-orange-500'} />
                  기술적 지표 분석
                </span>
                {analysisLoading && (
                  <Loader className="w-4 h-4 animate-spin text-muted-foreground" />
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {weightedAnalysis ? (
                <>
                  {/* 최종 신호 */}
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">최종 신호</span>
                    <Badge
                      variant={
                        weightedAnalysis.final_signal === 'BUY'
                          ? 'default'
                          : weightedAnalysis.final_signal === 'SELL'
                          ? 'destructive'
                          : 'secondary'
                      }
                      className="text-base py-1 px-3"
                    >
                      {weightedAnalysis.final_signal}
                    </Badge>
                  </div>

                  {/* 신뢰도 */}
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">신뢰도</span>
                      <span className="text-sm font-semibold">
                        {(weightedAnalysis.final_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={weightedAnalysis.final_confidence * 100}
                      className="h-2"
                    />
                  </div>

                  {/* AI 예측 */}
                  {weightedAnalysis.ai_prediction && (
                    <div className="pt-2 border-t">
                      <div className="text-xs font-semibold text-muted-foreground mb-2">
                        🤖 AI 예측
                      </div>
                      {weightedAnalysis.ai_prediction.confidence === 0 || 
                       weightedAnalysis.ai_prediction.analysis?.includes('모델이 없습니다') ? (
                        <div className="text-sm text-amber-600 bg-amber-50 dark:bg-amber-900/20 rounded p-2">
                          ⚠️ {selectedSymbol} AI 모델이 없습니다
                          <div className="text-xs text-muted-foreground mt-1">
                            기술적 지표만 참고하세요
                          </div>
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-muted-foreground">신호:</span>
                            <Badge variant="outline" className="ml-1">
                              {weightedAnalysis.ai_prediction.signal}
                            </Badge>
                          </div>
                          <div>
                            <span className="text-muted-foreground">신뢰도:</span>
                            <span className="ml-1 font-semibold">
                              {(weightedAnalysis.ai_prediction.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* 기술적 지표 */}
                  {weightedAnalysis.weighted_signal && (
                    <div className="pt-2 border-t">
                      <div className="text-xs font-semibold text-muted-foreground mb-2">
                        기술적 지표
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">신호:</span>
                          <Badge variant="outline" className="ml-1">
                            {weightedAnalysis.weighted_signal.signal}
                          </Badge>
                        </div>
                        <div>
                          <span className="text-muted-foreground">점수:</span>
                          <span
                            className={cn(
                              'ml-1 font-semibold',
                              weightedAnalysis.weighted_signal.score > 0
                                ? 'text-green-600'
                                : weightedAnalysis.weighted_signal.score < 0
                                ? 'text-red-600'
                                : 'text-gray-600'
                            )}
                          >
                            {weightedAnalysis.weighted_signal.score.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {/* 개별 지표 점수 */}
                      {weightedAnalysis.weighted_signal.indicators && 
                        Object.keys(weightedAnalysis.weighted_signal.indicators).length > 0 && (
                        <div className="mt-3 pt-2 border-t border-dashed">
                          <div className="text-xs font-semibold text-muted-foreground mb-2">
                            개별 지표 점수
                          </div>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            {Object.entries(weightedAnalysis.weighted_signal.indicators).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-muted-foreground">
                                  {key.replace('_score', '').toUpperCase()}:
                                </span>
                                <span
                                  className={cn(
                                    'font-mono',
                                    Number(value) > 0.3 ? 'text-green-600' :
                                    Number(value) < -0.3 ? 'text-red-600' : 'text-gray-500'
                                  )}
                                >
                                  {Number(value).toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {weightedAnalysis.weighted_signal.recommendation && (
                        <p className="text-xs text-muted-foreground mt-2 italic">
                          {weightedAnalysis.weighted_signal.recommendation}
                        </p>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  {analysisLoading ? (
                    <div className="flex flex-col items-center gap-2">
                      <Loader className="w-8 h-8 animate-spin" />
                      <span>분석 로딩 중...</span>
                    </div>
                  ) : (
                    <span>분석 데이터 없음</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* 모니터링 코인 목록 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                {marketType === 'spot' ? (
                  <Coins className="w-4 h-4 text-blue-600" />
                ) : (
                  <LineChart className="w-4 h-4 text-orange-500" />
                )}
                모니터링 중인 {marketType === 'spot' ? '현물' : '선물'} 코인
              </CardTitle>
            </CardHeader>
            <CardContent>
              {currentMarketCoins.length === 0 ? (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  모니터링 중인 코인이 없습니다
                </div>
              ) : (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {currentMarketCoins.map((coin) => (
                    <div
                      key={coin.id}
                      className={cn(
                        'p-2 rounded-lg border flex items-center justify-between cursor-pointer hover:bg-accent transition-colors',
                        selectedSymbol === coin.symbol && (
                          marketType === 'spot' 
                            ? 'bg-blue-500/10 border-blue-500' 
                            : 'bg-orange-500/10 border-orange-500'
                        )
                      )}
                      onClick={() => setSelectedSymbol(coin.symbol)}
                    >
                      <div className="flex items-center gap-2">
                        <div>
                          <div className="font-medium text-sm">{coin.symbol}</div>
                          <div className="text-xs text-muted-foreground">
                            {coin.candle_count} 캔들
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-sm">
                          ${coin.current_price?.toLocaleString(undefined, { maximumFractionDigits: 4 }) || '-'}
                        </div>
                        <div className={cn(
                          'text-xs',
                          (coin.price_change_24h || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {(coin.price_change_24h || 0) >= 0 ? '+' : ''}{(coin.price_change_24h || 0).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        </div>
        </TabsContent>

        {/* 모델 성능 탭 */}
        <TabsContent value="model" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 사용 가능한 모델 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Brain className="w-5 h-5 text-purple-600" />
                  {selectedSymbol} AI 모델
                </CardTitle>
              </CardHeader>
              <CardContent>
                {modelLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader className="w-8 h-8 animate-spin" />
                  </div>
                ) : (
                  <div className="space-y-3">
                    {availableModels.filter(m => 
                      typeof m === 'string' && m.toLowerCase().includes(selectedSymbol.toLowerCase().replace('usdt', ''))
                    ).length > 0 ? (
                      availableModels.filter(m => 
                        typeof m === 'string' && m.toLowerCase().includes(selectedSymbol.toLowerCase().replace('usdt', ''))
                      ).map((model) => (
                        <div key={String(model)} className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {String(model).includes('xgboost') && (
                                <Badge variant="outline" className="bg-green-500/10 text-green-600">XGBoost</Badge>
                              )}
                              {String(model).includes('lstm') && (
                                <Badge variant="outline" className="bg-blue-500/10 text-blue-600">LSTM</Badge>
                              )}
                              <span className="font-mono text-sm">{String(model)}</span>
                            </div>
                            <Badge variant="secondary">활성</Badge>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-center py-8">
                        <AlertCircle className="w-12 h-12 mx-auto mb-3 text-yellow-500" />
                        <p className="text-lg font-medium mb-2">{selectedSymbol} 모델이 아직 없습니다</p>
                        <p className="text-sm text-muted-foreground mb-4">
                          데이터 수집 후 AI 모델을 학습해야 예측이 가능합니다
                        </p>
                        
                        {isTraining ? (
                          <div className="space-y-3">
                            <div className="flex items-center justify-center gap-2">
                              <Loader className="w-5 h-5 animate-spin text-blue-500" />
                              <span className="text-blue-600 font-medium">
                                {trainingStatus?.step || '학습 준비 중...'}
                              </span>
                            </div>
                            {trainingStatus?.progress !== undefined && (
                              <Progress value={trainingStatus.progress} className="w-48 mx-auto" />
                            )}
                          </div>
                        ) : (
                          <Button
                            onClick={handleStartTraining}
                            className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
                          >
                            <Sparkles className="w-4 h-4 mr-2" />
                            자동 학습 시작
                          </Button>
                        )}
                        
                        <div className="mt-4 text-xs text-muted-foreground">
                          약 3-5분 소요 (데이터 수집 → 피처 생성 → 모델 학습)
                        </div>
                      </div>
                    )}
                    
                    {/* 전체 모델 수 */}
                    <div className="pt-3 border-t text-sm text-muted-foreground">
                      전체 모델: {availableModels.length}개
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 실거래 기반 정확도 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Target className="w-5 h-5 text-blue-600" />
                  실거래 기반 성과 (30일)
                </CardTitle>
              </CardHeader>
              <CardContent>
                {modelLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader className="w-8 h-8 animate-spin" />
                  </div>
                ) : modelAccuracy && modelAccuracy.total_trades > 0 ? (
                  <div className="space-y-4">
                    {/* 전체 승률 */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>승률</span>
                        <span className="font-bold">{modelAccuracy.win_rate.toFixed(1)}%</span>
                      </div>
                      <Progress value={modelAccuracy.win_rate} className="h-3" />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>승리: {modelAccuracy.wins}</span>
                        <span>패배: {modelAccuracy.losses}</span>
                      </div>
                    </div>

                    {/* AI 예측 정확도 */}
                    {modelAccuracy.ai_predictions > 0 && (
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>AI 예측 정확도</span>
                          <span className={cn(
                            'font-bold',
                            modelAccuracy.ai_accuracy >= 55 ? 'text-green-600' : 'text-yellow-600'
                          )}>
                            {modelAccuracy.ai_accuracy.toFixed(1)}%
                          </span>
                        </div>
                        <Progress 
                          value={modelAccuracy.ai_accuracy} 
                          className={cn(
                            'h-3',
                            modelAccuracy.ai_accuracy >= 55 ? '[&>div]:bg-green-500' : '[&>div]:bg-yellow-500'
                          )} 
                        />
                        <div className="text-xs text-muted-foreground mt-1">
                          {modelAccuracy.ai_correct} / {modelAccuracy.ai_predictions} 예측 적중
                        </div>
                      </div>
                    )}

                    {/* 총 수익 */}
                    <div className="p-3 rounded-lg bg-muted/30">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">총 수익</span>
                        <span className={cn(
                          'font-bold',
                          modelAccuracy.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {modelAccuracy.total_pnl >= 0 ? '+' : ''}{modelAccuracy.total_pnl.toFixed(2)} USDT
                        </span>
                      </div>
                      <div className="flex justify-between text-sm mt-1">
                        <span className="text-muted-foreground">평균 수익률</span>
                        <span>{modelAccuracy.avg_pnl_percent.toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>거래 기록이 없습니다</p>
                    <p className="text-sm mt-1">거래 후 피드백이 기록됩니다</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 신호별 성과 */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Award className="w-5 h-5 text-amber-600" />
                  AI 신호별 성과
                </CardTitle>
              </CardHeader>
              <CardContent>
                {modelLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader className="w-8 h-8 animate-spin" />
                  </div>
                ) : signalStats && (signalStats.BUY || signalStats.SELL || signalStats.HOLD) ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* BUY 신호 */}
                    <div className="p-4 rounded-lg border bg-green-500/5 border-green-500/30">
                      <div className="flex items-center gap-2 mb-3">
                        <TrendingUp className="w-5 h-5 text-green-600" />
                        <span className="font-semibold">BUY 신호</span>
                      </div>
                      {signalStats.BUY?.count > 0 ? (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">거래 수</span>
                            <span className="font-bold">{signalStats.BUY.count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">승률</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.BUY.win_rate >= 50 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.BUY.win_rate.toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">평균 수익</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.BUY.avg_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.BUY.avg_pnl >= 0 ? '+' : ''}{signalStats.BUY.avg_pnl.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-muted-foreground text-sm">데이터 없음</p>
                      )}
                    </div>

                    {/* SELL 신호 */}
                    <div className="p-4 rounded-lg border bg-red-500/5 border-red-500/30">
                      <div className="flex items-center gap-2 mb-3">
                        <TrendingDown className="w-5 h-5 text-red-600" />
                        <span className="font-semibold">SELL 신호</span>
                      </div>
                      {signalStats.SELL?.count > 0 ? (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">거래 수</span>
                            <span className="font-bold">{signalStats.SELL.count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">승률</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.SELL.win_rate >= 50 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.SELL.win_rate.toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">평균 수익</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.SELL.avg_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.SELL.avg_pnl >= 0 ? '+' : ''}{signalStats.SELL.avg_pnl.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-muted-foreground text-sm">데이터 없음</p>
                      )}
                    </div>

                    {/* HOLD 신호 */}
                    <div className="p-4 rounded-lg border bg-gray-500/5 border-gray-500/30">
                      <div className="flex items-center gap-2 mb-3">
                        <Activity className="w-5 h-5 text-gray-600" />
                        <span className="font-semibold">HOLD 신호</span>
                      </div>
                      {signalStats.HOLD?.count > 0 ? (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">거래 수</span>
                            <span className="font-bold">{signalStats.HOLD.count}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">승률</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.HOLD.win_rate >= 50 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.HOLD.win_rate.toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">평균 수익</span>
                            <span className={cn(
                              'font-bold',
                              signalStats.HOLD.avg_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            )}>
                              {signalStats.HOLD.avg_pnl >= 0 ? '+' : ''}{signalStats.HOLD.avg_pnl.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      ) : (
                        <p className="text-muted-foreground text-sm">데이터 없음</p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Award className="w-12 h-12 mx-auto mb-2 opacity-50" />
                    <p>신호별 성과 데이터가 없습니다</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* 모델 개선 안내 */}
            <Card className="lg:col-span-2 bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-500/30">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="p-3 rounded-full bg-purple-500/20">
                    <Sparkles className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-lg mb-2">💡 모델 개선하기</h3>
                    <p className="text-muted-foreground mb-3">
                      거래 결과가 쌓이면 피드백 데이터로 모델을 재학습할 수 있습니다.
                    </p>
                    <div className="bg-muted/50 rounded-lg p-3 font-mono text-sm">
                      <p className="text-muted-foreground"># 피드백 기반 재학습</p>
                      <p>python scripts/train_with_feedback.py \</p>
                      <p className="pl-4">--symbol {selectedSymbol} \</p>
                      <p className="pl-4">--model models/xgboost_{selectedSymbol.toLowerCase()}_5m_v4.joblib \</p>
                      <p className="pl-4">--output models/xgboost_{selectedSymbol.toLowerCase()}_5m_v5.joblib</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* 코인 추가 모달 */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>코인 추가</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    setShowAddModal(false);
                    setSearchSymbol('');
                    setSearchResults([]);
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* 마켓 타입 선택 */}
              <div className="flex gap-2">
                <Button
                  variant={marketType === 'spot' ? 'default' : 'outline'}
                  className={cn('flex-1', marketType === 'spot' && 'bg-blue-600 hover:bg-blue-700')}
                  onClick={() => {
                    setMarketType('spot');
                    setSearchResults([]);
                  }}
                >
                  <Coins className="w-4 h-4 mr-2" />
                  현물
                </Button>
                <Button
                  variant={marketType === 'futures' ? 'default' : 'outline'}
                  className={cn('flex-1', marketType === 'futures' && 'bg-orange-500 hover:bg-orange-600')}
                  onClick={() => {
                    setMarketType('futures');
                    setSearchResults([]);
                  }}
                >
                  <LineChart className="w-4 h-4 mr-2" />
                  선물
                </Button>
              </div>

              {/* 검색 */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="심볼 검색 (예: BTC, ETH)"
                  value={searchSymbol}
                  onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                  className="pl-10"
                />
              </div>

              {/* 검색 결과 */}
              {searchLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader className="w-6 h-6 animate-spin" />
                </div>
              ) : searchResults.length > 0 ? (
                <div className="max-h-[300px] overflow-y-auto space-y-1">
                  {searchResults.map((result) => (
                    <Button
                      key={result.symbol}
                      variant="ghost"
                      className="w-full justify-between"
                      disabled={addLoading}
                      onClick={() => handleAddCoin(result.symbol)}
                    >
                      <span className="font-medium">{result.symbol}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          ${result.price?.toLocaleString()}
                        </span>
                        <span className={cn(
                          'text-xs',
                          result.priceChangePercent >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {result.priceChangePercent >= 0 ? '+' : ''}{result.priceChangePercent?.toFixed(2)}%
                        </span>
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </Button>
                  ))}
                </div>
              ) : searchSymbol && !searchLoading ? (
                <div className="text-center py-4 text-muted-foreground">
                  검색 결과가 없습니다
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
