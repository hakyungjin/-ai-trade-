import { useEffect, useState } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { aiApi, marketApi } from '../api/client';
import { TrendingUp, TrendingDown, Minus, RefreshCw, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { SpotTrading } from './trading/SpotTrading';
import { FuturesTrading } from './trading/FuturesTrading';
import { PriceChart } from './market';

export function Dashboard() {
  const {
    currentSignal,
    selectedSymbol,
    setCurrentSignal,
    setSelectedSymbol,
  } = useTradingStore();

  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [tradingMode, setTradingMode] = useState<'spot' | 'futures'>('spot');
  const [weightedAnalysis, setWeightedAnalysis] = useState<any>(null);

  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];

  const fetchData = async () => {
    setLoading(true);
    try {
      const [tickerRes, signalRes, weightedRes] = await Promise.all([
        marketApi.getTicker(selectedSymbol),
        aiApi.predict(selectedSymbol),
        aiApi.combinedAnalysis(selectedSymbol),
      ]);

      console.log('Ticker Response:', tickerRes);
      console.log('Signal Response:', signalRes);
      console.log('Weighted Analysis:', weightedRes);

      // 마켓 데이터 처리
      if (tickerRes?.data) {
        const tickerData = tickerRes.data.data || tickerRes.data;
        console.log('Ticker Data:', tickerData);
        if (tickerData?.price !== undefined) {
          setCurrentPrice(tickerData.price);
          setPriceChange(tickerData.priceChangePercent || 0);
        } else {
          console.warn('Price not found in ticker data');
        }
      }
      
      // AI 신호 데이터 처리
      if (signalRes?.data) {
        console.log('Signal Data:', signalRes.data);
        setCurrentSignal(signalRes.data);
      }

      // 가중치 분석 데이터 처리
      if (weightedRes?.data) {
        console.log('Weighted Analysis Data:', weightedRes.data);
        setWeightedAnalysis(weightedRes.data);
      }
    } catch (error) {
      console.error('데이터 조회 실패:', error);
      if (error instanceof Error) {
        console.error('Error message:', error.message);
      }
      // 기본값 설정
      setCurrentPrice(0);
      setPriceChange(0);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('Fetching data for symbol:', selectedSymbol);
    // 심볼 변경 시 이전 상태 초기화
    setCurrentPrice(0);
    setPriceChange(0);
    
    // 즉시 데이터 조회
    fetchData();
    
    // 10초마다 업데이트 (더 자주)
    const interval = setInterval(() => {
      console.log('Auto-fetching data for:', selectedSymbol);
      fetchData();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const getSignalIcon = () => {
    if (!currentSignal) return <Minus className="w-6 h-6 text-muted-foreground" />;
    switch (currentSignal.signal) {
      case 'BUY':
        return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'SELL':
        return <TrendingDown className="w-6 h-6 text-red-500" />;
      default:
        return <Minus className="w-6 h-6 text-muted-foreground" />;
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header & Trading Mode Toggle */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h1 className="text-2xl font-bold">대시보드</h1>
        <div className="flex items-center gap-3">
          {/* Trading Mode Toggle */}
          <Tabs value={tradingMode} onValueChange={(value: any) => setTradingMode(value)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="spot" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white">
                현물 (Spot)
              </TabsTrigger>
              <TabsTrigger value="futures" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">
                선물 (Futures)
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <Button
            onClick={fetchData}
            disabled={loading}
            variant="outline"
            className="min-h-[44px] gap-2"
          >
            <RefreshCw className={cn('w-4 h-4', loading && 'animate-spin')} />
            새로고침
          </Button>
        </div>
      </div>

      {/* Symbol Selector */}
      <Tabs value={selectedSymbol} onValueChange={setSelectedSymbol} className="w-full">
        <TabsList className="w-full h-auto flex-wrap justify-start gap-1 bg-transparent p-0">
          {symbols.map((symbol) => (
            <TabsTrigger
              key={symbol}
              value={symbol}
              className="min-h-[44px] px-4 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              {symbol.replace('USDT', '')}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Stats Card - 현재가만 표시 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            현재가
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="text-2xl md:text-3xl font-bold">
              ${currentPrice.toLocaleString()}
            </div>
            <Badge
              variant={priceChange >= 0 ? 'default' : 'destructive'}
              className="text-sm"
            >
              {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
            </Badge>
          </div>
          <Badge variant="outline" className="mt-2">{selectedSymbol}</Badge>
        </CardContent>
      </Card>

      {/* 가중치 기반 신호 카드 */}
      {weightedAnalysis && (
        <Card className="border-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              📊 가중치 기반 분석 (AI + 기술적 지표)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
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
                  AI 예측
                </div>
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
              </div>
            )}

            {/* 기술적 지표 신호 */}
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
                {weightedAnalysis.weighted_signal.recommendation && (
                  <p className="text-xs text-muted-foreground mt-2 italic">
                    💡 {weightedAnalysis.weighted_signal.recommendation}
                  </p>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* 실시간 차트 */}
      <PriceChart symbol={selectedSymbol} />

      {/* AI Signal Card */}
      <Card className={cn(
        'border-2',
        currentSignal?.signal === 'BUY' && 'border-green-500/50 bg-green-500/5',
        currentSignal?.signal === 'SELL' && 'border-red-500/50 bg-red-500/5',
        (!currentSignal || currentSignal.signal === 'HOLD') && 'border-border'
      )}>
        <CardContent className="p-4 md:p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={cn(
                'p-3 rounded-full',
                currentSignal?.signal === 'BUY' && 'bg-green-500/20',
                currentSignal?.signal === 'SELL' && 'bg-red-500/20',
                (!currentSignal || currentSignal.signal === 'HOLD') && 'bg-muted'
              )}>
                {getSignalIcon()}
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">AI 신호</div>
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'text-2xl font-bold',
                    currentSignal?.signal === 'BUY' && 'text-green-500',
                    currentSignal?.signal === 'SELL' && 'text-red-500'
                  )}>
                    {currentSignal?.signal || 'HOLD'}
                  </span>
                  {currentSignal?.signal === 'BUY' && (
                    <Badge className="bg-green-500">매수</Badge>
                  )}
                  {currentSignal?.signal === 'SELL' && (
                    <Badge className="bg-red-500">매도</Badge>
                  )}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-muted-foreground mb-1">신뢰도</div>
              <div className="text-xl font-bold">
                {currentSignal
                  ? `${(currentSignal.confidence * 100).toFixed(1)}%`
                  : '-'}
              </div>
              {currentSignal && (
                <Progress
                  value={currentSignal.confidence * 100}
                  className="w-24 h-2 mt-2"
                />
              )}
            </div>
          </div>
          {currentSignal?.analysis && (
            <div className="mt-4 p-3 bg-background/50 rounded-lg border">
              <div className="text-sm text-muted-foreground">{currentSignal.analysis}</div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trading Mode Specific Components */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {tradingMode === 'spot' ? <SpotTrading /> : <FuturesTrading />}
        </div>

        {/* Side Panel: Chart Preview & Info */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                {tradingMode === 'spot' ? (
                  <>
                    <Badge variant="outline" className="bg-blue-500/20 text-blue-600">현물</Badge>
                    기본 정보
                  </>
                ) : (
                  <>
                    <Badge variant="outline" className="bg-orange-500/20 text-orange-600">선물</Badge>
                    위험도 정보
                  </>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {tradingMode === 'spot' ? (
                <>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">심볼</div>
                    <div className="font-semibold">{selectedSymbol}</div>
                  </div>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">현재가</div>
                    <div className="font-semibold text-lg">${currentPrice.toLocaleString()}</div>
                  </div>
                  <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">거래 방식</div>
                    <div className="font-semibold">즉시 현물 거래</div>
                    <div className="text-xs text-muted-foreground mt-1">보유 자산 범위 내</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">레버리지 주의</div>
                    <div className="font-semibold text-sm">1배 ~ 20배</div>
                    <div className="text-xs text-muted-foreground mt-1">높을수록 리스크 ↑</div>
                  </div>
                  <div className="p-3 bg-red-500/10 border border-red-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">청산 위험</div>
                    <div className="font-semibold text-sm">증증금 부족시</div>
                    <div className="text-xs text-muted-foreground mt-1">자동 포지션 청산</div>
                  </div>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">필수 설정</div>
                    <div className="font-semibold text-sm">스탑로스 & 익절</div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
